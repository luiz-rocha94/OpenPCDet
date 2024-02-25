import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from PIL import Image
from scipy.io import loadmat
import open3d as o3d
import matplotlib.pyplot as plt


def get_annos(sequence_path, cams=[], name='*.png'):
    sequence_path = Path(sequence_path)
    assert sequence_path.exists()
    depth_path = sequence_path / 'images' / 'depthRender'
    class_path = sequence_path / 'images' / 'groundtruth'
    gt_file = sequence_path / 'groundtruth.mat'
    digit_filter = lambda x: int(''.join(filter(str.isdigit, x.name if isinstance(x, Path) else x)))
    stop = digit_filter(name) if name != '*.png' else None
    start = stop - 1 if stop else 0
    gt_data = loadmat(gt_file)
    cameras = gt_data['cameras']
    annos = {}
    for cam in cams if cams else cameras.dtype.names:
        depth_files = sorted((depth_path / cam).glob(name), key=digit_filter)
        class_files = sorted((class_path / cam).glob(name), key=digit_filter)
        frames = cameras[cam].flat[0]['frames'].flat[0].squeeze()[start:stop]
        annos[cam] = [{'rotation': x['rotate'].flat[0].squeeze().astype(np.float32), 
                       'translation': x['translate'].flat[0].squeeze().astype(np.float32),
                       'depth_file': str(depth_files[i]),
                       'class_file': str(class_files[i])} 
                       for i, x in enumerate(frames)]
    
    posture = gt_data['joints'].squeeze()[start:stop]
    columns = OrderedDict()
    old_keys = ('HeadPGX', 'Neck1PGX', 'RightShoulderPGX', 'Spine1PGX', 
                'SpinePGX', 'RightUpLegPGX', 'RightLegPGX', 'RightFootPGX', 
                'RightToeBasePGX', 'LeftLegPGX', 'LeftFootPGX', 
                'LeftToeBasePGX', 'RightForeArmPGX', 'RightHandPGX', 
                'RightFingerBasePGX', 'LeftForeArmPGX', 'LeftHandPGX', 
                'LeftFingerBasePGX')
    new_keys = ('Head', 'Neck', 'Spine2', 'Spine1', 'Spine', 'Hip', 'RHip',
                'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'RShoulder',
                'RElbow', 'RHand', 'LShoulder', 'LElbow', 'LHand')
    for old_key, new_key in zip(old_keys, new_keys):
        columns[old_key] = new_key

    annos['Posture'] = []
    annos['BBox3D'] = []
    for joints in posture:
        pose = OrderedDict()
        for old_key, new_key in columns.items():
            pose[new_key] = joints[old_key].flat[0].squeeze().astype(np.float32)[12:15]
        joints = np.stack([pose[key] for key in pose])
        center = joints[5]
        half = joints[6] - joints[9]
        angle = np.arctan(half[2]/half[0])
        angle = angle if angle > 0 else 2*np.pi + angle
        if np.abs(center[0]) >= np.abs(center[2]):
            if angle <= np.pi/2:
                angle -= np.sign(center[0])*np.pi/2
            else:
                angle += np.sign(center[0])*np.pi/2
        else:
            angle += np.sign(center[2])*np.pi/2
        
        rot = Rotation.from_euler('y', -angle)
        joints = rot.apply(joints).astype(np.float32)
        min_, max_ = np.min(joints, 0), np.max(joints, 0)
        whl = max_ - min_
        box3d = np.concatenate([center, whl, [angle, 1]])
        annos['Posture'].append(pose)
        annos['BBox3D'].append(box3d)

    annos = [{key:annos[key][i] for key in annos.keys()} for i in range(len(posture))]
    return annos


def get_mapper(base_path):
    base_path = Path(base_path)
    mapper = np.load(base_path / 'mapper.npy')
    return mapper


def get_points(anno, mapper):
    depth_file  = anno['depth_file']
    class_file  = anno['class_file']
    translation = anno['translation']
    rotation    = anno['rotation']
    x_map, y_map = mapper[..., 0], mapper[..., 1]
    
    frame = np.array(Image.open(depth_file).convert('L'))
    colors = np.array(Image.open(class_file))
    mask = colors.sum(-1) != 0
    #plt.imshow(frame, cmap='gray'); plt.show()
    #plt.imshow(colors, cmap='gray'); plt.show()
    y, x = frame.shape
    min_, max_ = 50, 800
    zz = 1.03 * ((max_ - min_) * (frame / 255) + min_).astype(np.float32)
    xx, yy = x_map*zz, y_map*zz 
    points = np.stack([xx, yy, zz], axis=-1)
    points = points[mask].reshape((-1, 3))
    labels = (colors[mask] / 255).reshape((-1, 4)).astype(np.float32)
    
    rotyx = rotation[0]
    if rotyx > 0:
        rotyx = 180-rotyx
    else:
        rotyx = -rotyx
    rotyy = rotation[1]
    if translation[[0, 2]].prod() > 0:
        if rotyy < 0:
            rotyy = -rotyy
        else:
            rotyy = -180+rotyy
    else:
        if rotyy < 0:
            rotyy = -180+rotyy
        else:
            rotyy = -rotyy
    
    rot = Rotation.from_euler('xyz', [rotyx, rotyy, 0], degrees=True)
    points = rot.apply(points).astype(np.float32)
    points += translation.reshape((1,-1))
    points = np.concatenate([points, labels], -1)
    return points    


def get_joints(anno):
    pose = anno['Posture']
    joints = np.stack([pose[key] for key in pose])
    return joints


def plot_point_cloud(points, labels, joints):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(90, 0, 90)
    ax.scatter(points[:,0], points[:,1], points[:,2],
               s=0.1, c=labels)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def draw_point_cloud(points, labels, joints, box3d):
    geometries = []
    lines = [( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4), ( 4,  5), ( 5,  6), 
             ( 6,  7), ( 7,  8), ( 5,  9), ( 9, 10), (10, 11), (12, 13),
             (13, 14), (15, 16), (16, 17), ( 1, 12), (1, 15)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([0, 1, 0])
    geometries.append(line_set)
    for joint in joints:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.paint_uniform_color([0, 1, 0])
        mesh_sphere.translate(joint)
        geometries.append(mesh_sphere)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(labels[:, :3])
    geometries.append(pcd)
    
    center = box3d[0:3]
    lwh = box3d[3:6]
    angle = box3d[6]
    axis_angles = np.array([0, angle + 1e-10, 0])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    oriented_box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)  
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box3d)
    line_set.paint_uniform_color([0, 255, 0])
    geometries.append(line_set)
    
    forward = 10*np.array([np.cos(angle), 0, np.sin(angle)], dtype=center.dtype)
    normal = np.stack([center, center + forward])
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(normal), 
                                    lines=o3d.utility.Vector2iVector([(0, 1)]))
    line_set.paint_uniform_color([255, 0, 0])
    geometries.append(line_set)    
    
    
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(10)
    geometries.append(coords)
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_path', type=str, default='D:/mestrado/OpenPCDet/data/ubc3v')
    parser.add_argument('--subset_path', type=str, default='easy-pose')
    parser.add_argument('--split_path', type=str, default='train')
    parser.add_argument('--sequence_path', type=str, default='150')
    parser.add_argument('--cam', type=str, default='Cam3')
    parser.add_argument('--frame', type=str, default='mayaProject.000001.png')
    args = parser.parse_args()
    
    sequence_path = Path(args.base_path) / args.subset_path / args.split_path / args.sequence_path
    mapper = get_mapper(args.base_path)
    anno = get_annos(sequence_path, [args.cam], args.frame)[0]
    anno.update(anno.pop(args.cam))
    #labels = get_labels(anno)
    points = get_points(anno, mapper)
    joints = get_joints(anno)
    box3d = anno['BBox3D']
    #plot_point_cloud(points[:, :3], points[:, 3:], joints)
    draw_point_cloud(points[:, :3], points[:, 3:], joints, box3d)
