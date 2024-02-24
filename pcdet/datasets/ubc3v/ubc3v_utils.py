import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image
from scipy.io import loadmat
import open3d as o3d
import matplotlib.pyplot as plt


def get_annos(base, subset='easy-pose', split='train', seq='150'):
    base_path = Path(base)
    subset_path = base_path / subset
    split_path = subset_path / split
    sequence_path = split_path / seq
    assert sequence_path.exists()
    depth_path = sequence_path / 'images' / 'depthRender'
    class_path = sequence_path / 'images' / 'groundtruth'
    gt_file = sequence_path / 'groundtruth.mat'
    digit_filter = lambda x: int(''.join(filter(str.isdigit, x.name)))
    gt_data = loadmat(gt_file)
    cameras = gt_data['cameras']
    annos = {}
    for cam in cameras.dtype.names:
        depth_files = sorted((depth_path / cam).glob('*.png'), key=digit_filter)
        class_files = sorted((class_path / cam).glob('*.png'), key=digit_filter)
        frames = cameras[cam].flat[0]['frames'].flat[0].squeeze()
        assert len(depth_files) == len(frames)
        annos[cam] = [{'rotation': x['rotate'].flat[0].squeeze().astype(np.float32), 
                       'translation': x['translate'].flat[0].squeeze().astype(np.float32),
                       'depth_file': str(depth_files[i]),
                       'class_file': str(class_files[i])} 
                       for i, x in enumerate(frames)]
    
    posture = gt_data['joints'].squeeze()
    assert len(annos['Cam1']) == len(posture)
    columns = {'HeadPGX': 'Head',
               'Neck1PGX': 'Neck',
               'RightShoulderPGX': 'Spine2',
               'Spine1PGX': 'Spine1',
               'SpinePGX': 'Spine',
               'RightUpLegPGX': 'Hip',
               'RightLegPGX': 'RHip',
               'RightFootPGX': 'RKnee',
               'RightToeBasePGX': 'RFoot',
               'LeftLegPGX': 'LHip',
               'LeftFootPGX': 'LKnee',
               'LeftToeBasePGX': 'LFoot',
               'RightForeArmPGX': 'RShoulder',
               'RightHandPGX': 'RElbow',
               'RightFingerBasePGX': 'RHand',
               'LeftForeArmPGX': 'LShoulder',
               'LeftHandPGX': 'LElbow',
               'LeftFingerBasePGX': 'LHand'}
    annos['Posture'] = [{new_key:joints[old_key].flat[0].squeeze().astype(np.float32)[12:15] 
                         for old_key, new_key in columns.items()} 
                        for joints in posture]

    annos = [{key:annos[key][i] for key in annos.keys()} for i in range(len(posture))]
    mapper = np.load(base_path / 'mapper.npy')
    return annos, mapper


def get_points(anno, mapper, cam='Cam3'):
    camera      = anno[cam]
    depth_file  = camera['depth_file']
    class_file  = camera['class_file']
    translation = camera['translation']
    rotation    = camera['rotation']
    x_map, y_map = mapper[..., 0], mapper[..., 1]
    mask = np.array(Image.open(class_file)).sum(-1) != 0
    
    frame = np.array(Image.open(depth_file).convert('L'))
    #plt.imshow(frame, cmap='gray'); plt.show()
    y, x = frame.shape
    min_, max_ = 50, 800
    zz = 1.03 * ((max_ - min_) * (frame / 255) + min_).astype(np.float32)
    xx, yy = x_map*zz, y_map*zz 
    points = np.stack([xx, yy, zz], axis=-1)
    points = points[mask].reshape((-1, 3))
    
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
    return points    


def get_labels(anno, cam='Cam3'):    
    camera = anno[cam]
    class_file = camera['class_file']
    colors = np.array(Image.open(class_file))
    #plt.imshow(colors); plt.show()
    mask = colors.sum(-1) != 0
    labels = (colors[mask] / 255).reshape((-1, 4)).astype(np.float32)
    return labels


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


def draw_point_cloud(points, labels, joints):
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
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_path', type=str, default='D:/mestrado/OpenPCDet/data/ubc3v')
    parser.add_argument('--subset_path', type=str, default='easy-pose')
    parser.add_argument('--split_path', type=str, default='train')
    parser.add_argument('--sequence_path', type=str, default='150')
    parser.add_argument('--cam', type=str, default='Cam3')
    parser.add_argument('--frame', type=int, default=8)
    args = parser.parse_args()
    
    annos, mapper = get_annos(args.base_path, args.subset_path, 
                              args.split_path, args.sequence_path)
    anno = annos[args.frame]
    labels = get_labels(anno, args.cam)
    points = get_points(anno, mapper, args.cam)
    joints = get_joints(anno)
    #plot_point_cloud(points, labels, joints)
    draw_point_cloud(points, labels, joints)
