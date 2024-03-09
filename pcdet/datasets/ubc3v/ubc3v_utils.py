import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from PIL import Image
from scipy.io import loadmat
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import pairwise_distances


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
                       'class_file': str(class_files[i]),
                       'idx': frame_filter(depth_files[i])} 
                       for i, x in enumerate(frames)]
    
    posture = gt_data['joints'].squeeze()[start:stop]
    names = get_joints_name()
    posture = np.array([np.stack([joints[key].flat[0].squeeze().astype(np.float32)[12:15] 
                        for key in names]) for joints in posture])
    posture[:, :, [1, 2]] = posture[:, :, [2, 1]]
    posture = posture/100
    center = posture[:, 5]
    half = posture[:, 6] - posture[:, 9]
    angle = np.arctan(half[:, 1]/half[:, 0])
    angle = angle + (angle < 0)*2*np.pi
    forward = np.abs(center[:, 0]) >= np.abs(center[:, 1])
    direction = -1*(angle <= np.pi/2) + 1*(angle > np.pi/2)
    x, y = np.sign(center[:, 0])*np.pi/2, np.sign(center[:, 1])*np.pi/2
    angle = angle + forward*direction*x + ~forward*y
    min_, max_ = posture.min(1), posture.max(1)
    whl = max_ - min_
    box3d = np.concatenate([center, whl, angle[:, np.newaxis]], 
                           axis=1).astype(np.float32)
    annos['Posture'] = posture
    annos['Label'] = ['Pedestrian' for _ in range(len(posture))]
    annos['BBox3D'] = box3d
    annos = [{key:annos[key][i] for key in annos.keys()} for i in range(len(posture))]
    return annos


def get_mapper(base_path):
    base_path = Path(base_path)
    mapper = np.load(base_path / 'mapper.npy')
    return mapper


def get_points(anno, mapper):
    depth_file  = Path(anno['depth_file'])
    class_file  = Path(anno['class_file'])
    translation = anno['translation']
    rotation    = anno['rotation']
    x_map, y_map = mapper[..., 0].reshape(-1), mapper[..., 1].reshape(-1)
    
    points = np.zeros((0, 3), dtype=np.float32)
    labels = np.zeros((0, 4), dtype=np.float32)
    
    frame = np.array(Image.open(depth_file).convert('L'))
    colors = np.array(Image.open(class_file))
    mask = colors.sum(-1) != 0
    #plt.imshow(frame, cmap='gray'); plt.show()
    #plt.imshow(colors, cmap='gray'); plt.show()
    mask = mask.reshape(-1)
    frame = frame.reshape(-1)[mask]
    colors = colors.reshape((-1, 4))[mask] 
    min_, max_ = 50, 800
    zz = 1.03 * ((max_ - min_) * (frame / 255) + min_).astype(np.float32)
    xx, yy = x_map[mask]*zz, y_map[mask]*zz 
    points = np.concatenate([points, np.stack([xx, yy, zz], axis=-1)])
    labels = np.concatenate([labels, (colors / 255).astype(np.float32)])
    
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
    points = points/100
    points[:, [1, 2]] = points[:, [2, 1]]
    points = np.concatenate([points, labels], -1)
    return points    


def get_color_maps(cmap='gist_rainbow', plot=False):
    src_map = np.array([[255, 106,   0], 
                        [255,   0,   0],
                        [255, 178, 127],
                        [255, 127, 127],
                        [182, 255,   0],
                        [218, 255, 127],
                        [255, 216,   0],
                        [255, 233, 127],
                        [  0, 148, 255],
                        [ 72,   0, 255],
                        [ 48,  48,  48],
                        [ 76, 255,   0],
                        [  0, 255,  33],
                        [  0, 255, 255],
                        [  0, 255, 144],
                        [178,   0, 255],
                        [127, 116,  63],
                        [127,  63,  63],
                        [127, 201, 255],
                        [127, 255, 255],
                        [165, 255, 127],
                        [127, 255, 197],
                        [214, 127, 255],
                        [161, 127, 255],
                        [107,  63, 127],
                        [ 63,  73, 127],
                        [ 63, 127, 127],
                        [109, 127,  63],
                        [255, 127, 237],
                        [127,  63, 118],
                        [  0,  74, 127],
                        [255,   0, 110],
                        [  0, 127,  70],
                        [127,   0,   0],
                        [ 33,   0, 127],
                        [127,   0,  55],
                        [ 38, 127,   0],
                        [127,  51,   0],
                        [ 64,  64,  64],
                        [ 73,  73,  73],
                        [  0,   0,   0],
                        [191, 168, 247],
                        [192, 192, 192],
                        [127,  63,  63],
                        [127, 116,  63]], np.uint8)
    
    colors_dict = OrderedDict()
    colors_dict['head'] = [38, 39, 40, 41, 42, 43, 44]
    colors_dict['torso'] = [10, 16, 17, 8, 9, 15, 0, 1, 4, 6]
    colors_dict['left_leg'] = [2, 5, 18, 20, 22, 24, 26]
    colors_dict['right_leg'] = [3, 7, 19, 21, 23, 25, 27]
    colors_dict['left_arm'] = [12, 13, 30, 31, 32, 33, 28]
    colors_dict['right_arm'] = [11, 14, 34, 35, 36, 37, 29]
    colors_list = []
    [colors_list.extend(list(x)) for x in colors_dict.values()]
    src_map = src_map[colors_list]
    src_map = (src_map / 255).astype(np.float32)
    lens = [len(x) for x in colors_dict.values()]
    space_range = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    color_space = np.zeros(0, dtype=np.float32)
    for i, len_i in enumerate(lens):
        color_space = np.concatenate([color_space, 
                                      np.linspace(space_range[i], space_range[i+1], len_i, dtype=np.float32, endpoint=False)])
    
    cmap = cm.ScalarMappable(cmap=cmap)
    dst_map = cmap.to_rgba(color_space)[:, :3].astype(np.float32)
    if plot:
        width, height, channels = 450, 50, 3
        rows, cols = np.mgrid[:width, :channels]
        cols = cols.reshape(-1)
        rows = rows.reshape(-1)
        cumsum = 0
        for key, value in colors_dict.items():
            scale = width / len(value)
            scaled_rows = (rows/scale).astype(np.int32) + cumsum
            src_image = src_map[np.newaxis, scaled_rows, cols].reshape((1, -1, 3))
            src_image = np.repeat(src_image, height, 0)
            dst_image = dst_map[np.newaxis, scaled_rows, cols].reshape((1, -1, 3))
            dst_image = np.repeat(dst_image, height, 0)
            cumsum += len(value)
            
            plt.subplot(2, 1, 1)
            plt.imshow(src_image)
            plt.title('UBC3V %s color map' % key)
            plt.subplot(2, 1, 2)
            plt.imshow(dst_image)
            plt.title('VPSPose %s color map' % key)
            plt.show()
    
    return src_map, dst_map, color_space


def apply_color_map(colors, **kwargs):
    src_map, dst_map, color_space = get_color_maps(**kwargs)
    distances = pairwise_distances(colors, src_map)
    idx = np.argmin(distances, 1)
    colors = dst_map[idx]
    return colors


def get_normals(points, colors, joints, threshold=0.25):
    src_map, dst_map, color_space = get_color_maps()
    distances = pairwise_distances(colors, dst_map)
    idx = np.argmin(distances, 1)
    labels = color_space[idx]
    mask = lambda array, min, max : np.bitwise_and(array >= min, array <  max)
    distances = pairwise_distances(points, joints)
    space_range = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    joint_range = [0,   2,   6,   9,  12,  15,  18]
    idx = np.zeros(len(points), dtype=np.int32)
    for i in range(len(space_range) - 1):
        mask_label = mask(labels, space_range[i], space_range[i+1])
        dist_label = distances[mask_label, joint_range[i]:joint_range[i+1]]
        min_label = np.argmin(dist_label, 1) + joint_range[i]
        idx[mask_label] = min_label

    min_distances = distances[np.arange(0, len(points)), idx]
    normals = joints[idx] - points 
    normals[min_distances > threshold] = 0
    return normals
        

def get_joints_name():
    names = OrderedDict()
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
        names[old_key] = new_key
    return names


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


def draw_point_cloud(points, colors, normals, joints, box3d):
    geometries = []
    lines = [( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4), ( 4,  5), ( 5,  6), 
             ( 6,  7), ( 7,  8), ( 5,  9), ( 9, 10), (10, 11), (12, 13),
             (13, 14), (15, 16), (16, 17), ( 1, 12), (1, 15)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([0, 1, 0])
    geometries.append(line_set)
    for joint in joints:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.paint_uniform_color([0, 1, 0])
        mesh_sphere.translate(joint)
        geometries.append(mesh_sphere)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)
    
    norm = o3d.geometry.PointCloud()
    norm.points = o3d.utility.Vector3dVector(points + normals)
    ids = [(i,i) for i in range(len(points))]
    vectors = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, norm, ids)
    geometries.append(vectors)
    
    center = box3d[0:3]
    lwh = box3d[3:6]
    angle = box3d[6]
    axis_angles = np.array([0, 0, angle + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    oriented_box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)  
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box3d)
    line_set.paint_uniform_color([0, 255, 0])
    geometries.append(line_set)
    
    forward = 0.1*np.array([np.cos(angle), np.sin(angle), 0], dtype=center.dtype)
    normal = np.stack([center, center + forward])
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(normal), 
                                    lines=o3d.utility.Vector2iVector([(0, 1)]))
    line_set.paint_uniform_color([255, 0, 0])
    geometries.append(line_set)    
    
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geometries.append(coords)
    o3d.visualization.draw_geometries(geometries, width=1080, height=1080, 
                                      lookat=center, up=[0,0,1], front=[0,1,0], zoom=0.6)


frame_filter = lambda x: int(''.join(filter(str.isdigit, x.parts[-5]+x.parts[-2]+x.parts[-1])))


def map_files(subset_path, save_path, num_workers=4):
    import concurrent.futures as futures
    mapper = get_mapper(subset_path)
    
    def process_single_scene(sequence_path):
        split = sequence_path.parts[-2]
        print('%s sequence: %s' % (split, sequence_path.name))
        cams = ['Cam3']
        annos = get_annos(sequence_path, cams)
        sample_id_list = []
        for anno in annos:
            for cam in cams:
                cam_anno = anno[cam]
                points = get_points(cam_anno, mapper)
                sample_idx = cam_anno['idx']
                points_file = save_path / split / '{}.npy'.format(sample_idx)
                np.save(points_file, points)
                sample_id_list.append(sample_idx)
        return sample_id_list
    
    for split in ['train', 'test', 'valid']:
        split_path = subset_path / split
        (save_path / split).mkdir(exist_ok=True)
        sequences = sorted(split_path.glob('*'), key=lambda x: int(x.name))
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_id_list = executor.map(process_single_scene, sequences)
        
        id_list = []
        for sample_id_list in sequence_id_list:
            id_list.extend(sample_id_list)    
        
        id_list = sorted(id_list)
        with open(save_path / (split+'.txt'), 'w') as f:
            f.writelines('\n'.join(map(str, id_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_path', type=str, default='D:/mestrado/OpenPCDet/data/ubc3v')
    parser.add_argument('--subset_path', type=str, default='easy-pose')
    parser.add_argument('--split_path', type=str, default='train')
    parser.add_argument('--sequence_path', type=str, default='150')
    parser.add_argument('--cam', type=str, default='Cam3')
    parser.add_argument('--frame', type=str, default='mayaProject.000005.png')
    args = parser.parse_args()
    
    subset_path = Path(args.base_path) / args.subset_path
    save_path = Path(args.base_path) / 'pose'
    sequence_path = Path(args.base_path) / args.subset_path / args.split_path / args.sequence_path
    mapper = get_mapper(Path(args.base_path) / args.subset_path)
    anno = get_annos(sequence_path, [args.cam], args.frame)[0]
    anno.update(anno.pop(args.cam))
    points = get_points(anno, mapper)
    points, colors = points[:, :3], points[:, 3:6]
    joints = anno['Posture']
    box3d = anno['BBox3D']
    #plot_point_cloud(points[:, :3], points[:, 3:], joints)
    colors = apply_color_map(colors)
    normals = get_normals(points, colors, joints)
    draw_point_cloud(points, colors, normals, joints, box3d)
