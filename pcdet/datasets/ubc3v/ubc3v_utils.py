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
from matplotlib.colors import rgb_to_hsv, Normalize
from sklearn.metrics import pairwise_distances


def depth_transform(depth):
    min_, max_ = 50, 800
    zz = 1.03 * ((max_ - min_) * (depth / 255) + min_).astype(np.float32)
    return zz


def points_transform(points, rotation, translation):
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
    new_points = rot.apply(points).astype(np.float32)
    new_points += translation.reshape((1,-1))
    new_points = new_points/100
    new_points = new_points[:, [2, 0, 1]]
    return new_points


def get_angle(joints):
    names = list(get_joints_name().keys())
    if len(joints.shape) == 2:
        joints = joints[None, :, :]
    center = joints[:, names.index('Hip')]
    dist = joints[:, names.index('LHip')] - joints[:, names.index('RHip')]
    angle = np.arctan(dist[:, 1]/dist[:, 0])
    angle = angle + (angle < 0)*2*np.pi
    forward = np.abs(center[:, 0]) >= np.abs(center[:, 1])
    direction = -1*(angle <= np.pi/2) + 1*(angle > np.pi/2)
    x, y = np.sign(center[:, 0])*np.pi/2, np.sign(center[:, 1])*np.pi/2
    angle = angle + forward*direction*x + ~forward*y
    angle = angle + (angle < 0)*2*np.pi - (angle >= 2*np.pi)*2*np.pi
    return angle


def get_angle2(joints, right=False, plot=False):
    names = list(get_joints_name().keys())
    if len(joints.shape) == 2:
        joints = joints[None, :, :]
    center = joints[:, names.index('Hip')].copy()
    rhip = joints[:, names.index('RHip')].copy()
    lhip = joints[:, names.index('LHip')].copy()
    rhip -= center
    lhip -= center
    rhip[:, 2] = 1
    lhip[:, 2] = 1
    dist = np.cross(rhip, lhip) if right else np.cross(lhip, rhip)
    angle = np.arctan2(dist[:, 1], dist[:, 0]) # y / x
    angle = angle + (angle < 0)*2*np.pi # [0, 2pi]
    if plot:
        origin = np.zeros((2,3))
        dest = np.concatenate([rhip, lhip, dist])[:, :2].T
        plt.quiver(*origin, *dest, color=['r','b','g'], scale=0.75)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return angle


def get_bouding_box(points, joints):
    names = list(get_joints_name().keys())
    angle = get_angle2(joints)
    rot = Rotation.from_euler('z', angle)
    joint_center = joints[names.index('Hip')].copy()    
    new_points = points - joint_center
    new_points = rot.apply(new_points, inverse=True).astype(np.float32)
    max_, min_ = new_points.max(0), new_points.min(0)
    lwh = max_ - min_
    center = (max_ + min_) / 2
    center = rot.apply(center[None]).astype(np.float32)[0] + joint_center
    box3d = np.concatenate([center, lwh, angle], 
                           axis=0).astype(np.float32)
    return box3d
    

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
                        for key in names.values()]) for joints in posture])
    posture = posture/100 # cm to m
    posture = posture[..., [2, 0, 1]]
    
    """
    # Fix joint side   
    names = list(names.keys())
    center = posture[:, names.index('Hip')].copy()
    rhip = posture[:, names.index('RHip')].copy()
    angle = get_angle2(posture)
    for i in range(len(angle)):
        rot = Rotation.from_euler('z', -angle[i])
        #posture[i] -= center[i]
        #posture[i] = rot.apply(posture[i]).astype(np.float32)
        rhip[i] -= center[i]
        rhip[i] = rot.apply(rhip[i]).astype(np.float32)
    
    angle = np.arctan2(rhip[:, 1], rhip[:, 0]) # y / x
    angle = angle + (angle < 0)*2*np.pi # [0, 2pi]
    np.sum(np.sin(angle) < 0)     
    
    right_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    left_index = [0, 1, 2, 3, 4, 5, 7, 6, 10, 11, 8, 9, 15, 16, 17, 12, 13, 14]
    #joint_index = np.stack([right_index if x < 0 else left_index for x in np.sin(angle)])
    joint_index = np.tile(left_index, (len(posture), 1))
    index = np.mgrid[:len(posture), :len(names)][0]
    posture = posture[index, joint_index]
    """

    annos['Index'] = [[annos[key][i]['idx'] for key in annos.keys()][0] 
                      for i in range(len(posture))]
    annos['Posture'] = posture
    annos['Label'] = ['Pedestrian' for _ in range(len(posture))]
    annos = [{key:annos[key][i] for key in annos.keys()} for i in range(len(posture))]
    return annos


def get_mapper(base_path):
    base_path = Path(base_path)
    mapper = np.load(base_path / 'mapper.npy')
    return mapper


def get_points(anno, mapper):
    points = np.zeros((0, 7), dtype=np.float32)
    for cam in [c for c in anno.keys() if 'Cam' in c]:
        anno_cam = anno[cam]
        depth_file  = Path(anno_cam['depth_file'])
        class_file  = Path(anno_cam['class_file'])
        translation = anno_cam['translation']
        rotation    = anno_cam['rotation']
        x_map, y_map = mapper[..., 0].reshape(-1), mapper[..., 1].reshape(-1)
        
        frame = np.array(Image.open(depth_file).convert('L'))
        colors = np.array(Image.open(class_file))
        mask = colors.sum(-1) != 0
        #plt.imshow(frame, cmap='gray'); plt.show()
        #plt.imshow(colors, cmap='gray'); plt.show()
        mask = mask.reshape(-1)
        frame = frame.reshape(-1)[mask]
        colors = colors.reshape((-1, 4))[mask] 
        zz = depth_transform(frame)
        xx, yy = x_map[mask]*zz, y_map[mask]*zz 
        cam_points = np.stack([xx, yy, zz], axis=-1)
        cam_labels = (colors / 255).astype(np.float32)
        cam_points = points_transform(cam_points, rotation, translation)        
        cam_points = np.concatenate([cam_points, cam_labels], -1)
        points = np.concatenate([points, cam_points], 0)
    return points    


def get_color_maps(cmap='hsv', plot=False, **kwargs):
    cmap = 'hsv' if cmap == 'src' else cmap
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
    
    #"""
    colors_dict = OrderedDict()
    colors_dict['head'] = [38, 39, 40, 41, 42]
    colors_dict['neck'] = [43, 44, 10, 16, 17, 8, 9, 15]
    colors_dict['torso'] = [0, 1, 4, 6]
    colors_dict['hip'] = [3, 7, 2, 5]
    colors_dict['left_leg'] = [19, 21, 23]
    colors_dict['left_foot'] = [25, 27]
    colors_dict['right_leg'] = [18, 20, 22]
    colors_dict['right_foot'] = [24, 26]
    colors_dict['left_shoulder'] = [11, 14]
    colors_dict['left_arm'] = [34, 35, 36]
    colors_dict['left_hand'] = [37, 29]
    colors_dict['right_shoulder'] = [12, 13]
    colors_dict['right_arm'] = [30, 31, 32]
    colors_dict['right_hand'] = [33, 28]
    joint_dict = {0:0, 1:5, 2:13, 3:13, 4:13, 5:17, 6:17, 7:17, 8:21, 9:24, 10:26, 
                  11:29, 12:31, 13:33, 14:36, 15:38, 16:40, 17:43}
    #"""
    
    """
    colors_dict = OrderedDict()
    colors_dict['right_hip'] = [2, 5]
    colors_dict['left_hip'] = [3, 7]
    colors_dict['right_leg'] = [18]
    colors_dict['left_leg'] = [19]
    colors_dict['right_foot'] = [20, 22, 24, 26]
    colors_dict['left_foot'] = [21, 23, 25, 27]
    colors_dict['torso'] = [0, 1, 4, 6, 8, 9, 10, 15, 16, 17, 43, 44]
    colors_dict['head'] = [38, 39, 40, 41, 42]
    colors_dict['right_shoulder'] = [12, 13]
    colors_dict['left_shoulder'] = [11, 14]
    colors_dict['right_arm'] = [30, 31]
    colors_dict['left_arm'] = [34, 35]
    colors_dict['right_hand'] = [32, 33, 28]    
    colors_dict['left_hand'] = [36, 37, 29]
    joint_dict = {'left_hip':1, 'right_hip':2,
                'left_leg':3, 'right_leg':4,
                'left_foot':5, 'right_foot':6,
                'torso':7, 'head':8, 
                'left_shoulder':9, 'right_shoulder':10,
                'left_arm':11, 'right_arm':12,
                'left_hand':13, 'right_hand':14}
    #"""
    
    colors_list = []
    [colors_list.extend(list(x)) for x in colors_dict.values()]
    src_map = src_map[colors_list]
    src_map = (src_map / 255).astype(np.float32)
    lens = [len(x) for x in colors_dict.values()]
    space_range = np.linspace(0, 1, len(colors_dict)+1, endpoint=True)
    #color_space = np.zeros(len(src_map), dtype=np.float32)
    #for i, (part_key, idx_list) in enumerate(colors_dict.items()):
    #    new_color = space_range[i]
    #    for idx in idx_list:
    #        color_space[idx] = new_color
    
    color_space = np.zeros(0, dtype=np.float32)
    for i, len_i in enumerate(lens):
        color_space = np.concatenate([color_space, 
                                      np.repeat(space_range[i+1], len_i)])
    
    part_dict = {key:value 
                 for key, value in zip(colors_dict, np.cumsum([0]+lens)[:-1])}
    [[part_dict.update({v:k}) for v in list_v] for k,list_v in colors_dict.items()]
    cmap = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    dst_map = cmap.to_rgba(color_space)[:, :3].astype(np.float32)
    src_cmap = plt.cm.colors.ListedColormap(src_map)
    src_cmap = plt.cm.ScalarMappable(cmap=src_cmap, norm=Normalize(vmin=0, vmax=1))
    if plot:
        fig, ax = plt.subplots(figsize=(2, 6), layout='constrained')
        cbar = fig.colorbar(src_cmap, cax=ax, orientation='vertical', ticks=np.cumsum(lens)/len(src_map))
        cbar.ax.set_yticklabels(list(colors_dict)) 
        plt.show()
        
        fig, ax = plt.subplots(figsize=(2, 6), layout='constrained')
        cbar = fig.colorbar(cmap, cax=ax, orientation='vertical', ticks=space_range[1:])
        cbar.ax.set_yticklabels(list(colors_dict)) 
        plt.show()
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
        print(dst_map[np.cumsum([0]+lens)[:-1]])
    return src_map, dst_map, color_space, part_dict, joint_dict


def apply_color_map(colors, return_joint_idx=False, **kwargs):    
    use_src = kwargs['cmap'] == 'src'
    src_map, dst_map, color_space, part_dict, joint_dict = get_color_maps(**kwargs)
    distances = pairwise_distances(colors, src_map)
    src_idx = np.argmin(distances, 1)
    min_dist = distances[np.arange(0,len(src_idx)), src_idx]
    dist_mask = min_dist <= 3**0.5/255
    color_map = src_map if use_src else dst_map
    new_colors = color_map[src_idx] * 1.0*dist_mask[:, None]
    src_idx[~dist_mask] = -1
    new_colors = np.concatenate([new_colors, 1+src_idx[:, None]], axis=1).astype(np.float32)
    if return_joint_idx:
        names = np.zeros(len(new_colors), dtype=np.float32)
        for joint_index, part_idx in joint_dict.items():
            part_mask = ~(new_colors[:, :3] != color_map[part_idx]).any(1)
            names[part_mask] = joint_index
          
        new_colors = np.concatenate([new_colors, names[:, None]], axis=1).astype(np.float32)
    
    if kwargs.get('part'):
        part_idx = part_dict[kwargs['part']]
        new_colors[(new_colors[:, :3] != color_map[part_idx]).any(1), :] = 0
    return new_colors


def get_normals(points, colors, joints, threshold=0.20):
    src_map, dst_map, color_space, part_dict, joint_dict = get_color_maps()
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

    min_distances = distances[np.arange(0, len(points)), idx] > threshold
    normals = joints[idx] - points 
    normals[min_distances] = 0
    idx[min_distances] = -1
    return normals, idx
        

def get_joints_name():
    # from matlab example code
    names = OrderedDict()
    old_keys = ('HeadPGX', 'Neck1PGX', 
                'RightShoulderPGX', 'Spine1PGX', 'SpinePGX', 
                'RightUpLegPGX', 
                'LeftLegPGX', 'RightLegPGX', 
                'LeftFootPGX', 'LeftToeBasePGX', 
                'RightFootPGX', 'RightToeBasePGX', 
                'LeftForeArmPGX', 'LeftHandPGX', 'LeftFingerBasePGX',
                'RightForeArmPGX', 'RightHandPGX', 'RightFingerBasePGX')
    new_keys = {0:'Head', 1:'Neck', 
                2:'Spine2', 3:'Spine1', 4:'Spine', 
                5:'Hip', 
                6:'LHip', 7:'RHip', 
                8:'LKnee', 9:'LFoot', 
                10:'RKnee', 11:'RFoot', 
                12:'LShoulder', 13:'LElbow', 14:'LHand', 
                15:'RShoulder', 16:'RElbow', 17:'RHand'}
    
    for old_key, new_key in zip(old_keys, new_keys.values()):
        names[new_key] = old_key
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
    lines = [( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4), ( 4,  5)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([0, 1, 0])
    geometries.append(line_set)
    # left
    lines = [( 5,  6), ( 6,  8), ( 8,  9), ( 1, 12), (12, 13), (13, 14)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([1, 0, 0])
    geometries.append(line_set)
    # right
    lines = [( 5,  7), ( 7, 10), (10, 11), (1, 15), (15, 16), (16, 17)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([0, 0, 1])
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
    #center = joints[5]
    lwh = box3d[3:6]
    angle = box3d[6]
    axis_angles = np.array([0, 0, angle + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    oriented_box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)  
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 255, 0])
    geometries.append(line_set)
    
    forward = 0.5*np.array([np.cos(angle), np.sin(angle), 0], dtype=center.dtype)
    normal = np.stack([center, center + forward])
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(normal), 
                                    lines=o3d.utility.Vector2iVector([(0, 1)]))
    line_set.paint_uniform_color([255, 0, 0])
    #geometries.append(line_set)    
    
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geometries.append(coords)
    o3d.visualization.draw_geometries(geometries, width=1080, height=1080, 
                                      lookat=center, up=[0,0,1], front=[0,-1,0], zoom=0.6)


frame_filter = lambda x: int(''.join(filter(str.isdigit, x.parts[-5]+x.parts[-1])))


def map_files(subset_path, save_path, num_workers=4):
    import concurrent.futures as futures
    mapper = get_mapper(subset_path)
    
    def process_single_scene(sequence_path):
        split = sequence_path.parts[-2]
        annos = get_annos(sequence_path)
        sample_id_list = []
        for i, anno in enumerate(annos):
            print('split: {}; sequence: {}; step {}/{}'.format(split, sequence_path.name,
                                                               i+1, len(annos)))
            points = get_points(anno, mapper)
            sample_idx = anno['Index']
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
    parser.add_argument('--cam', type=list, default=[])
    parser.add_argument('--frame', type=str, default='mayaProject.000452.png')
    args = parser.parse_args()
    
    subset_path = Path(args.base_path) / args.subset_path
    save_path = Path(args.base_path) / 'pose'
    sequence_path = Path(args.base_path) / args.subset_path / args.split_path / args.sequence_path
    mapper = get_mapper(Path(args.base_path) / args.subset_path)
    anno = get_annos(sequence_path, name=args.frame, cams=args.cam)[0]
    points = get_points(anno, mapper)
    points, colors = points[:, :3], points[:, 3:6]
    joints = anno['Posture']
    box3d = get_bouding_box(points, joints)
    #plot_point_cloud(points[:, :3], points[:, 3:], joints)
    colors = apply_color_map(colors, return_joint_idx=True, plot=True, cmap='hsv')
    normals, _ = get_normals(points, colors[:, :3], joints)
    #joint_index = get_joint_index(colors)
    #mask = colors[:, 4] == 14
    #draw_point_cloud(points[mask], colors[mask, :3], normals[mask], joints, box3d)
    draw_point_cloud(points, colors[:, :3], normals, joints, box3d)
    #map_files(subset_path, save_path, num_workers=4)

    #model_points = np.concatenate([points, colors[:, 4, None]], axis=-1)
    #model_points = model_points[model_points[:,3] != 0]
    #model_joints = joints[[5, 7, 6, 10, 8, 11, 9, 1, 0, 15, 12, 16, 13, 17, 14]]
    #np.save('model_points.npy', model_points)
    #np.save('model_joints.npy', model_joints)
