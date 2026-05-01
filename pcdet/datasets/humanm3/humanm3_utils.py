import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from pcdet.datasets.ubc3v.ubc3v_utils import get_bouding_box, get_angle2


JOINT_NAMES = [
    'pelvis','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
    'neck','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'
]
UBC3V_ID = [4, 6, 7, 8, 10, 9, 11, 1, 0, 12, 15, 13, 16, 14, 17]

# Conexões do esqueleto
SKELETON = [
    ( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4), ( 4,  5),
    ( 5,  6), ( 6,  8), ( 8,  9), ( 1, 12), (12, 13), (13, 14),
    ( 5,  7), ( 7, 10), (10, 11), (1, 15), (15, 16), (16, 17),
]


def draw_point_cloud(points, poses, bboxes):
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0,0,0])
    geometries.append(pcd)
    
    if len(poses.shape) == 2:
        poses = [poses]
    
    for joints in poses:
        joint_lines, joint_points = create_skeleton(joints)
        geometries.append(joint_lines)
        geometries.append(joint_points)
        for joint in joints:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mesh_sphere.paint_uniform_color([0, 1, 0])
            mesh_sphere.translate(joint)
            geometries.append(mesh_sphere)
    
    if len(bboxes.shape) == 1:
        bboxes = [bboxes]
    
    for box in bboxes:
        box_lines = create_box(box)
        geometries.append(box_lines)    
    
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geometries.append(coords)
    o3d.visualization.draw_geometries(geometries, width=1080, height=1080, 
                                      lookat=[0,0,0], up=[0,0,1], front=[0,-1,0], zoom=0.6)


def map_files(split_path):
    split_path = Path(split_path)
    split = split_path.name
    pcd_files = sorted(split_path.glob('*/pointcloud/*.pcd'), key=lambda x: int(x.parts[-3]+x.stem[2:]))

    lines = [str(pcd_file.relative_to(split_path.parent))+'\n' for pcd_file in pcd_files]
    split_file = split_path.parent / ('%s.txt' % split)

    with open(split_file, 'w') as f:
        f.writelines(lines)


def align_points(input_points, input_pose, target_pose):
    output_points = input_points.copy()
    output_pose = input_pose.copy()
    root = 5
    # root translation
    translation = target_pose[root] - output_pose[root]
    output_pose += translation
    output_points[:, :3] += translation
    
    # root rotation
    output_pose_angle = get_angle2(output_pose)
    target_pose_angle = get_angle2(target_pose)
    rot = Rotation.from_euler('z', target_pose_angle - output_pose_angle)
    local = output_points[:, :3] - output_pose[root]
    local = rot.apply(local)
    output_points[:, :3] = local + output_pose[root]
    local = output_pose - output_pose[root]
    local = rot.apply(local)
    output_pose = local + output_pose[root]
    
    # align
    for start, stop in SKELETON:
        if stop in (6,7):
            continue
        # mask
        stop_mask = output_points[:, -1] == stop
        # vectors
        v_src = output_pose[stop] - output_pose[start]
        v_tgt = target_pose[stop] - target_pose[start]
        # rotation
        rot, _ = Rotation.align_vectors(v_src, v_tgt)
        #rot.apply(v_src, inverse=True)        
        # scale
        scale = np.linalg.norm(v_tgt) / np.linalg.norm(v_src)
        # align points
        local = output_points[stop_mask, :3] - output_pose[start]
        local = rot.apply(local, inverse=True)*1#(1 if stop == 8 else scale)
        output_points[stop_mask, :3] = output_pose[start] + local
        # translation
        translation = target_pose[start] - output_pose[start]
        output_points[stop_mask, :3] += translation   
    
    return output_points, output_pose


def load_data_json(path, pcd_file):
    with open(pcd_file, 'r') as f:
        lines = f.readlines()
        
    points = np.vstack([np.array(line.replace('\n', '').split(' '), dtype=np.float32) 
                        for line in lines[11:]])[:, :3]
    
    
    with open(path) as f:
        data = json.load(f)
    
    if len(data) == 0:
        return None

    index = []
    labels = []
    poses = np.zeros((len(data),18,3), dtype=np.float32)
    for i, (obj_id, obj_joints) in enumerate(data.items()):
        poses[i, UBC3V_ID] = np.array(obj_joints, dtype=np.float32)
        index.append(int(obj_id))
        labels.append('Pedestrian')
    
    poses[:, 5] = poses[:, [6,7]].mean(1)
    poses[:, 2] = poses[:, [1,4]].mean(1)
    poses[:, 3] = poses[:, [2,4]].mean(1)        
    
    dist = np.linalg.norm(points[:, None, None] - poses[None], axis=-1)
    bboxes = np.zeros((len(data),7), dtype=np.float32)
    for i in range(len(data)):
        mask = (dist[:, i] < 0.3).any(1)
        pose_points = np.concatenate([points[mask], poses[i]], axis=0)
        box3d = get_bouding_box(pose_points, poses[i])
        bboxes[i] = box3d
    
    data_json = {'Posture':poses, 'Label':labels, 'ID':index, 'BBox3D': bboxes}
    return data_json


def get_annos(sequence_path, name='*.pcd'):
    sequence_path = Path(sequence_path)
    split, subset_path = sequence_path.parts[-2:]
    split_file = Path(__file__).resolve().parents[3] / 'data' / 'humanm3' / ('%s.txt' % split)    
    pcd_file = sequence_path / 'pointcloud' / name
    if pcd_file.exists():
        pcd_files = [pcd_file]
    else:
        with open(split_file, 'r') as f:
            lines = f.readlines()
        pcd_files = [split_file.parent / line.replace('\n', '') for line in lines 
                     if (subset_path in line and name.replace('*.pcd', '') in line)]
    
    pose_folder = sequence_path / "pose_calib"
    annos = []
    for pcd_file in pcd_files:
        frame_id = pcd_file.stem[-4:]
        pose_file = pose_folder / f"{frame_id}.json"
        if not pose_file.exists():
            continue
        
        data_json = load_data_json(pose_file, pcd_file)
        if data_json is None:
            continue
            
        anno = {'Index':sequence_path.name + frame_id, 
                'pcd_file':str(pcd_file), 'pose_file':str(pose_file)}
        anno.update(data_json)
        annos.append(anno)
    
    return annos


def create_skeleton(joints, color=[0,1,0]):
    angle = get_angle2(joints)[0]
    center = joints[5]
    forward = center + 0.1*np.array([np.cos(angle), np.sin(angle), 0], dtype=center.dtype)
    points = np.concatenate([joints, forward[None]], axis=0)
    lines = SKELETON+[(5,18)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints)
    pcd.paint_uniform_color(color)

    return line_set, pcd


def create_box(box3d):
    center = box3d[0:3]
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
    return line_set


def visualize_sequence(sequence_path, **kwargs):
    annos = get_annos(sequence_path, **kwargs)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 4.0

    for anno in annos:        
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        pcd = o3d.io.read_point_cloud(anno['pcd_file'])
        pcd.paint_uniform_color([0,0,0])

        vis.clear_geometries()
        vis.add_geometry(axis_pcd)
        vis.add_geometry(pcd)

        for joints in anno['Posture']:
            lines, points = create_skeleton(joints)
            vis.add_geometry(lines)
            vis.add_geometry(points)
        
        for box in anno['BBox3D']:
            lines = create_box(box)
            vis.add_geometry(lines)

        vis.get_view_control().set_lookat([0, 0, 1])
        vis.get_view_control().set_up([0, 0, 1])
        vis.get_view_control().set_front([-1, 0, 0])
        vis.get_view_control().set_zoom(2)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        #break
        time.sleep(0.05)
        

    vis.destroy_window()


def visualize_model():
    data_path = Path(r"D:\mestrado\OpenPCDet\data\humanm3")
    input_points = np.load(data_path / 'model_points.npy')
    input_pose = np.load(data_path / 'model_joints.npy')
    target_pose = np.load(data_path / 'target_joints.npy') 
    output_points, output_pose = align_points(input_points, input_pose, target_pose)    

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 4.0
     
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points[:, :3])
    pcd.paint_uniform_color([0,0,0])

    vis.clear_geometries()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd)

    for joints, color in zip([input_pose, target_pose, output_pose], [[1,0,0],[0,1,0], [0,0,1]]):
        lines, points = create_skeleton(joints, color)
        vis.add_geometry(lines)
        vis.add_geometry(points)

    vis.get_view_control().set_lookat([0, 0, 1])
    vis.get_view_control().set_up([0, 0, 1])
    vis.get_view_control().set_front([-1, 0, 0])
    vis.get_view_control().set_zoom(1)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
        

    vis.destroy_window()


if __name__ == "__main__":
    dataset = Path(r"D:\mestrado\OpenPCDet\data\humanm3\test\01")
    #visualize_sequence(dataset, name='001800.pcd')
    visualize_model()
