import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


JOINT_NAMES = [
    'pelvis','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
    'neck','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'
]

# Conexões do esqueleto
SKELETON = [
    (0,1),(1,3),(3,5),      # left leg
    (0,2),(2,4),(4,6),      # right leg
    (0,7),(7,8),            # spine
    (7,9),(9,11),(11,13),   # left arm
    (7,10),(10,12),(12,14)  # right arm
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
    
    for box3d in bboxes:
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
        geometries.append(line_set)
        
        forward = 0.5*np.array([np.cos(angle), np.sin(angle), 0], dtype=center.dtype)
        normal = np.stack([center, center + forward])
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(normal), 
                                        lines=o3d.utility.Vector2iVector([(0, 1)]))
        line_set.paint_uniform_color([255, 0, 0])
        geometries.append(line_set)    
    
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geometries.append(coords)
    o3d.visualization.draw_geometries(geometries, width=1080, height=1080, 
                                      lookat=center, up=[0,0,1], front=[0,-1,0], zoom=0.6)


def map_files(split_path):
    split_path = Path(split_path)
    split = split_path.name
    pcd_files = sorted(split_path.glob('*/pointcloud/*.pcd'), key=lambda x: int(x.parts[-3]+x.stem[2:]))

    lines = [str(pcd_file.relative_to(split_path.parent))+'\n' for pcd_file in pcd_files]
    split_file = split_path.parent / ('%s.txt' % split)

    with open(split_file, 'w') as f:
        f.writelines(lines)


def get_angle(pose, right=True, plot=False):
    if len(pose.shape) == 2:
        pose = pose[None, :, :]
    center = pose[:, JOINT_NAMES.index('pelvis')].copy()
    rhip = pose[:, JOINT_NAMES.index('right_hip')].copy()
    lhip = pose[:, JOINT_NAMES.index('left_hip')].copy()
    rhip -= center
    lhip -= center
    rhip[:, 2] = 1
    lhip[:, 2] = 1
    dist = np.cross(rhip, lhip) if right else np.cross(lhip, rhip)
    angle = np.arctan2(dist[:, 1], dist[:, 0]) # y / x
    angle = angle + (angle < 0)*2*np.pi # [0, 2pi]
    origin = np.zeros((2,3))
    dest = np.concatenate([rhip, lhip, dist])[:, :2].T
    if plot:
        plt.quiver(*origin, *dest, color=['r','b','g'], scale=0.75)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return angle


def load_data_json(path):
    with open(path) as f:
        data = json.load(f)
    
    if len(data) == 0:
        return None

    index = []
    poses = []
    labels = []
    bboxes = []
    for obj_id, obj_joints in data.items():
        pose = np.array(obj_joints, dtype=np.float32)
        angle = get_angle(pose, False)
        max_, min_ = pose.max(0), pose.min(0)
        lwh = max_ - min_
        center = min_ + lwh/2
        box3d = np.concatenate([center, lwh, angle], 
                               axis=0).astype(np.float32)
        
        index.append(int(obj_id))
        poses.append(pose)
        labels.append('Pedestrian')
        bboxes.append(box3d)
    poses = np.stack(poses, axis=0)
    bboxes = np.stack(bboxes, axis=0)
    data_json = {'Posture':poses, 'Label':labels, 'ID':index, 'BBox3D': bboxes}
    return data_json


def get_annos(sequence_path, name='*.pcd'):
    sequence_path = Path(sequence_path)
    split, subset_path = sequence_path.parts[-2:]
    split_file = Path(__file__).resolve().parents[3] / 'data' / 'humanm3' / 'm3' / ('%s.txt' % split)    
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
        
        data_json = load_data_json(pose_file)
        if data_json is None:
            continue
            
        anno = {'Index':sequence_path.name + frame_id, 
                'pcd_file':str(pcd_file), 'pose_file':str(pose_file)}
        anno.update(data_json)
        annos.append(anno)
    
    return annos


def create_skeleton(points):

    colors = [[1,0,0] for _ in SKELETON]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(SKELETON)
    )

    line_set.colors = o3d.utility.Vector3dVector(colors)

    joints = o3d.geometry.PointCloud()
    joints.points = o3d.utility.Vector3dVector(points)
    joints.paint_uniform_color([0,1,0])

    return line_set, joints


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


if __name__ == "__main__":
    dataset = Path(r"D:\mestrado\OpenPCDet\data\humanm3\m3\test\01")
    visualize_sequence(dataset, name='001800.pcd')
