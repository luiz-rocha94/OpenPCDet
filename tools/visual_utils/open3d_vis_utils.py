"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d as o3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, 
                point_colors=None, normals=None, ref_poses=None, gt_poses=None, joint_colors=[1, 0, 0], 
                draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 4.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        if isinstance(point_colors, torch.Tensor):
            point_colors = point_colors.cpu().numpy()
        if point_colors.shape[1] == 4:
            point_colors = point_colors[:, :3] #* point_colors[:, 3, np.newaxis]
        pts.colors = o3d.utility.Vector3dVector(point_colors)
    
    if normals is not None:
        if isinstance(normals, torch.Tensor):
            normals = normals.cpu().numpy()
        
        if normals.shape != points.shape:
            normals = normals.reshape((-1, 18, 3))
            min_normals = np.linalg.norm(normals, axis=-1).argmin(-1)
            normals = normals[np.arange(len(points)), min_normals]
        norm = o3d.geometry.PointCloud()
        norm.points = o3d.utility.Vector3dVector(points + normals)
        ids = [(i,i) for i in range(len(points))]
        vectors = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pts, norm, ids)
        vectors.paint_uniform_color([1, 0, 0])
        vis.add_geometry(vectors)
    
    if gt_poses is not None:
        vis = draw_poses(vis, gt_poses, [0, 1, 0])
    
    if ref_poses is not None:
        vis = draw_poses(vis, ref_poses, joint_colors)
        
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 1, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 0), ref_labels, ref_scores)
    
    vis.get_view_control().set_lookat([0, 0, 1])
    vis.get_view_control().set_up([0, 0, 1])
    vis.get_view_control().set_front([-1, 0, 0])
    vis.get_view_control().set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()


def translate_boxes_to_o3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_o3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_poses(vis, poses, color=[0, 1, 0]):
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().numpy()
    
    for joints in poses:
        lines = [( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4), ( 4,  5), ( 5,  6), 
                 ( 6,  8), ( 8,  9), ( 5,  7), ( 7, 10), (10, 11), (12, 13),
                 (13, 14), (15, 16), (16, 17), ( 1, 12), (1, 15)]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joints), 
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)
        for i, joint in enumerate(joints):
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mesh_sphere.paint_uniform_color(color[i] if len(color) == len(joints) else color)
            mesh_sphere.translate(joint)
            vis.add_geometry(mesh_sphere)
    
    return vis

