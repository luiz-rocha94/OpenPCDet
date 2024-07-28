# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils


def pearson(x, y):
    x_mean = torch.mean(x, 1).unsqueeze(1)
    y_mean = torch.mean(y, 1).unsqueeze(1)
    x_norm = x - x_mean
    y_norm = y - y_mean
    coef_pearson = torch.sum(x_norm * y_norm, 1) / (torch.sqrt(torch.sum(x_norm ** 2, 1)) * torch.sqrt(torch.sum(y_norm ** 2, 1)))
    #coef_pearson = (coef_pearson + 1) / 2
    torch.nan_to_num(coef_pearson, out=coef_pearson)
    return coef_pearson.unsqueeze(1)


def box_scores(func):
    def wrapper(points, input_boxes, **kwargs):
        points, is_numpy = common_utils.check_numpy_to_torch(points)
        input_boxes, is_numpy = common_utils.check_numpy_to_torch(input_boxes)
        new_axis = True if len(points.shape) == 2 else False
        if is_numpy:
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points[..., 0:3], input_boxes)
            points_indices = (point_indices - 1) # TO DO 
            points = points.unsqueeze(0)
            input_boxes = input_boxes.unsqueeze(0)
        else:
            if new_axis:
                points = points.unsqueeze(0)
                input_boxes = input_boxes.unsqueeze(0)
            
            point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(points[..., 0:3], input_boxes)
        
        output_box = func(points, input_boxes, point_indices, **kwargs)
        
        if new_axis:
            output_box = output_box.squeeze(0)
            points = points.squeeze(0)
            input_boxes = input_boxes.squeeze(0)
        
        torch.nan_to_num(output_box, out=output_box)
        return output_box.numpy() if is_numpy else output_box
    
    return wrapper


@box_scores
def pearson_in_boxes(points, input_boxes, point_indices):
    batch_size, num_objects, _ = input_boxes.shape
    output_box = torch.zeros((batch_size, num_objects), dtype=input_boxes.dtype, device=input_boxes.device)
    for batch_index in range(batch_size):
        for i in range(num_objects):
            output_box[batch_index, i] = points[batch_index, point_indices[batch_index] == i, -1].mean()
    return output_box


def jpe_in_boxes(pred_joints, gt_joints):
    num_objects, num_joints, _ = gt_joints.shape
    output_box = torch.zeros((num_objects, num_joints), dtype=gt_joints.dtype, device=gt_joints.device)
    for i in range(num_objects):
        output_box[i] = torch.linalg.norm(gt_joints[i] - pred_joints[i], dim=-1)
    return output_box

def hue_joint_index(rgb):
    # [0,1]
    #rgb = rgb / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val

    h = torch.zeros_like(max_val)
    s = torch.zeros_like(max_val)
    v = max_val

    # H
    h[max_val == r] = ((60 * ((g - b) / diff) + 360) % 360)[max_val == r]
    h[max_val == g] = ((60 * ((b - r) / diff) + 120) % 360)[max_val == g]
    h[max_val == b] = ((60 * ((r - g) / diff) + 240) % 360)[max_val == b]
    h[min_val == max_val] = 0.0
    h = 1 - h / 360

    space_range = torch.Tensor([0.05714286, 0.1, 
                                0.14, 0.16, 0.18, 0.2, 
                                0.25714287, 0.34285715, 0.4, 
                                0.45714286, 0.54285717, 0.6, 
                                0.6571429 , 0.74285716, 0.8, 
                                0.85714287, 0.94285715, 1.0]).to(h.device)
    h_dist = space_range[None, :] - h[:, None]
    h_dist[h_dist < 0] = 1.0
    _, joint_index = h_dist.min(1)

    return joint_index


def cartesian_to_spherical(xyz):
    shape = xyz.shape
    xyz = xyz.view(-1, 3)
    rho = torch.linalg.norm(xyz, axis=-1)
    theta = torch.atan2(xyz[..., 1], xyz[..., 0]) + torch.pi
    phi = torch.acos(xyz[..., 2] / rho)
    rtp = torch.stack([rho, theta, phi], dim=-1)
    rtp = rtp.view(shape)
    return rtp


def spherical_to_cartesian(rtp):
    shape = rtp.shape
    rtp = rtp.view(-1, 3)
    rtp[..., 1] += torch.pi
    x = rtp[..., 0]*torch.sin(rtp[..., 2])*torch.cos(rtp[..., 1])
    y = rtp[..., 0]*torch.sin(rtp[..., 2])*torch.sin(rtp[..., 1])
    z = rtp[..., 0]*torch.cos(rtp[..., 2])
    xyz = torch.stack([x, y, z], dim=-1)
    xyz = xyz.view(shape)
    return xyz 
    

def color_joint_index(rgb):
    src_map = torch.Tensor([[1.        , 0.41691217, 0.        ], # Head
                            [1.        , 0.83382434, 0.        ], # Neck
                            [0.74926347, 1.        , 0.        ], # Spine2
                            [0.74926347, 1.        , 0.        ], # Spine1
                            [0.74926347, 1.        , 0.        ], # Spine
                            [0.3091895 , 1.        , 0.        ], # Hip
                            [0.3091895 , 1.        , 0.        ], # RHip
                            [0.3091895 , 1.        , 0.        ], # LHip
                            [0.        , 1.        , 0.10772241], # RKnee
                            [0.        , 1.        , 0.524632  ], # RFoot
                            [0.        , 1.        , 0.96470314], # LKnee
                            [0.        , 0.6183849 , 1.        ], # LFoot
                            [0.        , 0.2014727 , 1.        ], # RShoulder
                            [0.21543948, 0.        , 1.        ], # RElbow
                            [0.65551347, 0.        , 1.        ], # RHand
                            [1.        , 0.        , 0.92757434], # LShoulder
                            [1.        , 0.        , 0.5106622 ], # LElbow
                            [1.        , 0.        , 0.09375   ], # LHand
                            ]).to(rgb.device)
    _, joint_index = torch.linalg.norm(rgb.view(-1, 1, 3) - src_map.view(1, -1, 3), axis=-1).min(1)
    joint_index = joint_index.view(-1, 18)
    mask = torch.zeros(joint_index.shape, dtype=torch.bool)
    ids = joint_index[:, 0].clone()
    mask[torch.arange(0, len(ids)), ids] = True
    joint_index[~mask] = -1 # other joints
    joint_index[ids==2, 2:5] = torch.tensor([[2,3,4]], dtype=joint_index.dtype, device=joint_index.device) # torso joints
    joint_index[ids==5, 5:8] = torch.tensor([[5,6,7]], dtype=joint_index.dtype, device=joint_index.device) # hip joints
    mean_mask = torch.zeros(18, dtype=torch.bool)
    mean_mask[joint_index.unique()[1:]] = True
    mean_index = torch.arange(0, 18, dtype=joint_index.dtype, device=joint_index.device)[~mean_mask]
    if len(mean_index):
        joint_index[:, mean_index] = mean_index.view(1, -1)
    joint_index = joint_index.view(-1)
    return joint_index


@box_scores
def pose_estimation(points, input_boxes, point_indices, point_part=None,  point_dist=None):
    batch_size, num_objects, _ = input_boxes.shape
    num_joints = 18
    point_part, is_numpy = common_utils.check_numpy_to_torch(point_part)
    point_dist, is_numpy = common_utils.check_numpy_to_torch(point_dist)
    joint_index = color_joint_index(point_part).view(1, -1, 1)
    point_dist = point_dist.view(1, -1, 1)
    output_box = torch.zeros((batch_size, num_objects, num_joints, 3), dtype=input_boxes.dtype, device=input_boxes.device)
    for batch_index in range(batch_size):
        for i in range(num_objects):
            object_indices = point_indices[batch_index] == i
            box_points = points[batch_index, :, :3]
            box_index = joint_index[batch_index, :, 0]
            box_dist = point_dist[batch_index, :, 0]
            index_mask = torch.logical_and(box_index != -1, box_dist) 
            box_points = box_points[index_mask]
            box_index = box_index[index_mask]
            num_points = index_mask.sum()
            pose = torch.zeros((num_joints, num_points, 3), dtype=points.dtype, device=points.device)
            mask = torch.zeros((num_joints, num_points, 1), dtype=torch.bool, device=points.device)
            pose[box_index, torch.arange(0, num_points)] = box_points
            mask[box_index, torch.arange(0, num_points)] = True
            normalizer = mask.sum(1)
            pose = pose.sum(1) / normalizer
            torch.nan_to_num(pose, out=pose)
            output_box[batch_index, i] = pose

    return output_box    
