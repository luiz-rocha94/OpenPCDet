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
    coef_pearson = (coef_pearson + 1) / 2
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


@box_scores
def pose_estimation(points, input_boxes, point_indices, joint_index=None):
    batch_size, num_objects, _ = input_boxes.shape
    num_joints = 18
    joint_index, is_numpy = common_utils.check_numpy_to_torch(joint_index)
    joint_index = joint_index.reshape((batch_size, -1, 1))
    output_box = torch.zeros((batch_size, num_objects, num_joints, 3), dtype=input_boxes.dtype, device=input_boxes.device)
    for batch_index in range(batch_size):
        for i in range(num_objects):
            object_indices = point_indices[batch_index] == i
            box_points = points[batch_index, object_indices, :3]
            box_index = joint_index[batch_index, object_indices, 0]
            num_points = object_indices.sum()
            pose = torch.zeros((num_joints, num_points, 3), dtype=points.dtype, device=points.device)
            mask = torch.zeros((num_joints, num_points, 1), dtype=torch.bool, device=points.device)
            pose[box_index, torch.arange(0, num_points)] = box_points
            mask[box_index, torch.arange(0, num_points)] = True
            normalizer = mask.sum(1)
            pose = pose.sum(1) / normalizer
            torch.nan_to_num(pose, out=pose)
            output_box[batch_index, i] = pose
    return output_box
        
