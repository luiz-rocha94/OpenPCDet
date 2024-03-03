# -*- coding: utf-8 -*-
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils


def pearson(x, y):
    x_mean = torch.mean(x, 1).unsqueeze(1)
    y_mean = torch.mean(y, 1).unsqueeze(1)
    x_norm = x - x_mean
    y_norm = y - y_mean
    coef_pearson = torch.sum(x_norm * y_norm, 1) / (torch.sqrt(torch.sum(x_norm ** 2, 1)) * torch.sqrt(torch.sum(y_norm ** 2, 1)))
    coef_pearson = (coef_pearson + 1) / 2
    return coef_pearson.unsqueeze(1)


def pearson_in_boxes(points, boxes):
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)
    new_axis = True if len(points.shape) == 2 else False
    if is_numpy:
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points[..., 0:3], boxes)
        points_indices = (point_indices - 1) # TO DO 
        points = points.unsqueeze(0)
        boxes = boxes.unsqueeze(0)
    else:
        if new_axis:
            points = points.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
        
        point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(points[..., 0:3], boxes)
        
    batch_size, num_objects, _ = boxes.shape
    pearson_box = torch.zeros((batch_size, num_objects), dtype=boxes.dtype, device=boxes.device)
    for batch_index in range(batch_size):
        for i in range(num_objects):
            pearson_box[batch_index, i] = points[batch_index, point_indices[batch_index] == i, -1].mean()
    
    if new_axis:
        pearson_box = pearson_box.squeeze(0)
        points = points.squeeze(0)
        boxes = boxes.squeeze(0)
    
    return pearson_box.numpy() if is_numpy else pearson_box
        
