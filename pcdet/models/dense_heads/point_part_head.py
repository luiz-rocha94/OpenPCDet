import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils
from .point_intra_part_head import PointIntraPartOffsetHead
from pcdet.datasets.ubc3v.ubc3v_utils import get_color_maps


class PointPartOffsetHead(PointIntraPartOffsetHead):
    """
    """
    def __init__(self, num_class, input_channels, **kwargs):
        super().__init__(num_class, input_channels, **kwargs)
        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=input_channels,
            output_channels=self.model_cfg.TARGET_CONFIG.COLORS
        )
    
    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        
        point_part_labels_index = self.forward_ret_dict['point_part_labels_index'].view(-1)
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.cross_entropy(point_part_preds, point_part_labels_index, reduction='none')
        point_loss_part = (point_loss_part * pos_mask.float()).sum() / pos_normalizer
        
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict
    
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=not self.model_cfg.TARGET_CONFIG.LABELS, ret_box_labels=(self.box_layers is not None)
        )
        
        if self.model_cfg.TARGET_CONFIG.LABELS:
            targets_dict['point_part_labels'] = input_dict['point_part_labels']
            targets_dict['point_part_labels_index'] = input_dict['point_part_labels_index']
        
        if self.model_cfg.TARGET_CONFIG.NORMALS:
            from ..model_utils.vps_pose_utils import cartesian_to_spherical
            targets_dict['point_normal_labels'] = cartesian_to_spherical(input_dict['point_normal_labels'])
        
        if self.model_cfg.TARGET_CONFIG.JOINTS:
            targets_dict['point_joint_labels'] = input_dict['gt_poses']

        return targets_dict
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_part_preds = self.part_reg_layers(point_features)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_part_preds': point_part_preds,
        }
        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features)
            ret_dict['point_box_preds'] = point_box_preds
        
        if self.normal_layers is not None:
            from ..model_utils.vps_pose_utils import spherical_to_cartesian
            point_normal_preds = self.normal_layers(point_features)
            ret_dict['point_normal_preds'] = point_normal_preds
            batch_dict['point_normal_preds'] = spherical_to_cartesian(torch.relu(point_normal_preds))
        
        if self.joint_layers is not None:
            point_coords = (batch_dict['point_coords'][:, 1:].view(-1, 512, 1, 3) + 
                            point_normal_preds.view(-1, 512, 18, 3))
            point_part = torch.sigmoid(point_part_preds).view(-1, 512, 1, 3).repeat_interleave(18, dim=2)
            point_coords_part = torch.concatenate([point_coords, point_part], dim=-1)
            point_coords_part = point_coords_part.swapaxes(2, 1).contiguous().view(-1, 3072)
            point_joint_preds = self.joint_layers(point_coords_part)
            ret_dict['point_joint_preds'] = point_joint_preds
            batch_dict['point_joint_preds'] = point_joint_preds

        point_cls_scores = torch.sigmoid(point_cls_preds)
        
        _, point_part_index = F.softmax(point_part_preds, 1).max(1)
        src_map, dst_map, color_space, part_dict = get_color_maps()
        color_map = torch.ones((len(src_map)+1, 3), device=point_part_preds.device, 
                               dtype=point_part_preds.dtype)
        color_map[1:] = torch.tensor(src_map)
        point_part_offset = color_map[point_part_index]
        
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels_index'] = targets_dict.get('point_part_labels_index')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')
            ret_dict['point_normal_labels'] = targets_dict.get('point_normal_labels')
            ret_dict['point_joint_labels'] = targets_dict.get('point_joint_labels')

        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict
        return batch_dict
