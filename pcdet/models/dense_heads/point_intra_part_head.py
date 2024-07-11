import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointIntraPartOffsetHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=input_channels,
            output_channels=3
        )
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
                **target_cfg.BOX_CODER_CONFIG
            )
            self.box_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=input_channels,
                output_channels=self.box_coder.code_size
            )
        else:
            self.box_layers = None
        
        if target_cfg.NORMALS:
            self.normal_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.NORM_FC,
                input_channels=input_channels,
                output_channels=target_cfg.NORMALS
            )
        else:
            self.normal_layers = None
        
        if target_cfg.JOINTS:
            self.joint_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.JOINT_FC,
                input_channels=target_cfg.JOINTS,
                output_channels=3
            )
        else:
            self.joint_layers = None

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
        
        if self.model_cfg.TARGET_CONFIG.NORMALS:
            targets_dict['point_normal_labels'] = input_dict['point_normal_labels']
        
        if self.model_cfg.TARGET_CONFIG.JOINTS:
            targets_dict['point_joint_labels'] = input_dict['gt_poses']

        return targets_dict
    
    def get_normal_layer_loss(self, tb_dict=None):
        from ..model_utils.vps_pose_utils import spherical_to_cartesian
        target_cfg = self.model_cfg.TARGET_CONFIG
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        
        point_normal_labels = spherical_to_cartesian(self.forward_ret_dict['point_normal_labels'])
        point_normal_preds = spherical_to_cartesian(self.forward_ret_dict['point_normal_preds'])
        point_loss_normal = F.smooth_l1_loss(point_normal_preds, point_normal_labels, reduction='none') 
        point_loss_normal = (point_loss_normal.sum(dim=-1) * pos_mask.float()).sum() / (target_cfg.NORMALS * pos_normalizer)
        
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_normal = point_loss_normal * loss_weights_dict['point_normal_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_normal': point_loss_normal.item()})
        return point_loss_normal, tb_dict
    
    def get_joint_layer_loss(self, tb_dict=None):
        target_cfg = self.model_cfg.TARGET_CONFIG
        pos_mask = (self.forward_ret_dict['point_cls_labels'] > 0).view(-1, 512).any(1)
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        
        point_joint_labels = self.forward_ret_dict['point_joint_labels'].view(-1, target_cfg.NORMALS)
        point_joint_preds = self.forward_ret_dict['point_joint_preds'].view(-1, target_cfg.NORMALS)
        point_loss_joint = F.smooth_l1_loss(point_joint_preds, point_joint_labels, reduction='none') 
        point_loss_joint = (point_loss_joint.sum(dim=-1) * pos_mask.float()).sum() / (target_cfg.NORMALS * pos_normalizer)
        
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_joint = point_loss_joint * loss_weights_dict['point_joint_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_joint': point_loss_joint.item()})
        return point_loss_joint, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_part

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        
        if self.normal_layers is not None:
            point_loss_normal, tb_dict = self.get_normal_layer_loss(tb_dict)
            point_loss += point_loss_normal
        
        if self.joint_layers is not None:
            point_loss_joint, tb_dict = self.get_joint_layer_loss(tb_dict)
            point_loss += point_loss_joint
        
        tb_dict['point_loss'] = point_loss.item()
        return point_loss, tb_dict

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
            point_normal_preds = self.normal_layers(point_features)
            ret_dict['point_normal_preds'] = point_normal_preds
            batch_dict['point_normal_preds'] = point_normal_preds
        
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
        point_part_offset = torch.sigmoid(point_part_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
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
