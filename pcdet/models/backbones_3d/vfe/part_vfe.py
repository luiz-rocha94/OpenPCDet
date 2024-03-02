import torch

from .vfe_template import VFETemplate


class PartVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = self.model_cfg.NUM_POINT_FEATURES

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features = batch_dict['voxels'][..., :self.num_point_features].contiguous()
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_colors = batch_dict['voxels'][..., self.num_point_features:].contiguous()
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        colors_mean = voxel_colors[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        colors_mean = colors_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_labels'] = colors_mean.contiguous()

        return batch_dict
