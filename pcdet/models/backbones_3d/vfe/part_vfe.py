import torch

from .vfe_template import VFETemplate


class PartVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.ch = self.model_cfg.NUM_POINT_FEATURES

    def get_output_feature_dim(self):
        return self.ch[0]

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
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_features = batch_dict['voxels']
        features_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        features_mean = features_mean / normalizer
        batch_dict['voxel_features']    = features_mean[:, :self.ch[0]].contiguous()
        if len(self.ch) >= 2:
            batch_dict['voxel_colors']  = features_mean[:, self.ch[0]:self.ch[1]].contiguous()
        if len(self.ch) >= 3:
            batch_dict['voxel_normals']  = features_mean[:, self.ch[1]:self.ch[2]].contiguous()
        return batch_dict
