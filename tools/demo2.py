import argparse
import glob
from pathlib import Path
from easydict import EasyDict

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import get_cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader, DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])[:, [0, 1, 2, -1]]
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--det_cfg_file', type=str, default='cfgs/humanm3_models/second_iou.yaml',
                        help='specify the config for demo')
    parser.add_argument('--det_ckpt', type=str, default='cfgs/humanm3_models/second_iou_latest.pth', help='specify the pretrained model')
    parser.add_argument('--pose_cfg_file', type=str, default='cfgs/humanm3_models/vps_pose_left_crop_128.yaml',
                        help='specify the config for demo')
    parser.add_argument('--pose_ckpt', type=str, default='D:/mestrado/OpenPCDet/output/ubc3v_models/vps_pose_left_128_noise/default/ckpt/latest_model.pth', help='specify the pretrained model')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    det_cfg = get_cfg()
    cfg_from_yaml_file(args.det_cfg_file, det_cfg)
    pose_cfg = get_cfg()
    cfg_from_yaml_file(args.pose_cfg_file, pose_cfg)
    cfg = EasyDict({'det_cfg':det_cfg, 'pose_cfg':pose_cfg})

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    demo_dataset, demo_loader, sampler = build_dataloader(
        dataset_cfg=cfg.pose_cfg.DATA_CONFIG,
        class_names=cfg.pose_cfg.CLASS_NAMES,
        batch_size=1, dist=False, logger=logger, training=False
    )
    pose_model = build_network(model_cfg=cfg.pose_cfg.MODEL, num_class=len(cfg.pose_cfg.CLASS_NAMES), dataset=demo_dataset)
    pose_model.load_params_from_file(filename=args.pose_ckpt, logger=logger, to_cpu=True)
    pose_model.cuda()
    pose_model.eval()
    
    demo_dataset, demo_loader, sampler = build_dataloader(
        dataset_cfg=cfg.det_cfg.DATA_CONFIG,
        class_names=cfg.det_cfg.CLASS_NAMES,
        batch_size=1, dist=False, logger=logger, training=False
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    det_model = build_network(model_cfg=cfg.det_cfg.MODEL, num_class=len(cfg.det_cfg.CLASS_NAMES), dataset=demo_dataset)
    det_model.load_params_from_file(filename=args.det_ckpt, logger=logger, to_cpu=True)
    det_model.cuda()
    det_model.eval()
    
    with torch.no_grad():
        for idx in range(0, len(demo_dataset)):
            data_dict = demo_dataset[idx]
            logger.info(f'Visualized sample index: \t{idx}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = det_model.forward(data_dict)
            """
            logger.info(('pearson_scores: {:.3f}; ' 
                         'normals_scores: {:.3f}; ' 
                         'jpe_scores: {} ' 
                         'mean: {:.3f}; '
                         'jap_scores: {:.3f}').format(pred_dicts[0]['pearson_scores'].cpu().numpy()[0], 
                                                      pred_dicts[0]['normals_scores'].cpu().numpy()[0], 
                                                      ', '.join(['{}:{:.3f}'.format(j, x) for j, x in enumerate(pred_dicts[0]['jpe_scores'].cpu().numpy()[0])]), 
                                                      pred_dicts[0]['jpe_scores'].cpu().numpy()[0].mean(),
                                                      pred_dicts[0]['jap_scores'].cpu().numpy()[0]))                                        
            """
            pred_boxes = data_dict['gt_boxes'][0, :, :7]
            #pred_boxes = pred_dicts[0]['pred_boxes']
            points = data_dict['points'].detach().clone()
            point_indices = pose_model.instance_indices(points[:, 1:4], pred_boxes)[0]
            unique_ids = torch.unique(point_indices)
            
            indices_mask = point_indices >= 0
            point_indices = point_indices[indices_mask]
            points = points[indices_mask]
            unique_ids = unique_ids[unique_ids >= 0]
            
            min_pos = torch.full((unique_ids.shape[0], 3), float('inf'), device=unique_ids.device)
            max_pos = torch.full((unique_ids.shape[0], 3), float('-inf'), device=unique_ids.device)
            min_pos.index_reduce_(0, point_indices, points[:, 1:4], 'amin', include_self=False)
            max_pos.index_reduce_(0, point_indices, points[:, 1:4], 'amax', include_self=False)
            max_pos[:, 2] = min_pos[:, 2]
            center_pos = (max_pos + min_pos) / 2
            
            points[:, 1:4] -= center_pos[point_indices]
            points[:, 0] = point_indices
            points_npy = points.cpu().numpy()
            unique_ids_npy = unique_ids.detach().cpu().numpy()
            gt_poses_npy = data_dict['gt_poses'].detach().cpu().numpy()
            batch_dict = [pose_model.dataset.prepare_data({'points':points_npy[points_npy[:,0] == id][:, 1:], 'gt_poses':gt_poses_npy[:, i], 
                                                           'frame_id':i}) 
                         for i, id in enumerate(unique_ids_npy)]
            batch_dict = pose_model.dataset.collate_batch(batch_dict)
            load_data_to_gpu(batch_dict)
            
            pred_dicts, _ = pose_model.forward(batch_dict)
            
            ref_boxes = torch.zeros(pred_boxes.shape, dtype=pred_boxes.dtype, device=pred_boxes.device)
            ref_poses = torch.zeros((len(pred_boxes),18,3), dtype=pred_boxes.dtype, device=pred_boxes.device)
            for id, pred_dict in enumerate(pred_dicts):
                if len(pred_dict['pred_boxes']):
                    ref_boxes[id] = pred_dict['pred_boxes'][0]
                    ref_poses[id] = pred_dict['pose_estimation'][0]
            
            ref_boxes[:, :3] += center_pos
            ref_poses += center_pos[:, None]
            V.draw_scenes(
                points=data_dict['points'][:, 1:],
                #points=batch_dict['voxels'][..., :3].view((-1, 3)),
                #points=data_dict['voxel_features'][..., :3].view((-1, 3)), #point_colors=data_dict['voxel_colors'][..., :3].view((-1, 3))
                #points=data_dict['points'][:, 1:4], point_colors=data_dict['points'][:, 4:7],
                #points=batch_dict['point_coords'][:, 1:],
                #point_colors=data_dict['point_part_labels'],
                #point_colors=pred_dicts[0]['part_segmentation'],
                #normals=data_dict['point_normal_labels'].view(-1, 18, 3)[:, :],
                #normals=pred_dicts[0]['normals'],#.view(-1, 18, 3)[:, idx % 18], 
                gt_poses=data_dict['gt_poses'][0],
                ref_poses=ref_poses,
                #gt_boxes=data_dict['gt_boxes'][0],
                #ref_boxes=ref_boxes, 
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
