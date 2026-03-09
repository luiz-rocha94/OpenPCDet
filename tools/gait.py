import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

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

from pcdet.config import cfg, cfg_from_yaml_file
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
        data_file_list = root_path.glob(f'*{self.ext}') if self.root_path.is_dir() else [self.root_path]
        data_file_list = sorted(data_file_list, key=lambda x: int(''.join(filter(str.isdigit, x.name))))
 
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])[:, [0, 1, 2, -1]]
        elif self.ext == '.ply':
            with open(self.sample_file_list[index], 'r') as f:
                lines = f.readlines()[8:]
            
            points = np.array([line.strip().split(' ') for line in lines], dtype=np.float32).reshape((-1, 4))
            points[:, 0] -= 9
            points[:, 2] = -1*(points[:, 2] - 1.5)
            points[:, 3] = 1
            points = points[np.logical_and(points[:, 1] > -3, 
                                           points[:, 1] < 3)]
            #points = points[points[:, 2] >= 0.05]
            points = points[np.logical_and(points[:, 0] > -0.5, 
                                           points[:, 0] < 0.5)]
            
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
    parser.add_argument('--cfg_file', type=str, default='cfgs/xenomatix_models/vps_pose.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=r'D:\mestrado\OpenPCDet\data\xenomatix\seq2',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='D:/mestrado/OpenPCDet/output/ubc3v_models/vps_pose/default/ckpt/latest_model.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.ply', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    if args.data_path:
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
    else:
        demo_dataset, demo_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1, dist=False, logger=logger, training=False
        )
    
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    poses = np.zeros((0, 18, 3), dtype=np.float32)
    files = []
    with torch.no_grad():
        for idx in range(7, 61, 9):
            data_dict = demo_dataset[idx]
            file = demo_dataset.sample_file_list[idx]
            files.append(file.name)
            logger.info(f'Visualized sample index: \t{idx} \t{file}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            logger.info('detected: {}'.format(len(pred_dicts[0]['pred_boxes'])))
            #"""
            V.draw_scenes(
                points=data_dict['points'][:, 1:4],
                #points=data_dict['voxels'][..., :3].view((-1, 3)),
                #points=data_dict['point_coords'][:, 1:], 
                #point_colors=pred_dicts[0]['part_segmentation'],
                #normals=pred_dicts[0]['normals'], 
                #ref_boxes=pred_dicts[0]['pred_boxes'], 
                ref_poses=pred_dicts[0]['pose_estimation'],
            )
            #"""

            poses = np.concatenate([poses, pred_dicts[0]['pose_estimation'].cpu().numpy()], axis=0)
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    print(files[::9])
    time = 0.05*np.arange(len(poses))
    r_foot = poses[::18, 11]
    l_foot = poses[9::18, 9]
    r_time = time[::18]
    l_time = time[9::18]
    plt.scatter(r_foot[:, 0], r_foot[:, 1], c='red')
    plt.scatter(l_foot[:, 0], l_foot[:, 1], c='blue')
    plt.xlim([-0.5, 0.5])
    plt.show()
    
    lp = l_foot - r_foot
    lp = np.linalg.norm(lp[..., :2], axis=-1)
    logger.info('{:.3f}'.format(lp.mean()))
    
    ls = np.concatenate([r_foot[1:] - r_foot[:-1], 
                         l_foot[1:] - l_foot[:-1]])
    ls = np.linalg.norm(ls[..., :2], axis=-1)
    logger.info('{:.3f}'.format(ls.mean()))
    
    cp = len(l_time)+len(r_time)
    ct = np.maximum(l_time, r_time).max() - np.minimum(l_time, r_time).min()
    ct = np.maximum(l_time, r_time).max() - np.minimum(l_time, r_time).min()
    cd = np.linalg.norm(l_foot[-1, :2] - r_foot[0, :2])
    cv = 60*(cp)/ct
    logger.info('{} {:.3f} {:.3f} {:.3f}'.format(cp, ct, cd, cv))
    
    tp = l_time - r_time
    logger.info('{:.3f}'.format(tp.mean()))
    
    ts = np.concatenate([r_time[1:] - r_time[:-1], 
                         l_time[1:] - l_time[:-1]])
    logger.info('{:.3f}'.format(ts.mean()))
    
    v = ls/ts
    logger.info('{:.3f}'.format(v.mean()))
    
    rot = Rotation.from_euler('x', np.pi)
    right = rot.apply(poses[:, 10] - poses[:, 7]).astype(np.float32)  
    left = rot.apply(poses[:, 8] - poses[:, 6]).astype(np.float32) 
    t = np.linspace(0, 100, len(poses), dtype=np.int32)
    right_a = np.rad2deg(np.arctan2(right[:, 1], right[:, 2]))
    left_a = np.rad2deg(np.arctan2(left[:, 1], left[:, 2]))
    left_interpolation = interp1d(t, left_a, kind = "cubic")
    right_interpolation = interp1d(t, right_a, kind = "cubic")
    t = np.linspace(0, 100, 100, dtype=np.int32)
    right_a_interpolated = right_interpolation(t)
    left_a_interpolated = left_interpolation(t)
    plt.plot(t, right_a_interpolated, label='Lado D.', color='r')
    plt.plot(t, left_a_interpolated, label='Lado E.', color='b')
    for i in range(1,7):
        line_color = '--r' if i % 2 == 0 else '--b'
        label = 'Pé D.' if i % 2 == 0 else 'Pé E.'
        plt.plot([16.667*i, 16.667*i], [-20, 50], line_color, label=label if i <= 2 else None)
    plt.title('Flexão de Quadril')
    plt.xlabel('Ciclo de marcha [%]')
    plt.ylabel('Ângulo [°]')
    plt.legend()
    plt.show()
    
    right_n = (right_a_interpolated - right_a_interpolated.min())/(right_a_interpolated.max() - right_a_interpolated.min())
    left_n = (left_a_interpolated - left_a_interpolated.min())/(left_a_interpolated.max() - left_a_interpolated.min())
    si_n = np.abs(200*(right_n - left_n)/(right_n + left_n))
    plt.plot(t, si_n, color='k', label='Simetria')
    for i in range(1,7):
        line_color = '--r' if i % 2 == 0 else '--b'
        label = 'Pé D.' if i % 2 == 0 else 'Pé E.'
        plt.plot([16.667*i, 16.667*i], [-5, 200], line_color, label=label if i <= 2 else None)
    plt.plot([0, 100], [15, 15], '-g', label='Referência (15%)')
    plt.title('Flexão de Quadril')
    plt.xlabel('Ciclo de marcha [%]')
    plt.ylabel('SI Norm [%]')
    plt.legend()
    plt.show()
    
    logger.info('{:.3f} {:.3f} {:.3f} {:.3f}'.format(si_n.min(),
                                                     si_n.max(),
                                                     si_n.mean(),
                                                     si_n.std()))

    logger.info('Gait done.')
    
    

if __name__ == '__main__':
    main()
