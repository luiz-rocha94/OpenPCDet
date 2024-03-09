# -*- coding: utf-8 -*-
import yaml
from pathlib import Path
from easydict import EasyDict

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import box_utils, common_utils
from pcdet.models import build_network
from train_utils.train_utils import save_checkpoint, checkpoint_state
from pcdet.datasets.ubc3v.ubc3v_dataset import UBC3VDataset


dataset_cfg = cfg_from_yaml_file('cfgs/dataset_configs/ubc3v_dataset.yaml', cfg)
dataset = UBC3VDataset(
    dataset_cfg=dataset_cfg, class_names=['Pedestrian'], 
    root_path=dataset_cfg.ROOT_DIR / 'data' / 'ubc3v' / 'pose',
    training=False, logger=common_utils.create_logger()
)

second_ckpt = dataset_cfg.ROOT_DIR / 'tools/cfgs/ubc3v_models/second_7862.pth'
pv_rcnn_ckpt = dataset_cfg.ROOT_DIR / 'tools/cfgs/ubc3v_models/pv_rcnn_8369.pth'
part_a2_ckpt = dataset_cfg.ROOT_DIR / 'tools/cfgs/ubc3v_models/PartA2_free_7872.pth'
vps_pose_ckpt = dataset_cfg.ROOT_DIR / 'tools/cfgs/ubc3v_models/vps_pose_latest.pth'

model_cfg = cfg_from_yaml_file('cfgs/ubc3v_models/vps_pose.yaml', cfg)
model_ckpt = dataset_cfg.ROOT_DIR / 'tools/cfgs/ubc3v_models/vps_pose.pth'
model = build_network(model_cfg=model_cfg.MODEL, num_class=len(model_cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=part_a2_ckpt, logger=dataset.logger)
model.load_params_from_file(filename=pv_rcnn_ckpt, logger=dataset.logger)
model.load_params_from_file(filename=second_ckpt, logger=dataset.logger)
model.load_params_from_file(filename=vps_pose_ckpt, logger=dataset.logger)
model.cuda()
model.eval()

save_checkpoint(checkpoint_state(model, epoch=0, it=0), model_ckpt.with_suffix(''))
