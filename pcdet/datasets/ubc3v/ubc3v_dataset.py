import copy
import pickle
import os
import numpy as np
try:
    from ...ops.roiaware_pool3d import roiaware_pool3d_utils
    from ...utils import box_utils, common_utils
    from ..dataset import DatasetTemplate
    from .ubc3v_utils import get_annos, get_bouding_box, get_joints_name, draw_point_cloud, apply_color_map, get_normals
except:
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils import box_utils, common_utils
    from pcdet.datasets.dataset import DatasetTemplate
    from ubc3v_utils import get_annos, get_bouding_box, get_joints_name, draw_point_cloud, apply_color_map, get_normals


class UBC3VDataset(DatasetTemplate):    
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.set_split(split, False)
        self.ubc3v_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        self.logger.info('Loading UBC3V dataset.')
        ubc3v_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ubc3v_infos.extend(infos)

        self.ubc3v_infos.extend(ubc3v_infos)
        self.logger.info('Total samples for UBC3V dataset: %d' % (len(ubc3v_infos)))

    def get_anno(self, idx):
        idx = str(idx)
        name = 'mayaProject.{}.png'.format(idx[-6:])
        cams = ['Cam{}'.format(idx[-7])]
        sequence_path = self.root_path / self.split / idx[:-7]
        anno = get_annos(sequence_path, cams, name)[0]
        return anno
    
    def draw(self, index):
        info = copy.deepcopy(self.ubc3v_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        draw_point_cloud(points[:, :3], points[:, 3:], info['annos']['pose'][0], info['annos']['gt_boxes_lidar'][0])

    def get_label(self, point_features, joints):
        normals, idx = get_normals(point_features[:, 0:3], point_features[:, 3:6], joints)
        normals = np.concatenate([normals, idx[:, np.newaxis]], axis=1)
        return normals

    def get_lidar(self, idx, return_offset=False):
        point_features = np.load(self.root_path / self.split / '{}.npy'.format(idx))
        offset = point_features[:, 2].min()
        point_features[:, 2] -= offset
        colors = apply_color_map(point_features[:, 3:6])
        point_features = np.concatenate([point_features[:, :3], colors], axis=1)
        
        if return_offset:
            return point_features, offset
        
        return point_features

    def set_split(self, split, call=True):
        if call:
            super().__init__(
                dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
                root_path=self.root_path, logger=self.logger
            )
        
        self.split = split
        split_dir = self.root_path / (self.split+'.txt')
        assert split_dir.exists()
        #self.sample_id_list = sorted([UBC3VDataset.frame_filter(file) for file in split_dir.glob('*/images/depthRender/*/*.png')])
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.ubc3v_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.ubc3v_infos)

        info = copy.deepcopy(self.ubc3v_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': sample_idx,
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            gt_poses = annos['pose']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_poses': gt_poses
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples),
                'pearson_scores': np.zeros(num_samples, np.float32), 'normals_scores': np.zeros(num_samples, np.float32),
                'jpe_scores': np.zeros(num_samples, np.float32), 'jpa_scores': np.zeros(num_samples, np.float32)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pearson_scores = box_dict['pearson_scores'].cpu().numpy()
            normals_scores = box_dict['normals_scores'].cpu().numpy()
            jpe_scores = box_dict['jpe_scores'].cpu().numpy()
            jap_scores = box_dict['jap_scores'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            pred_dict['pearson_scores'] = pearson_scores
            pred_dict['normals_scores'] = normals_scores
            pred_dict['jpe_scores'] = jpe_scores
            pred_dict['jap_scores'] = jap_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.ubc3v_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in map_name_to_kitti]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.ubc3v_infos]

        eval_metrics = kwargs['eval_metric'] if isinstance(kwargs['eval_metric'], list) else [kwargs['eval_metric']]
        result_str, result_dict = '\n', {}
        for eval_metric in eval_metrics:
            if eval_metric == 'kitti':
                ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
                result_str += ap_result_str 
                result_dict.update(ap_dict)
            elif eval_metric == 'pearson':
                mean_pearson_scores = np.mean([anno['pearson_scores'].mean() for anno in eval_det_annos])
                result_str += 'Pearson Coef [-1, 1]: {:.3f}\n'.format(mean_pearson_scores)
                result_dict.update({'pearson': mean_pearson_scores})
            elif eval_metric == 'normals':
                mean_normals_scores = np.mean([anno['normals_scores'].mean() for anno in eval_det_annos])
                result_str += 'Normals [m]: {:.3f}\n'.format(mean_normals_scores)
                result_dict.update({'normals': mean_normals_scores})
            elif eval_metric == 'jpe':
                jpe_scores = np.concatenate([anno['jpe_scores'] for anno in eval_det_annos])
                for j_id in range(18):
                    j_jpe_scores = jpe_scores[:, j_id]
                    result_str += 'Joint Position Error J{} mean [m]: {:.3f}\n'.format(j_id, j_jpe_scores.mean())
                    result_str += 'Joint Position Error J{} std [m]: {:.3f}\n'.format(j_id, j_jpe_scores.std())
                    result_str += 'Joint Average Precision J{} mean [%]: {:.3f}\n'.format(j_id, (j_jpe_scores <= 0.1).mean())
                 
                mean_jpe_scores = jpe_scores.mean(-1)
                result_str += 'Joint Position Error mean [m]: {:.3f}\n'.format(mean_jpe_scores.mean())
                result_str += 'Joint Position Error std [m]: {:.3f}\n'.format(mean_jpe_scores.std())
                result_dict.update({'jpe': mean_jpe_scores.mean()})
                mean_jap_scores = np.concatenate([anno['jap_scores'] for anno in eval_det_annos]).mean()
                result_str += 'Joint Average Precision [%]: {:.3f}\n'.format(mean_jap_scores)
                result_dict.update({'jap': mean_jap_scores})
            else:
                raise NotImplementedError

        return result_str, result_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sequence_path):
            print('%s sequence: %s' % (self.split, sequence_path.name))
            annos = get_annos(sequence_path)
            infos = []
            for anno in annos:
                info = {}
                sample_idx = anno['Index']
                pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
                info['point_cloud'] = pc_info
    
                if has_label:
                    points, z_offset = self.get_lidar(sample_idx, return_offset=True)
                    annotations = {}
                    joints = anno['Posture']
                    joints[:, 2] -= z_offset
                    gt_boxes_lidar = get_bouding_box(points[:, :3], joints)
                    annotations['pose'] = joints.reshape((-1, 18, 3))
                    annotations['name'] = np.array(anno['Label']).reshape(-1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar.reshape((-1, 7))
                    info['annos'] = annotations
                
                infos.append(info)

            return infos

        split_path = self.root_path.parent / self.dataset_cfg.DATA_SRC / self.split
        sequences = sorted(split_path.glob('*'), key=lambda x: int(x.name))
        
        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_info_list = executor.map(process_single_scene, sequences)
        
        infos = []
        for info in sequence_info_list:
            infos.extend(info)       
        return infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ubc3v_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_ubc3v_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = UBC3VDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split, val_split = 'train', 'valid'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('ubc3v_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ubc3v_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    ubc3v_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(ubc3v_infos_train, f)
    print('UBC3V info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    ubc3v_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(ubc3v_infos_val, f)
    print('UBC3V info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    #dataset.set_split(train_split)
    #dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_ubc3v_infos':
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        create_ubc3v_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Pedestrian'],
        data_path=ROOT_DIR / 'data' / 'ubc3v' / 'pose',
        save_path=ROOT_DIR / 'data' / 'ubc3v' / 'pose',
        )
    else:
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg = EasyDict(yaml.safe_load(open(ROOT_DIR / 'tools/cfgs/dataset_configs/ubc3v_dataset.yaml')))
        dataset = UBC3VDataset(
            dataset_cfg=dataset_cfg, class_names=['Pedestrian'], 
            root_path=ROOT_DIR / 'data' / 'ubc3v' / 'pose',
            training=False, logger=common_utils.create_logger()
        )
