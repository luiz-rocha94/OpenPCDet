import copy
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
try:
    from ...ops.roiaware_pool3d import roiaware_pool3d_utils
    from ...utils import box_utils, common_utils
    from ..dataset import DatasetTemplate
    from .humanm3_utils import get_annos, draw_point_cloud, align_points
    from ..ubc3v.ubc3v_utils import get_color_maps
except:
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils import box_utils, common_utils
    from pcdet.datasets.dataset import DatasetTemplate
    from humanm3_utils import get_annos, draw_point_cloud, align_points
    from pcdet.datasets.ubc3v.ubc3v_utils import get_color_maps


class HumanM3Dataset(DatasetTemplate):    
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
        self.set_split(split)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self):
        self.logger.info('Loading HumanM3 dataset.')
        self.humanm3_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[self.split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            
            if isinstance(infos, dict):
                infos = infos['Pedestrian']
                infos = [info for info in infos if info['num_points_in_gt'] >= 128]
            self.humanm3_infos.extend(infos)

        self.logger.info('Total samples for HumanM3 dataset: %d' % (len(self.humanm3_infos)))

    def set_split(self, split):        
        self.split = split
        self.include_data()
        split_file = self.root_path / (self.split+'.txt')
        assert split_file.exists()
        with  open(split_file, 'r') as f:
            lines = f.readlines()
        lines = [line.replace('\n','') for line in lines] 
            
        self.sample_id_list = {line[-24:-22]+line[-8:-4]:line for line in lines}

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.humanm3_infos)
    
    def get_anno(self, idx):
        pcd_file = self.root_path / self.sample_id_list[idx]
        sequence_path, name = pcd_file.parents[1], pcd_file.name
        anno = get_annos(sequence_path, name)[0]
        return anno
    
    def draw(self, index):
        info = copy.deepcopy(self.humanm3_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        draw_point_cloud(points[:, :3], info['annos']['pose'], info['annos']['gt_boxes_lidar'])

    def get_lidar(self, idx, return_offset=False):
        pcd_file = self.root_path / self.sample_id_list[idx]
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
            
        point_features = np.vstack([np.array(line.replace('\n', '').split(' '), dtype=np.float32) 
                                    for line in lines[11:]])
        if return_offset:
            offset = np.zeros(3, dtype=np.float32)
            # center
            max_, min_ = point_features.max(0)[:3], point_features.min(0)[:3]
            center = (max_ + min_) / 2
            # kitti offset
            offset[0] = min_[0]
            offset[1] = center[1]
            offset[2] = min_[2] + 3
            point_features[:, :3] = point_features[:, :3] - offset[None, :]
            return point_features, offset
        
        return point_features
    
    def draw_skeleton(self, target_pose):
        input_pose = np.load(self.root_path / 'model_joints.npy')
        input_points = np.load(self.root_path / 'model_points.npy')
        output_points, _ = align_points(input_points, input_pose, target_pose)
        return output_points[:, :-1]

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.humanm3_infos)

        info = copy.deepcopy(self.humanm3_infos[index])
        data_src = self.dataset_cfg.get('DATA_SRC')
        if data_src:
            sample_idx = index
            points = np.load(self.root_path / info['path'])
            feat = [0,1,2]
            if data_src == 'align':
                mask = points[:, -1] == 0
                feat += [3,4,5]
            elif data_src == 'crop':
                mask = points[:, -1] == 1
            elif data_src == 'color':
                mask = points[:, -1] == 1
                feat += [3,4,5]
            
            feat += [6]
            points = points[mask][:,feat]
            offset = np.zeros(3, dtype=np.float32)
            offset[2] = points[:, 2].min()
            points[:, :3] -= offset[None]
        else:
            sample_idx = info['point_cloud']['lidar_idx']
            points, offset = self.get_lidar(sample_idx, return_offset=True)
        
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
            
            gt_poses = gt_poses - offset[None, None, :]
            gt_boxes_lidar[:, :3] = gt_boxes_lidar[:, :3] - offset[None, :]
            
            cmap = 'hsv'
            _, color_map, _, _, _ = get_color_maps(cmap=cmap)
            
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_poses': gt_poses,
                'cmap': color_map,
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
                'jpe_scores': np.zeros((num_samples, 18), np.float32), 'jap_scores': np.zeros(num_samples, np.float32)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
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
            pred_dict['normals_scores'] = normals_scores
            pred_dict['jpe_scores'] = jpe_scores
            pred_dict['jap_scores'] = jap_scores
            
            if 'pearson_scores' in box_dict:
                pearson_scores = box_dict['pearson_scores'].cpu().numpy()
                pred_dict['pearson_scores'] = pearson_scores

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
        if 'annos' not in self.humanm3_infos[0].keys():
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

        eval_metrics = kwargs['eval_metric'] if isinstance(kwargs['eval_metric'], list) else [kwargs['eval_metric']]
        result_str, result_dict = '\n', {}
        for eval_metric in eval_metrics:
            if eval_metric == 'kitti':
                eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.humanm3_infos]
                ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
                result_str += ap_result_str 
                result_dict.update(ap_dict)
            elif eval_metric == 'pearson':
                mean_pearson_scores = np.concatenate([anno['pearson_scores'] for anno in eval_det_annos]).mean()
                result_str += 'Pearson Coef [-1, 1]: {:.3f}\n'.format(mean_pearson_scores)
                result_dict.update({'pearson': mean_pearson_scores})
            elif eval_metric == 'normals':
                mean_normals_scores = np.concatenate([anno['normals_scores'] for anno in eval_det_annos]).mean()
                result_str += 'Normals [m]: {:.3f}\n'.format(mean_normals_scores)
                result_dict.update({'normals': mean_normals_scores})
            elif eval_metric == 'jpe':
                jpe_scores = np.concatenate([anno['jpe_scores'] for anno in eval_det_annos])
                result_str += 'Joint Position Shape {}\n'.format(jpe_scores.shape)
                jap_tp = (jpe_scores <= 0.1).sum(0)
                jap_fp = (jpe_scores > 0.1).sum(0)
                jap_scores = jap_tp / (jap_tp + jap_fp)
                for j_id in range(18):
                    j_jpe_scores = jpe_scores[:, j_id].mean()
                    j_jap_scores = jap_scores[j_id]
                    result_str += 'Joint Position Error J{} mean [m]: {:.3f}\n'.format(j_id, j_jpe_scores)
                    result_str += 'Joint Average Precision J{} mean [%]: {:.3f}\n'.format(j_id, 100.0*j_jap_scores)
                 
                result_str += 'Joint Position Error mean [m]: {:.3f}\n'.format(jpe_scores.mean())
                result_dict.update({'jpe': jpe_scores.mean()})
                result_str += 'Joint Average Precision [%]: {:.3f}\n'.format(100.0*jap_scores.mean())
                result_dict.update({'jap': jap_scores.mean()})
                
                data_format = lambda data: ['ID: {}; E {:.3f}mm'.format(i, v) for i, v in data]
                mean_jpe_scores = jpe_scores.mean(1)*1e3
                sorted_idx = np.argsort(mean_jpe_scores)
                sorted_mean_jpe_scores = mean_jpe_scores[sorted_idx]
                q1 = np.percentile(sorted_mean_jpe_scores, 25)
                q2 = np.percentile(sorted_mean_jpe_scores, 50)
                q3 = np.percentile(sorted_mean_jpe_scores, 75)
                iqr = q3 - q1
                q0 = max(q3 - 1.5 * iqr, sorted_mean_jpe_scores.min())
                q0 = np.min(sorted_mean_jpe_scores)
                q4 = min(q3 + 1.5 * iqr, sorted_mean_jpe_scores.max())
                q5 = 1e2
                q6 = sorted_mean_jpe_scores.max()
                p_max = 0
                p_idx = np.zeros(0, dtype=np.float32)
                for i, (ql, qr) in enumerate([(q0,q1), (q1,q2), (q2,q3), (q3,q4), (q4,q5)]):
                    mask = (sorted_mean_jpe_scores >= ql) & (sorted_mean_jpe_scores < qr)
                    q_jpe_scores = sorted_mean_jpe_scores[mask]
                    q_idx = sorted_idx[mask]
                    p = p_max
                    p_max = p + mask.mean()*1e2
                    p_idx = np.concatenate([p_idx, np.linspace(p, p_max, mask.sum())])
                    k=3
                    result_str += '\nQ{} {}-{}% {:.3f}-{:.3f}mm {:.3f}% data'.format(i, i*25, (i+1)*25, ql, qr, (1*mask).mean()*1e2)
                    result_str += '\nBest\n'+'\n'.join(data_format(zip(q_idx[:k], q_jpe_scores[:k])))
                    result_str += '\nWorst\n'+'\n'.join(data_format(zip(q_idx[-k:], q_jpe_scores[-k:])))
                
                mask = sorted_mean_jpe_scores >= q5
                p_idx = np.concatenate([p_idx, np.linspace(p_max, 100, mask.sum())])
                p_idx = np.linspace(0, 100, len(sorted_mean_jpe_scores))
                plt.plot(p_idx, sorted_mean_jpe_scores, '-k')
                plt.plot([0, 0], [0, 1e3], '-b', label='Min {:.3f}mm'.format(q0))
                plt.plot([25, 25], [0, 1e3], '-b', label='25% {:.3f}mm'.format(q1))
                plt.plot([50, 50], [0, 1e3], '-b', label='50% {:.3f}mm'.format(q2))
                plt.plot([75, 75], [0, 1e3], '-b', label='75% {:.3f}mm'.format(q3))
                plt.plot([p, p], [0, 1e3], '-b', label='Max ({:.0f}%) {:.3f}mm'.format(p, q4))
                plt.plot([100, 100], [0, 1e3], '-b', label='100% {:.3f}mm'.format(q6))
                plt.plot([0, 100], [q5, q5], '-r', label='Limite 100mm')
                plt.xlabel('Distribuição [%]')
                plt.xticks(range(0, 101, 25))
                plt.ylabel('mPJPE [mm]')
                plt.ylim(0, 200)
                plt.legend()
                handler = [handler for handler in self.logger.handlers if isinstance(handler, logging.FileHandler)][0]
                log_file = Path(handler.baseFilename)
                parts = ['dist'] + log_file.name.split('_')[1:]
                dist_file = '_'.join(parts)
                dist_file = log_file.with_name(dist_file).with_suffix('.png')
                plt.savefig(str(dist_file))
                #plt.show()
            else:
                raise NotImplementedError

        return result_str, result_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sequence_path):
            annos = get_annos(sequence_path)
            infos = []
            for i, anno in enumerate(annos):
                print('split: {}; sequence: {}; step {}/{}'.format(self.split, sequence_path.name,
                                                                   i+1, len(annos)))
                info = {}
                sample_idx = anno['Index']
                pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
                info['point_cloud'] = pc_info
    
                if has_label:
                    #points = self.get_lidar(sample_idx)
                    annotations = {}
                    joints = anno['Posture']
                    gt_boxes_lidar = anno['BBox3D']
                    annotations['pose'] = joints
                    annotations['name'] = np.array(anno['Label']).reshape(-1)
                    annotations['id'] = np.array(anno['ID']).reshape(-1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                    info['annos'] = annotations
                
                infos.append(info)

            return infos

        split_path = self.root_path / self.split
        sequences = sorted(split_path.glob('*'))
        
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
        db_info_save_path = Path(self.root_path) / ('humanm3_dbinfos_%s.pkl' % split)

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
            gt_poses = annos['pose']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.npy' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                offset = gt_boxes[i, :3] - np.array([0,0,gt_boxes[i, 5]/2], dtype=np.float32)
                gt_points[:, :3] -= offset[None]
                gt_poses[i, :] -= offset[None]
                gt_boxes[i, :3] -= offset
                #draw_point_cloud(gt_points[:, :3], gt_poses[i][None], gt_boxes[i][None])
                pose_points = self.draw_skeleton(gt_poses[i])
                dist = np.linalg.norm(gt_points[:, None, :3] - pose_points[None, :, :3], axis=-1)
                min_idx = np.argmin(dist, 1)
                colors = pose_points[min_idx, 3:]
                num_points_in_gt = len(gt_points)
                is_gt = np.ones((num_points_in_gt,1), dtype=gt_points.dtype)
                gt_points = np.concatenate([gt_points[:,:3], colors, is_gt], axis=1)
                is_pose = np.zeros((len(pose_points),1), dtype=gt_points.dtype)
                pose_points = np.concatenate([pose_points, is_pose], axis=1)
                gt_points = np.concatenate([gt_points, pose_points], axis=0)
                np.save(filepath, gt_points)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'path': db_path, 'gt_idx': i, 'box3d_lidar': gt_boxes[i], 'num_points_in_gt': num_points_in_gt,
                               'offset': offset, 
                               'annos':{'name':np.array(names[i]).reshape(-1), 'pose':gt_poses[i][None], 'gt_boxes_lidar':gt_boxes[i][None]}}
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


def create_humanm3_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = HumanM3Dataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split, test_split = 'train', 'test'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('humanm3_infos_%s.pkl' % train_split)
    test_filename = save_path / ('humanm3_infos_%s.pkl' % test_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    humanm3_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(humanm3_infos_train, f)
    print('HumanM3 info train file is saved to %s' % train_filename)

    dataset.set_split(test_split)
    humanm3_infos_test = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(humanm3_infos_test, f)
    print('HumanM3 info test file is saved to %s' % test_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    dataset.set_split(test_split)
    dataset.create_groundtruth_database(test_filename, split=test_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    ROOT_DIR = Path(__file__).resolve().parents[3]
    dataset_cfg = EasyDict(yaml.safe_load(open(ROOT_DIR / 'tools/cfgs/dataset_configs/humanm3_dataset.yaml')))
    data_path = Path(dataset_cfg['DATA_PATH'])
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_humanm3_infos':
        create_humanm3_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Pedestrian'],
        data_path=ROOT_DIR / data_path,
        save_path=ROOT_DIR / data_path,
        )
    else:
        dataset = HumanM3Dataset(
            dataset_cfg=dataset_cfg, class_names=['Pedestrian'], 
            root_path=data_path,
            training=False, logger=common_utils.create_logger()
        )
