'''
Modified from https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py
'''
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from evaluators.result2kitti import result2kitti, kitti_evaluation, result2kitti_dair, result2kitti_rope3d

__all__ = ['RoadSideEvaluator']


class RoadSideEvaluator():
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(
        self,
        class_names,
        current_classes,
        data_root,
        gt_label_path,
        modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),
        output_dir=None,
    ) -> None:
        self.class_names = class_names
        self.current_classes = current_classes
        self.data_root = data_root
        self.gt_label_path = gt_label_path
        self.modality = modality
        self.output_dir = output_dir

    def format_results(self,
                       results,
                       img_metas,
                       result_names=['img_bbox'],
                       jsonfile_prefix=None,
                       **kwargs):
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = dict()
        for rasult_name in result_names:
            if '2d' in rasult_name:
                continue
            print(f'\nFormating bboxes of {rasult_name}')
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, self.output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        img_metas,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['img_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        result_files, tmp_dir = self.format_results(results, img_metas,
                                                    result_names,
                                                    jsonfile_prefix)
        print(result_files, tmp_dir)
        results_path = "outputs" 
        if 'dair' in self.data_root:
            pred_label_path = result2kitti_dair(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)
        elif 'rope3d' in self.data_root:
            pred_label_path = result2kitti_rope3d(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)
        else:
            pred_label_path = result2kitti(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)            
        kitti_evaluation(pred_label_path, self.gt_label_path, current_classes=self.current_classes, metric_path="outputs/metrics")

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    box_yaw=box_yaw,
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path