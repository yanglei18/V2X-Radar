# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def calculate_center_distance(box):
    center = np.mean(box, axis=0)
    distance = np.linalg.norm(center[:2])  # Only consider x and y for distance
    return distance

def _print_scene_stats(scene_info, iou_dist, stats_collected):
    """Print aggregated stats for a scene."""
    if iou_dist and len(iou_dist) > 0:
        iou_array = np.array(iou_dist)
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(iou_array, bins=bins)
        hist_str = " ".join([f"{count}" for count in hist])
    else:
        hist_str = "0 0 0 0 0 0 0 0 0 0"

    print(f"[{scene_info}] IoU_hist: {hist_str}")
    print(f"[{scene_info}] Confusion Matrix:")
    for thresh in sorted(stats_collected.keys()):
        stats = stats_collected[thresh]
        prec = stats['tp']/(stats['tp']+stats['fp']) if (stats['tp']+stats['fp']) > 0 else 0
        recall = stats['tp']/(stats['tp']+stats['fn']) if (stats['tp']+stats['fn']) > 0 else 0
        print(f"  IoU@{thresh}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, GT={stats['gt']}, Prec={prec:.3f}, Rec={recall:.3f}")
    print("-" * 80)

def _init_scene_collection(scene_info, iou_distribution):
    """Initialize stats collection for a new scene."""
    caluclate_tp_fp._iou_dist_collected = iou_distribution
    caluclate_tp_fp._stats_collected = {}
    caluclate_tp_fp._thresh_count = 0
    caluclate_tp_fp._current_scene = scene_info

def _handle_scene_switch(scene_info, iou_distribution):
    """Handle scene switch: print previous scene and init new one."""
    if caluclate_tp_fp._thresh_count >= 3 and caluclate_tp_fp._current_scene:
        _print_scene_stats(caluclate_tp_fp._current_scene, caluclate_tp_fp._iou_dist_collected, caluclate_tp_fp._stats_collected)
    _init_scene_collection(scene_info, iou_distribution)

def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, distance_ranges, scene_info=None, collect_stats=False):
    """
    Calculate the true positive and false positive numbers of the current
    frames for different distance ranges.

    Args:
        scene_info: Scene information for logging (optional)
        collect_stats: If True, collect stats for all IoU thresholds and print summary
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score : torch.Tensor
        The confidence score for each predicted bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number for each distance range.
    iou_thresh : float
        The iou threshold.
    distance_ranges : list of tuples
        The list of distance ranges.
    """

    # GT 必须存在，否则无法评估
    assert gt_boxes is not None, "GT boxes must be provided"
    gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)
    gt_count_overall = gt_boxes.shape[0]
    
    # 当没有预测框时，构造空数组，确保仍记录 GT 数量
    if det_boxes is None:
        # 当没有预测框时，所有 GT 都是 FN，直接填充统计结果
        result_stat['overall'][iou_thresh]['score'] += []
        result_stat['overall'][iou_thresh]['fp'] += []
        result_stat['overall'][iou_thresh]['tp'] += []
        result_stat['overall'][iou_thresh]['gt'] += gt_count_overall

        # Handle stats collection mode for empty detections
        if collect_stats:
            if not hasattr(caluclate_tp_fp, '_iou_dist_collected') or caluclate_tp_fp._iou_dist_collected is None:
                _init_scene_collection(scene_info, [])
            elif caluclate_tp_fp._current_scene != scene_info:
                _handle_scene_switch(scene_info, [])

            caluclate_tp_fp._stats_collected[iou_thresh] = {'tp': 0, 'fp': 0, 'gt': gt_count_overall, 'fn': gt_count_overall}
            caluclate_tp_fp._thresh_count += 1

            if caluclate_tp_fp._thresh_count >= 3 and scene_info:
                _print_scene_stats(scene_info, caluclate_tp_fp._iou_dist_collected, caluclate_tp_fp._stats_collected)
                caluclate_tp_fp._iou_dist_collected = None
                caluclate_tp_fp._stats_collected = {}
                caluclate_tp_fp._thresh_count = 0
                caluclate_tp_fp._current_scene = None
            # Continue to distance-specific statistics (don't return here)

        # Distance-specific statistics
        gt_center_distances = [calculate_center_distance(box) for box in gt_boxes]

        for dist_range in distance_ranges:
            gt_count = sum(1 for dist in gt_center_distances if dist_range[0] <= dist < dist_range[1])
            result_stat[str(dist_range)][iou_thresh]['score'] += []
            result_stat[str(dist_range)][iou_thresh]['fp'] += []
            result_stat[str(dist_range)][iou_thresh]['tp'] += []
            result_stat[str(dist_range)][iou_thresh]['gt'] += gt_count
        return
    else:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)

        # Sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]  # from high to low
        det_boxes = det_boxes[score_order_descend]  

        # Convert boxes to polygons
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # Overall statistics
        fp_overall = []
        tp_overall = []
        
        # collect IoU distribution for analysis
        iou_distribution = []

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[i]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            # record max IoU for this prediction
            if len(ious) > 0:
                max_iou = np.max(ious)
                iou_distribution.append(max_iou)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp_overall.append(1)
                tp_overall.append(0)
                continue

            fp_overall.append(0)
            tp_overall.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat['overall'][iou_thresh]['score'] += det_score.tolist()
        result_stat['overall'][iou_thresh]['fp'] += fp_overall
        result_stat['overall'][iou_thresh]['tp'] += tp_overall
        result_stat['overall'][iou_thresh]['gt'] += gt_count_overall

        # Handle stats collection mode
        if collect_stats:
            if not hasattr(caluclate_tp_fp, '_iou_dist_collected') or caluclate_tp_fp._iou_dist_collected is None:
                _init_scene_collection(scene_info, iou_distribution)
            elif caluclate_tp_fp._current_scene != scene_info:
                _handle_scene_switch(scene_info, iou_distribution)

            caluclate_tp_fp._stats_collected[iou_thresh] = {
                'tp': sum(tp_overall),
                'fp': sum(fp_overall),
                'gt': gt_count_overall,
                'fn': max(gt_count_overall - sum(tp_overall), 0)
            }
            caluclate_tp_fp._thresh_count += 1

            if caluclate_tp_fp._thresh_count >= 3 and scene_info:
                _print_scene_stats(scene_info, caluclate_tp_fp._iou_dist_collected, caluclate_tp_fp._stats_collected)
                caluclate_tp_fp._iou_dist_collected = None
                caluclate_tp_fp._stats_collected = {}
                caluclate_tp_fp._thresh_count = 0
                caluclate_tp_fp._current_scene = None
            # Continue to distance-specific statistics (don't return here)

        # Distance-specific statistics
        # Calculate the center distances
        det_center_distances = [calculate_center_distance(box) for box in det_boxes]
        gt_center_distances = [calculate_center_distance(box) for box in gt_boxes]
        
        for dist_range in distance_ranges:
            fp = []
            tp = []
            scores = []
            gt_count = sum(1 for dist in gt_center_distances if dist_range[0] <= dist < dist_range[1])
            
            # Filter GT boxes and polygons by distance range
            current_gt_boxes = [box for box, dist in zip(gt_boxes, gt_center_distances) if dist_range[0] <= dist < dist_range[1]]
            current_gt_polygon_list = list(common_utils.convert_format(current_gt_boxes))
            # Filter DET boxes and polygons by distance range
            current_det_indices = [i for i, dist in enumerate(det_center_distances) if dist_range[0] <= dist < dist_range[1]]
            current_det_boxes = [det_boxes[i] for i in current_det_indices]
            current_det_polygon_list = list(common_utils.convert_format(current_det_boxes))
            current_det_scores = [det_score[i] for i in current_det_indices]

            for i in range(len(current_det_polygon_list)):
                det_polygon = current_det_polygon_list[i]
                ious = common_utils.compute_iou(det_polygon, current_gt_polygon_list)

                if len(current_gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                    fp.append(1)
                    tp.append(0)
                    scores.append(current_det_scores[i])
                    continue

                fp.append(0)
                tp.append(1)
                scores.append(current_det_scores[i])

                gt_index = np.argmax(ious)
                current_gt_polygon_list.pop(gt_index)

            result_stat[str(dist_range)][iou_thresh]['score'] += scores
            result_stat[str(dist_range)][iou_thresh]['fp'] += fp
            result_stat[str(dist_range)][iou_thresh]['tp'] += tp
            result_stat[str(dist_range)][iou_thresh]['gt'] += gt_count


def calculate_ap_for_range(iou_stat):
    """
    Calculate the average precision for a given IoU statistics range.
    Parameters
    ----------
    iou_stat : dict
        A dictionary contains fp, tp and gt number.
    """
    fp = np.array(iou_stat['fp'])
    tp = np.array(iou_stat['tp'])
    score = np.array(iou_stat['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_stat['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def calculate_ap(result_stat, iou, distance_ranges):
    """
    Calculate the average precision and recall for different distance ranges and overall.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number for each distance range and overall.
    iou : float
    distance_ranges : list of tuples
        The list of distance ranges.
    """
    ap_results = {}

    ap_results['overall'] = calculate_ap_for_range(result_stat['overall'][iou])

    for dist_range in distance_ranges:
        ap_results[str(dist_range)] = calculate_ap_for_range(result_stat[str(dist_range)][iou])

    return ap_results


def eval_final_results(result_stat, save_path, distance_ranges, infer_info=None):
    dump_dict = {}

    ap_results_30 = calculate_ap(result_stat, 0.30, distance_ranges)
    ap_results_50 = calculate_ap(result_stat, 0.50, distance_ranges)
    ap_results_70 = calculate_ap(result_stat, 0.70, distance_ranges)

    for range_key, (ap_30, mrec_30, mpre_30) in ap_results_30.items():
        dump_dict[f'ap30_{range_key}'] = ap_30
        dump_dict[f'mprec_30_{range_key}'] = mpre_30
        dump_dict[f'mrec_30_{range_key}'] = mrec_30
    for range_key, (ap_50, mrec_50, mpre_50) in ap_results_50.items():
        dump_dict[f'ap50_{range_key}'] = ap_50
        dump_dict[f'mprec_50_{range_key}'] = mpre_50
        dump_dict[f'mrec_50_{range_key}'] = mrec_50
    for range_key, (ap_70, mrec_70, mpre_70) in ap_results_70.items():
        dump_dict[f'ap70_{range_key}'] = ap_70
        dump_dict[f'mprec_70_{range_key}'] = mpre_70
        dump_dict[f'mrec_70_{range_key}'] = mrec_70
        
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'piecewise_eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'piecewise_eval_{infer_info}.yaml'))

    for range_key, (ap_30, _, _) in ap_results_30.items():
        print(f'The Average Precision at IOU 0.3 for {range_key} is %.4f' % ap_30)
    for range_key, (ap_50, _, _) in ap_results_50.items():
        print(f'The Average Precision at IOU 0.5 for {range_key} is %.4f' % ap_50)
    for range_key, (ap_70, _, _) in ap_results_70.items():
        print(f'The Average Precision at IOU 0.7 for {range_key} is %.4f' % ap_70)

    return ap_results_30, ap_results_50, ap_results_70