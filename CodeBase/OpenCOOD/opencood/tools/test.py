# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, test_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils_modify
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')
from collections import defaultdict
from opencood.tools.train_utils import check_missing_key
from opencood.tools.train_utils import set_seed
from opencood.visualization.visual_by_step import visualize_step
from opencood.visualization.visual_by_step import extract_timestamp_and_index

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, default='opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign_whole.yaml', help='data generation yaml file needed')
    parser.add_argument('--ckpt_path', type=str, default='opencood/work_dirs/v2x_radar_collab_radaronly_radarpillarnet_coalign_whole_2026_01_02_09_00_26/net_epoch24.pth', help='Continued training path')
    parser.add_argument('--fusion_method', type=str, default='intermediate', help='no, single, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=10, help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true', help='whether to save prediction and gt result in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4", help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true', help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 
    hypes = yaml_utils.config_parser(opt)
    hypes['fusion']['fusion_method'] = opt.fusion_method # NOTE: need to update fusion method in hypes
    set_seed(opt.seed)

    # update range by command line argument
    x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
    y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
    opt.note += f"_{x_max}_{y_max}"
    new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
    hypes = update_dict(hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range})

    # reload anchor config
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
    hypes = parser_func(hypes)
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization. left hand: OPV2V, V2XSet; right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    if 'box_align' in hypes.keys(): hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('============== Creating Model ==============')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # saved_path = os.path.dirname(opt.ckpt_path)
    saved_path = train_utils.setup_train(hypes)
    loaded_state_dict = torch.load(opt.ckpt_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
    print(f"load checkpoint from {opt.ckpt_path}")
    model.to(device)
    model.eval()
    print('TOTAL NUMBER OF PARAMETERS: %d' % sum(p.numel() for p in model.parameters()))

    # build dataset for inference
    print('============== Dataset Building ==============')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset, batch_size=1, num_workers=4, collate_fn=opencood_dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
    distance_ranges = [(0, 30), (30, 50), (50, 100)]
    result_stat = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [],  'score': [], 'gt': 0 }))
    for iou_thresh in [0.3, 0.5, 0.7]:
        result_stat['overall'][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
        for dist_range in distance_ranges:
            result_stat[str(dist_range)][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
    infer_info = opt.fusion_method + opt.note

    print('============== Inference Start ===============')
    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None: continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            # Extract timestamp and index from yaml path
            yaml_file_path_list = batch_data['ego']['yaml_file_path']
            timestamp, index = extract_timestamp_and_index(yaml_file_path_list)
            if opt.fusion_method == 'early' or opt.fusion_method == 'intermediate': infer_result, output_dict = test_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            if opt.fusion_method == 'late': infer_result, output_dict = test_utils.inference_late_fusion(batch_data, model, opencood_dataset)
            if opt.fusion_method == 'no': infer_result, output_dict = test_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            if opt.fusion_method == 'single': infer_result, output_dict = test_utils.inference_no_fusion(batch_data, model, opencood_dataset, single_gt=True)
            
            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            for iou_thresh in [0.3, 0.5, 0.7]:
                eval_utils_modify.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou_thresh, distance_ranges, scene_info=f"{timestamp}_{index}", collect_stats=True)
        
            cav_box_np, agent_modality_list = test_utils.get_cav_box(batch_data)
            infer_result.update({"cav_box_np": cav_box_np, "agent_modality_list": agent_modality_list})
            if not opt.no_score: infer_result.update({'score_tensor': pred_score})
            
            if opt.save_npy:
                npy_save_path = os.path.join(saved_path, 'npy')
                if not os.path.exists(npy_save_path): os.makedirs(npy_save_path)
                test_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], i, npy_save_path)

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(saved_path, f'vis_infer_{infer_info}')
                if not os.path.exists(vis_save_path_root): os.makedirs(vis_save_path_root)
                vis_save_path = os.path.join(vis_save_path_root, f'bev_{timestamp}_{index}.png')
                simple_vis.visualize(infer_result, batch_data['ego']['origin_lidar'][0], hypes['postprocess']['gt_range'], vis_save_path, method='bev', left_hand=left_hand)
                visualize_step(output_dict['ego'], batch_data, opencood_dataset, hypes, 0, i, vis_save_path_root)
            
    ap_results_30, ap_results_50, ap_results_70 = eval_utils_modify.eval_final_results(result_stat, saved_path, distance_ranges, infer_info)

if __name__ == '__main__':
    main()
