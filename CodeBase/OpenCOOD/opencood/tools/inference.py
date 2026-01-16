import argparse
import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, test_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization.visual_by_step import visualize_step
from opencood.utils import eval_utils_modify


def inference_parser():
    parser = argparse.ArgumentParser(description="Model inference and evaluation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument("--model_path", "-m", type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument("--visualize", action='store_true', default=False, help='Enable visualization')
    parser.add_argument("--fusion_method", type=str, default="no", help='Fusion method: early, late, no, single, intermediate')
    opt = parser.parse_args()
    return opt


def main():
    opt = inference_parser()
    hypes = yaml_utils.config_parser(opt)
    
    print('================ Dataset Building ================')
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)
    validate_loader = DataLoader(opencood_validate_dataset, batch_size=1, num_workers=4, 
                                 collate_fn=opencood_validate_dataset.collate_batch_test, drop_last=False, pin_memory=False)
    distance_ranges = [(0, 30), (30, 50), (50, 100)]
    
    print('================ Creating Model  ================')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loaded_state_dict = torch.load(opt.model_path, map_location='cpu')
    train_utils.check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
    print(f"Loaded model from {opt.model_path}")
    
    model.to(device)
    model.eval()
    
    print('============== Evaluation Start ==============')
    result_stat = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [], 'score': [], 'gt': 0}))
    for iou_thresh in [0.3, 0.5, 0.7]:
        result_stat['overall'][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
        for dist_range in distance_ranges:
            result_stat[str(dist_range)][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
    
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(validate_loader)):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            
            if opt.fusion_method == 'early' or opt.fusion_method == 'intermediate':
                infer_result, output_dict = test_utils.inference_early_fusion(batch_data, model, opencood_validate_dataset)
            elif opt.fusion_method == 'late':
                infer_result, output_dict = test_utils.inference_late_fusion(batch_data, model, opencood_validate_dataset)
            elif opt.fusion_method == 'no' or opt.fusion_method == 'single':
                infer_result, output_dict = test_utils.inference_no_fusion(batch_data, model, opencood_validate_dataset, single_gt=(opt.fusion_method == 'single'))
            else:
                raise ValueError(f"Invalid fusion method: {opt.fusion_method}")
            
            pred_box_tensor, gt_box_tensor, pred_score = infer_result['pred_box_tensor'], infer_result['gt_box_tensor'], infer_result['pred_score']
            
            for iou_thresh in [0.3, 0.5, 0.7]:
                eval_utils_modify.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou_thresh, distance_ranges)
            
            if opt.visualize:
                print(f"Visualizing batch {i} of {len(validate_loader)}")
                visualize_step(output_dict['ego'], batch_data, opencood_validate_dataset, hypes, 0, i, os.path.dirname(opt.model_path), suffix="vis_inference")
    
    print('============== Evaluation Results ==============')
    save_path = os.path.dirname(opt.model_path)
    eval_utils_modify.eval_final_results(result_stat, save_path, distance_ranges)


if __name__ == '__main__':
    main()