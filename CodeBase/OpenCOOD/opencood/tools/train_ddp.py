import argparse
import os
import statistics
import glob
import time
import pickle
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, test_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import multi_gpu_utils
from opencood.tools.train_utils import set_seed
from opencood.tools.train_utils import worker_init_fn
from opencood.tools.train_utils import check_missing_key
from opencood.tools.train_utils import calculate_eta
from opencood.tools.train_utils import gather_and_merge_eval_results
from opencood.tools.train_utils import calculate_and_log_ap_results
from opencood.tools.train_utils import sync_all_ranks
from opencood.tools.train_utils import save_checkpoint_sync
from opencood.tools.train_utils import reinitialize_dataset_sync
from opencood.visualization.visual_by_step import visualize_step
from opencood.utils import eval_utils_modify
from icecream import ic

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help='data generation yaml file needed ')
    parser.add_argument('--resume_dir', default='', help='Continued training path')
    parser.add_argument('--visualize', type=int, default=200, help='Visualize frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.config_parser(opt)
    opt.fusion_method = hypes['fusion']['fusion_method']
    multi_gpu_utils.init_distributed_mode(opt)
    
    # Set seed for reproducibility (after distributed init, same seed for all ranks)
    set_seed(opt.seed)
    generator = torch.Generator()
    generator.manual_seed(opt.seed)

    print('================ Dataset Building ================')
    opencood_training_dataset = build_dataset(hypes, visualize=True, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)
    sampler_training = DistributedSampler(opencood_training_dataset, seed=opt.seed)
    sampler_validate = DistributedSampler(opencood_validate_dataset, shuffle=False, seed=opt.seed)  # Fixed: Add seed to ensure consistent data distribution across epochs
    training_loader = DataLoader(opencood_training_dataset,  batch_size=hypes['train_params']['batch_size'], sampler=sampler_training, num_workers=4, collate_fn=opencood_training_dataset.collate_batch_train, worker_init_fn=worker_init_fn, generator=generator, drop_last=True)
    validate_loader = DataLoader(opencood_validate_dataset,  batch_size=1, sampler=sampler_validate, num_workers=4,  collate_fn=opencood_training_dataset.collate_batch_test, worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False)
    distance_ranges = [(0, 30), (30, 50), (50, 100)]
    
    print('================ Creating Model  ================')
    # Ensure deterministic model initialization by resetting seed before model creation
    # This is important because model initialization may use random operations
    set_seed(opt.seed)
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = train_utils.create_loss(hypes)

    if opt.resume_dir:
        loaded_state_dict = torch.load(opt.resume_dir, map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        init_epoch = int(opt.resume_dir.split('epoch')[-1].split('.')[0])
        saved_path = os.path.dirname(opt.resume_dir)
        saved_path_rel = os.path.relpath(saved_path, os.getcwd())
        print(f"resume from {init_epoch} epoch, saved to {saved_path_rel}")
    else: # train from scratch
        init_epoch, saved_path = 0, None
        # All ranks need saved_path, but only rank 0 creates the directory
        # setup_train is safe to call on all ranks as it handles directory creation with exception handling
        if opt.rank == 0:
            saved_path = train_utils.setup_train(hypes)
            saved_path_rel = os.path.relpath(saved_path, os.getcwd())
            print(f"train from scratch, saved to {saved_path_rel}")

    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_training_dataset, "supervise_single") \
        else opencood_training_dataset.supervise_single
    model.to(device)
    # Synchronize all ranks before wrapping with DDP to ensure consistent initialization
    torch.distributed.barrier()
    # Use broadcast_buffers=True to ensure BN buffers are synchronized
    # This helps maintain consistency across ranks
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[opt.gpu], 
        find_unused_parameters=True,  # Keep False - unused parameters are handled in HGTCavAttention
        broadcast_buffers=True,  # Synchronize BN buffers across ranks
        gradient_as_bucket_view=True  # Use gradient buckets for better determinism
    )
    model_without_ddp = model.module
    print('TOTAL NUMBER OF PARAMETERS: %d' % sum(p.numel() for p in model_without_ddp.parameters()))
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    # Setup learning rate scheduler
    lr_schedule_config = hypes.get('lr_scheduler', {})
    by_epoch = lr_schedule_config.get('by_epoch', True)
    iters_per_epoch = len(training_loader) if not by_epoch else None
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch, iters_per_epoch)
    if opt.rank==0: writer = SummaryWriter(saved_path)
    assert opt.half==False, "half precision training is not encouraged"
    
    print('================ Training Start ================')
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups: print('learning rate %f' % param_group["lr"])
        sampler_training.set_epoch(epoch)
        model.train()
        try: model_without_ddp.model_train_init()
        except: print("No model_train_init function")
        
        epoch_start_time = time.time()
        for i, batch_data in enumerate(training_loader):
            # Calculate local skip decision
            lr = optimizer.param_groups[0]["lr"]
            local_skip = batch_data is None or batch_data['ego']['object_bbx_mask'].sum() == 0
            # Synchronize skip decision across all ranks in distributed training
            local_skip_tensor = torch.as_tensor(int(local_skip), dtype=torch.int, device=device)
            torch.distributed.all_reduce(local_skip_tensor, op=torch.distributed.ReduceOp.MAX)
            global_skip = (local_skip_tensor.item() > 0)
            # Skip batch if any rank decides to skip (to maintain synchronization)
            if global_skip: continue
            
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            eta_str = calculate_eta(epoch_start_time, i, len(training_loader))
            
            # calculate final loss and log
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            if opt.rank==0: criterion.logging(epoch + 1, i, len(training_loader), writer, eta_str=eta_str, lr=lr)   
            if supervise_single_flag: 
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                if opt.rank==0: criterion.logging(epoch + 1, i, len(training_loader), writer, eta_str=eta_str, lr=lr, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()
            if not by_epoch: scheduler.step()
            
            if opt.rank == 0 and i % opt.visualize == 0 and i > 0:
                visualize_step(ouput_dict, batch_data, opencood_training_dataset, hypes, epoch + 1, i, saved_path, suffix="vis_training")
        
        # Update learning rate by epoch if by_epoch=True
        if by_epoch: scheduler.step()
        
        # Save checkpoint periodically
        if (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            save_checkpoint_sync(model_without_ddp, saved_path, epoch, opt.rank, message="checkpoint")

        if (epoch + 1) % hypes['train_params']['eval_freq'] == 0:
            if opt.rank == 0: print('============== Evaluation Start ==============')
            # Fixed: Remove set_epoch for validation to ensure consistent data order across epochs
            # sampler_validate.set_epoch(epoch)  # Commented out to prevent evaluation result fluctuations
            model_without_ddp.eval()
            result_stat = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [], 'score': [], 'gt': 0}))
            for iou_thresh in [0.3, 0.5, 0.7]:
                result_stat['overall'][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
                for dist_range in distance_ranges: result_stat[str(dist_range)][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
            with torch.no_grad():
                for i, batch_data in enumerate(validate_loader):
                    if batch_data is None: continue
                    batch_data = train_utils.to_device(batch_data, device)
                    if opt.fusion_method == 'early' or opt.fusion_method == 'intermediate': infer_result, output_dict = test_utils.inference_early_fusion(batch_data, model, opencood_validate_dataset)
                    if opt.fusion_method == 'late': infer_result, output_dict = test_utils.inference_late_fusion(batch_data, model, opencood_validate_dataset)
                    if opt.fusion_method == 'no': infer_result, output_dict = test_utils.inference_no_fusion(batch_data, model, opencood_validate_dataset)
                    if opt.fusion_method == 'single': infer_result, output_dict = test_utils.inference_no_fusion(batch_data, model, opencood_validate_dataset, single_gt=True)    
                    pred_box_tensor, gt_box_tensor, pred_score = infer_result['pred_box_tensor'], infer_result['gt_box_tensor'], infer_result['pred_score']
                    for iou_thresh in [0.3, 0.5, 0.7]: eval_utils_modify.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou_thresh, distance_ranges)
                    if opt.rank == 0 and i % opt.visualize == 0:
                        print("processing batch %d of %d" % (i, len(validate_loader)))
                        visualize_step(output_dict['ego'], batch_data, opencood_training_dataset, hypes, epoch + 1, i, saved_path, suffix="vis_validation")
            result_stat = gather_and_merge_eval_results(result_stat, opt, distance_ranges)
            if opt.rank == 0: calculate_and_log_ap_results(result_stat, epoch + 1, distance_ranges, eval_utils_modify, writer)
            model_without_ddp.train()
        reinitialize_dataset_sync(opencood_training_dataset, epoch, opt.rank)
    save_checkpoint_sync(model_without_ddp, saved_path, epoch, opt.rank, message="final checkpoint")

if __name__ == '__main__':
    main()
