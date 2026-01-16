# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic
from opencood.tools.train_utils import set_seed, worker_init_fn
from opencood.tools.train_utils import check_missing_key
from opencood.tools.train_utils import calculate_eta
from opencood.visualization.visual_by_step import visualize_step
torch.backends.cudnn.enabled = False

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, default='opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_lxl_coalign.yaml', help='data generation yaml file needed ')
    parser.add_argument('--resume_dir', default='', help='Continued training path')
    parser.add_argument('--visualize', type=int, default=1, help='Visualize frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.config_parser(opt)
    set_seed(opt.seed)
    generator = torch.Generator()
    generator.manual_seed(opt.seed)

    print('============== Dataset Building ==============')
    opencood_training_dataset = build_dataset(hypes, visualize=True, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)
    training_loader = DataLoader(opencood_training_dataset, batch_size=hypes['train_params']['batch_size'], num_workers=0, collate_fn=opencood_training_dataset.collate_batch_train, shuffle=True, pin_memory=True,  drop_last=True, worker_init_fn=worker_init_fn, generator=generator)
    validate_loader = DataLoader(opencood_validate_dataset, batch_size=hypes['train_params']['batch_size'], num_workers=0, collate_fn=opencood_training_dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False, worker_init_fn=worker_init_fn)

    print('============== Creating Model  ==============')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = train_utils.create_loss(hypes)

    if opt.resume_dir:
        loaded_state_dict = torch.load(opt.resume_dir, map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        init_epoch = int(opt.resume_dir.split('epoch')[-1].split('.')[0])
        saved_path = os.path.dirname(opt.resume_dir)
        print(f"resume from {init_epoch} epoch, saved to {saved_path}")
    else: # train from scratch
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        print(f"train from scratch, saved to {saved_path}")
        
    optimizer = train_utils.setup_optimizer(hypes, model)
    
    # Setup learning rate scheduler
    lr_schedule_config = hypes.get('lr_scheduler', {})
    by_epoch = lr_schedule_config.get('by_epoch', True)
    iters_per_epoch = len(training_loader) if not by_epoch else None
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch, iters_per_epoch)
    
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_training_dataset, "supervise_single") \
        else opencood_training_dataset.supervise_single
    model.to(device)
    print('TOTAL NUMBER OF PARAMETERS: %d' % sum(p.numel() for p in model.parameters()))
    writer = SummaryWriter(saved_path)

    print('============== Training Start ==============')
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups: print('learning rate %f' % param_group["lr"])
        model.train()
        try:  model.model_train_init()
        except: print("No model_train_init function")
        
        epoch_start_time = time.time()
        lr = optimizer.param_groups[0]["lr"]
        for i, batch_data in enumerate(training_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0: continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            eta_str = calculate_eta(epoch_start_time, i, len(training_loader))
            
            # calculate final loss and log
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(training_loader), writer, eta_str=eta_str, lr=lr)
            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(training_loader), writer, eta_str=eta_str, lr=lr, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()
            
            # Update learning rate by iteration if by_epoch=False
            if not by_epoch:
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
            
            if i % opt.visualize == 0 and i > 0:
                visualize_step(ouput_dict, batch_data, opencood_training_dataset, hypes, epoch, i, saved_path, suffix="vis_training")
        
        # Update learning rate by epoch if by_epoch=True
        if by_epoch:
            scheduler.step()
        
        if (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%02d.pth' % (epoch + 1)))

        # reinitialize the dataset by shuffle ego
        opencood_training_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

if __name__ == '__main__':
    main()
