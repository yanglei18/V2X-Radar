# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
import random
import numpy as np


def worker_init_fn(worker_id):
    """Initialize worker with seed for reproducibility
    
    Note: torch.initial_seed() returns the seed set by the DataLoader's generator,
    which is already seeded with the main seed. However, we need to ensure
    that each worker gets a unique but deterministic seed.
    """
    # Get the base seed from the DataLoader's generator
    # This is set by generator.manual_seed(opt.seed) in train_ddp.py
    worker_seed = torch.initial_seed() % 2**32
    # Add worker_id to ensure different workers get different seeds
    # but the same worker always gets the same seed across runs
    worker_seed = (worker_seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Also set torch seed for this worker
    torch.manual_seed(worker_seed)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set CUBLAS workspace config for deterministic behavior (required for CUDA >= 10.2)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Enable deterministic algorithms (PyTorch 1.8+)
    # Note: Some operations may not support deterministic mode
    # try:
    #     torch.use_deterministic_algorithms(True, warn_only=True)
    # except AttributeError:
    #     # For older PyTorch versions
    #     pass
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Additional environment variables for reproducibility
    os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering
    # Set torch default dtype to ensure consistent behavior
    torch.set_default_dtype(torch.float32)

def calculate_eta(epoch_start_time, current_iter, total_iter):
    """
    Calculate estimated time remaining (ETA) for training.
    
    Parameters
    ----------
    epoch_start_time : float
        Start time of the epoch (from time.time())
    current_iter : int
        Current iteration index (0-based)
    total_iter : int
        Total number of iterations in the epoch
    
    Returns
    -------
    str
        ETA string in format "ETA: HH:MM:SS" or empty string if cannot calculate
    """
    if current_iter <= 0:
        return ""
    
    import time
    elapsed = time.time() - epoch_start_time
    avg_time = elapsed / (current_iter + 1)
    remaining = (total_iter - (current_iter + 1)) * avg_time
    h, m, s = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
    return f"ETA: {h:02d}:{m:02d}:{s:02d}"

def _convert_defaultdict_to_dict(d):
    """Convert nested defaultdict to regular dict for pickling."""
    from collections import defaultdict
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        return {k: _convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d

def sync_all_ranks(rank, message="", verbose=True):
    """
    Synchronize all ranks using barrier and print status messages.
    
    This function ensures all ranks reach the same point before continuing,
    which is critical for preventing NCCL timeout in distributed training.
    
    Before calling barrier, this function ensures all CUDA operations and
    DDP communications are complete to prevent timeout issues.
    
    Parameters
    ----------
    rank : int
        Current process rank
    message : str, optional
        Optional message to print before synchronization (default: "")
    verbose : bool, optional
        Whether to print status messages (default: True)
    
    Examples
    --------
    >>> # After training loop
    >>> sync_all_ranks(opt.rank, f"Training loop completed for epoch {epoch + 1}")
    """
    if verbose and rank == 0 and message:
        print(f"Rank {rank}: {message}")
    
    # Ensure all CUDA operations complete before barrier
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Perform a dummy all_reduce to ensure all DDP communications are complete
    # This is critical when find_unused_parameters=True, as DDP may have pending communications
    try:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        dummy_tensor = torch.tensor([0.0], device=device)
        torch.distributed.all_reduce(dummy_tensor, op=torch.distributed.ReduceOp.SUM, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        if verbose and rank == 0:
            print(f"Warning: Pre-barrier synchronization failed: {e}")
    
    # Now perform the barrier to ensure all ranks reach this point
    try:
        torch.distributed.barrier()
    except Exception as e:
        if verbose and rank == 0:
            print(f"Error: Barrier synchronization failed: {e}")
        raise
    
    if verbose and rank == 0 and message:
        print(f"All ranks completed: {message}")

def save_checkpoint_sync(model_without_ddp, saved_path, epoch, rank, message="checkpoint"):
    """
    Save checkpoint with synchronization to prevent NCCL timeout.
    
    This function ensures all ranks are synchronized before and after checkpoint saving,
    which prevents NCCL timeout when only rank 0 performs I/O operations.
    
    Parameters
    ----------
    model_without_ddp : torch.nn.Module
        Model without DDP wrapper
    saved_path : str
        Path to save checkpoint
    epoch : int
        Current epoch number (0-based, will be converted to 1-based for filename)
    rank : int
        Current process rank
    message : str, optional
        Message to display during synchronization (default: "checkpoint")
    
    Examples
    --------
    >>> # Save checkpoint periodically
    >>> save_checkpoint_sync(model_without_ddp, saved_path, epoch, opt.rank, message="checkpoint")
    >>> # Save final checkpoint
    >>> save_checkpoint_sync(model_without_ddp, saved_path, epoch, opt.rank, message="final checkpoint")
    """
    sync_all_ranks(rank, f"Ready to save {message} for epoch {epoch + 1}", verbose=True)
    if rank == 0:
        checkpoint_path = os.path.join(saved_path, 'net_epoch%02d.pth' % (epoch + 1))
        torch.save(model_without_ddp.state_dict(), checkpoint_path)
    sync_all_ranks(rank, f"{message.capitalize()} saved for epoch {epoch + 1}", verbose=True)

def reinitialize_dataset_sync(dataset, epoch, rank):
    """
    Reinitialize dataset with synchronization to prevent NCCL timeout.
    
    This function ensures all ranks are synchronized before and after dataset reinitialization,
    which prevents NCCL timeout when dataset operations may take different time on different ranks.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to reinitialize
    epoch : int
        Current epoch number (0-based, will be converted to 1-based for message)
    rank : int
        Current process rank
    
    Examples
    --------
    >>> # Reinitialize dataset after epoch
    >>> reinitialize_dataset_sync(opencood_training_dataset, epoch, opt.rank)
    """
    import time
    # Ensure all CUDA operations complete before reinitialize
    torch.cuda.synchronize()
    
    # Synchronize before reinitialize
    sync_all_ranks(rank, f"Ready to reinitialize dataset after epoch {epoch + 1}", verbose=True)
    
    # Reinitialize with timing
    start_time = time.time()
    if rank == 0:
        print(f"Rank {rank}: Starting dataset reinitialize for epoch {epoch + 1}...")
    
    try:
        dataset.reinitialize()
        elapsed_time = time.time() - start_time
        if rank == 0:
            print(f"Rank {rank}: Dataset reinitialize completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Rank {rank}: Error during dataset reinitialize: {e}")
        raise
    
    # Ensure all CUDA operations complete after reinitialize
    torch.cuda.synchronize()
    
    # Synchronize after reinitialize
    sync_all_ranks(rank, f"Dataset reinitialized for epoch {epoch + 1}", verbose=True)

def gather_and_merge_eval_results(result_stat, opt, distance_ranges):
    """
    Gather evaluation results from all ranks and merge them on rank 0.
    
    Parameters
    ----------
    result_stat : dict
        Evaluation statistics from current rank
    opt : argparse.Namespace
        Options containing rank and world_size
    distance_ranges : list of tuples
        List of distance ranges for evaluation
    
    Returns
    -------
    dict
        Merged result_stat on rank 0, original result_stat on other ranks
    """
    import pickle
    from collections import defaultdict
    
    # Convert defaultdict to regular dict for pickling (lambda functions can't be pickled)
    result_stat_dict = _convert_defaultdict_to_dict(result_stat)
    
    # Gather results from all ranks to rank 0
    # Add barrier before all_gather_object to ensure all ranks are ready
    torch.distributed.barrier()
    result_stat_bytes = pickle.dumps(result_stat_dict)
    result_stat_list = [None] * opt.world_size
    torch.distributed.all_gather_object(result_stat_list, result_stat_bytes)
    
    # Merge results on rank 0
    if opt.rank == 0:
        merged_result_stat = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [], 'score': [], 'gt': 0}))
        for iou_thresh in [0.3, 0.5, 0.7]:
            merged_result_stat['overall'][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
            for dist_range in distance_ranges: 
                merged_result_stat[str(dist_range)][iou_thresh] = {'tp': [], 'fp': [], 'score': [], 'gt': 0}
        
        for rank_result_bytes in result_stat_list:
            rank_result_stat = pickle.loads(rank_result_bytes)
            for range_key in ['overall'] + [str(dr) for dr in distance_ranges]:
                for iou_thresh in [0.3, 0.5, 0.7]:
                    merged_result_stat[range_key][iou_thresh]['tp'] += rank_result_stat[range_key][iou_thresh]['tp']
                    merged_result_stat[range_key][iou_thresh]['fp'] += rank_result_stat[range_key][iou_thresh]['fp']
                    merged_result_stat[range_key][iou_thresh]['score'] += rank_result_stat[range_key][iou_thresh]['score']
                    merged_result_stat[range_key][iou_thresh]['gt'] += rank_result_stat[range_key][iou_thresh]['gt']
        
        # Debug print for TP / FP / FN statistics on rank0
        # TP/FP are stored为列表，这里用长度作为数量，FN = GT - TP_num
        print("=== Eval TP/FP/FN (merged across ranks) ===")
        for range_key in ['overall'] + [str(dr) for dr in distance_ranges]:
            for iou_thresh in [0.3, 0.5, 0.7]:
                tp_num = sum(merged_result_stat[range_key][iou_thresh]['tp'])
                fp_num = sum(merged_result_stat[range_key][iou_thresh]['fp'])
                gt_num = merged_result_stat[range_key][iou_thresh]['gt']
                fn_num = max(gt_num - tp_num, 0)
                print(f"[{range_key}] IoU={iou_thresh}: TP={tp_num}, FP={fp_num}, FN={fn_num}, GT={gt_num}")

        return merged_result_stat
    else:
        return result_stat

def calculate_and_log_ap_results(result_stat, epoch, distance_ranges, eval_utils_modify, writer=None):
    """
    Calculate AP, print and log results to console and TensorBoard.
    
    Parameters
    ----------
    result_stat : dict
        Merged evaluation statistics
    epoch : int
        Current epoch number
    distance_ranges : list of tuples
        List of distance ranges for evaluation
    eval_utils_modify : module
        Evaluation utils module containing calculate_ap function
    writer : SummaryWriter, optional
        TensorBoard writer for logging
    
    Returns
    -------
    tuple
        (ap_results_30, ap_results_50, ap_results_70) dictionaries
    """
    # Calculate AP for different IoU thresholds
    ap_results_30 = eval_utils_modify.calculate_ap(result_stat, 0.30, distance_ranges)
    ap_results_50 = eval_utils_modify.calculate_ap(result_stat, 0.50, distance_ranges)
    ap_results_70 = eval_utils_modify.calculate_ap(result_stat, 0.70, distance_ranges)
    
    # Print AP results for all distance ranges
    print(f'============== Epoch {epoch} Evaluation Results ==============')
    for range_key in ['overall'] + [str(dr) for dr in distance_ranges]:
        ap30, _, _ = ap_results_30[range_key]
        ap50, _, _ = ap_results_50[range_key]
        ap70, _, _ = ap_results_70[range_key]
        print(f'  {range_key:15s} - AP@0.3: {ap30:.4f}, AP@0.5: {ap50:.4f}, AP@0.7: {ap70:.4f}')
    print('=' * 60)
    
    # Log to tensorboard (overall and all distance ranges)
    if writer is not None:
        for range_key in ['overall'] + [str(dr) for dr in distance_ranges]:
            ap30, _, _ = ap_results_30[range_key]
            ap50, _, _ = ap_results_50[range_key]
            ap70, _, _ = ap_results_70[range_key]
            # Convert range_key to TensorBoard-safe name (replace parentheses and spaces)
            if range_key == 'overall': 
                suffix = ''
            else:  
                suffix = '_' + range_key.replace('(', '').replace(')', '').replace(', ', '_').replace(' ', '_')
            writer.add_scalar(f'Eval/AP_30{suffix}', ap30, epoch)
            writer.add_scalar(f'Eval/AP_50{suffix}', ap50, epoch)
            writer.add_scalar(f'Eval/AP_70{suffix}', ap70, epoch)
    
    return ap_results_30, ap_results_50, ap_results_70

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def check_missing_key(model_state_dict, ckpt_state_dict):
    checkpoint_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    # Get unique module names for missing and extra keys
    missing_key_modules = set()
    for key in missing_keys:
        module_name = key.split('.')[0]
        # Check if the module name actually represents a module in the model
        # by verifying if any key in model_keys starts with this module name
        if any(k.startswith(f"{module_name}.") for k in model_keys):
            missing_key_modules.add(module_name)

    extra_key_modules = set()
    for key in extra_keys:
        module_name = key.split('.')[0]
        # Check if the module name actually represents a module in the checkpoint
        # by verifying if any key in checkpoint_keys starts with this module name
        if any(k.startswith(f"{module_name}.") for k in checkpoint_keys):
            extra_key_modules.add(module_name)

    # print("------ Loading Checkpoint ------")
    if len(missing_key_modules) == 0 and len(extra_key_modules) ==0:
        return

    print("Missing keys from ckpt:")
    print(*missing_key_modules,sep='\n',end='\n\n')
    # Uncomment the next line to see full missing keys
    # print(*missing_keys,sep='\n',end='\n\n')

    print("Extra keys from ckpt:")
    print(*extra_key_modules,sep='\n',end='\n\n')
    # Print full extra keys
    print(*extra_keys,sep='\n',end='\n\n')

    print("You can go to tools/train_utils.py to print the full missing key name!")
    print("--------------------------------")


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epoch = int(result[0].split('_')[0]) if "_" in result[0] else int(result[0])
                print("result[0]: ", result[0], epoch)

                epochs_exist.append(epoch)
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    print(file_list)
    if file_list:    
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch").split('_')[0]), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../work_dirs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']
    target_model_name = backbone_name.replace('_', '').lower()
    model = None

    # First, try direct import from dedicated module (fastest path)
    model_filename = "opencood.models." + backbone_name
    import pkgutil
    import opencood.models as models_pkg
    
    # Get all module names and prioritize similar names
    module_list = []
    for importer, modname, ispkg in pkgutil.iter_modules(models_pkg.__path__):
        if not ispkg:  # Only process .py files, skip sub-packages
            module_list.append(modname)
    
    # Prioritize modules with similar names to backbone_name
    # Count how many parts of backbone_name appear in module name
    def similarity_score(name):
        name_lower = name.lower()
        backbone_parts = backbone_name.split('_')
        matches = sum(1 for part in backbone_parts if part in name_lower)
        return -matches  # Negative for descending order (more matches = higher priority)
    module_list.sort(key=lambda x: (similarity_score(x), x))
    
    for modname in module_list:
        try:
            module = importlib.import_module(f"opencood.models.{modname}")
            for name, cls in module.__dict__.items():
                if isinstance(cls, type) and name.lower() == target_model_name:
                    model = cls
                    break
            if model is not None:
                break
        except (ImportError, AttributeError):
            continue

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']
    target_loss_name = loss_func_name.replace('_', '').lower()
    loss_func = None

    # First, try direct import from dedicated module (fastest path)
    loss_filename = "opencood.models.loss_setting." + loss_func_name

    # If not found, search in all modules under opencood.models.loss_setting
    import pkgutil
    import opencood.models.loss_setting as loss_pkg
    
    # Get all module names and prioritize similar names
    module_list = []
    for importer, modname, ispkg in pkgutil.iter_modules(loss_pkg.__path__):
        if not ispkg:  # Only process .py files, skip sub-packages
            module_list.append(modname)
    
    # Prioritize modules with similar names to loss_func_name
    def similarity_score(name):
        name_lower = name.lower()
        loss_parts = loss_func_name.split('_')
        matches = sum(1 for part in loss_parts if part in name_lower)
        return -matches  # Negative for descending order (more matches = higher priority)
    module_list.sort(key=lambda x: (similarity_score(x), x))
    
    for modname in module_list:
        try:
            module = importlib.import_module(f"opencood.models.loss_setting.{modname}")
            for name, lfunc in module.__dict__.items():
                if isinstance(lfunc, type) and name.lower() == target_loss_name:
                    loss_func = lfunc
                    break
            if loss_func is not None:
                break
        except (ImportError, AttributeError):
            continue

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, iters_per_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.
    optimizer : torch.optimizer
    init_epoch : int, optional
        Initial epoch number (for resume training)
    iters_per_epoch : int, optional
        Number of iterations per epoch (required when by_epoch=False)
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    by_epoch = lr_schedule_config.get('by_epoch', True)
    
    # Calculate total_iters and init_iter for by_epoch=False
    if not by_epoch:
        if iters_per_epoch is None:
            raise ValueError("iters_per_epoch must be provided when by_epoch=False")
        total_iters = iters_per_epoch * hypes['train_params']['epoches']
        init_iter = init_epoch * iters_per_epoch if init_epoch is not None and init_epoch > 0 else None
    else:
        total_iters = None
        init_iter = None
    
    # Warmup configuration
    warmup = lr_schedule_config.get('warmup', None)
    warmup_iters = lr_schedule_config.get('warmup_iters', 0)
    warmup_ratio = lr_schedule_config.get('warmup_ratio', 0.1)
    
    # Get base learning rate
    base_lr = optimizer.param_groups[0]['lr']
    min_lr_ratio = lr_schedule_config.get('min_lr_ratio', 0.0)
    min_lr = base_lr * min_lr_ratio
    
    # Ensure initial_lr is set in param_groups for resume training
    # PyTorch schedulers require initial_lr when last_epoch >= 0
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'CosineAnnealing':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        if by_epoch:
            T_max = hypes['train_params']['epoches']
            # If warmup is enabled, adjust T_max to exclude warmup epochs
            if warmup is not None and warmup_iters > 0:
                T_max = max(1, T_max - warmup_iters)
            # For epoch-based, last_epoch is the epoch number (0-indexed)
            last_epoch_for_scheduler = max(0, last_epoch - 1) if not (warmup and warmup_iters > 0) else -1
        else:
            # total_iters should have been calculated above if by_epoch=False
            T_max = total_iters
            # If warmup is enabled, adjust T_max to exclude warmup iterations
            if warmup is not None and warmup_iters > 0:
                T_max = max(1, T_max - warmup_iters)
            # For iteration-based, calculate initial iteration (0-indexed)
            # If warmup is enabled and we're resuming after warmup, adjust init_iter
            if warmup is not None and warmup_iters > 0:
                # Calculate initial iteration from init_epoch or init_iter
                if init_iter is not None:
                    # If resuming after warmup, adjust init_iter for base scheduler
                    if init_iter > warmup_iters:
                        # We're past warmup, so base scheduler should start from (init_iter - warmup_iters)
                        last_epoch_for_scheduler = max(0, init_iter - warmup_iters - 1)
                    else:
                        # Still in warmup, base scheduler starts from 0
                        last_epoch_for_scheduler = -1
                elif init_epoch is not None and init_epoch > 0:
                    # Estimate: assume each epoch has similar number of iterations
                    estimated_iters_per_epoch = total_iters // hypes['train_params']['epoches']
                    estimated_init_iter = init_epoch * estimated_iters_per_epoch
                    if estimated_init_iter > warmup_iters:
                        # We're past warmup, so base scheduler should start from (estimated_init_iter - warmup_iters)
                        last_epoch_for_scheduler = max(0, estimated_init_iter - warmup_iters - 1)
                    else:
                        # Still in warmup, base scheduler starts from 0
                        last_epoch_for_scheduler = -1
                else:
                    last_epoch_for_scheduler = -1
            else:
                # Calculate initial iteration from init_epoch or init_iter
                if init_iter is not None:
                    last_epoch_for_scheduler = max(0, init_iter - 1)
                elif init_epoch is not None and init_epoch > 0:
                    # Estimate: assume each epoch has similar number of iterations
                    estimated_iters_per_epoch = total_iters // hypes['train_params']['epoches']
                    last_epoch_for_scheduler = max(0, init_epoch * estimated_iters_per_epoch - 1)
                else:
                    last_epoch_for_scheduler = -1
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=last_epoch_for_scheduler)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    # Apply warmup if configured
    if warmup is not None and warmup_iters > 0:
        # Combine warmup with base scheduler
        class WarmupScheduler:
            def __init__(self, base_scheduler, optimizer, warmup_iters, warmup_ratio, by_epoch, base_lr, init_epoch=0, init_iter=0):
                self.base_scheduler = base_scheduler
                self.optimizer = optimizer
                self.warmup_iters = warmup_iters
                self.warmup_ratio = warmup_ratio
                self.by_epoch = by_epoch
                self.base_lr = base_lr
                self.current_iter = init_iter
                self.current_epoch = init_epoch
                
            def step(self, epoch=None):
                if self.by_epoch:
                    self.current_epoch += 1
                    if self.current_epoch <= self.warmup_iters:
                        # Warmup phase: linear warmup
                        lr = self.base_lr * (self.warmup_ratio + (1.0 - self.warmup_ratio) * (self.current_epoch / self.warmup_iters))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        # After warmup, use base scheduler
                        self.base_scheduler.step()
                else:
                    self.current_iter += 1
                    if self.current_iter <= self.warmup_iters:
                        # Warmup phase: linear warmup
                        lr = self.base_lr * (self.warmup_ratio + (1.0 - self.warmup_ratio) * (self.current_iter / self.warmup_iters))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        # After warmup, use base scheduler
                        self.base_scheduler.step()
            
            def get_last_lr(self):
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
        
        # Calculate initial iteration for warmup scheduler
        if init_iter is None:
            init_iter = 0
        scheduler = WarmupScheduler(scheduler, optimizer, warmup_iters, warmup_ratio, by_epoch, base_lr, 
                                   init_epoch=last_epoch, init_iter=init_iter)
    else:
        # No warmup: step through epochs/iterations if needed for resume training
        if by_epoch:
            # Step through epochs
            for _ in range(last_epoch):
                scheduler.step()
        else:
            # Step through iterations
            if init_iter is None and init_epoch is not None:
                # If only init_epoch is provided, estimate init_iter
                # This requires training_loader length, which we don't have here
                # So we'll rely on init_iter being passed explicitly
                pass
            if init_iter is not None and init_iter > 0:
                for _ in range(init_iter):
                    scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
