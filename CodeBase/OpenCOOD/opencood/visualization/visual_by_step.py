"""
Training visualization utilities for OpenCOOD.
Provides functions to visualize BEV features, LiDAR point clouds, and camera images with projected bounding boxes.
"""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image
from opencood.utils.camera_utils import indices_to_depth
VISUALIZE_SCORE_THRESHOLD = 0.2

def extract_timestamp_and_index(yaml_file_path_list):
    """
    Extract timestamp and index from yaml file path for visualization naming.

    Parameters
    ----------
    yaml_file_path_list : str or list
        Path or list of paths to yaml files

    Returns
    -------
    tuple
        (timestamp, index) for file naming
    """
    # Extract timestamp and index from yaml path (consistent with training_visualization.py)
    if isinstance(yaml_file_path_list, list):
        yaml_file_path = yaml_file_path_list[0]
    if isinstance(yaml_file_path, list):
        yaml_file_path = yaml_file_path[0]

    yaml_file_path = yaml_file_path.split("/")
    timestamp = yaml_file_path[-3]
    index = yaml_file_path[-1].replace(".yaml", "")

    return timestamp, index

def visualize_step(output_dict, batch_data, dataset, hypes, epoch, iter_idx, saved_path, suffix=""):
    """
    Visualize training step including BEV features, LiDAR point clouds, and camera images with GT boxes.
    
    Args:
        output_dict: Model output dictionary containing 'fused_feature'
        batch_data: Batch data dictionary containing 'ego' key with 'origin_lidar' and 'inputs_m1'
        dataset: Training dataset object with post_processor
        hypes: Hyperparameters dictionary
        epoch: Current epoch number
        iter_idx: Current iteration index
        saved_path: Path to save visualization results
    """
    modality_setting_list = {}
    for modality in hypes["heter"]["modality_setting"]:
        modality_setting_list.update({modality: hypes["heter"]["modality_setting"][modality]["sensor_type"]})
    
    vis_save_dir = os.path.join(saved_path, suffix)
    os.makedirs(vis_save_dir, exist_ok=True)
    batch_size = output_dict["fused_feature"].shape[0]
    
    if "fused_feature" in output_dict:
        visualize_bev_features(output_dict, batch_data, vis_save_dir, epoch, iter_idx, batch_size)
        
    if "origin_lidar" in batch_data["ego"]:
        visualize_pointcloud_with_boxes_per_batch(output_dict, batch_data, dataset, hypes, vis_save_dir, epoch, iter_idx, batch_size, pc_type="lidar")
    
    if "origin_radar" in batch_data["ego"]:
        visualize_pointcloud_with_boxes_per_batch(output_dict, batch_data, dataset, hypes, vis_save_dir, epoch, iter_idx, batch_size, pc_type="radar")
        
    if "camera" in hypes["input_source"]:
        camera_index = list(modality_setting_list.values()).index("camera")
        camera_name = list(modality_setting_list.keys())[camera_index]
        camera_input_name = f"inputs_{camera_name}"
        fusion_method = hypes['fusion']['fusion_method']
        if fusion_method == 'intermediate' or fusion_method == 'early': # need pairwise_t_matrix
            visualize_camera_images_with_boxes_per_batch(output_dict, batch_data, dataset, hypes, vis_save_dir, epoch, iter_idx, batch_size, camera_input_name)
        depth_items_key = f"depth_items_{camera_name}"
        if depth_items_key in output_dict:
            visualize_depth_per_batch(output_dict[depth_items_key], batch_data, hypes, vis_save_dir, epoch, iter_idx, batch_size, camera_name)


def visualize_bev_features(output_dict, batch_data, vis_save_dir, epoch, iter_idx, batch_size):
    """
    Visualize BEV features as heatmap.
    
    Args:
        fused_feature: Fused BEV features tensor [B, C, H, W]
        batch_data: Batch data dictionary containing 'ego' key with 'yaml_file_path'
        vis_save_dir: Directory to save visualization
        epoch: Current epoch number
        iter_idx: Current iteration index
        batch_size: Batch size
    """
    yaml_file_path_list = batch_data["ego"]["yaml_file_path"]
    bev_feats_show_raw = output_dict["fused_feature"].max(1, keepdim=True).values
    bev_score_show_raw = torch.sigmoid(output_dict["cls_preds"]).max(1, keepdim=True).values
    bev_feats_show_raw = torch.flip(bev_feats_show_raw, dims=[2])
    bev_score_show_raw = torch.flip(bev_score_show_raw, dims=[2])
    
    bev_feats_show = bev_score_show_raw
    for batch_idx in range(batch_size):
        bev_feats_tmp = bev_feats_show[batch_idx:batch_idx+1, :, :, :]
        bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min()) / (bev_feats_tmp.max() - bev_feats_tmp.min())
        bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
        bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3]
        bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
        yaml_file_path = yaml_file_path_list[batch_idx]
        if isinstance(yaml_file_path, list):
            yaml_file_path = yaml_file_path[0]
        yaml_file_path = yaml_file_path.split("/")
        timestamp, index = yaml_file_path[-3], yaml_file_path[-1].replace(".yaml", "")
        save_image(bev_feats_tmp_colored, os.path.join(vis_save_dir, f"epoch{epoch}_iter{iter_idx}_batch{batch_idx}_bev_score_{timestamp}_{index}.png"))

    bev_feats_show = bev_feats_show_raw
    for batch_idx in range(batch_size):
        bev_feats_tmp = bev_feats_show[batch_idx:batch_idx+1, :, :, :]
        bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min()) / (bev_feats_tmp.max() - bev_feats_tmp.min())
        bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
        bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3]
        bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
        yaml_file_path = yaml_file_path_list[batch_idx]
        if isinstance(yaml_file_path, list):
            yaml_file_path = yaml_file_path[0]
        yaml_file_path = yaml_file_path.split("/")
        timestamp, index = yaml_file_path[-3], yaml_file_path[-1].replace(".yaml", "")
        save_image(bev_feats_tmp_colored, os.path.join(vis_save_dir, f"epoch{epoch}_iter{iter_idx}_batch{batch_idx}_bev_feats_{timestamp}_{index}.png"))


def visualize_pointcloud_with_boxes_per_batch(output_dict, batch_data, dataset, hypes, vis_save_dir, epoch, iter_idx, batch_size, pc_type):
    """
    Visualize LiDAR point clouds with GT and predicted bounding boxes, separated by batch.
    
    Args:
        output_dict: Model output dictionary
        batch_data: Batch data dictionary containing 'ego' key
        dataset: Training dataset object with post_processor
        hypes: Hyperparameters dictionary
        vis_save_dir: Directory to save visualization
        epoch: Current epoch number
        iter_idx: Current iteration index
        batch_size: Batch size
    """
    if pc_type == "lidar": pc = batch_data["ego"]["origin_lidar"].cpu().detach().numpy()
    if pc_type == "radar": pc = batch_data["ego"]["origin_radar"].cpu().detach().numpy()
    yaml_file_path_list = batch_data["ego"]["yaml_file_path"]
    
    for batch_idx in range(batch_size):
        if not batch_size == 1: 
            object_ids = batch_data["ego"]["object_ids"][batch_idx]
            if isinstance(object_ids, list):
                if len(object_ids) > 0:
                    if isinstance(object_ids[0], list):
                        object_ids = [item for sublist in object_ids for item in iter(sublist)]
        else: 
            object_ids = batch_data["ego"]["object_ids"]
            # Handle nested list case even when batch_size == 1
            if isinstance(object_ids, list):
                if len(object_ids) > 0:
                    if isinstance(object_ids[0], list):
                        object_ids = [item for sublist in object_ids for item in iter(sublist)]

        transformation_matrix_clean = batch_data["ego"]["transformation_matrix_clean"]
        if transformation_matrix_clean.dim() == 3:
            transformation_matrix_clean = transformation_matrix_clean[batch_idx]
        if isinstance(transformation_matrix_clean, torch.Tensor):
            transformation_matrix_clean = transformation_matrix_clean.cpu()
        
        transformation_matrix = batch_data["ego"]["transformation_matrix"][batch_idx]
        if isinstance(transformation_matrix, torch.Tensor):
            transformation_matrix = transformation_matrix.cpu()
        
        object_bbx_center = batch_data["ego"]["object_bbx_center"][batch_idx]
        if isinstance(object_bbx_center, torch.Tensor):
            object_bbx_center = object_bbx_center.cpu()
            if object_bbx_center.dim() > 2:
                object_bbx_center = object_bbx_center.squeeze(0)
        
        object_bbx_mask = batch_data["ego"]["object_bbx_mask"][batch_idx]
        if isinstance(object_bbx_mask, torch.Tensor):
            object_bbx_mask = object_bbx_mask.cpu().int()
            if object_bbx_mask.dim() > 1:
                object_bbx_mask = object_bbx_mask.squeeze(0)
        
        anchor_box = batch_data["ego"]["anchor_box"]
        single_batch_data = {
            "ego": {
                'object_bbx_center': object_bbx_center, 
                'object_bbx_mask': object_bbx_mask, 
                'object_ids': object_ids, 
                'transformation_matrix_clean': transformation_matrix_clean, 
                'transformation_matrix': transformation_matrix, 
                'anchor_box': anchor_box
            }
        }
        
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(single_batch_data)
        gt_box_np = gt_box_tensor.cpu().numpy() if gt_box_tensor is not None else None
        
        single_output_dict = {"ego": {}}
        for key in output_dict.keys():
            if isinstance(output_dict[key], torch.Tensor) and output_dict[key].dim() > 0:
                single_output_dict["ego"][key] = output_dict[key][batch_idx:batch_idx+1]
            else:
                single_output_dict["ego"][key] = output_dict[key]
        
        try:
            pred_box_tensor, pred_score = dataset.post_processor.post_process(single_batch_data, single_output_dict)
            pred_box_tensor = pred_box_tensor[pred_score > VISUALIZE_SCORE_THRESHOLD]
            pred_score = pred_score[pred_score > VISUALIZE_SCORE_THRESHOLD]
            pred_box_np = pred_box_tensor.cpu().numpy() if (pred_box_tensor is not None and len(pred_box_tensor) > 0) else None
        except Exception:
            pred_box_np = None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        pc_colors = pc[batch_idx, :, 3]
        rsu_mask = pc_colors == 0
        car_mask = pc_colors == 1
        if pc_type == "lidar":
            ax.scatter(pc[batch_idx, rsu_mask, 0], pc[batch_idx, rsu_mask, 1], s=0.01, c="blue", alpha=0.5, label="RSU")
            ax.scatter(pc[batch_idx, car_mask, 0], pc[batch_idx, car_mask, 1], s=0.01, c="red", alpha=0.5, label="car")
        if pc_type == "radar":
            ax.scatter(pc[batch_idx, rsu_mask, 0], pc[batch_idx, rsu_mask, 1], s=5.00, c="blue", alpha=0.3, marker='x', label="RSU")
            ax.scatter(pc[batch_idx, car_mask, 0], pc[batch_idx, car_mask, 1], s=5.00, c="red", alpha=0.3, marker='x', label="car")
        
        if gt_box_np is not None and len(gt_box_np) > 0:
            for box_idx in range(len(gt_box_np)):
                box_corners = gt_box_np[box_idx, :4, :2]
                ax.fill(box_corners[:, 0], box_corners[:, 1], color="green",
                        alpha=0.5,
                        label=("GT" if box_idx == 0 else ""))
                box_corners_closed = np.vstack([box_corners, box_corners[0]])
                ax.plot(box_corners_closed[:, 0], box_corners_closed[:, 1], "-", linewidth=1.5,
                        alpha=0.3,
                        color="orange")
                
        if pred_box_np is not None and len(pred_box_np) > 0:
            for box_idx in range(len(pred_box_np)):
                box_corners = pred_box_np[box_idx, :4, :2]
                box_corners_closed = np.vstack([box_corners, box_corners[0]])
                ax.plot(box_corners_closed[:, 0], box_corners_closed[:, 1], "-", linewidth=2.0,
                        alpha=0.8,
                        color="orange",
                        label=("Pred" if box_idx == 0 else ""))
        
        ax.set_xlim(hypes["postprocess"]["gt_range"][0], hypes["postprocess"]["gt_range"][3])
        ax.set_ylim(hypes["postprocess"]["gt_range"][1], hypes["postprocess"]["gt_range"][4])
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        yaml_file_path = yaml_file_path_list[batch_idx]
        if isinstance(yaml_file_path, list):
            yaml_file_path = yaml_file_path[0]
        yaml_file_path = yaml_file_path.split("/")
        timestamp, index = yaml_file_path[-3], yaml_file_path[-1].replace(".yaml", "")
        fig.savefig(os.path.join(vis_save_dir, f"epoch{epoch}_iter{iter_idx}_batch{batch_idx}_{pc_type}_{timestamp}_{index}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def visualize_camera_images_with_boxes_per_batch(output_dict, batch_data, dataset, hypes, vis_save_dir, epoch, iter_idx, batch_size, camera_input_name):
    """
    Visualize camera images with projected GT and predicted bounding boxes, separated by batch.
    
    Args:
        output_dict: Model output dictionary
        batch_data: Batch data dictionary containing 'ego' key
        dataset: Training dataset object with post_processor
        hypes: Hyperparameters dictionary
        vis_save_dir: Directory to save visualization
        epoch: Current epoch number
        iter_idx: Current iteration index
        batch_size: Current batch size
    """
    inputs_m1 = batch_data["ego"][camera_input_name]
    # 根据每个 batch 的 record_len（CAV 数）对 [B * N_cav, ...] 维度进行切分，而不是简单按 batch_size 均分
    record_len_list = batch_data["ego"]["record_len"].cpu().numpy().tolist()
    imgs_all = inputs_m1["imgs"]
    intrins_all = inputs_m1["intrins"]
    extrinsics_all = inputs_m1["extrinsics"]
    post_rots_all = inputs_m1["post_rots"]
    post_trans_all = inputs_m1["post_trans"]

    batch_images, batch_intrins, batch_extrinsics = [], [], []
    batch_post_rots, batch_post_trans = [], []
    start_idx = 0
    for rl in record_len_list:
        end_idx = start_idx + rl
        batch_images.append(imgs_all[start_idx:end_idx])
        batch_intrins.append(intrins_all[start_idx:end_idx])
        batch_extrinsics.append(extrinsics_all[start_idx:end_idx])
        batch_post_rots.append(post_rots_all[start_idx:end_idx])
        batch_post_trans.append(post_trans_all[start_idx:end_idx])
        start_idx = end_idx
    yaml_file_path_list = batch_data["ego"]["yaml_file_path"]
    
    for batch_idx, (image, intrins, extrinsics, post_rots, post_trans) in enumerate(
        zip(batch_images, batch_intrins, batch_extrinsics, batch_post_rots, batch_post_trans)
    ):
        if not batch_size == 1: 
            object_ids = batch_data["ego"]["object_ids"][batch_idx]
            if isinstance(object_ids, list):
                if len(object_ids) > 0:
                    if isinstance(object_ids[0], list):
                        object_ids = [item for sublist in object_ids for item in iter(sublist)]
        else: 
            object_ids = batch_data["ego"]["object_ids"]
            # Handle nested list case even when batch_size == 1
            if isinstance(object_ids, list):
                if len(object_ids) > 0:
                    if isinstance(object_ids[0], list):
                        object_ids = [item for sublist in object_ids for item in iter(sublist)]
        
        transformation_matrix_clean = batch_data["ego"]["transformation_matrix_clean"]
        if transformation_matrix_clean.dim() == 3:
            transformation_matrix_clean = transformation_matrix_clean[batch_idx]
        if isinstance(transformation_matrix_clean, torch.Tensor):
            transformation_matrix_clean = transformation_matrix_clean.cpu()
        
        transformation_matrix = batch_data["ego"]["transformation_matrix"][batch_idx]
        if isinstance(transformation_matrix, torch.Tensor):
            transformation_matrix = transformation_matrix.cpu()
        
        object_bbx_center = batch_data["ego"]["object_bbx_center"][batch_idx]
        if isinstance(object_bbx_center, torch.Tensor):
            object_bbx_center = object_bbx_center.cpu()
            if object_bbx_center.dim() > 2:
                object_bbx_center = object_bbx_center.squeeze(0)
        
        object_bbx_mask = batch_data["ego"]["object_bbx_mask"][batch_idx]
        if isinstance(object_bbx_mask, torch.Tensor):
            object_bbx_mask = object_bbx_mask.cpu().int()
            if object_bbx_mask.dim() > 1:
                object_bbx_mask = object_bbx_mask.squeeze(0)
        
        anchor_box = batch_data["ego"]["anchor_box"]
        single_batch_data = {
            "ego": {
                'object_bbx_center': object_bbx_center, 
                'object_bbx_mask': object_bbx_mask, 
                'object_ids': object_ids, 
                'transformation_matrix_clean': transformation_matrix_clean, 
                'transformation_matrix': transformation_matrix, 
                'anchor_box': anchor_box
            }
        }
        
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(single_batch_data)
        gt_box_np = gt_box_tensor.cpu().numpy() if gt_box_tensor is not None else None
        
        single_output_dict = {"ego": {}}
        for key in output_dict.keys():
            if isinstance(output_dict[key], torch.Tensor) and output_dict[key].dim() > 0:
                single_output_dict["ego"][key] = output_dict[key][batch_idx:batch_idx+1]
            else:
                single_output_dict["ego"][key] = output_dict[key]
        
        try:
            pred_box_tensor, pred_score = dataset.post_processor.post_process(single_batch_data, single_output_dict)
            pred_box_tensor = pred_box_tensor[pred_score > VISUALIZE_SCORE_THRESHOLD]
            pred_score = pred_score[pred_score > VISUALIZE_SCORE_THRESHOLD]
            pred_box_np = pred_box_tensor.cpu().numpy() if (pred_box_tensor is not None and len(pred_box_tensor) > 0) else None
        except Exception:
            pred_box_np = None
        
        pairwise_t_matrix = batch_data["ego"]["pairwise_t_matrix"][batch_idx].cpu().numpy()
        record_len = batch_data["ego"]["record_len"][batch_idx].item()
        
        if image.shape[2] > 3:
            image = image[:, :, :3, :, :]
        
        image = image.flatten(0, 1)
        intrins = intrins.flatten(0, 1)
        extrinsics = extrinsics.flatten(0, 1)
        post_rots = post_rots.flatten(0, 1)
        post_trans = post_trans.flatten(0, 1)
        
        N_cam_total = image.shape[0]
        N_cam_per_cav = N_cam_total // record_len
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.cpu().numpy()
        N_cam = image_np.shape[0]
        H, W = image_np.shape[2], image_np.shape[3]
        img_list = []
        
        for cam_idx in range(N_cam):
            img_cam = image_np[cam_idx].transpose(1, 2, 0)
            img_cam = (img_cam * 255).astype(np.uint8)
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_RGB2BGR)
            img_cam = np.ascontiguousarray(img_cam)
            cav_idx = cam_idx // N_cam_per_cav
            
            from opencood.utils.box_utils import project_box3d
            
            if gt_box_np is not None:
                if cav_idx == 0:
                    gt_box_np_cav = gt_box_np
                else:
                    gt_box_tensor_cav = torch.from_numpy(gt_box_np).float()
                    T_ego_to_cav = pairwise_t_matrix[0, cav_idx]
                    T_ego_to_cav_torch = torch.from_numpy(T_ego_to_cav).float()
                    gt_box_tensor_cav = project_box3d(gt_box_tensor_cav, T_ego_to_cav_torch)
                    gt_box_np_cav = gt_box_tensor_cav.cpu().numpy()
            else:
                gt_box_np_cav = None
            
            if pred_box_np is not None:
                if cav_idx == 0:
                    pred_box_np_cav = pred_box_np
                else:
                    pred_box_tensor_cav = torch.from_numpy(pred_box_np).float()
                    T_ego_to_cav = pairwise_t_matrix[0, cav_idx]
                    T_ego_to_cav_torch = torch.from_numpy(T_ego_to_cav).float()
                    pred_box_tensor_cav = project_box3d(pred_box_tensor_cav, T_ego_to_cav_torch)
                    pred_box_np_cav = pred_box_tensor_cav.cpu().numpy()
            else:
                pred_box_np_cav = None
            
            int_matrix_3x3 = intrins[cam_idx].cpu().numpy()
            ext_matrix = extrinsics[cam_idx].cpu().numpy()
            post_rot = post_rots[cam_idx].cpu().numpy()
            post_tran = post_trans[cam_idx].cpu().numpy()
            
            if post_rot.ndim > 2:
                post_rot = post_rot.squeeze()
            if post_tran.ndim > 1:
                post_tran = post_tran.squeeze()
            
            if pred_box_np_cav is not None:
                pred_box2d, pred_box2d_mask = project_boxes_to_image(pred_box_np_cav, int_matrix_3x3, ext_matrix, post_rot, post_tran, H, W)
                for box_idx in range(len(pred_box2d)):
                    if pred_box2d_mask[box_idx]:
                        box_2d = pred_box2d[box_idx]
                        edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                                 [4, 5], [5, 6], [6, 7], [7, 4],
                                 [0, 4], [1, 5], [2, 6], [3, 7]]
                        for edge in edges:
                            pt1 = tuple(box_2d[edge[0]].astype(int))
                            pt2 = tuple(box_2d[edge[1]].astype(int))
                            if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
                                cv2.line(img_cam, pt1, pt2, (0, 0, 255), 1)
            if gt_box_np_cav is not None:
                gt_box2d, gt_box2d_mask = project_boxes_to_image(gt_box_np_cav, int_matrix_3x3, ext_matrix, post_rot, post_tran, H, W)
                for box_idx in range(len(gt_box2d)):
                    if gt_box2d_mask[box_idx]:
                        box_2d = gt_box2d[box_idx]
                        edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                                 [4, 5], [5, 6], [6, 7], [7, 4],
                                 [0, 4], [1, 5], [2, 6], [3, 7]]
                        for edge in edges:
                            pt1 = tuple(box_2d[edge[0]].astype(int))
                            pt2 = tuple(box_2d[edge[1]].astype(int))
                            if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
                                cv2.line(img_cam, pt1, pt2, (0, 255, 0), 1)
            
            img_list.append(img_cam)
        
        nrow = N_cam_per_cav
        ncol = math.ceil(len(img_list) / nrow)
        grid_h = H * ncol
        grid_w = W * nrow
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx, img in enumerate(img_list):
            row = idx // nrow
            col = idx % nrow
            y_start = row * H
            y_end = y_start + H
            x_start = col * W
            x_end = x_start + W
            grid_img[y_start:y_end, x_start:x_end] = img
        
        yaml_file_path = yaml_file_path_list[batch_idx]
        if isinstance(yaml_file_path, list):
            yaml_file_path = yaml_file_path[0]
        yaml_file_path = yaml_file_path.split("/")
        timestamp, index = yaml_file_path[-3], yaml_file_path[-1].replace(".yaml", "")
        img_save_path = os.path.join(vis_save_dir, f"epoch{epoch}_iter{iter_idx}_batch{batch_idx}_camera_{timestamp}_{index}.png")
        cv2.imwrite(img_save_path, grid_img)


def visualize_depth_per_batch(depth_items, batch_data, hypes, vis_save_dir, epoch, iter_idx, batch_size, modality_name):
    """
    Visualize depth prediction and GT depth.
    
    Args:
        depth_items: Tuple of (depth_logit, depth_gt_indices)
            - depth_logit: [B*N, D, fH, fW] - predicted depth logits
            - depth_gt_indices: [B*N, fH, fW] - GT depth bin indices
        batch_data: Batch data dictionary
        hypes: Hyperparameters dictionary
        vis_save_dir: Directory to save visualization
        epoch: Current epoch number
        iter_idx: Current iteration index
        batch_size: Batch size
        modality_name: Modality name (e.g., 'm1')
    """
    if depth_items is None or not isinstance(depth_items, tuple) or len(depth_items) != 2:
        return
    
    depth_logit, depth_gt_indices = depth_items
    grid_conf = hypes["model"]["args"][modality_name]["encoder_args"]["grid_conf"]
    d_min = grid_conf["ddiscr"][0]
    d_max = grid_conf["ddiscr"][1]
    num_bins = grid_conf["ddiscr"][2]
    mode = grid_conf["mode"]
    camera_input_name = f"inputs_{modality_name}"
    
    if camera_input_name not in batch_data["ego"]:
        return
    
    inputs = batch_data["ego"][camera_input_name]
    imgs = inputs["imgs"]  # [sum(record_len_list), N_cam, C, H, W]
    B_N_cav, N_cam, C, H, W = imgs.shape
    fusion_method = hypes['fusion']['fusion_method']
    yaml_file_path_list = batch_data["ego"]["yaml_file_path"]

    # 目前深度分支只在 intermediate / early 下使用，这里按样本逐个、根据 record_len 动态处理，
    # 允许同一 batch 内不同样本有不同数量的 CAV
    if fusion_method not in ['intermediate', 'early']:
        return

    record_len_list = batch_data["ego"]["record_len"].cpu().numpy().astype(int).tolist()
    B = len(record_len_list)
    assert len(yaml_file_path_list) == B, "yaml_file_path_list 与 record_len_list 长度不一致"
    assert sum(record_len_list) == B_N_cav, \
        f"sum(record_len_list)={sum(record_len_list)} 与 imgs.shape[0]={B_N_cav} 不一致"

    fH, fW = depth_logit.shape[2], depth_logit.shape[3]
    # depth_logit / depth_gt_indices 第一维应为 sum(record_len_list) * N_cam
    total_entries = sum(record_len_list) * N_cam
    assert depth_logit.shape[0] == total_entries and depth_gt_indices.shape[0] == total_entries, \
        f"depth_logit/gt 第一维 ({depth_logit.shape[0]}, {depth_gt_indices.shape[0]}) 与 sum(record_len_list)*N_cam={total_entries} 不一致"

    # 先计算深度 bin 中心
    # 后续按样本逐个切分 depth_logit / depth_gt_indices，不再假设 N_cav 固定
    # softmax 仍然在 depth 维（num_bins）上进行
    
    if mode == "UD":
        bin_size = (d_max - d_min) / num_bins
        depth_bin_centers = torch.arange(num_bins, device=depth_logit.device, dtype=depth_logit.dtype) * bin_size + d_min + bin_size / 2
    elif mode == "LID":
        bin_size = 2 * (d_max - d_min) / (num_bins * (1 + num_bins))
        depth_bin_centers = torch.zeros(num_bins, device=depth_logit.device, dtype=depth_logit.dtype)
        for i in range(num_bins):
            depth_bin_centers[i] = d_min + bin_size * (i * (i + 1)) / 2
    elif mode == "SID":
        depth_bin_centers = torch.zeros(num_bins, device=depth_logit.device, dtype=depth_logit.dtype)
        for i in range(num_bins):
            depth_bin_centers[i] = (1 + d_min) * ((1 + d_max) / (1 + d_min)) ** (i / num_bins) - 1
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    
    depth_bin_centers = depth_bin_centers.view(1, 1, num_bins, 1, 1)  # 适配 [N_cav, N_cam, num_bins, fH, fW]
    
    # 逐个样本处理，支持一个 batch 内 CAV 数不同
    start_entry = 0  # 以 N_cav * N_cam 为单位在 depth_logit / depth_gt_indices 上推进
    for batch_idx, n_cav in enumerate(record_len_list):
        n_entry = n_cav * N_cam
        end_entry = start_entry + n_entry

        depth_logit_b = depth_logit[start_entry:end_entry]          # [n_cav * N_cam, num_bins, fH, fW]
        depth_gt_indices_b = depth_gt_indices[start_entry:end_entry]# [n_cav * N_cam, fH, fW]

        depth_logit_b = depth_logit_b.view(n_cav, N_cam, num_bins, fH, fW)
        depth_dist_b = F.softmax(depth_logit_b, dim=2)  # 在 num_bins 维度做 softmax
        depth_pred_b = (depth_dist_b * depth_bin_centers).sum(dim=2)  # [n_cav, N_cam, fH, fW]

        depth_gt_indices_flat_b = depth_gt_indices_b.view(-1, fH, fW)
        depth_gt_flat_b = indices_to_depth(depth_gt_indices_flat_b, d_min, d_max, num_bins, mode)
        depth_gt_b = depth_gt_flat_b.view(n_cav, N_cam, fH, fW)

        start_entry = end_entry

        yaml_file_path = yaml_file_path_list[batch_idx]
        if isinstance(yaml_file_path, list):
            yaml_file_path = yaml_file_path[0]
        yaml_file_path = yaml_file_path.split("/")
        timestamp, index = yaml_file_path[-3], yaml_file_path[-1].replace(".yaml", "")
        depth_pred_list = []
        depth_gt_list = []
        
        for cav_idx in range(n_cav):
            for cam_idx in range(N_cam):
                depth_pred_cam = depth_pred_b[cav_idx, cam_idx].cpu().detach().numpy()
                depth_gt_cam = depth_gt_b[cav_idx, cam_idx].cpu().detach().numpy()
                depth_pred_norm = (depth_pred_cam - d_min) / (d_max - d_min)
                depth_pred_norm = np.clip(depth_pred_norm, 0, 1)
                depth_gt_norm = (depth_gt_cam - d_min) / (d_max - d_min)
                depth_gt_norm = np.clip(depth_gt_norm, 0, 1)
                depth_pred_colored = plt.cm.jet(depth_pred_norm)[..., :3]
                depth_gt_colored = plt.cm.jet(depth_gt_norm)[..., :3]
                depth_pred_list.append(depth_pred_colored)
                depth_gt_list.append(depth_gt_colored)
        
        ncol = n_cav * N_cam
        fH, fW = depth_pred_list[0].shape[:2]
        grid_h = fH * 2
        grid_w = fW * ncol
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx, depth_img in enumerate(depth_pred_list):
            row = 0
            col = idx
            y_start = row * fH
            y_end = y_start + fH
            x_start = col * fW
            x_end = x_start + fW
            grid_img[y_start:y_end, x_start:x_end] = (depth_img * 255).astype(np.uint8)
        
        for idx, depth_img in enumerate(depth_gt_list):
            row = 1
            col = idx
            y_start = row * fH
            y_end = y_start + fH
            x_start = col * fW
            x_end = x_start + fW
            grid_img[y_start:y_end, x_start:x_end] = (depth_img * 255).astype(np.uint8)
        
        grid_img_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
        depth_save_path = os.path.join(vis_save_dir, f"epoch{epoch}_iter{iter_idx}_batch{batch_idx}_depth_{timestamp}_{index}.png")
        cv2.imwrite(depth_save_path, grid_img_bgr)


def project_boxes_to_image(gt_box_np, int_matrix_3x3, ext_matrix, post_rot, post_tran, H, W):
    """
    Project 3D bounding boxes from LiDAR coordinate to 2D image coordinates.
    
    Args:
        gt_box_np: Ground truth bounding boxes numpy array [N, 8, 3] in LiDAR coordinate
        int_matrix_3x3: Camera intrinsic matrix [3, 3]
        ext_matrix: Camera extrinsic matrix [4, 4] - T_camera_to_lidar
        post_rot: Post augmentation rotation matrix [3, 3]
        post_tran: Post augmentation translation vector [3]
        H: Image height
        W: Image width
    
    Returns:
        gt_box2d: Projected 2D bounding boxes [N, 8, 2]
        gt_box2d_mask: Validity mask [N] indicating which boxes are valid
    """
    N = gt_box_np.shape[0]
    xyz = gt_box_np.reshape(-1, 3)
    xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    ext_matrix_inv = np.linalg.inv(ext_matrix)[:3, :4]
    int_matrix_4x4 = np.eye(4, dtype=np.float32)
    int_matrix_4x4[:3, :3] = int_matrix_3x3
    img_pts = (int_matrix_4x4[:3, :3] @ ext_matrix_inv @ xyz_hom.T).T
    depth = img_pts[:, 2]
    uv_original = img_pts[:, :2] / (depth[:, None] + 1e-08)
    uv_hom = np.concatenate([uv_original, np.ones((uv_original.shape[0], 1))], axis=1)
    uv_aug_hom = uv_hom @ post_rot.T
    uv_aug = uv_aug_hom[:, :2] + post_tran[:2]
    uv_aug = uv_aug.reshape(N, 8, 2)
    depth = depth.reshape(N, 8)
    uv_int = uv_aug.round().astype(np.int32)
    valid_mask1 = (uv_int[:, :, 0] >= 0) & (uv_int[:, :, 0] < W) & (uv_int[:, :, 1] >= 0) & (uv_int[:, :, 1] < H)
    valid_mask2 = (depth > 0.5) & (depth < 100)
    gt_box2d_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1)
    uv_int[:, :, 0] = np.clip(uv_int[:, :, 0], 0, W - 1)
    uv_int[:, :, 1] = np.clip(uv_int[:, :, 1], 0, H - 1)
    gt_box2d = uv_int
    return gt_box2d, gt_box2d_mask
