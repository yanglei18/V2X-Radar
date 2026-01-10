'''
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
''' 

import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
import cv2
import os
from torchvision.utils import save_image
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_lidar_points,
    mask_ego_radar_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor


def getUnifiedFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class getUnifiedFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.dataset_type = 'v2x-r' if 'v2x-r' in params['root_dir'].split('/') else 'v2x-radar' # 'v2x-radar' or 'v2x-r'
            self.fusion_method = params['fusion']['fusion_method']
            self.heterogeneous = True
            self.proj_first = False if 'proj_first' not in params['fusion']['args'] else params['fusion']['args']['proj_first']
            if self.fusion_method == 'early': assert self.proj_first, "proj_first must be True for early fusion"
            # Check if supervise_single is enabled in model args
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) else False
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)
            self.ego_modality = params['heter']['ego_modality'] # "m1" or "m1&m2" or "m3"
            self.modality_assignment = None if ('assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) else read_json(params['heter']['assignment_path'])
            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()
            self.adaptor = Adaptor(ego_modality=self.ego_modality,
                                   model_modality_list=self.modality_name_list,
                                   modality_assignment=self.modality_assignment,
                                   lidar_channels_dict=params['heter'].get('lidar_channels_dict', OrderedDict()),
                                   mapping_dict=params['heter']['mapping_dict'],
                                   cav_preference=params['heter'].get("cav_preference", None),
                                   train=train)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar' or modal_setting['sensor_type'] == 'radar':
                    setattr(self, f"pre_processor_{modality_name}", build_preprocessor(params, train))
                    if 'filter_point_setting' in modal_setting.keys():
                        setattr(self, f"filter_point_setting_{modality_name}", modal_setting['filter_point_setting'])
                if modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])

            self.reinitialize()
            print("dataset len:", self.len_record[-1])
            print('-'*50)
        
        def filter_points_by_image(self, lidar_np, selected_cav_base):
            """
            Filter LiDAR points that cannot be projected to any camera image.
            
            Parameters
            ----------
            lidar_np : np.ndarray
                LiDAR point cloud, shape [N, 4] (x, y, z, intensity)
            selected_cav_base : dict
                Dictionary containing CAV information including 'camera_data' and 'params'
            
            Returns
            -------
            filtered_lidar_np : np.ndarray
                Filtered LiDAR point cloud
            """
            params = selected_cav_base["params"]
            lidar_xyz = lidar_np[:, :3]  # [N, 3]
            N = lidar_xyz.shape[0]
            
            # Get image dimensions from data_aug_conf or actual image
            # Try to get from any modality's data_aug_conf
            ############ default values ###########
            broaden_horizontal = 2
            if self.dataset_type == 'v2x-r': 
                image_list = [0,1,2,3] # 0
                orig_H, orig_W = 600, 800
            if self.dataset_type == 'v2x-radar': 
                image_list = [0,1,2] # 1
                orig_H, orig_W = 864, 1536
            #######################################
            # 如果某个 modality 没有定义 filter_point_setting_*，则跳过，继续使用上面的默认值
            for modality_name in self.modality_name_list:
                attr_name = f"filter_point_setting_{modality_name}"
                if not hasattr(self, attr_name): continue
                aug_conf = getattr(self, attr_name)
                orig_H, orig_W = aug_conf['H'], aug_conf['W']
                image_list = aug_conf['image_list']
                broaden_horizontal = aug_conf['broaden_horizontal']        
            
            # Enlarge the image size symmetrically on both sides to avoid the points 
            # being projected to the edge of the image
            # Calculate expansion: expand from center symmetrically
            expand_W = orig_W * (broaden_horizontal - 1) / 2
            expand_H = orig_H * (broaden_horizontal - 1) / 2
            
            # New boundaries (symmetric expansion)
            u_min, u_max = -expand_W, orig_W + expand_W
            v_min, v_max = -expand_H, orig_H + expand_H
            
            # Convert to homogeneous coordinates
            lidar_xyz_hom = np.concatenate([lidar_xyz, np.ones((N, 1))], axis=1)  # [N, 4]
            
            # Track which points are valid (can be projected to at least one camera)
            valid_mask = np.zeros(N, dtype=bool)
            
            # Check each camera
            for idx in image_list:
                camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx, self.dataset_type)
                
                # Compute lidar2cam (inverse of camera_to_lidar)
                lidar2cam = np.linalg.inv(camera_to_lidar)[:3, :4]  # [3, 4]
                
                # Project points: int_matrix @ lidar2cam @ xyz_hom.T
                # camera_intrinsic is [3, 3], lidar2cam is [3, 4], xyz_hom is [N, 4]
                img_pts = (camera_intrinsic @ lidar2cam @ lidar_xyz_hom.T).T  # [N, 3] = (u*z, v*z, z)
                
                # Extract depth and normalize
                depth = img_pts[:, 2]  # [N]
                valid_depth = depth > 0.1  # Filter by depth
                
                # Normalize by depth to get pixel coordinates
                uv = img_pts[:, :2] / (depth[:, None] + 1e-8)  # [N, 2] = (u, v)
                
                # Check if points are within image bounds (with symmetric expansion)
                in_bounds = (uv[:, 0] >= u_min) & (uv[:, 0] < u_max) & \
                            (uv[:, 1] >= v_min) & (uv[:, 1] < v_max)
                
                # Combined valid mask for this camera
                camera_valid_mask = valid_depth & in_bounds
                valid_mask = valid_mask | camera_valid_mask  # OR operation: valid if visible in any camera
            
            # Filter points: keep only those that can be projected to at least one camera
            filtered_lidar_np = lidar_np[valid_mask]
            
            return filtered_lidar_np
        
        def get_processed_features_single_car(self, selected_cav_base, ego_cav_base, sensor_type, modality_name, selected_cav_processed):
            
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']
            transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose) # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            
            # lidar, # camera for visualization
            if sensor_type == "lidar" or self.visualize: 
                # process lidar # remove points that hit itself
                lidar_np = copy.copy(selected_cav_base['lidar_np'])
                lidar_np = shuffle_points(lidar_np)
                lidar_np = mask_ego_lidar_points(lidar_np)
                # filter lidar points that are not in the image
                # if sensor_type == "lidar": # lidar only, has filter_point_setting in modality setting
                lidar_np = self.filter_points_by_image(lidar_np, selected_cav_base)
                assert lidar_np.shape[0] > 0, f"Warning: No LiDAR points after filtering for CAV {selected_cav_base['yaml_file_path']}"
                    
                # data augmentation, seems very important for single agent training, 
                # because lack of data diversity, only work for lidar modality in training.
                # if sensor_type == "lidar" and self.data_augmentor and (self.fusion_method == "late" or self.fusion_method == "no" or self.fusion_method == "single"):
                #     object_bbx_center, object_bbx_mask = selected_cav_processed['single_object_bbx_center'], selected_cav_processed['single_object_bbx_mask']
                #     lidar_np, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask)
                #     selected_cav_processed.update({'single_object_bbx_center': object_bbx_center, 'single_object_bbx_mask': object_bbx_mask})
    
                # project the lidar to ego space first if needed or for visualization
                projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                if self.proj_first: lidar_np[:, :3] = projected_lidar # NOTE here
                if selected_cav_base['params']['RSU']: indicates = np.zeros(projected_lidar.shape[0])
                else: indicates = np.ones(projected_lidar.shape[0]) # for visualization
                projected_lidar = np.concatenate([projected_lidar, indicates[:, np.newaxis]], axis=1)
                selected_cav_processed.update({'projected_lidar': projected_lidar})
        
                if sensor_type == "lidar": # above is for visualization, below is truly process lidar features
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_lidar})
            
            if sensor_type == "radar" or self.visualize:
                radar_np = copy.copy(selected_cav_base['radar_np'])
                # First transform radar points from radar coordinate to lidar coordinate
                if 'radar' in selected_cav_base['params'].keys():
                    radar_coords = np.array(selected_cav_base['params']['radar']['cords']).astype(np.float32)
                elif 'radar_pose' in selected_cav_base['params'].keys():
                    radar_coords = np.array(selected_cav_base['params']['radar_pose']).astype(np.float32)
                lidar_pose = selected_cav_base['params']['lidar_pose']
                radar_to_lidar_matrix = x1_to_x2(radar_coords, lidar_pose)  # T_lidar_radar
                # Transform radar points to lidar coordinate system
                radar_xyz_hom = np.concatenate([radar_np[:, :3], np.ones((radar_np.shape[0], 1))], axis=1)
                radar_np[:, :3] = (radar_to_lidar_matrix @ radar_xyz_hom.T).T[:, :3]
                radar_np = shuffle_points(radar_np)
                radar_np = mask_ego_radar_points(radar_np)
                radar_np = self.filter_points_by_image(radar_np, selected_cav_base)
                assert radar_np.shape[0] > 0, f"Warning: No radar points after filtering for CAV {selected_cav_base['yaml_file_path']}"
                
                # project the radar to ego space first if needed or for visualization
                projected_radar = box_utils.project_points_by_matrix_torch(radar_np[:, :3], transformation_matrix)
                if self.proj_first: radar_np[:, :3] = projected_radar # NOTE here
                if selected_cav_base['params']['RSU']: indicates = np.zeros(projected_radar.shape[0])
                else: indicates = np.ones(projected_radar.shape[0]) # for visualization
                projected_radar = np.concatenate([projected_radar, indicates[:, np.newaxis]], axis=1)
                selected_cav_processed.update({'projected_radar': projected_radar})
        
                if sensor_type == "radar": # above is for visualization, below is truly process radar features
                    processed_radar = eval(f"self.pre_processor_{modality_name}").preprocess(radar_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_radar})
                
            # camera
            if sensor_type == "camera":
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []
                lidar2img_matrices = []  # Store final_lidar2img matrices
                data_aug_conf = eval(f"self.data_aug_conf_{modality_name}")
                cams = data_aug_conf['image_list']
                
                for idx in cams:
                    img = camera_data_list[idx]
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx, self.dataset_type)      

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # Compute final_lidar2img matrix (integrated with data augmentation)
                    # Reference: BEVAug3D approach
                    # Step 1: Build cam2img matrix (3x4) from intrinsic
                    cam2img = torch.zeros(3, 4, dtype=intrin.dtype, device=intrin.device)
                    cam2img[:3, :3] = intrin  # [3, 3]
                    cam2img[:2, :3] = post_rot[:2, :2] @ cam2img[:2, :3]
                    cam2img[:2, 2] = post_tran[:2] + cam2img[:2, 2]
                    lidar2cam = torch.linalg.inv(torch.from_numpy(camera_to_lidar).to(cam2img.device))  # [4, 4]
                    final_lidar2img_3x4 = cam2img @ lidar2cam  # [3, 4] @ [4, 4] = [3, 4]
                    dtype = final_lidar2img_3x4.dtype
                    device = final_lidar2img_3x4.device
                    final_lidar2img = torch.zeros(4, 4, dtype=dtype, device=device)
                    final_lidar2img[:3, :4] = final_lidar2img_3x4  # [3, 4]
                    final_lidar2img[3, 3] = 1.0  # [4, 4] - homogeneous coordinates

                    # Generate depth image from LiDAR point cloud projection (after data augmentation)
                    lidar_np = selected_cav_base.get('lidar_np', None)
                    # Project LiDAR points to image coordinates
                    lidar_xyz = lidar_np[:, :3]  # [N, 3]
                    lidar_xyz_hom = np.concatenate([lidar_xyz, np.ones((lidar_xyz.shape[0], 1))], axis=1)  # [N, 4]
                    lidar_xyz_hom_torch = torch.from_numpy(lidar_xyz_hom).float().to(device)  # [N, 4]
                    final_lidar2img_3x4 = final_lidar2img[:3, :4]  # [3, 4]
                    img_pts_hom = lidar_xyz_hom_torch @ final_lidar2img_3x4.T  # [N, 3] = (u*z, v*z, z)
                    depth = img_pts_hom[:, 2]  # [N] - depth values
                    valid_mask = depth > 0.1  # Filter points with valid depth
                    img_pts = img_pts_hom[:, :2] / (depth[:, None] + 1e-8)  # [N, 2] = (u, v)
                    
                    # Get image dimensions (after augmentation)
                    # PIL Image.size returns (width, height), so W = size[0], H = size[1]
                    W, H = img_src[0].size  # PIL Image: (W, H)
                    
                    # Filter points within image bounds
                    img_pts_np = img_pts.cpu().numpy()  # [N, 2]
                    depth_np = depth.cpu().numpy()  # [N]
                    valid_mask_np = valid_mask.cpu().numpy()  # [N]
                    
                    # Additional filtering: points within image bounds
                    in_bounds = (img_pts_np[:, 0] >= 0) & (img_pts_np[:, 0] < W) & \
                                (img_pts_np[:, 1] >= 0) & (img_pts_np[:, 1] < H)
                    final_valid_mask = valid_mask_np & in_bounds
                    
                    # Get valid points
                    valid_pts = img_pts_np[final_valid_mask]  # [M, 2]
                    valid_depth = depth_np[final_valid_mask]  # [M]
                    
                    # Generate depth image: for each pixel, use the minimum depth (closest point)
                    depth_img = np.full((H, W), np.inf, dtype=np.float32)
                    u_int = np.clip(valid_pts[:, 0].astype(np.int32), 0, W - 1)
                    v_int = np.clip(valid_pts[:, 1].astype(np.int32), 0, H - 1)
                    # Use minimum depth for each pixel (closest point wins)
                    for i in range(len(valid_pts)):
                        u, v = u_int[i], v_int[i]
                        if valid_depth[i] < depth_img[v, u]:
                            depth_img[v, u] = valid_depth[i]
                    # Replace inf with 0 (no point projected to this pixel)
                    depth_img[depth_img == np.inf] = 0
                    depth_max = 100.0  # Maximum expected depth in meters
                    depth_normalized = np.clip(depth_img / depth_max * 255.0, 0, 255).astype(np.uint8)
                    depth_img_pil = Image.fromarray(depth_normalized, mode='L')  # Grayscale image
                    # Add depth image to img_src list
                    img_src.append(depth_img_pil)

                    # decouple RGB and Depth
                    img_src[0] = normalize_img(img_src[0])
                    img_src[1] = img_to_tensor(img_src[1])

                    # # Visualize projected lidar points  
                    # # Project lidar_np to image coordinates using final_lidar2img
                    # lidar_np = selected_cav_base['lidar_np']  # [N, 4] - (x, y, z, intensity)
                    # lidar_xyz = lidar_np[:, :3]  # [N, 3] - (x, y, z)
                    # lidar_xyz_hom = np.concatenate([lidar_xyz, np.ones((lidar_xyz.shape[0], 1))], axis=1)  # [N, 4]
                    # lidar_xyz_hom_torch = torch.from_numpy(lidar_xyz_hom).float().to(device)  # [N, 4]
                    # # Project to image coordinates: [N, 4] @ [4, 4]^T = [N, 4]
                    # # Note: final_lidar2img is [4, 4], we need to use [:3, :4] for 3x4 projection
                    # final_lidar2img_3x4 = final_lidar2img[:3, :4]  # [3, 4]
                    # img_pts_hom = lidar_xyz_hom_torch @ final_lidar2img_3x4.T  # [N, 3] = (u*z, v*z, z)
                    # # Normalize by depth: [N, 2] = (u, v)
                    # depth = img_pts_hom[:, 2]  # [N]
                    # valid_mask = depth > 0.1  # Filter points with valid depth
                    # img_pts = img_pts_hom[:, :2] / (depth[:, None] + 1e-8)  # [N, 2] = (u, v)
                    
                    # # Denormalize image and draw projected points
                    # # img_src[0] is normalized, need to denormalize for visualization
                    # img_denorm = img_src[0].clone()  # [C, H, W]
                    # # Reverse ImageNet normalization
                    # mean = torch.tensor([0.485, 0.456, 0.406], device=img_denorm.device).view(3, 1, 1)
                    # std = torch.tensor([0.229, 0.224, 0.225], device=img_denorm.device).view(3, 1, 1)
                    # img_denorm = img_denorm * std + mean
                    # img_denorm = torch.clamp(img_denorm, 0, 1)
                    
                    # # Convert to numpy for OpenCV drawing
                    # img_np = img_denorm.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                    # img_np = (img_np * 255).astype(np.uint8)
                    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
                    
                    # # Get image dimensions
                    # H, W = img_np.shape[:2]
                    
                    # # Filter points within image bounds
                    # img_pts_np = img_pts.cpu().numpy()  # [N, 2]
                    # depth_np = depth.cpu().numpy()  # [N]
                    # valid_mask_np = valid_mask.cpu().numpy()  # [N]
                    
                    # # Additional filtering: points within image bounds
                    # in_bounds = (img_pts_np[:, 0] >= 0) & (img_pts_np[:, 0] < W) & \
                    #            (img_pts_np[:, 1] >= 0) & (img_pts_np[:, 1] < H)
                    # final_valid_mask = valid_mask_np & in_bounds
                    
                    # # Get valid points
                    # valid_pts = img_pts_np[final_valid_mask]  # [M, 2]
                    # valid_depth = depth_np[final_valid_mask]  # [M]
                    
                    # # Normalize depth for colormap (0-255)
                    # depth_min, depth_max = valid_depth.min(), valid_depth.max()
                    # depth_normalized = ((valid_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8) 
                    # # Draw points with depth-based coloring
                    # for i, (pt, d_norm) in enumerate(zip(valid_pts, depth_normalized)):
                    #     u, v = int(pt[0]), int(pt[1])
                    #     # If 90-degree rotation persists, try swapping: u, v = int(pt[1]), int(pt[0])
                    #     # Use colormap for depth visualization (BGR format for OpenCV)
                    #     color = cv2.applyColorMap(np.array([[d_norm]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                    #     color = tuple(map(int, color))  # Convert to tuple
                    #     cv2.circle(img_np, (u, v), 1, color, -1)  # Draw filled circle
                    # cv2.imwrite(f"lidar_proj_cam{idx}.png", img_np)

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    lidar2img_matrices.append(final_lidar2img)
                    

                selected_cav_processed.update(
                    {
                    f"image_inputs_{modality_name}": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                            "lidar2img": torch.stack(lidar2img_matrices),  # [Ncam, 4, 4] - final lidar2img matrix with data augmentation (homogeneous coordinates)
                        }
                    }
                )
             
            return selected_cav_processed
        
        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            # preparation and calculate the transformation matrix
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']
            transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose) # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            selected_cav_processed.update({
                "modality_name": selected_cav_base['modality_name'],
                "yaml_file_path": selected_cav_base['yaml_file_path'],
                "transformation_matrix": transformation_matrix,
                "transformation_matrix_clean": transformation_matrix_clean})
            
            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], selected_cav_base['params']['lidar_pose'])
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask)
            selected_cav_processed.update({
                "single_label_dict": label_dict,
                "single_object_bbx_center": object_bbx_center,
                "single_object_bbx_mask": object_bbx_mask,
                "single_object_ids": object_ids})

            # note the reference pose is ego, we use ego pose for label transformation
            # we transform all cav's gt to ego, we just append here, then stack object_bbx_center
            # and generate label_dict later outside，also care about exculude all repetitve objects 
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], ego_pose_clean)
            selected_cav_processed.update({
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids})

            return selected_cav_processed

        def __getitem__(self, idx):
            # preparation for data loading
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            # find ego vehicle, using ego flag in base_data_dict
            ego_id, ego_lidar_pose, ego_cav_base = None, [], None
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
            assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
            assert ego_id != None
            assert len(ego_lidar_pose) > 0

            # for stack single cav's attributes
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            projected_lidar_stack = []
            projected_radar_stack = []
            cav_id_list = []
            exclude_agent = []
            yaml_file_path_list = []
            agent_modality_list = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue # if distance is too far, we will just skip this agent
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue # if modality not match
                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            # get cav number and exclude agents that are too far or modality not match
            cav_num = len(cav_id_list)
            if len(cav_id_list) == 0: return None
            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            # after excluding agents, get pairwise transformation matrix
            pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)
            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # initialize input lists for each modality dynamically
            for modality_name in self.modality_name_list:
                exec(f"input_list_{modality_name} = []")
            
            # merge preprocessed features from different cavs into the same dict
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name_used = selected_cav_base['modality_name'].split("&")
                agent_modality_list.append(selected_cav_base['modality_name'])
                yaml_file_path_list.append(selected_cav_base['yaml_file_path'])
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)
                for modality_name in modality_name_used:
                    sensor_type = self.sensor_type_dict[modality_name]
                    selected_cav_processed = self.get_processed_features_single_car(selected_cav_base, ego_cav_base, sensor_type, modality_name, selected_cav_processed) 
                    if sensor_type == "lidar" or sensor_type == "radar": eval(f"input_list_{modality_name}").append(selected_cav_processed[f"processed_features_{modality_name}"])
                    if sensor_type == "camera": eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])                      
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                projected_radar_stack.append(selected_cav_processed['projected_radar'])
                # NOTE: single is for gather to ego, while cav_id: selected_cav_processed is for single cav
                single_label_list.append(selected_cav_processed['single_label_dict'])
                single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                selected_cav_processed.update({'cav_id_list': [cav_id]})
                processed_data_dict.update({cav_id: selected_cav_processed})

            # generate single view GT label, NOTE here, rank by cav_id_list
            single_label_dicts = self.post_processor.collate_batch(single_label_list)
            single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
            single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
            processed_data_dict['ego'].update({
                "single_label_dict_torch": single_label_dicts,
                "single_object_bbx_center_torch": single_object_bbx_center,
                "single_object_bbx_mask_torch": single_object_bbx_mask})
            
            # exculude all repetitve objects, DAIR-V2X, NOTE: which gt source?
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
            # make sure bounding boxes across all frames have the same number
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=mask)

            # merge all modalities' features as tensor
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar" or self.sensor_type_dict[modality_name] == "radar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}")) 
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict}) # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"), merge='stack')
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict}) # maybe None

            # update processed_data_dict
            processed_data_dict['ego'].update({
                'label_dict': label_dict,
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses,
                'origin_lidar': np.vstack(projected_lidar_stack),
                'origin_radar': np.vstack(projected_radar_stack),
                'yaml_file_path': yaml_file_path_list,
                'agent_modality_list': agent_modality_list,
                'cav_num': cav_num,
                'sample_idx': idx, 
                'cav_id_list': cav_id_list})

            return processed_data_dict

        def collate_batch_train(self, batch):
            
            # for intermediate and early fusion, we use aggregated ego as ego, e.g. final_output_dict {"ego(true_ego aggregated)": {}}
            if self.fusion_method == 'intermediate' or self.fusion_method == 'early':
                output_dict = {'ego': {}}
                label_dict_list = []
                object_bbx_center = []
                object_bbx_mask = []
                object_ids = []
                lidar_pose_list = []
                lidar_pose_clean_list = []
                pairwise_t_matrix_list = []
                origin_lidar = []
                origin_radar = []
                record_len = []
                agent_modality_list = []
                yaml_file_path_list = []
                # single view GT label
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []
                # for ego vehicle, the transformation matrix is identity
                batch_size = len(batch)
                identity_4x4 = np.identity(4).astype(np.float32)
                transformation_matrix_torch = torch.from_numpy(np.tile(identity_4x4, (batch_size, 1, 1))).float()  # [B, 4, 4]
                transformation_matrix_clean_torch = torch.from_numpy(np.tile(identity_4x4, (batch_size, 1, 1))).float()  # [B, 4, 4]
                
                # initialize inputs lists for each modality dynamically
                for modality_name in self.modality_name_list:
                    exec(f"inputs_list_{modality_name} = []")

                for i in range(len(batch)):
                    ego_dict = batch[i]['ego']
                    label_dict_list.append(ego_dict['label_dict'])
                    object_bbx_center.append(ego_dict['object_bbx_center'])
                    object_bbx_mask.append(ego_dict['object_bbx_mask'])
                    object_ids.append(ego_dict['object_ids'])
                    lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                    lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                    origin_lidar.append(ego_dict['origin_lidar'])
                    origin_radar.append(ego_dict['origin_radar'])
                    record_len.append(ego_dict['cav_num'])
                    agent_modality_list.extend(ego_dict['agent_modality_list'])
                    yaml_file_path_list.append(ego_dict['yaml_file_path'])
                    pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

                # convert to numpy, (B, max_num, 7)
                label_torch_dict =  self.post_processor.collate_batch(label_dict_list)
                object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
                object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
                pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list)) # (B, max_cav)
                record_len = torch.from_numpy(np.array(record_len, dtype=int))
                lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
                lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
                origin_lidar = torch.from_numpy(np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar)))
                origin_radar = torch.from_numpy(np.array(downsample_lidar_minimum(pcd_np_list=origin_radar)))
                
                # merge all modalities' features as tensor
                for modality_name in self.modality_name_list:
                    for i in range(len(batch)):
                        ego_dict = batch[i]['ego']
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'input_{modality_name}']) # OrderedDict() if empty?
                    if self.sensor_type_dict[modality_name] == "lidar" or self.sensor_type_dict[modality_name] == "radar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})
                    if self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge='cat')
                        output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})

                # all gt under ego
                output_dict['ego'].update({
                    'label_dict': label_torch_dict,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'object_ids': object_ids,
                    'anchor_box': self.anchor_box_torch,
                    'pairwise_t_matrix': pairwise_t_matrix,
                    'lidar_pose_clean': lidar_pose_clean,
                    'lidar_pose': lidar_pose,
                    'origin_lidar': origin_lidar,
                    'origin_radar': origin_radar,
                    'record_len': record_len,
                    'yaml_file_path': yaml_file_path_list,
                    'agent_modality_list': agent_modality_list,
                    'transformation_matrix': transformation_matrix_torch,
                    'transformation_matrix_clean': transformation_matrix_clean_torch,
                    'label_dict_single': { # NOTE here, single view GT label
                        "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0),
                        "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                        "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)}})
                
                final_output_dict = {}
                final_output_dict['ego'] = output_dict['ego'] # we only need ego
            
            # for single cav training, we need to select a random cav as ego, e.g. final_output_dict {"ego(-1)": {}, "142": {}}
            if self.fusion_method == 'late' or self.fusion_method == 'no' or self.fusion_method == 'single':  
                # get all cav_ids except 'ego'
                cav_id_list = [k for k in batch[0].keys() if k != 'ego']
                output_dict = {}
                # process each cav's features
                for cav_id in cav_id_list:
                    object_bbx_center = []
                    object_bbx_mask = []
                    object_ids = []
                    label_dict_list = []
                    origin_lidar = []
                    origin_radar = []
                    yaml_file_path_list = []
                    agent_modality_list = []
                    transformation_matrix_list = []
                    transformation_matrix_clean_list = []
                    for modality_name in self.modality_name_list:
                        exec(f"inputs_list_{modality_name} = []")
                    for i in range(len(batch)):
                        ego_dict = batch[i][cav_id]
                        object_bbx_center.append(ego_dict['single_object_bbx_center'])
                        object_bbx_mask.append(ego_dict['single_object_bbx_mask'])
                        label_dict_list.append(ego_dict['single_label_dict'])
                        object_ids.append(ego_dict['single_object_ids'])
                        origin_lidar.append(ego_dict['projected_lidar'])
                        origin_radar.append(ego_dict['projected_radar'])
                        transformation_matrix_list.append(torch.from_numpy(ego_dict['transformation_matrix']).float().unsqueeze(0))
                        transformation_matrix_clean_list.append(torch.from_numpy(ego_dict['transformation_matrix_clean']).float().unsqueeze(0))
                        yaml_file_path_list.append(ego_dict['yaml_file_path'])
                        agent_modality_list.append(ego_dict['modality_name'])
                    object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
                    object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
                    label_torch_dict = self.post_processor.collate_batch(label_dict_list)
                    origin_lidar = torch.from_numpy(np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar)))
                    origin_radar = torch.from_numpy(np.array(downsample_lidar_minimum(pcd_np_list=origin_radar)))
                    transformation_matrix_torch = torch.cat(transformation_matrix_list, dim=0)
                    transformation_matrix_clean_torch = torch.cat(transformation_matrix_clean_list, dim=0)
                    output_dict.update({cav_id: {
                        'object_bbx_center': object_bbx_center,
                        'object_bbx_mask': object_bbx_mask,
                        'object_ids': object_ids,
                        'anchor_box': torch.from_numpy(self.anchor_box),
                        'label_dict': label_torch_dict,
                        'origin_lidar': origin_lidar,
                        'origin_radar': origin_radar,
                        'yaml_file_path': yaml_file_path_list,
                        'agent_modality_list': agent_modality_list,
                        'transformation_matrix': transformation_matrix_torch,
                        'transformation_matrix_clean': transformation_matrix_clean_torch}})
                    for modality_name in self.modality_name_list:
                        for i in range(len(batch)):
                            ego_dict = batch[i][cav_id]
                            if f'processed_features_{modality_name}' in ego_dict:
                                eval(f"inputs_list_{modality_name}").append(ego_dict[f'processed_features_{modality_name}']) 
                            if f'image_inputs_{modality_name}' in ego_dict:
                                eval(f"inputs_list_{modality_name}").append(ego_dict[f'image_inputs_{modality_name}']) 
                    if self.sensor_type_dict[modality_name] == "lidar" or self.sensor_type_dict[modality_name] == "radar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_dict)
                        output_dict[cav_id].update({f'inputs_{modality_name}': processed_lidar_torch_dict})
                    if self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge='stack')
                        output_dict[cav_id].update({f'inputs_{modality_name}': merged_image_inputs_dict})
                
                final_output_dict = {}
                selected_cav_id_for_ego = random.choice(cav_id_list)
                if not self.train:
                    selected_cav_id_for_ego = '142'
                final_output_dict['ego'] = output_dict[selected_cav_id_for_ego]
                # Add all other agents except original 'ego' and selected one
                for cav_id in output_dict.keys():
                    if cav_id != selected_cav_id_for_ego:
                        final_output_dict[cav_id] = output_dict[cav_id]  

            return final_output_dict

        def collate_batch_test(self, batch):
            
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None: return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None: return None
            
            # NOTE here, object_ids is only used during inference, where batch size is 1.
            sample_idx = batch[0]['ego']['sample_idx'] # raw ego idx
            cav_id_list = list(output_dict.keys())
            for cav_id in cav_id_list:
                output_dict[cav_id]['object_ids'] = output_dict[cav_id]['object_ids'][0] # batch size is 1
                output_dict[cav_id]['sample_idx'] = sample_idx
                if len(output_dict[cav_id]['agent_modality_list']) == 1: # for single cav, need to add modality name
                    output_dict[cav_id]['modality_name'] = output_dict[cav_id]['agent_modality_list'][0]

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego["ego"] = data_dict["ego"]
            pred_box_tensor, pred_score = self.post_processor.post_process(data_dict_ego, output_dict_ego)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
            return pred_box_tensor, pred_score, gt_box_tensor

    return getUnifiedFusionDataset


