# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from opencood.models.sub_modules.lss_submodule import Up, CamEncode, BevEncode, CamEncode_Resnet101
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.pillar_vfe import PillarVFE, PillarVFERadar
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.second_backbone import second_backbone, second_fpn
from mmdet.models.backbones.resnet import BasicBlock
from torch.nn import functional as F
from torchvision.utils import save_image
from packages.Voxelization.bev_pool import bev_pool
from packages.Voxelization.bev_pool_v2 import bev_pool_v2

class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])


    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
    
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict['spatial_features'] # H0, W0
        return lidar_feature_2d
    
class RadarPillarNet(nn.Module):
    def __init__(self, args):
        super(RadarPillarNet, self).__init__()
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFERadar(
            args['pillar_vfe'],
            num_point_features=5,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.pts_backbone=second_backbone(
            in_channels=64,
            layer_nums=[1, 1, 1],
            layer_strides=[1, 2, 2],
            out_channels=[64, 64, 64])
        self.pts_neck=second_fpn(
            in_channels=[64, 64, 64],
            upsample_strides=[1, 2, 4],
            out_channels=[64, 64, 64])
        self.shrink_head=nn.Sequential(
            nn.Conv2d(64*3, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1))

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
        batch_size = voxel_coords[:,0].max() + 1
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        x = batch_dict['spatial_features'] # H0, W0
        x = self.pts_backbone(x)
        x = self.pts_neck(x)[0]
        x = self.shrink_head(x)
        return x

class SECOND(nn.Module):
    def __init__(self, args):
        super(SECOND, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                                np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                            args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
        batch_size = voxel_coords[:,0].max() + 1


        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'batch_size': batch_size}

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        return batch_dict['spatial_features']

class VoxelPainting(nn.Module):
    def __init__(self, args):
        super(VoxelPainting, self).__init__()
        self.use_depth = args['use_depth']
        self.num_in_height = 8
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        self.D = self.grid_conf['ddiscr'][2] # depth discretization 98 
        self.point_cloud_range = [\
            self.grid_conf['xbound'][0], self.grid_conf['ybound'][0], self.grid_conf['zbound'][0],
            self.grid_conf['xbound'][1], self.grid_conf['ybound'][1], self.grid_conf['zbound'][1]]
        self.voxel_size = [self.grid_conf['xbound'][2],  self.grid_conf['ybound'][2],  self.grid_conf['zbound'][2]]
        self.depth_supervision = args['depth_supervision']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.bev_h_ = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        self.bev_w_ = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        self.pts_dim = 3
        
        self.voxelpainting_points, self.voxel_coords = \
            self.generate_pillar_ref_points(self.num_in_height)
            
        in_channels = self.num_in_height * self.camC
        self.adaptive_collapse_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.camC, kernel_size=1),
            BasicBlock(self.camC, self.camC))
        
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
            checkpoint_path = args['pretrained_path']
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
            missing_keys, unexpected_keys = self.camencode.load_state_dict(pretrained_dict, strict=False)
            if len(missing_keys) > 0: print("Missing keys:", missing_keys)
            if len(unexpected_keys) > 0: print("Unexpected keys:", unexpected_keys)
            print("Loaded pretrained weights from %s"%(checkpoint_path))
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
    
    def generate_pillar_ref_points(self, num_in_height):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        voxel_x, voxel_y, _ = self.voxel_size
        
        # Calculate the center points for the grid
        x_centers = torch.arange(x_min + voxel_x / 2, x_max, voxel_x)
        y_centers = torch.arange(y_min + voxel_y / 2, y_max, voxel_y)
        z_step = (z_max - z_min) / num_in_height
        z_centers = torch.arange(z_min + z_step / 2, z_max, z_step)
        assert x_centers.shape[0] == self.bev_h_
        assert y_centers.shape[0] == self.bev_w_

        # Create a mesh grid for x, y, z
        xv, yv, zv = torch.meshgrid(x_centers, y_centers, z_centers)
        # Stack the grid coordinates
        ref_points = torch.stack((xv, yv, zv), dim=-1) # shape: (H, W, Z, 3)
        
        # indices
        hv, wv, zv = torch.meshgrid(torch.arange(self.bev_h_), torch.arange(self.bev_w_), torch.arange(num_in_height))
        idx = torch.arange(self.bev_h_ * self.bev_w_ * num_in_height)
        voxel_coords = torch.cat([hv.reshape(-1, 1), wv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)
        return ref_points, voxel_coords
    
    def voxel_painting_backward(self, context, points, lidar2img, depth_logits, use_depth=False):
        """
        Memory-efficient voxel painting backward pass supporting multiple cameras.
        Only processes points that project into each camera's field of view.
        
        Args:
            context: [(cav_num in batch)*N, C, H, W] camera context features
            points: list of [N, 3] point clouds for each batch
            lidar2img: [(cav_num in batch), N, 4, 4] transformation matrices (N is number of cameras)
            depth_logits: [(cav_num in batch)*N, D, H, W] depth logits
        
        Returns:
            painted_points: list of [N, C+pts_dim] decorated point features for each batch
        """
        epsilon = 1e-6
        device = lidar2img.device
        ddiscr, mode = self.grid_conf['ddiscr'], self.grid_conf['mode']
        cam_depth_range = [ddiscr[0], ddiscr[1], int((ddiscr[1]-ddiscr[0]) / ddiscr[2])]
        
        points_tensor = torch.stack([p[:, :3] for p in points], dim=0)  # [B, N_points, 3]
        B, N_points = points_tensor.shape[:2]
        N_cams = lidar2img.shape[1]  # Number of cameras
        C = context.shape[1]
        D, H, W = depth_logits.shape[1], depth_logits.shape[2], depth_logits.shape[3]
        
        # Initialize output features: [B, N_points, C+pts_dim]
        context_features = torch.zeros((B, N_points, C + self.pts_dim), device=device, dtype=context.dtype)
        context_features[:, :, :3] = points_tensor  # Keep same as radar points
        
        # Initialize camera count for each point: [B, N_points]
        # This counts how many cameras see each point, for mean aggregation
        cam_count = torch.zeros((B, N_points), device=device, dtype=torch.float)
        
        # Prepare homogeneous points: [B, N_points, 4]
        pts_hom = torch.cat((points_tensor, torch.ones((B, N_points, 1), device=device)), dim=2)  # [B, N_points, 4]
        
        # Process each batch separately, then process all cameras for that batch together
        # This is more efficient because cameras in the same batch share the same point cloud space
        for b in range(B):
            # Get batch-specific points: [1, N_points, 4]
            batch_pts_hom = pts_hom[b:b+1]  # [1, N_points, 4]
            
            # Project points to all cameras for this batch: [N_cams, N_points, 4] @ [N_cams, 4, 4]^T -> [N_cams, N_points, 4]
            batch_lidar2img = lidar2img[b]  # [N_cams, 4, 4]
            # Expand points to match camera dimension: [1, N_points, 4] -> [N_cams, N_points, 4]
            batch_pts_hom_expanded = batch_pts_hom.expand(N_cams, -1, -1)  # [N_cams, N_points, 4]
            # Batch matrix multiplication: [N_cams, N_points, 4] @ [N_cams, 4, 4]^T -> [N_cams, N_points, 4]
            img_pts_all_cams = torch.bmm(batch_pts_hom_expanded, batch_lidar2img.transpose(1, 2))  # [N_cams, N_points, 4]
            depth_values_all_cams = img_pts_all_cams[:, :, 2]  # [N_cams, N_points]
            img_pts_all_cams[:, :, :2] = img_pts_all_cams[:, :, :2] / (img_pts_all_cams[:, :, 2:3] + epsilon)  # [N_cams, N_points, 2]
            
            # Find valid points for each camera: [N_cams, N_points]
            valid_mask_all_cams = (img_pts_all_cams[:, :, 2] > epsilon) & \
                                  (img_pts_all_cams[:, :, 0] >= 0) & (img_pts_all_cams[:, :, 0] < W) & \
                                  (img_pts_all_cams[:, :, 1] >= 0) & (img_pts_all_cams[:, :, 1] < H)  # [N_cams, N_points]
            
            # Process each camera for this batch
            for cam_idx in range(N_cams):
                valid_mask = valid_mask_all_cams[cam_idx]  # [N_points]
                if not valid_mask.any():
                    continue
                
                # Get camera-specific context and depth_logits: [1, C, H, W] and [1, D, H, W]
                cam_context = context[cam_idx + b * N_cams:cam_idx + b * N_cams + 1]  # [1, C, H, W]
                cam_depth_logits = depth_logits[cam_idx + b * N_cams:cam_idx + b * N_cams + 1] if use_depth else None  # [1, D, H, W]
                
                # Get valid points for this camera
                valid_point_indices = torch.where(valid_mask)[0]  # [N_valid]
                valid_img_pts = img_pts_all_cams[cam_idx, valid_point_indices, :2]  # [N_valid, 2]
                valid_depth_values = depth_values_all_cams[cam_idx, valid_point_indices]  # [N_valid]
                
                # Normalize image coordinates for grid_sample: [N_valid, 2] -> [-1, 1]
                valid_img_pts_norm = valid_img_pts.clone()
                valid_img_pts_norm[:, 0] = (valid_img_pts_norm[:, 0] / (W - 1)) * 2 - 1
                valid_img_pts_norm[:, 1] = (valid_img_pts_norm[:, 1] / (H - 1)) * 2 - 1
                valid_img_pts_norm = torch.clamp(valid_img_pts_norm, -1, 1)
                valid_img_pts_norm = valid_img_pts_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, N_valid, 2]
                
                # Batch grid_sample for context features: [1, C, 1, N_valid] -> [N_valid, C]
                sampled_context = F.grid_sample(
                    cam_context, valid_img_pts_norm, align_corners=True
                )  # [1, C, 1, N_valid]
                sampled_context = sampled_context.squeeze(0).squeeze(1).permute(1, 0)  # [N_valid, C]
                
                if use_depth:
                    # Batch grid_sample for depth logits: [1, D, 1, N_valid] -> [N_valid, D]
                    sampled_depth_probs = F.grid_sample(
                        cam_depth_logits, valid_img_pts_norm, align_corners=True
                    )  # [1, D, 1, N_valid]
                    sampled_depth_probs = sampled_depth_probs.squeeze(0).squeeze(1).permute(1, 0)  # [N_valid, D]
                    
                    # Normalize depth probabilities
                    power_exponent = 1.0
                    sampled_depth_probs = (sampled_depth_probs ** power_exponent) / \
                                        (sampled_depth_probs.sum(dim=1, keepdim=True) + epsilon)  # [N_valid, D]
                    
                    # Calculate depth indices and interpolate
                    depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2])  # [N_valid]
                    lower_indices = torch.clamp(torch.floor(depth_indices).long(), 0, D - 1)  # [N_valid]
                    upper_indices = torch.clamp(torch.ceil(depth_indices).long(), 0, D - 1)  # [N_valid]
                    upper_weight = depth_indices - lower_indices.float()  # [N_valid]
                    lower_weight = 1 - upper_weight.float()  # [N_valid]
                    
                    # Gather depth probabilities
                    lower_prob_values = sampled_depth_probs.gather(1, lower_indices.unsqueeze(1)).squeeze(1)  # [N_valid]
                    upper_prob_values = sampled_depth_probs.gather(1, upper_indices.unsqueeze(1)).squeeze(1)  # [N_valid]
                    depth_prob_values = lower_weight * lower_prob_values + upper_weight * upper_prob_values  # [N_valid]
                    
                    # Re-weight context features
                    sampled_context.mul_(depth_prob_values.unsqueeze(1))  # [N_valid, C]
                    
                    # Free memory
                    del sampled_depth_probs, depth_prob_values, lower_prob_values, upper_prob_values
                
                # Accumulate features: add features from multiple cameras for overlapping points
                # Note: Advanced indexing returns a copy, not a view, so we need to assign back
                existing_features = context_features[b, valid_point_indices, self.pts_dim:]
                context_features[b, valid_point_indices, self.pts_dim:] = existing_features + sampled_context
                
                # Count how many cameras see each point
                # Note: Advanced indexing returns a copy, so we need to assign back
                existing_count = cam_count[b, valid_point_indices]
                cam_count[b, valid_point_indices] = existing_count + 1.0
                
                # Free memory
                del sampled_context, valid_img_pts_norm
            
            # Normalize by camera count to get mean (avoid division by zero)
            # Only normalize points that were seen by at least one camera
            valid_points_mask = cam_count[b] > 0  # [N_points]
            if valid_points_mask.any():
                context_features[b, valid_points_mask, self.pts_dim:] /= cam_count[b, valid_points_mask].unsqueeze(1)
            
            # Free memory for this batch
            del img_pts_all_cams, depth_values_all_cams, valid_mask_all_cams, batch_pts_hom_expanded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check for NaN and handle
        if torch.isnan(context_features).any():
            print("NaN detected in %s, num=%d" % ('painted_points', torch.isnan(context_features).sum()))
            context_features = torch.nan_to_num(context_features, nan=0.0)
        
        # Convert to list format for compatibility
        painted_points = [context_features[i] for i in range(B)]
        
        return painted_points
    
    # def voxel_painting_backward(self, context, points, lidar2img, depth_logits, use_depth=False):
    #     """
    #     Batch version of voxel painting backward pass.
    #     Processes all batches in parallel without for loop.
        
    #     Args:
    #         context: [(cav_num in batch)*N, C, H, W] camera context features
    #         points: list of [N, 3] point clouds for each batch
    #         lidar2img: [(cav_num in batch), N, 4, 4] transformation matrices (N is number of cameras)
    #         depth_logits: [(cav_num in batch)*N, D, H, W] depth logits
        
    #     Returns:
    #         painted_points: list of [N, C+pts_dim] decorated point features for each batch
    #     """
    #     epsilon = 1e-6
    #     B, D, H, W = depth_logits.shape
    #     device = lidar2img.device
    #     ddiscr, mode = self.grid_conf['ddiscr'], self.grid_conf['mode']
    #     cam_depth_range = [ddiscr[0], ddiscr[1], int((ddiscr[1]-ddiscr[0]) / ddiscr[2])]
    #     points_tensor = torch.stack([p[:, :3] for p in points], dim=0)  # [B, N, 3]
    #     N_points = points_tensor.shape[1]
    #     N_cams = lidar2img.shape[1]  # Number of cameras
        
    #     assert N_cams == 1, "currently only support one camera"
    #     assert mode == 'UD', "currently only support UD mode"
    #     lidar2img = lidar2img.squeeze(1)  # [B, 4, 4]
        
    #     # Initialize output features: [B, N_points, C+pts_dim]
    #     context_features = torch.zeros((B, N_points, context.shape[1] + self.pts_dim), device=device, dtype=context.dtype)
    #     context_features[:, :, :3] = points_tensor  # Keep same as radar points
        
    #     # Batch parallel projection: [B, N, 3] -> [B, N, 4]
    #     pts_hom = torch.cat((points_tensor, torch.ones((B, N_points, 1), device=device)), dim=2)  # [B, N, 4]
    #     # Project all points: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 4]
    #     img_pts = torch.bmm(pts_hom, lidar2img.transpose(1, 2))  # [B, N, 4]
    #     img_pts[:, :, :2] = img_pts[:, :, :2] / (img_pts[:, :, 2:3] + epsilon)  # [B, N, 2]
    #     depth_values = img_pts[:, :, 2]  # [B, N]
        
    #     # Batch parallel validity check
    #     valid_mask = (img_pts[:, :, 0] >= 0) & (img_pts[:, :, 0] < W) & \
    #                  (img_pts[:, :, 1] >= 0) & (img_pts[:, :, 1] < H)  # [B, N]
        
    #     # Normalize image coordinates for grid_sample: [B, N, 2]
    #     img_pts_norm = img_pts[:, :, :2].clone()
    #     img_pts_norm[:, :, 0] = (img_pts_norm[:, :, 0] / (W - 1)) * 2 - 1
    #     img_pts_norm[:, :, 1] = (img_pts_norm[:, :, 1] / (H - 1)) * 2 - 1
    #     img_pts_norm = torch.clamp(img_pts_norm, -1, 1)
        
    #     # Reshape for grid_sample: [B, N, 2] -> [B, N, 1, 2] (grid_sample expects [B, H, W, 2])
    #     img_pts_norm_grid = img_pts_norm.unsqueeze(2)  # [B, N, 1, 2]
        
    #     # Batch parallel grid_sample for context features: [B, C, H, W] -> [B, C, 1, N] -> [B, N, C]
    #     all_context_features = F.grid_sample(context, img_pts_norm_grid, align_corners=True)  # [B, C, 1, N]
    #     all_context_features = all_context_features.squeeze(-1).permute(0, 2, 1)  # [B, N, C]
        
    #     if use_depth:
    #         # Batch parallel grid_sample for depth logits: [B, D, H, W] -> [B, D, 1, N] -> [B, N, D]
    #         all_depth_probs = F.grid_sample(depth_logits, img_pts_norm_grid, align_corners=True)  # [B, D, 1, N]
    #         all_depth_probs = all_depth_probs.squeeze(3).permute(0, 2, 1)  # [B, N, D]
            
    #         # Normalize depth probabilities: [B, N, D]
    #         power_exponent = 1.0
    #         all_depth_probs = (all_depth_probs ** power_exponent) / (all_depth_probs.sum(dim=2, keepdim=True) + epsilon)  # [B, N, D]
            
    #         # Calculate depth indices and interpolate: [B, N]
    #         depth_indices = ((depth_values - cam_depth_range[0]) / cam_depth_range[2])  # [B, N]
    #         lower_indices = torch.clamp(torch.floor(depth_indices).long(), 0, D - 1)  # [B, N]
    #         upper_indices = torch.clamp(torch.ceil(depth_indices).long(), 0, D - 1)  # [B, N]
    #         upper_weight = depth_indices - lower_indices.float()  # [B, N]
    #         lower_weight = 1 - upper_weight.float()  # [B, N]
            
    #         # Gather depth probabilities using gather: [B, N, D] -> [B, N]
    #         lower_prob_values = all_depth_probs.gather(2, lower_indices.unsqueeze(2)).squeeze(2)  # [B, N]
    #         upper_prob_values = all_depth_probs.gather(2, upper_indices.unsqueeze(2)).squeeze(2)  # [B, N]
    #         depth_prob_values = lower_weight * lower_prob_values + upper_weight * upper_prob_values  # [B, N]
            
    #         # Free memory for all_depth_probs before re-weighting
    #         del all_depth_probs
    #         torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    #         # Re-weight context features: [B, N, C] * [B, N, 1] -> [B, N, C]
    #         all_context_features.mul_(depth_prob_values.unsqueeze(2))
        
    #     # Only update features for valid points to save memory
    #     # Apply mask in-place to avoid creating new tensor
    #     all_context_features.mul_(valid_mask.unsqueeze(2).float())  # [B, N, C] in-place
    #     # Only update the context part (skip pts_dim which is already set)
    #     context_features[:, :, self.pts_dim:].add_(all_context_features)
    #     # Free memory
    #     del all_context_features
        
    #     # Check for NaN and handle
    #     if torch.isnan(context_features).any():
    #         print("NaN detected in %s, num=%d" % ('painted_points', torch.isnan(context_features).sum()))
    #         context_features = torch.nan_to_num(context_features, nan=0.0)
        
    #     # Convert to list format for compatibility
    #     painted_points = [context_features[i] for i in range(B)]
        
    #     return painted_points
    
    def get_depth_dist(self, x):
        return x.softmax(dim=1)
    
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352
        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, volume, context = self.camencode.get_context(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        return context, depth_items
    
    def forward(self, data_dict, modality_name):
        # x: [4,4,3,256,352]
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        x = image_inputs_dict['imgs']
        lidar2img = image_inputs_dict['lidar2img']
        device = x.device
        B, N, _, H, W = x.shape # B: CAVs_num  N: Cams_num  C: 3(or4)
        context, depth_items = self.get_cam_feats(x)
        depth_prob = self.get_depth_dist(depth_items[0])
        h, w, z = self.voxelpainting_points.shape[:3]
        input_points = [self.voxelpainting_points.reshape(-1, 3).to(device) for _ in range(B)]
        matrix = torch.eye(4).to(device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        matrix[:, :, 0, 0] = matrix[:, :, 0, 0] / self.downsample
        matrix[:, :, 1, 1] = matrix[:, :, 1, 1] / self.downsample
        projection = matrix @ lidar2img

        all_decorated_points = self.voxel_painting_backward(context, input_points, projection, depth_prob, use_depth=False)
        all_decorated_points = [all_decorated_points[i][:, self.pts_dim:] for i in range(B)]
        paint_bev = [x.view(1, h, w, z, -1) for x in all_decorated_points]
        paint_bev = torch.cat(paint_bev, dim=0) # B H W Z C
        paint_bev = paint_bev.view(B, h, w, -1).permute(0, 3, 2, 1).contiguous() # B C*Z H W
        img_bev_feats_3Dto2D = self.adaptive_collapse_conv(paint_bev)
        if self.depth_supervision: self.depth_items = depth_items
        # save_image(img_bev_feats_3Dto2D.max(1, keepdim=True).values, "img_bev_feats_3Dto2D.png")
        return img_bev_feats_3Dto2D
    
        # Visualize projected lidar points  
        # Project lidar_np to image coordinates using final_lidar2img
        cav_batch_idx, cam_idx = 0, 0
        final_lidar2img = projection[cav_batch_idx, cam_idx]
        lidar_np = self.voxelpainting_points.reshape(-1, 3)  # [N, 4] - (x, y, z, intensity)
        indices = torch.randperm(lidar_np.shape[0], device=device)[:5000]
        lidar_np = lidar_np[indices]
        lidar_xyz = lidar_np[:, :3]  # [N, 3] - (x, y, z)
        lidar_xyz_hom = np.concatenate([lidar_xyz, np.ones((lidar_xyz.shape[0], 1))], axis=1)  # [N, 4]
        lidar_xyz_hom_torch = torch.from_numpy(lidar_xyz_hom).float().to(device)  # [N, 4]
        # Project to image coordinates: [N, 4] @ [4, 4]^T = [N, 4]
        # Note: final_lidar2img is [4, 4], we need to use [:3, :4] for 3x4 projection
        final_lidar2img_3x4 = final_lidar2img[:3, :4]  # [3, 4]
        img_pts_hom = lidar_xyz_hom_torch @ final_lidar2img_3x4.T  # [N, 3] = (u*z, v*z, z)
        # Normalize by depth: [N, 2] = (u, v)
        depth = img_pts_hom[:, 2]  # [N]
        valid_mask = depth > 0.1  # Filter points with valid depth
        img_pts = img_pts_hom[:, :2] / (depth[:, None] + 1e-8)  # [N, 2] = (u, v)
        
        # Convert to numpy for OpenCV drawing
        B, N, _, H, W = x.shape # B: CAVs_num  N: Cams_num  C: 3(or4)
        img_np = np.zeros((H//self.downsample, W//self.downsample, 3))  # [H, W, 3]
        img_np = (img_np * 255).astype(np.uint8)
        # Get image dimensions
        H, W = img_np.shape[:2]

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

        # Normalize depth for colormap (0-255)
        depth_min, depth_max = valid_depth.min(), valid_depth.max()
        depth_normalized = ((valid_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8) 
        # Draw points with depth-based coloring
        for i, (pt, d_norm) in enumerate(zip(valid_pts, depth_normalized)):
            u, v = int(pt[0]), int(pt[1])
            # If 90-degree rotation persists, try swapping: u, v = int(pt[1]), int(pt[0])
            # Use colormap for depth visualization (BGR format for OpenCV)
            color = cv2.applyColorMap(np.array([[d_norm]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            color = tuple(map(int, color))  # Convert to tuple
            cv2.circle(img_np, (u, v), 1, color, -1)  # Draw filled circle
        cv2.imwrite(f"lidar_proj_cam_{cav_batch_idx}_{cam_idx}.png", img_np)    
    
class LiftSplatShoot(nn.Module):
    def __init__(self, args): 
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        dx, bx, nx, lx = gen_dx_bx(self.grid_conf['xbound'], # [0, 69.12, 0.08]
                               self.grid_conf['ybound'], # [-39.68, 39.68, 0.08]
                               self.grid_conf['zbound']) # [-3, 1, 4.0]
        self.dx = torch.tensor(dx.clone().detach().numpy(), requires_grad=False) # voxel bin size
        self.bx = torch.tensor(bx.clone().detach().numpy(), requires_grad=False) # 
        self.nx = torch.tensor(nx.clone().detach().numpy(), requires_grad=False) # grid map size
        self.lx = torch.tensor(lx.clone().detach().numpy(), requires_grad=False) # grid_lower_bound
        # bev pool v2 settings
        self.grid_lower_bound = self.lx
        self.grid_interval = self.dx
        self.grid_size = self.nx
        
        self.depth_supervision = args['depth_supervision']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)
        self.use_quickcumsum = True
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
            checkpoint_path = args['pretrained_path']
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
            missing_keys, unexpected_keys = self.camencode.load_state_dict(pretrained_dict, strict=False)
            if len(missing_keys) > 0: print("Missing keys:", missing_keys)
            if len(unexpected_keys) > 0: print("Unexpected keys:", unexpected_keys)
            print("Loaded pretrained weights from %s"%(checkpoint_path))
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        # depth_items, x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        depth_items, volume, context = self.camencode.get_context(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        volume = volume.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        volume = volume.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return depth_items, volume, context

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def voxel_pooling_v1(self, geom_feats, x):
        # geom_feats: (B x N x D x H x W x 3): ego cordinates
        # x: (B x N x D x fH x fW x C) image features
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)
        dx = self.dx.to(x.device)
        nx = self.nx.to(x.device)
        bx = self.bx.to(x.device)
        # flatten indices
        # Convert geom_feats to grid-relative voxel indices by subtracting bx - dx / 2 
        # and dividing by dx. These operations basically map geometric features 
        # (positions in the vehicle coordinate system) to indices in a voxel grid.
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] => [b, c, x, y, z]
        img_bev_feats = bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])
        img_bev_feats = img_bev_feats.permute(0, 1, 3, 4, 2).contiguous()
        img_bev_feats = img_bev_feats.mean(-1)
        img_bev_feats = img_bev_feats.permute(0, 1, 3, 2).contiguous()
        return img_bev_feats

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        
        cav_num, N, _, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        h, w = H//self.downsample, W//self.downsample
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        depth_items, volume, context = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        depth_prob = self.get_depth_dist(depth_items[0])
        
        # bev pool raw in this repo
        # bev_feat = self.voxel_pooling(geom, volume)

        # bev pool v1
        # bev_feat = self.voxel_pooling_v1(geom, volume)
        
        # bev pool v2
        # geom       (B, N, D, H, W, 3)
        # context    (B, N, C, H, W)
        # depth_prob (B, N, D, H, W) 
        depth_prob = depth_prob.view(cav_num, N, self.D, h, w)
        context = context.view(cav_num, N, -1, h, w)
        bev_feat = self.voxel_pooling_v2(geom, depth_prob, context)
        return bev_feat, depth_items

    def forward(self, data_dict, modality_name):
        # x: [4,4,3,256,352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        
        if self.depth_supervision:
            self.depth_items = depth_items

        return x

class LiftSplatShootVoxel(LiftSplatShoot):
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        # final = torch.max(final.unbind(dim=2), 1)[0]  # 消除掉z维
        final = torch.max(final, 2)[0]  # 消除掉z维
        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W 