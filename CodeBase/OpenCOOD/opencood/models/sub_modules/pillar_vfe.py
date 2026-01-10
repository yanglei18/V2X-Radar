"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayerRadar(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        self.in_channels1 = 8
        self.in_channels2 = 2
        self.in_channels3 = 2
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units1 = out_channels // 2
        self.units2 = out_channels // 4
        self.units3 = out_channels // 4
        self.norm1 = build_norm_layer(norm_cfg, self.units1)[1] # 需要归一化的维度为32
        self.norm2 = build_norm_layer(norm_cfg, self.units2)[1]# 需要归一化的维度为15
        self.norm3 = build_norm_layer(norm_cfg, self.units3)[1]# 需要归一化的维度为16
        self.linear1 = nn.Linear(self.in_channels1, self.units1, bias=False)
        self.linear2 = nn.Linear(self.in_channels2, self.units2, bias=False)
        self.linear3 = nn.Linear(self.in_channels3, self.units3, bias=False)
        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        # Feature order in inputs (after PillarVFERadar.forward):
        # [0-4]: original (x, y, z, v, r) - 5维 (always present)
        # [5-7]: f_cluster (x_c, y_c, z_c) if with_cluster_center - 3维
        # [8-9]: f_center (x_p, y_p) if with_voxel_center - 2维
        # [10]: points_dist if with_distance - 1维 (may be missing)
        # [11-12]: velocity_snr_center (v_c, r_c) if with_velocity_snr_center - 2维 (may be missing)
        # Possible totals:
        #   10 dims: 5 + 3 + 2 (no distance, no velocity_snr_center)
        #   11 dims: 5 + 3 + 2 + 1 (with distance, no velocity_snr_center)
        #   12 dims: 5 + 3 + 2 + 2 (no distance, with velocity_snr_center)
        #   13 dims: 5 + 3 + 2 + 1 + 2 (all enabled)
        
        # Check feature dimension to avoid index out of bounds
        feature_dim = inputs.shape[2]
        
        # Determine which features are present based on dimension
        # Base: 5 (original)
        # +3 if cluster_center: 8 total
        # +2 if voxel_center: 10 total
        # +1 if distance: 11 total
        # +2 if velocity_snr_center: 12 or 13 total
        
        has_cluster = feature_dim >= 8
        has_voxel_center = feature_dim >= 10
        has_distance = feature_dim >= 11
        has_velocity_snr = feature_dim >= 12
        
        # Ensure we have at least 10 dimensions (minimum: original + cluster + voxel_center)
        if feature_dim < 10:
            raise RuntimeError(
                f"PFNLayerRadar requires at least 10 feature dimensions, "
                f"but got {feature_dim}. Please ensure feature decorations are enabled: "
                f"with_cluster_center=True, with_voxel_center=True"
            )
        
        # Build spatio indices: [0,1,2] (x,y,z) + cluster + voxel_center
        spatio_indices = [0, 1, 2]
        if has_cluster:
            spatio_indices.extend([5, 6, 7])  # f_cluster
        if has_voxel_center:
            spatio_indices.extend([8, 9])  # f_center
        
        # Ensure we have exactly 8 spatio features
        if len(spatio_indices) != 8:
            # Pad with zeros or use available features
            if len(spatio_indices) < 8:
                # If missing features, repeat the last available feature
                while len(spatio_indices) < 8:
                    spatio_indices.append(spatio_indices[-1] if spatio_indices else 0)
            else:
                spatio_indices = spatio_indices[:8]
        
        spatio_input = inputs.index_select(2, torch.tensor(spatio_indices, device=inputs.device))
        
        # Velocity features: [3] (v) + second feature
        velocity_indices = [3]
        if has_distance:
            velocity_indices.append(10)  # points_dist
        elif has_velocity_snr:
            velocity_indices.append(10)  # velocity_snr_center[0] (v_c)
        else:
            # If no distance and no velocity_snr, use a zero feature or repeat v
            # For now, we'll use index 3 (v) again, but this should be handled by padding
            velocity_indices.append(3)  # Fallback: use v again
        
        velocity_input = inputs.index_select(2, torch.tensor(velocity_indices, device=inputs.device))
        
        # SNR features: [4] (r) + second feature
        snr_indices = [4]
        if has_distance and has_velocity_snr:
            snr_indices.append(11)  # velocity_snr_center[0] (v_c) when distance exists
        elif has_velocity_snr:
            snr_indices.append(11)  # velocity_snr_center[1] (r_c) when no distance
        elif has_distance:
            # If distance but no velocity_snr, we need to handle this
            # For now, use a zero or repeat r
            snr_indices.append(4)  # Fallback: use r again
        else:
            # No distance and no velocity_snr
            snr_indices.append(4)  # Fallback: use r again
        
        snr_input = inputs.index_select(2, torch.tensor(snr_indices, device=inputs.device))
        x1 = self.linear1(spatio_input)
        x1 = self.norm1(x1.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()  # N,C 或者 N C L
        x2 = self.linear2(velocity_input)
        x2 = self.norm2(x2.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x3 = self.linear3(snr_input)
        x3 = self.norm3(x3.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                             1).contiguous()
        x = torch.cat((x1,x2,x3),dim=-1)
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # 0是数，1是索引
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4]
            voxel_num_points: [M,]
            voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        return batch_dict
    
class PillarVFERadar(nn.Module):
    """Pillar Feature Net for Radar data.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. Modified to match PillarVFE interface.

    Args:
        model_cfg (dict): Model configuration dictionary containing:
            - feat_channels: Number of features in each PFNLayer
            - with_distance: Whether to include Euclidean distance
            - with_cluster_center: Whether to include cluster center features
            - with_voxel_center: Whether to include voxel center features
            - with_velocity_snr_center: Whether to include velocity/SNR center features
            - norm_cfg: Normalization config
            - mode: Pooling mode ('max' or 'avg')
            - legacy: Whether to use legacy behavior
        num_point_features (int): Number of input point features
        voxel_size (list): Size of voxels [x, y, z]
        point_cloud_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super(PillarVFERadar, self).__init__()
        self.model_cfg = model_cfg
        
        # Extract configuration from model_cfg with defaults
        self.feat_channels = self.model_cfg.get('feat_channels', [64])
        self.with_distance = self.model_cfg.get('with_distance', True)
        self.with_cluster_center = self.model_cfg.get('with_cluster_center', True)
        self.with_voxel_center = self.model_cfg.get('with_voxel_center', True)
        # For radar, velocity_snr_center is required for PFNLayerRadar to work properly
        self.with_velocity_snr_center = self.model_cfg.get('with_velocity_snr_center', True)
        
        # Ensure required features for PFNLayerRadar
        # PFNLayerRadar expects at least: cluster_center, voxel_center, and velocity_snr_center
        if not self.with_cluster_center:
            print("Warning: with_cluster_center=False, but PFNLayerRadar requires it. Enabling it.")
            self.with_cluster_center = True
        if not self.with_voxel_center:
            print("Warning: with_voxel_center=False, but PFNLayerRadar requires it. Enabling it.")
            self.with_voxel_center = True
        if not self.with_velocity_snr_center:
            print("Warning: with_velocity_snr_center=False, but PFNLayerRadar requires it. Enabling it.")
            self.with_velocity_snr_center = True
        self.norm_cfg = self.model_cfg.get('norm_cfg', dict(type='BN1d', eps=1e-3, momentum=0.01))
        self.mode = self.model_cfg.get('mode', 'max')
        self.legacy = self.model_cfg.get('legacy', True)
        
        assert len(self.feat_channels) > 0
        
        # Calculate input channels based on feature decorations
        in_channels = num_point_features
        if self.with_cluster_center:
            in_channels += 3
        if self.with_voxel_center:
            in_channels += 2
        if self.with_distance:
            in_channels += 1
        if self.with_velocity_snr_center:
            in_channels += 2
        
        self._with_distance = self.with_distance
        self._with_cluster_center = self.with_cluster_center
        self._with_voxel_center = self.with_voxel_center
        self._with_velocity_snr_center = self.with_velocity_snr_center
        self.fp16_enabled = False
        
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(self.feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayerRadar(
                    in_filters,
                    out_filters,
                    norm_cfg=self.norm_cfg,
                    last_layer=last_layer,
                    mode=self.mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range
        
        # Store num_filters for get_output_feature_dim
        self.num_filters = self.feat_channels

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """Forward function matching PillarVFE interface.

        Args:
            batch_dict (dict): Dictionary containing:
                - voxel_features: Point features in shape (N, M, C)
                - voxel_num_points: Number of points in each pillar (N,)
                - voxel_coords: Coordinates of each voxel (N, 4)

        Returns:
            dict: Updated batch_dict with 'pillar_features' key
        """
        voxel_features = batch_dict['voxel_features']
        voxel_num_points = batch_dict['voxel_num_points']
        coors = batch_dict['voxel_coords']
        
        features_ls = [voxel_features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = voxel_features[:, :, :3].sum(
                dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
                    -1, 1, 1)  # 每个非空pillar的均值xyz
            f_cluster = voxel_features[:, :, :3] - points_mean  # 所有点和均值的差包括补充的零点
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = voxel_features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(voxel_features[:, :, :2])  # 这里coor好像是bzyx
                f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +  # ：，：，3可以相减，对应每个voxel中心，包括补充的0点
                    self.x_offset)
                f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = voxel_features[:, :, :2]   # 这里是trick，改变了前两维特征变成了局部xy，相对于voxel中心
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(voxel_features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(voxel_features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        if self._with_velocity_snr_center:
            velocity_snr_mean = voxel_features[:, :, 3:5].sum(
                dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
                    -1, 1, 1)
            velocity_snr_center = voxel_features[:, :, 3:5] - velocity_snr_mean
            features_ls.append(velocity_snr_center)
        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # 最里面一维接起来（xyzvrXcYcZcXpYpVcRc）[X,10,12]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]  # maxpoints
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # 接起来的特征有的是0点的，需要把这些再归成0
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  #  这里无效点特征都会是0

        for pfn in self.pfn_layers:
            features = pfn(features, voxel_num_points)

        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
