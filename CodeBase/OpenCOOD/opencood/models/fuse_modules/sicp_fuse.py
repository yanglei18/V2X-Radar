# -*- coding: utf-8 -*-
# Author: Deyuan Qu <deyuanqu@my.unt.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import pickle
import yaml
import numpy as np

from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple

class SICPFusion(nn.Module):
    def __init__(self, feat_dim):
        """
        feat_dim : 单车特征通道数 C

        NOTE:
        - 在 SICP 中，我们总是对 (ego, neighbor) 这两个特征进行拼接，
          因此中间的 conv 实际输入通道数是 2*C。
        """
        super(SICPFusion, self).__init__()

        pair_channels = feat_dim * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
            )
        # 接收 (ego, neighbor) 拼接后的 2C 通道特征
        self.compChannels1 = nn.Sequential(
            nn.Conv2d(pair_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
            )
        # 同样接收 2C 通道，输出回单车特征维度 C
        self.compChannels2 = nn.Sequential(
            nn.Conv2d(pair_channels, feat_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=feat_dim),
            nn.ReLU()
            )

    def generate_overlap_selector(self, selector):
        overlap_sel = torch.mean(selector, 1).unsqueeze(0).cuda()
        return overlap_sel

    def generate_nonoverlap_selector(self, overlap_sel):
        non_overlap_sel = torch.tensor(np.where(overlap_sel.cpu() > 0, 0, 1)).cuda()
        return non_overlap_sel

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Multi-CAV Spatial Information Compensation and Propagation (SICP).

        Parameters
        ----------
        x : torch.Tensor
            shape: (N, C, H, W), features of N CAVs in the same batch.
            We assume index 0 is the ego / receiver.
        record_len : torch.Tensor
            shape: (1,), value == N for this batch. Kept for API consistency.
        pairwise_t_matrix : torch.Tensor
            shape: (1, L, L, 2, 3), normalized affine matrices.
            We only use the block [:, :N, :N, ...] and treat 0 as ego index.

        Strategy
        --------
        - Warp all CAV features into ego's BEV frame.
        - For each neighbor j>0, run the original 2-CAV SICP (ego + j) to
          obtain a pairwise fused map.
        - Aggregate all pairwise fused maps across neighbors (mean) to
          produce the final ego fused feature.
        """
        # x: (N, C, H, W)
        device = x.device
        N, C, H, W = x.shape

        if N < 2:
            # 单车场景，直接返回原始 ego 特征
            return x[0:1, ...]

        # ego feature
        rec_feature = x[0:1, ...]  # (1, C, H, W)

        # 取该 batch 的配准矩阵子块: (1, N, N, 2, 3)
        t_matrix = pairwise_t_matrix[0:1, :N, :N, :, :]

        pairwise_fused_list = []

        # 遍历所有 neighbor j>0，执行原来的 2 车 SICP 逻辑
        for j in range(1, N):
            sed_feature = x[j:j+1, ...]  # (1, C, H, W)

            # transfer sed to rec's space using T_{j->0}
            t_j0 = t_matrix[0, j, 0, :, :].unsqueeze(0)  # (1, 2, 3)
            t_sed_feature = warp_affine_simple(
                sed_feature, t_j0, (H, W)
            )  # (1, C, H, W)

            # generate overlap selector and non-overlap selector
            selector = torch.ones_like(sed_feature)
            selector = warp_affine_simple(selector, t_j0, (H, W))
            overlap_sel = self.generate_overlap_selector(selector)  # (1, 1, H, W)
            non_overlap_sel = self.generate_nonoverlap_selector(overlap_sel)  # (1, 1, H, W)

            # generate the weight map (same as original 2-CAV logic)
            cat_feature = torch.cat((rec_feature, t_sed_feature), dim=1)  # (1, 2C, H, W)
            comp_feature = self.compChannels1(cat_feature)                # (1, 1, H, W)
            f1 = self.conv1(comp_feature)
            f2 = self.conv2(f1)
            weight_map = comp_feature + f2                                # (1, 1, H, W)

            # normalize the weight map to [0,1]
            w_min = torch.min(weight_map)
            w_max = torch.max(weight_map)
            # 避免数值问题
            if (w_max - w_min) < 1e-6:
                normalize_weight_map = torch.zeros_like(weight_map)
            else:
                normalize_weight_map = (weight_map - w_min) / (w_max - w_min)

            # apply normalized weight map to rec_feature and t_sed_feature
            weight_to_rec = rec_feature * (normalize_weight_map * overlap_sel + non_overlap_sel)
            weight_to_t_sed = t_sed_feature * (1 - normalize_weight_map)

            # (1, 2C, H, W) -> (1, C, H, W)
            x_pair = torch.cat((weight_to_rec, weight_to_t_sed), dim=1)
            x_pair = self.compChannels2(x_pair)

            pairwise_fused_list.append(x_pair)

        # 将所有 neighbor 的 pairwise 融合结果聚合为最终 ego 特征
        # 这里采用 max 聚合（在所有邻车融合结果上取逐点最大值）。
        stacked = torch.stack(pairwise_fused_list, dim=0)   # (N-1, 1, C, H, W)
        fused_ego = torch.max(stacked, dim=0)[0]            # (1, C, H, W)

        return fused_ego