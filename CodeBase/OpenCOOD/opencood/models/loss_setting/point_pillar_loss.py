# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic

class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        # cls loss: default values for SigmoidFocalLoss
        # Support both flat format (cls_weight) and nested format (cls.weight)
        cls_weight = args.get('cls_weight', None)
        if cls_weight is None:
            cls_config = args.get('cls', {})
            cls_weight = cls_config.get('weight', 1.0)
        self.cls = {
            'type': 'SigmoidFocalLoss',
            'alpha': args.get('cls_alpha', args.get('cls', {}).get('alpha', 0.25)),
            'gamma': args.get('cls_gamma', args.get('cls', {}).get('gamma', 2.0)),
            'weight': cls_weight
        }

        # reg loss: default values for WeightedSmoothL1Loss
        # Support both flat format (reg_weight) and nested format (reg.weight)
        reg_weight = args.get('reg_weight', None)
        if reg_weight is None:
            reg_config = args.get('reg', {})
            reg_weight = reg_config.get('weight', 2.0)
        self.reg = {
            'type': 'WeightedSmoothL1Loss',
            'sigma': args.get('reg_sigma', args.get('reg', {}).get('sigma', 3.0)),
            'codewise': args.get('reg_codewise', args.get('reg', {}).get('codewise', True)),
            'weight': reg_weight
        }

        # dir loss: default values for WeightedSoftmaxClassificationLoss
        # Support both flat format (dir_weight) and nested format (dir.weight)
        dir_weight = args.get('dir_weight', None)
        dir_args = args.get('dir_args', None)
        if dir_weight is None or dir_args is None:
            dir_config = args.get('dir', {})
            if dir_weight is None:
                dir_weight = dir_config.get('weight', 0.2)
            if dir_args is None:
                dir_args = dir_config.get('args', {})
        
        if dir_weight is not None or dir_args is not None or 'dir' in args:
            self.dir = {
                'type': 'WeightedSoftmaxClassificationLoss',
                'weight': dir_weight if dir_weight is not None else 0.2,
                'args': dir_args if dir_args is not None else {}
            }
        else:
            self.dir = None

        if 'iou' in args:
            from packages.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
            self.iou_loss_func = aligned_boxes_iou3d_gpu
            self.iou = args['iou']
        else:
            self.iou = None
        
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
        # cared = torch.logical_or(positives, negatives)
        # cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()

        # rename variable 
        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
        if f'dm{suffix}' in output_dict:
            output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']

        total_loss = 0

        # cls loss
        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size


        ######## direction ##########
        if self.dir:
            dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
            dir_logits = output_dict[f"dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]

            dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
            dir_loss = dir_loss.flatten() * reg_weights.flatten() 
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
            total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss.item()})


        ######## IoU ###########
        if self.iou:
            iou_preds = output_dict["iou_preds{suffix}"].permute(0, 2, 3, 1).contiguous()
            pos_pred_mask = reg_weights.squeeze(dim=-1) > 0 # (4, 70400)
            iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
            boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().detach(),
                                                            output_dict['anchor_box'])[pos_pred_mask]
            boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict['targets'],
                                                            output_dict['anchor_box'])[pos_pred_mask]
            iou_weights = reg_weights[pos_pred_mask].view(-1)
            iou_pos_targets = self.iou_loss_func(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]], # hwl -> dx dy dz
                                                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
            iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
            iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])

            iou_loss = iou_loss.sum() * self.iou['weight'] / batch_size
            total_loss += iou_loss
            self.loss_dict.update({'iou_loss': iou_loss.item()})

        total_loss += reg_loss + cls_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': reg_loss.item(),
                               'cls_loss': cls_loss.item()})

        return total_loss


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets



    def logging(self, epoch, batch_id, batch_len, writer = None, eta_str="", lr=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        eta_str : str
            Estimated time of arrival string
        lr : float
            Learning rate
        suffix : str
            Suffix for the loss name
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)

        # Build log string
        log_str = "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f" \
                  " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len, suffix,
                      total_loss, cls_loss, reg_loss, dir_loss, iou_loss)
        
        if lr is not None:
            log_str += f" || LR: {lr:.6f}"
        if eta_str:
            log_str += f" || {eta_str}"
        
        print(log_str)

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            if lr is not None:
                writer.add_scalar('Train/LearningRate', lr, epoch*batch_len + batch_id)

class PointPillarDepthLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        # Support both flat format (depth_weight) and nested format (depth.weight)
        depth_weight = args.get('depth_weight', None)
        depth_config = args.get('depth', {})
        if depth_weight is None:
            depth_weight = depth_config.get('weight', 1.0)
        self.depth_weight = depth_weight
        
        # Support both flat format (depth_smooth_target, depth_use_fg_mask) and nested format
        self.smooth_target = args.get('depth_smooth_target', depth_config.get('smooth_target', False))
        self.use_fg_mask = args.get('depth_use_fg_mask', depth_config.get('use_fg_mask', False))
        self.fg_weight = 3.25
        self.bg_weight = 0.25
        if self.smooth_target:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none", smooth_target=True)
        else:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")


    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """

        total_loss = super().forward(output_dict, target_dict, suffix)
        all_depth_loss = 0
        depth_items_list = [x for x in output_dict.keys() if x.startswith(f"depth_items{suffix}")]
        ######## Depth Supervision ########
        for depth_item_name in depth_items_list:
            depth_item = output_dict[depth_item_name]

            # depth logdit: [N, D, H, W]
            # depth gt indices: [N, H, W]
            # fg_mask: [N, H, W]
            depth_logit, depth_gt_indices = depth_item[0], depth_item[1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices) 
            if self.use_fg_mask:
                fg_mask = depth_item[-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss *= weight_mask

            depth_loss = depth_loss.mean() * self.depth_weight 
            all_depth_loss += depth_loss

        total_loss += all_depth_loss
        self.loss_dict.update({'depth_loss': all_depth_loss}) # no update the total loss in dict
        
        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, eta_str="", lr=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        eta_str : str
            Estimated time of arrival string
        lr : float
            Learning rate
        suffix : str
            Suffix for the loss name
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)

        # Build log string
        log_str = "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f" \
                  " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len, suffix,
                      total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss)
        
        if lr is not None:
            log_str += f" || LR: {lr:.6f}"
        if eta_str:
            log_str += f" || {eta_str}"
        
        print(log_str)

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            if lr is not None:
                writer.add_scalar('Train/LearningRate', lr, epoch*batch_len + batch_id)

class PointPillarPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid = args['pyramid']

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid['relative_downsample']
        self.pyramid_weight = self.pyramid['weight']
        self.num_levels = len(self.relative_downsample)
    
    def forward(self, output_dict, target_dict, suffix=""):
        if output_dict['pyramid'] == 'collab': # intermediate fusion, pyramid collab.
            return self.forward_collab(output_dict, target_dict, suffix)
        elif output_dict['pyramid'] == 'single': # late fusion, pyramid single 
            return self.forward_single(output_dict, target_dict, suffix)
        raise
        
    def forward_single(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_single
        """
        batch_size = target_dict['pos_equal_one'].shape[0]
        total_loss = super().forward(output_dict, target_dict, suffix)

        occ_single_list = output_dict['occ_single_list']    
        occ_loss = self.calc_occ_loss(occ_single_list, target_dict['pos_equal_one'], target_dict['neg_equal_one'], batch_size)
        total_loss += occ_loss
        self.loss_dict.update({
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        })
        return total_loss

    def forward_collab(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_collab
        """
        if suffix == "": 
            return super().forward(output_dict, target_dict)
        assert suffix == "_single"
        batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        }
        return total_loss

    def calc_occ_loss(self, occ_single_list, positives, negatives, batch_size):
        total_occ_loss = 0
        occ_positives = torch.logical_or(positives[...,0], positives[...,1]).unsqueeze(-1).float() # N, H, W
        occ_negatives = torch.logical_and(negatives[...,0], negatives[...,1]).unsqueeze(-1).float() # N, H, W

        for i, occ_preds_single in enumerate(occ_single_list):
            """
            occ_preds_single: N, 1, H, W
            occ_positives: N, H, W, 1
            occ_negatives: N, H, W, 1
            """
            positives_level = F.max_pool2d(occ_positives.permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)
            negatives_level = 1 - F.max_pool2d((1 - occ_negatives).permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)
            occ_labls = positives_level.view(batch_size, -1, 1)
            positives_level = occ_labls
            negatives_level = negatives_level.view(batch_size, -1, 1)
            pos_normalizer = positives_level.sum(1, keepdim=True).float()
            occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,  1)
            occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            occ_weights /= torch.clamp(pos_normalizer, min=1.0)
            occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
            occ_loss = occ_loss.sum() / batch_size
            occ_loss *= self.pyramid_weight[i]

            total_occ_loss += occ_loss
        return total_occ_loss
    
    def logging(self, epoch, batch_id, batch_len, writer = None, eta_str="", lr=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        suffix : str
            Suffix for logging (e.g., "_single")
        eta_str : str
            Estimated time remaining string
        lr : float
            Learning rate
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)

        # Build log string
        log_str = "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f" \
                  " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len, suffix,
                      total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss)
        
        if lr is not None:
            log_str += f" || LR: {lr:.6f}"
        if eta_str:
            log_str += f" || {eta_str}"
        
        print(log_str)

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss, epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss, epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss, epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss, epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss, epoch*batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss, epoch*batch_len + batch_id)
            if lr is not None:
                writer.add_scalar('Train/LearningRate', lr, epoch*batch_len + batch_id)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss

def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction= 'none', smooth_target = False , eps = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_target = smooth_target
        self.eps = eps
        if self.smooth_target:
            self.smooth_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.smooth_kernel.weight = torch.nn.Parameter(torch.tensor([[[0.2, 0.9, 0.2]]]), requires_grad=False)
            self.smooth_kernel = self.smooth_kernel.to(torch.device("cuda"))

    def forward(self, input, target):
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        D = input.shape[1]
        if self.smooth_target:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).view(-1, D) # [N*H*W, D]
            target_one_hot = self.smooth_kernel(target_one_hot.float().unsqueeze(1)).squeeze(1) # [N*H*W, D]
            target_one_hot = target_one_hot.view(*target.shape, D).permute(0, 3, 1, 2)
        else:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).permute(0, 3, 1, 2)
        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss