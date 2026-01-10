""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import importlib
from collections import OrderedDict, Counter

import torch
import torch.nn as nn
import numpy as np
import torchvision
from icecream import ic

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.cross_modal_fusion import Cross_Modal_Fusion
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion,
    CoBEVT, Where2commFusion, Who2comFusion, AdaFusion, SICPFusion)
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from torchvision.utils import save_image

class HeterModelLate(nn.Module):
    def __init__(self, args):
        super(HeterModelLate, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            # setup layers (actual backbone)
            setattr(self, f"layers_{modality_name}", ResNetBEVBackbone(model_setting['layers_args']))
            setattr(self, f"layers_num_{modality_name}", len(model_setting['layers_args']['num_upsample_filter']))
            # setup shrink head
            setattr(self, f"shrink_conv_{modality_name}",  DownsampleConv(model_setting['shrink_header']))
            # setup detection head
            in_head = model_setting['head_args']['in_head']
            setattr(self, f'cls_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))

            # depth supervision for camera args
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else: setattr(self, f"depth_supervision_{modality_name}", False)

    def forward(self, data_dict):
        output_dict = {}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop((int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}"))))(feature)
            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})

        # multiscale fusion, Here we do not use layer0 of the "self.layers_{modality_name}"
        # We assume feature from the "self.backbone_{modality_name}" is the first-scale feature
        feature_list = [feature]
        for i in range(1, eval(f"self.layers_num_{modality_name}")):
            feature = eval(f"self.layers_{modality_name}").get_layer_i_feature(feature, layer_i=i)
            feature_list.append(feature)

        feature = eval(f"self.layers_{modality_name}").decode_multiscale_feature(feature_list)
        feature = eval(f"self.shrink_conv_{modality_name}")(feature)

        cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
        reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
        dir_preds = eval(f"self.dir_head_{modality_name}")(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'fused_feature': feature})

        return output_dict

class HeterModelBaseline(nn.Module):
    def __init__(self, args):
        super(HeterModelBaseline, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls
                    
            # build encoder, backbone, shrink_header
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], model_setting['backbone_args'].get('inplanes',64)))
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            # depth supervision for camera args
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:  setattr(self, f"depth_supervision_{modality_name}", False)

        # for feature transformation
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))

        # build fusion net, only one fusion method is used in last layer
        if args['fusion_method'] == "max": self.fusion_net = MaxFusion(args['max'])
        if args['fusion_method'] == "att": self.fusion_net = AttFusion(args['att'])
        if args['fusion_method'] == "disconet": self.fusion_net = DiscoFusion(args['disconet'])
        if args['fusion_method'] == "v2vnet": self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit': self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt': self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm': self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com': self.fusion_net = Who2comFusion(args['who2com'])
        if args['fusion_method'] == 'adafusion': self.fusion_net = AdaFusion(args['adafusion'])
        if args['fusion_method'] == 'sicp': self.fusion_net = SICPFusion(args['sicp'])

        # build cross-modal fusion, only initialize if combined modalities (e.g., "m1&m2") are used
        self.use_cross_modal_fusion = False
        if isinstance(args['ego_modality'], str) and "&" in args['ego_modality']:
            self.use_cross_modal_fusion = True
        if self.use_cross_modal_fusion:
            assert len(self.modality_name_list) == 2, "Cross-modal fusion requires exactly two modalities"
            cross_modal_feat_dim = 256
            backbone_args = args[self.modality_name_list[0]]['backbone_args']
            cross_modal_feat_dim_m1 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            backbone_args = args[self.modality_name_list[1]]['backbone_args']
            cross_modal_feat_dim_m2 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            self.cross_modal_fusion = Cross_Modal_Fusion(
                kernel_size=3,
                img_channels=cross_modal_feat_dim_m1,
                rad_channels=cross_modal_feat_dim_m2,
                out_channels=cross_modal_feat_dim )
        else: self.cross_modal_fusion = None

        # build shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # build shared heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
        
        # build compressor, only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'], args['compressor']['compress_ratio'])
            self.model_train_init()

        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        
        # get data
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # print(agent_modality_list)
        #  modality_count_dict = Counter(agent_modality_list)
        # count the number of each modality, including combined modalities
        modality_count_dict = {}
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, split and count each sub-modality
                sub_modalities = agent_modality.split("&")
                for sub_mod in sub_modalities:
                    modality_count_dict[sub_mod] = modality_count_dict.get(sub_mod, 0) + 1
            else:
                # if single modality, count directly
                modality_count_dict[agent_modality] = modality_count_dict.get(agent_modality, 0) + 1
        

        # get feature from each modality
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict: continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # crop/padd camera feature map
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})

        # assemble heter features
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, fuse multiple sub-modalities
                sub_modalities = agent_modality.split("&")
                # get feature from each sub-modality for the same CAV
                sub_features = []
                for sub_mod in sub_modalities:
                    if sub_mod in modality_feature_dict:
                        feat_idx = counting_dict[sub_mod]
                        sub_features.append(modality_feature_dict[sub_mod][feat_idx:feat_idx+1])
                        counting_dict[sub_mod] += 1
                # fuse multiple sub-modalities using Cross_Modal_Fusion
                assert len(sub_features) >= 2, "At least two modalities are required for cross-modal fusion"
                assert self.cross_modal_fusion is not None, "cross_modal_fusion is not initialized. Combined modalities require cross_modal_fusion."
                fused_feature = self.cross_modal_fusion(sub_features[0], sub_features[1])
                heter_feature_2d_list.append(fused_feature)
            else:
                # if single modality, process directly
                if agent_modality in modality_feature_dict:
                    feat_idx = counting_dict[agent_modality]
                    heter_feature_2d_list.append(modality_feature_dict[agent_modality][feat_idx:feat_idx+1])
                    counting_dict[agent_modality] += 1

        # stack heter features and compress if needed
        heter_feature_2d = torch.cat(heter_feature_2d_list, dim=0)
        if self.compress: heter_feature_2d = self.compressor(heter_feature_2d)

        # single supervision
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        # feature fusion (multiscale), we omit self.backbone's first layer
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        if self.shrink_flag: fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'fused_feature': fused_feature})

        return output_dict

class HeterModelBaselineMs(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineMs, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']
        self.stage2_added_modality = args.get('stage2_added_modality', None)

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            # depth supervision for camera args
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else: setattr(self, f"depth_supervision_{modality_name}", False)
            
        # for feature transformation
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # single supervision
        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))

        # build fusion net, by default multiscale fusion
        self.backbone = ResNetBEVBackbone(args['fusion_backbone'])
        self.fusion_net = nn.ModuleList()
        # fusion_method must be a list, one method per scale
        fusion_method_list = args['fusion_method']
        assert isinstance(fusion_method_list, list), f"fusion_method must be a list, got {type(fusion_method_list)}"
        assert len(fusion_method_list) == len(args['fusion_backbone']['layer_nums']), f"fusion_method list length ({len(fusion_method_list)}) must match fusion_backbone layer_nums length ({len(args['fusion_backbone']['layer_nums'])})"
        for i in range(len(args['fusion_backbone']['layer_nums'])):
            method = fusion_method_list[i]
            if method == "max": self.fusion_net.append(MaxFusion(args['fusion_method_config'][i]))
            if method == "att": self.fusion_net.append(AttFusion(args['fusion_method_config'][i]))
            if method == "disconet": self.fusion_net.append(DiscoFusion(args['fusion_method_config'][i]))
            if method == "v2vnet": self.fusion_net.append(V2VNetFusion(args['fusion_method_config'][i]))
            if method == 'v2xvit': self.fusion_net.append(V2XViTFusion(args['fusion_method_config'][i]))
            if method == 'cobevt': self.fusion_net.append(CoBEVT(args['fusion_method_config'][i]))
            if method == 'where2comm': self.fusion_net.append(Where2commFusion(args['fusion_method_config'][i]))
            if method == 'who2com': self.fusion_net.append(Who2comFusion(args['fusion_method_config'][i]))

        # build cross-modal fusion, only initialize if combined modalities (e.g., "m1&m2") are used
        self.use_cross_modal_fusion = False
        if isinstance(args['ego_modality'], str) and "&" in args['ego_modality']:
            self.use_cross_modal_fusion = True
        if self.use_cross_modal_fusion:
            assert len(self.modality_name_list) == 2, "Cross-modal fusion requires exactly two modalities"
            cross_modal_feat_dim = 64
            backbone_args = args[self.modality_name_list[0]]['backbone_args']
            cross_modal_feat_dim_m1 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            backbone_args = args[self.modality_name_list[1]]['backbone_args']
            cross_modal_feat_dim_m2 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            self.cross_modal_fusion = Cross_Modal_Fusion(
                kernel_size=3,
                img_channels=cross_modal_feat_dim_m1,
                rad_channels=cross_modal_feat_dim_m2,
                out_channels=cross_modal_feat_dim )
        else: self.cross_modal_fusion = None

        # build shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # build shared heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
        
        # build compressor, only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'], args['compressor']['compress_ratio'])
            self.model_train_init()

        # check again which module is not fixed.
        check_trainable_module(self)
    
    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        
        # get data
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # print(agent_modality_list)
        #  modality_count_dict = Counter(agent_modality_list)
        # count the number of each modality, including combined modalities
        modality_count_dict = {}
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, split and count each sub-modality
                sub_modalities = agent_modality.split("&")
                for sub_mod in sub_modalities:
                    modality_count_dict[sub_mod] = modality_count_dict.get(sub_mod, 0) + 1
            else:
                # if single modality, count directly
                modality_count_dict[agent_modality] = modality_count_dict.get(agent_modality, 0) + 1
        

        # get feature from each modality
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict: continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # crop/padd camera feature map
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})

        # assemble heter features
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, fuse multiple sub-modalities
                sub_modalities = agent_modality.split("&")
                # get feature from each sub-modality for the same CAV
                sub_features = []
                for sub_mod in sub_modalities:
                    if sub_mod in modality_feature_dict:
                        feat_idx = counting_dict[sub_mod]
                        sub_features.append(modality_feature_dict[sub_mod][feat_idx:feat_idx+1])
                        counting_dict[sub_mod] += 1
                # fuse multiple sub-modalities using Cross_Modal_Fusion
                assert len(sub_features) >= 2, "At least two modalities are required for cross-modal fusion"
                assert self.cross_modal_fusion is not None, "cross_modal_fusion is not initialized. Combined modalities require cross_modal_fusion."
                fused_feature = self.cross_modal_fusion(sub_features[0], sub_features[1])
                heter_feature_2d_list.append(fused_feature)
            else:
                # if single modality, process directly
                if agent_modality in modality_feature_dict:
                    feat_idx = counting_dict[agent_modality]
                    heter_feature_2d_list.append(modality_feature_dict[agent_modality][feat_idx:feat_idx+1])
                    counting_dict[agent_modality] += 1

        # stack heter features and compress if needed
        heter_feature_2d = torch.cat(heter_feature_2d_list, dim=0)
        if self.compress: heter_feature_2d = self.compressor(heter_feature_2d)

        # single supervision
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        # feature fusion (multiscale), we omit self.backbone's first layer
        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        # shrink feature if needed
        if self.shrink_flag: fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'fused_feature': fused_feature})

        return output_dict

class HeterPyramidCollab(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            
            # depth supervision for camera args
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}")}
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else: setattr(self, f"depth_supervision_{modality_name}", False)

        # for feature transformation
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # build fusion net, by default multiscale fusion
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        # build cross-modal fusion, only initialize if combined modalities (e.g., "m1&m2") are used
        self.use_cross_modal_fusion = False
        if isinstance(args['ego_modality'], str) and "&" in args['ego_modality']:
            self.use_cross_modal_fusion = True
        if self.use_cross_modal_fusion:
            assert len(self.modality_name_list) == 2, "Cross-modal fusion requires exactly two modalities"
            cross_modal_feat_dim = 64
            backbone_args = args[self.modality_name_list[0]]['backbone_args']
            cross_modal_feat_dim_m1 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            backbone_args = args[self.modality_name_list[1]]['backbone_args']
            cross_modal_feat_dim_m2 = backbone_args['num_filters'][-1] # 使用最后一层的通道数
            self.cross_modal_fusion = Cross_Modal_Fusion(
                kernel_size=3,
                img_channels=cross_modal_feat_dim_m1,
                rad_channels=cross_modal_feat_dim_m2,
                out_channels=cross_modal_feat_dim )
        else: self.cross_modal_fusion = None

        # build shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # build shared heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
        
        # build compressor, only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'], args['compressor']['compress_ratio'])
            self.model_train_init()
        
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        
        # get data
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_feature_dict = {}
        
        # print(agent_modality_list)
        #  modality_count_dict = Counter(agent_modality_list)
        # count the number of each modality, including combined modalities
        modality_count_dict = {}
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, split and count each sub-modality
                sub_modalities = agent_modality.split("&")
                for sub_mod in sub_modalities:
                    modality_count_dict[sub_mod] = modality_count_dict.get(sub_mod, 0) + 1
            else:
                # if single modality, count directly
                modality_count_dict[agent_modality] = modality_count_dict.get(agent_modality, 0) + 1
        
        # get feature from each modality
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict: continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # crop/padd camera feature map
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})

        # assemble heter features
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for agent_modality in agent_modality_list:
            if "&" in agent_modality:
                # if combined modality, fuse multiple sub-modalities
                sub_modalities = agent_modality.split("&")
                # get feature from each sub-modality for the same CAV
                sub_features = []
                for sub_mod in sub_modalities:
                    if sub_mod in modality_feature_dict:
                        feat_idx = counting_dict[sub_mod]
                        sub_features.append(modality_feature_dict[sub_mod][feat_idx:feat_idx+1])
                        counting_dict[sub_mod] += 1
                # fuse multiple sub-modalities using Cross_Modal_Fusion
                assert len(sub_features) >= 2, "At least two modalities are required for cross-modal fusion"
                assert self.cross_modal_fusion is not None, "cross_modal_fusion is not initialized. Combined modalities require cross_modal_fusion."
                fused_feature = self.cross_modal_fusion(sub_features[0], sub_features[1])
                heter_feature_2d_list.append(fused_feature)
            else:
                # if single modality, process directly
                if agent_modality in modality_feature_dict:
                    feat_idx = counting_dict[agent_modality]
                    heter_feature_2d_list.append(modality_feature_dict[agent_modality][feat_idx:feat_idx+1])
                    counting_dict[agent_modality] += 1
        
        
        # stack heter features and compress if needed
        heter_feature_2d = torch.cat(heter_feature_2d_list, dim=0)
        if self.compress: heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x, add croping information to collaboration module
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab( \
            heter_feature_2d, record_len, affine_matrix, agent_modality_list, self.cam_crop_info)

        # shrink feature if needed
        if self.shrink_flag: fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'fused_feature': fused_feature,
            'occ_single_list': occ_outputs})
 
        return output_dict
    
class HeterPyramidSingle(nn.Module):
    def __init__(self, args):
        super(HeterPyramidSingle, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # setup backbone (very light-weight)
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if args.get("fix_encoder", False): self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

            
            # depth supervision for camera args
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else: setattr(self, f"depth_supervision_{modality_name}", False)
            
        # build fusion net, by default multiscale fusion
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        # build shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.fix_modules.append('shrink_conv')

        # build shared heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2

        # only trainable compressor
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self) 

    def model_train_init(self):
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)

    def forward(self, data_dict):
        
        # get data
        output_dict = {'pyramid': 'single'}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1, "Only one modality is supported for single-modality training"
        modality_name = modality_name[0].lstrip('inputs_')

        # get feature from each modality
        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop((int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}"))))(feature)
            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})
        
        # multiscale fusion. 
        feature, occ_map_list = self.pyramid_backbone.forward_single(feature)
        # shrink feature if needed
        if self.shrink_flag: feature = self.shrink_conv(feature)

        cls_preds = self.cls_head(feature)
        reg_preds = self.reg_head(feature)
        dir_preds = self.dir_head(feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'occ_single_list': occ_map_list})
        
        return output_dict