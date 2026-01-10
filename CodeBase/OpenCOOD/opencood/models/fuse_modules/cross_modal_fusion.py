import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3, img_channels=64, rad_channels=64, out_channels=64):
        super(Cross_Modal_Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                img_channels+rad_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev, radar_bev],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV