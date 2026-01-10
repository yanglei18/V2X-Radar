
"""
Implementation of S-AdaFusion model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFusion(nn.Module):
    def __init__(self):
        super(AdaFusion, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 1, 3, stride=1, padding=1),
            # nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            x_max = torch.max(xx, dim=0, keepdim=True)[0]
            x_mean = torch.mean(xx, dim=0, keepdim=True)
            out.append(torch.cat((x_max, x_mean), dim=0).unsqueeze(0))
        out = torch.cat(out, dim=0)
        out = self.conv3d(out).squeeze(1)
        return out