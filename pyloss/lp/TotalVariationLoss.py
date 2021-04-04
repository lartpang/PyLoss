# -*- coding: utf-8 -*-
# @Time    : 2020/10/9
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        n, c, h, w = x.shape
        # 下面减去上面
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h - 1, :]), 2).mean()
        # 右面减去左面
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w - 1]), 2).mean()
        return self.weight * 2 * (h_tv + w_tv)


def total_variation_loss(x, is_sigmoid=False):
    if not is_sigmoid:
        x = torch.sigmoid(x)
    n, c, h, w = x.shape
    h_tv = (x[:, :, 1:, :] - x[:, :, : h - 1, :]).pow(2)
    w_tv = (x[:, :, :, 1:] - x[:, :, :, : w - 1]).pow(2)
    hw_tv = (h_tv + w_tv).sqrt().mean()
    return hw_tv


def total_variation_loss_ori(x, weight=1, is_sigmoid=False):
    if not is_sigmoid:
        x = torch.sigmoid(x)
    n, c, h, w = x.shape
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h - 1, :]), 2).mean()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w - 1]), 2).mean()
    return weight * 2 * (h_tv + w_tv)
