# -*- coding: utf-8 -*-
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim
# /__init__.py
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.misc import check_args, reduce_score


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    def __init__(self, channel=1, window_size=11, mean_on_loss=True):
        super(SSIM, self).__init__()
        """
        :param channel: the channel number of the input
        :param window_size: the size of the window
        :param mean_on_loss: calculate the mean of the loss or the mean of the score.
        """
        self.channel = channel
        self.window_size = window_size
        self.window = create_window(window_size, self.channel)
        self.mean_on_loss = mean_on_loss

    @check_args
    def forward(self, logits, gts):
        """
        :param logits: N,C,H,W
        :param gts: N,1,H,W
        """
        assert logits.shape[1] == self.channel, self.window.type() == logits.type()
        probs = logits.sigmoid()
        ssim_score = SSIM.cal_ssim(probs, gts, self.window, self.window_size, self.channel)
        return reduce_score(ssim_score, mean_on_loss=self.mean_on_loss)

    @classmethod
    def cal_ssim(cls, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        )
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean((1, 2, 3))


class LOGSSIM(nn.Module):
    def __init__(self, channel=1, window_size=11, mean_on_loss=True):
        """
        :param channel: the channel number of the input
        :param window_size: the size of the window
        :param mean_on_loss: calculate the mean of the loss or the mean of the score.
        """
        super(LOGSSIM, self).__init__()
        self.channel = channel
        self.window_size = window_size
        self.window = create_window(window_size, self.channel)
        self.mean_on_loss = mean_on_loss

    @check_args
    def forward(self, logits, gts):
        """
        :param logits: N,C,H,W
        :param gts: N,1,H,W
        """
        assert logits.shape[1] == self.channel, self.window.type() == logits.type()
        probs = logits.sigmoid()
        ssim_score = LOGSSIM.cal_logssim(probs, gts, self.window, self.window_size, self.channel)
        return reduce_score(ssim_score, mean_on_loss=self.mean_on_loss)

    @classmethod
    def cal_logssim(cls, img1, img2, window, window_size, channel, eps=1e-8):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        )
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim_map = (ssim_map - torch.min(ssim_map)) / (torch.max(ssim_map) - torch.min(ssim_map))
        ssim_map = -torch.log(ssim_map + eps)
        return ssim_map.mean((1, 2, 3))


@check_args
def cal_ssim_loss(logits, gts, window_size=11, mean_on_loss=True):
    """
    SSIM loss

    :param logits: N,C,H,W
    :param gts: N,1,H,W
    :param window_size: the size of the window
    :param mean_on_loss: calculate the mean of the loss or the mean of the score.
    """
    probs = logits.sigmoid()

    channel = probs.shape[1]
    window = create_window(window_size, channel)
    window = window.to(device=probs.device, dtype=probs.dtype)
    ssim_score = SSIM.cal_ssim(probs, gts, window, window_size, channel)
    return reduce_score(ssim_score, mean_on_loss=mean_on_loss)


@check_args
def cal_logssim_loss(logits, gts, window_size=11, mean_on_loss=True):
    """
    LOG SSIM loss

    :param logits: N,C,H,W
    :param gts: N,1,H,W
    :param window_size: the size of the window
    :param mean_on_loss: calculate the mean of the loss or the mean of the score.
    """
    probs = logits.sigmoid()

    channel = probs.shape[1]
    window = create_window(window_size, channel)
    window = window.to(device=probs.device, dtype=probs.dtype)
    ssim_score = LOGSSIM.cal_ssim(probs, gts, window, window_size, channel)
    return reduce_score(ssim_score, mean_on_loss=mean_on_loss)
