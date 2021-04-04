# -*- coding: utf-8 -*-
# @Time    : 2020/12/12
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch

from utils.misc import arg_check_for_loss


def _cal_dice_loss(seg_probs: torch.FloatTensor, seg_gts: torch.BoolTensor, smooth_factor: int = 1):
    """
    Args:
        seg_probs: N,HW
        seg_gts: N,HW
        smooth_factor: default 1

    Returns: N
    """
    numerator = 2 * (seg_probs * seg_gts).sum(-1) + smooth_factor
    denominator = (seg_probs + seg_gts).sum(dim=-1) + smooth_factor
    dice_loss = 1 - numerator / denominator
    return dice_loss


def cal_dice_loss(seg_logits, seg_gts):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        seg_logits: N,C,H,W
        seg_gts: N,1,H,W
    """
    arg_check_for_loss(seg_logits=seg_logits, seg_gts=seg_gts)

    seg_probs = seg_logits.sigmoid().flatten(2).permute(1, 0, 2)  # C,N,HW
    seg_gts = seg_gts.flatten(1)  # N,HW

    # 对类别平均
    num_classes = seg_probs.shape[0]
    if num_classes == 1:
        dice_loss = _cal_dice_loss(seg_probs=seg_probs[0], seg_gts=seg_gts == 1)
    else:
        dice_loss = 0
        for c in range(num_classes):
            dice_loss += _cal_dice_loss(seg_probs=seg_probs[c], seg_gts=seg_gts == c)
        dice_loss = dice_loss / num_classes
    # 对batch平均
    return dice_loss.mean()


if __name__ == "__main__":
    a = torch.randn(4, 2, 10, 10)
    b = torch.randint(low=0, high=2, size=(4, 1, 10, 10))
    print(cal_dice_loss(seg_logits=a, seg_gts=b))
