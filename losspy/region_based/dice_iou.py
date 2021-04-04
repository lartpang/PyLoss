# -*- coding: utf-8 -*-
# @Time    : 2020/12/12
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from ..utils.misc import check_args, reduce_loss


def _cal_dice_loss(probs, gts, smooth_factor: int = 1):
    numerator = 2 * (probs * gts).sum(-1) + smooth_factor
    denominator = (probs + gts).sum(-1) + smooth_factor
    dice_loss = 1 - numerator / denominator
    return dice_loss


@check_args
def cal_dice_loss(logits, gts, reduction="mean"):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    :param logits: N,C,H,W
    :param gts: N,1,H,W
    :param reduction: mean, sum or none
    :return: dice loss
    """
    probs = logits.sigmoid().flatten(2).permute(1, 0, 2)  # C,N,HW
    gts = gts.flatten(1)  # N,HW

    # 对类别平均
    num_classes = probs.shape[0]
    if num_classes == 1:
        loss = _cal_dice_loss(probs=probs[0], gts=gts)
    else:
        losses = [_cal_dice_loss(probs=probs[c], gts=gts == c) for c in range(num_classes)]
        loss = sum(losses) / num_classes
    return reduce_loss(loss, reduction)


def _cal_iou_loss(probs, gts, smooth_factor: int = 1):
    inter = (probs * gts).sum(dim=-1)
    union = (probs + gts).sum(dim=-1)
    iou_loss = 1 - (inter + smooth_factor) / (union - inter + smooth_factor)
    return iou_loss


@check_args
def cal_iou_loss(logits, gts, reduction="mean"):
    """
    IOU Loss

    :param logits: N,C,H,W
    :param gts: N,1,H,W
    :param reduction: mean, sum or none
    :return: iou loss
    """
    probs = logits.sigmoid().flatten(2).permute(1, 0, 2)  # C,N,HW
    gts = gts.flatten(1)  # N,HW

    # 对类别平均
    num_classes = probs.shape[0]
    if num_classes == 1:
        loss = _cal_iou_loss(probs=probs[0], gts=gts)
    else:
        losses = [_cal_iou_loss(probs=probs[c], gts=gts == c) for c in range(num_classes)]
        loss = sum(losses) / num_classes
    return reduce_loss(loss, reduction)
