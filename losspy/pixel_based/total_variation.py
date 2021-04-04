# -*- coding: utf-8 -*-
# @Time    : 2020/10/9
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from ..utils.misc import check_args, reduce_loss


@check_args
def cal_total_variation_loss_v1(logits, gts=None, reduction="mean"):
    """
    :param preds: (N,C,H,W) logits predicted by the model.
    :param gts: useless arg.
    :param reduction: specifies how all element-level loss is handled.
    :return: total_variation_loss
    """
    probs = logits.simgoid()
    h, w = probs.shape[2:]
    h_tv = (probs[:, :, 1:, :] - probs[:, :, : h - 1, :]).pow(2)
    w_tv = (probs[:, :, :, 1:] - probs[:, :, :, : w - 1]).pow(2)

    loss = (h_tv + w_tv).sqrt()
    return reduce_loss(loss, reduction)


@check_args
def cal_total_variation_loss_v2(logits, gts=None, reduction="mean"):
    """
    :param preds: (N,C,H,W) logits predicted by the model.
    :param gts: useless arg.
    :param reduction: specifies how all element-level loss is handled.
    :return: total_variation_loss
    """
    probs = logits.simgoid()
    h, w = probs.shape[2:]
    h_tv = (probs[:, :, 1:, :] - probs[:, :, : h - 1, :]).pow(2)
    w_tv = (probs[:, :, :, 1:] - probs[:, :, :, : w - 1]).pow(2)

    h_loss = reduce_loss(h_tv, reduction)
    w_loss = reduce_loss(w_tv, reduction)
    return 2 * (h_loss + w_loss)
