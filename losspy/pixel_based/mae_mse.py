# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from ..utils.misc import check_args, reduce_loss


@check_args
def cal_mae_loss(logits, gts, reduction):
    """
    :param preds: (N,C,H,W) logits predicted by the model.
    :param gts: (N,1,H,W) ground truths.
    :param reduction: specifies how all element-level loss is handled.
    :return: mae loss
    """
    probs = logits.sigmoid()
    loss = (probs - gts).abs()
    return reduce_loss(loss, reduction)


@check_args
def cal_mse_loss(logits, gts, reduction):
    """
    :param preds: (N,C,H,W) logits predicted by the model.
    :param gts: (N,1,H,W) ground truths.
    :param reduction: specifies how all element-level loss is handled.
    :return: mse loss
    """
    probs = logits.sigmoid()
    loss = (probs - gts).pow(2)
    return reduce_loss(loss, reduction)
