# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


def cal_sparse_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    seg_probs = seg_logits.sigmoid()

    loss_map = 1 - (seg_probs * 2 - 1).abs()
    return loss_map.mean()


def cal_smooth_sparse_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)

    # exp_x = seg_logits.exp()
    # loss_map = 4 * exp_x / (exp_x + 1).pow(2)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).pow(2)
    return loss_map.mean()
