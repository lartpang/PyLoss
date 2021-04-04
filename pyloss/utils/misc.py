# -*- coding: utf-8 -*-
# @Time    : 2020/12/12
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import numpy as np


def arg_check_for_loss(seg_logits, seg_gts):
    assert seg_logits.ndim == 4 and seg_gts.ndim == 4 and seg_gts.shape[1] == 1, (seg_logits.shape, seg_gts.shape)
    assert seg_logits.shape[1] >= seg_gts.max()


def cal_sparse_coef(curr_iter, num_iter, method="linear", extra_args=None):
    if extra_args is None:
        extra_args = {}

    def _linear(curr_iter, milestones=(0.3, 0.7), coef_range=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if curr_iter < (num_iter * min_point):
            coef = min_coef
        elif curr_iter > (num_iter * max_point):
            coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (num_iter * (max_point - min_point))
            coef = ratio * (curr_iter - num_iter * min_point)
        return coef

    def _cos(curr_iter, coef_range=(0, 1)):
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(curr_iter / num_iter * np.pi)) / 2
        coef = normalized_coef * (max_coef - min_coef) + min_coef
        return coef

    def _constant(curr_iter, constant=1.0):
        return constant

    _funcs = dict(
        linear=_linear,
        cos=_cos,
        constant=_constant,
    )

    coef = _funcs[method](curr_iter, **extra_args)
    return coef
