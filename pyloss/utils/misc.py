# -*- coding: utf-8 -*-
# @Time    : 2020/12/12
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from functools import wraps

import numpy as np
import torch


def reduce_score(score: torch.Tensor, mean_on_loss: bool = True):
    if mean_on_loss:
        loss = (1 - score).mean()
    else:
        loss = 1 - score.mean()
    return loss


def reduce_loss(loss: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    :param loss: loss tensor
    :param reduction: mean, sum, or none
    """
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError
    return loss


def check_args(func):
    """
    Decorator that checks the validity of the parameters.
    """

    @wraps(func)
    def wrapper(preds, gts=None, **kwargs):
        if not (preds.ndim == 4 and isinstance(preds, torch.FloatTensor)):
            raise ValueError("Only support N,C,H,W preds(FloatTensor)")
        if gts is not None:
            if not (
                gts.ndim == 4
                and gts.shape[1] == 1
                and isinstance(gts, (torch.BoolTensor, torch.IntTensor, torch.LongTensor))
            ):
                raise ValueError(
                    "Only support N,1,H,W gts(torch.BoolTensor, torch.IntTensor, torch.LongTensor)."
                )
            if preds.shape[2:] != gts.shape[2:]:
                raise ValueError("Preds and gts must have the same size.")
            if not 1 <= preds.shape[1] <= gts.max():
                raise ValueError("The num_classes of preds is not compatible with the one of gts.")
        return func(preds, gts, **kwargs)

    return wrapper


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
