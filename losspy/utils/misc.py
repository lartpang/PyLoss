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
    def wrapper(logits, gts=None, **kwargs):
        if logits.ndim != 4:
            raise ValueError(
                f"Only support N,C,H,W logits, but get {logits.shape} and {logits.type()}"
            )
        if gts is not None:
            if gts.ndim != 4 or gts.shape[1] != 1:
                raise ValueError(f"Only support N,1,H,W gts, but get {gts.shape} and {gts.type()}")
            if logits.shape[0] != gts.shape[0] or logits.shape[2:] != gts.shape[2:]:
                raise ValueError(
                    f"Logits {logits.shape} and gts {gts.shape} must have the same size."
                )
            if not 1 <= logits.shape[1] <= gts.max():
                raise ValueError(
                    "The num_classes of logits is not compatible with the one of gts."
                )
        return func(logits, gts, **kwargs)

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
