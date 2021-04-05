# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from ..utils.misc import check_args, reduce_loss


@check_args
def cal_bce_loss(logits, gts, reduction="mean"):
    """
    :param logits: (N,C,H,W) logits predicted by the model.
    :param gts: (N,1,H,W) ground truths.
    :param reduction: specifies how all element-level loss is handled.
    :return: bce loss
    """
    return binary_cross_entropy_with_logits(input=logits, target=gts, reduction=reduction)


@check_args
def cal_ce_loss(logits, gts, reduction="mean"):
    """
    :param logits: (N,C,H,W) logits predicted by the model.
    :param gts: (N,1,H,W) ground truths.
    :param reduction: specifies how all element-level loss is handled.
    :return: ce loss
    """
    return cross_entropy(input=logits, target=gts, reduction=reduction)


@check_args
def cal_sigmoid_focal_loss(logits, gts, reduction="mean", alpha=0.25, gamma=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    :param logits: (N,C,H,W) logits predicted by the model.
    :param gts: (N,1,H,W) ground truths.
    :param reduction: specifies how all element-level loss is handled.
    :param alpha: weighting factor in range (0,1) to balance positive vs negative examples
    :param gamma: exponent of the modulating factor (1 - p_t) to balance easy vs hard examples
    :return: focal loss based on bce loss
    """
    bce_loss = binary_cross_entropy_with_logits(logits, gts, reduction="none")

    probs = logits.sigmoid()
    p_t = probs * gts + (1 - probs) * (1 - gts)
    alpha_t = alpha * gts + (1 - alpha) * (1 - gts)
    loss = alpha_t * ((1 - p_t) ** gamma) * bce_loss
    return reduce_loss(loss, reduction)


class OHEMBCELoss(nn.Module):
    def __init__(self, threshold=None, min_kept_numel=100000, reduction="mean"):
        """
        Reference:
        https://github.com/open-mmlab/mmsegmentation/blob/4345ee3d55a5198dc223c47e21c73b73fb44a57d/mmseg/core/seg/sampler/ohem_pixel_sampler.py#L9

        Sample pixels that have high loss or with low prediction confidence.

        :param threshold: (float, optional) The threshold for hard example selection.
            Below which, are prediction with low confidence. If not specified, the hard
            examples will be pixels of top ``min_kept_numel`` loss.
        :param min_kept_numel: (int, optional) The minimum number of predictions to keep.
        :param reduction: mean, sum, or none
        """
        super(OHEMBCELoss, self).__init__()
        assert min_kept_numel > 1, min_kept_numel
        assert threshold is None or 0 < threshold < 1, threshold

        self.threshold = threshold
        self.min_kept_numel = min_kept_numel
        self.reduction = reduction

    @check_args
    def forward(self, logits, gts):
        """
        Sample the loss for pixels that:
        - [X] have high loss
        - [X] or with low prediction confidence

        :param logits: (N,C,H,W)
        :param gts: (N,1,H,W)
        :return: ohembceloss
        """
        batch_min_kept_numel = self.min_kept_numel * gts.size(0)
        bce_loss = binary_cross_entropy_with_logits(logits, gts, reduction="none")
        flatten_bce_loss = bce_loss.reshape(-1)

        with torch.no_grad():
            if self.threshold is None:
                # for high loss
                sorted_loss, indices = flatten_bce_loss.sort(descending=True)  # large -> small
                # select the largest loss
                loss_indices = indices[:batch_min_kept_numel]
            else:
                # 0&1: has the largest certainty, 0.5: has the smallest one
                confidence = (logits.sigmoid() - 0.5).abs() * 2
                sorted_conf, indices = confidence.reshape(-1).sort()  # small -> large
                min_threshold = sorted_conf[min(batch_min_kept_numel, sorted_conf.numel() - 1)]
                final_threshold = max(min_threshold, self.threshold)
                # select the largest loss
                loss_indices = indices[sorted_conf < final_threshold]
        loss = flatten_bce_loss[loss_indices]
        return reduce_loss(loss, self.reduction)


@check_args
def cal_ohembce_loss(logits, gts, threshold=None, min_kept_numel=100000, reduction="mean"):
    """
    Sample the loss for pixels that:
    - [X] have high loss
    - [X] or with low prediction confidence

    :param logits: (N,C,H,W)
    :param gts: (N,1,H,W)
    :param threshold: (float, optional) The threshold for hard example selection.
        Below which, are prediction with low confidence. If not specified, the hard
        examples will be pixels of top ``min_kept_numel`` loss.
    :param min_kept_numel: (int, optional) The minimum number of predictions to keep.
    :param reduction: mean, sum, or none
    :return: ohembceloss
    """
    batch_min_kept_numel = min_kept_numel * gts.size(0)
    bce_loss = binary_cross_entropy_with_logits(logits, gts, reduction="none")
    flatten_bce_loss = bce_loss.reshape(-1)

    with torch.no_grad():
        if threshold is None:
            # for high loss
            sorted_loss, indices = flatten_bce_loss.sort(descending=True)  # large -> small
            # select the largest loss
            loss_indices = indices[:batch_min_kept_numel]
        else:
            # 0&1: has the largest certainty, 0.5: has the smallest one
            confidence = (logits.sigmoid() - 0.5).abs() * 2
            sorted_conf, indices = confidence.reshape(-1).sort()  # small -> large
            min_threshold = sorted_conf[min(batch_min_kept_numel, sorted_conf.numel() - 1)]
            final_threshold = max(min_threshold, threshold)
            # select the largest loss
            loss_indices = indices[sorted_conf < final_threshold]
    loss = flatten_bce_loss[loss_indices]
    return reduce_loss(loss, reduction)
