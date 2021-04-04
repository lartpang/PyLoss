# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 下午10:02
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from torch import nn
from torch.nn.functional import avg_pool2d

from ..utils.misc import reduce_loss


class HEL(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        """
        https://github.com/lartpang/HDFNet/blob/master/loss/HEL.py

        ::

            @inproceedings{HDFNet-ECCV2020,
                author = {Youwei Pang and Lihe Zhang and Xiaoqi Zhao and Huchuan Lu},
                title = {Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection},
                booktitle = ECCV,
                year = {2020}
            }

        :param eps: Avoid dividing by 0.
        :param reduction: mean, sum, or none
        """
        super(HEL, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def edge_loss(self, probs, gts):
        # 强调边缘区域的一致性
        edge = gts - avg_pool2d(gts, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1

        numerator = (edge * (probs - gts).abs()).sum([2, 3])
        denominator = edge.sum([2, 3]) + self.eps
        return numerator / denominator

    def region_loss(self, probs, gts):
        # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
        numerator_fg = (gts - gts * probs).sum([2, 3])
        denominator_fg = gts.sum([2, 3]) + self.eps

        numerator_bg = ((1 - gts) * probs).sum([2, 3])
        denominator_bg = (1 - gts).sum([2, 3]) + self.eps
        return numerator_fg / denominator_fg + numerator_bg / denominator_bg

    def forward(self, logits, gts):
        """
        :param logits: N,C,H,W
        :param gts: N,1,H,W
        :return: hel loss
        """
        probs = logits.sigmoid()
        edge_loss = self.edge_loss(probs, gts)
        region_loss = self.region_loss(probs, gts)
        loss = edge_loss + region_loss
        return reduce_loss(loss, self.reduction)


def cal_hel(logits, gts, eps=1e-6, reduction="mean"):
    """
    https://github.com/lartpang/HDFNet/blob/master/loss/HEL.py

    ::

        @inproceedings{HDFNet-ECCV2020,
            author = {Youwei Pang and Lihe Zhang and Xiaoqi Zhao and Huchuan Lu},
            title = {Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection},
            booktitle = ECCV,
            year = {2020}
        }

    :param logits: N,C,H,W
    :param gts: N,1,H,W
    :param eps: Avoid dividing by 0.
    :param reduction: mean, sum, or none
    """

    def _edge_loss(probs, gts):
        # 强调边缘区域的一致性
        edge = gts - avg_pool2d(gts, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1

        numerator = (edge * (probs - gts).abs_()).sum([2, 3])
        denominator = edge.sum([2, 3]) + eps
        return numerator / denominator

    def _region_loss(probs, gts):
        # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
        numerator_fg = (gts - gts * probs).sum([2, 3])
        denominator_fg = gts.sum([2, 3]) + eps

        numerator_bg = ((1 - gts) * probs).sum([2, 3])
        denominator_bg = (1 - gts).sum([2, 3]) + eps
        return numerator_fg / denominator_fg + numerator_bg / denominator_bg

    probs = logits.sigmoid()
    edge_loss = _edge_loss(probs, gts)
    region_loss = _region_loss(probs, gts)
    loss = edge_loss + region_loss
    return reduce_loss(loss, reduction)
