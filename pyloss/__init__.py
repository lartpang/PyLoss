# -*- coding: utf-8 -*-
from .lp.L12Loss import cal_mae_loss, cal_mse_loss
from .lp.SelfRegularizationLoss import cal_smooth_sparse_loss, cal_sparse_loss
from .region_based.DiceLoss import cal_dice_loss
from .region_based.IOULoss import cal_iou_loss, cal_weighted_iou_loss
from .region_based.SSIM import cal_ssim_loss
