# -*- coding: utf-8 -*-
from .pixel_based.cross_entropy import (
    cal_bce_loss,
    cal_ce_loss,
    cal_ohembce_loss,
    cal_sigmoid_focal_loss,
)
from .pixel_based.mae_mse import cal_mae_loss, cal_mse_loss
from .pixel_based.total_variation import (
    cal_total_variation_loss_v1,
    cal_total_variation_loss_v2,
)
from .region_based.dice_iou import cal_dice_loss, cal_iou_loss
from .region_based.hel import HEL, cal_hel
from .region_based.ssim import LOGSSIM, SSIM, cal_logssim_loss, cal_ssim_loss
