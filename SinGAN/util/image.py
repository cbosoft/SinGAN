import math

from skimage import io as img
from skimage import color
import torch

from .imresize import norm, imresize, np2torch


def read_image(fn, cfg):
    x = img.imread(fn)
    if len(x.shape) != 3 or x.shape[-1] == 1:
        x = color.gray2rgb(x)
    x = np2torch(x, cfg)
    x = x[:, 0:3, :, :]
    return x


def adjust_scales_to_image(image, cfg):
    B, C, H, W = image.shape
    Sn = min([H, W])
    Sx = max([H, W])
    cfg.num_scales = math.ceil((math.log(math.pow(cfg.min_size / Sn, 1), cfg.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([cfg.max_size, Sn]) / Sx, cfg.scale_factor_init))
    cfg.stop_scale = cfg.num_scales - scale2stop
    cfg.scale1 = min(cfg.max_size / Sx, 1)
    real = imresize(image, cfg.scale1, cfg)
    cfg.scale_factor = math.pow(cfg.min_size/Sn, 1/cfg.stop_scale)
    scale2stop = math.ceil(math.log(min([cfg.max_size, Sx]) / Sx, cfg.scale_factor_init))
    cfg.stop_scale = cfg.num_scales - scale2stop
    return real
