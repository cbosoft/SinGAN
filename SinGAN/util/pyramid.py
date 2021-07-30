import math
import os

import torch

from .imresize import imresize


def create_reals_pyramid(real, reals, cfg):
    real = real[:,0:3,:,:]
    for i in range(cfg.stop_scale+1):
        scale = math.pow(cfg.scale_factor, cfg.stop_scale - i)
        curr_real = imresize(real, scale, cfg)
        reals.append(curr_real)


def load_trained_pyramid(dn):
    assert os.path.exists(dn), f'No trained model found at "{dn}".'

    Gs = torch.load(f'{dn}/Gs.pth')
    Zs = torch.load(f'{dn}/Zs.pth')
    reals = torch.load(f'{dn}/reals.pth')
    NoiseAmp = torch.load(f'{dn}/NoiseAmp.pth')

    return Gs, Zs, reals, NoiseAmp

