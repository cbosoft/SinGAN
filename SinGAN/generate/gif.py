import argparse
import os
import random
import math

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
from skimage import io as img
from skimage import color
import imageio
import matplotlib.pyplot as plt

import SinGAN.util as util

# def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):
#
#     in_s = torch.full(Zs[0].shape, 0, device=opt.device)
#     images_cur = []
#     count = 0
#
#     for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
#         pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#         nzx = Z_opt.shape[2]
#         nzy = Z_opt.shape[3]
#         #pad_noise = 0
#         #m_noise = nn.ZeroPad2d(int(pad_noise))
#         m_image = nn.ZeroPad2d(int(pad_image))
#         images_prev = images_cur
#         images_cur = []
#         if count == 0:
#             z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
#             z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
#             z_prev1 = 0.95*Z_opt +0.05*z_rand
#             z_prev2 = Z_opt
#         else:
#             z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
#             z_prev2 = Z_opt
#
#         for i in range(0,100,1):
#             if count == 0:
#                 z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
#                 z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
#                 diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
#             else:
#                 diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))
#
#             z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
#             z_prev2 = z_prev1
#             z_prev1 = z_curr
#
#             if images_prev == []:
#                 I_prev = in_s
#             else:
#                 I_prev = images_prev[i]
#                 I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
#                 I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
#                 #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
#                 I_prev = m_image(I_prev)
#             if count < start_scale:
#                 z_curr = Z_opt
#
#             z_in = noise_amp*z_curr+I_prev
#             I_curr = G(z_in.detach(),I_prev)
#
#             if (count == len(Gs)-1):
#                 I_curr = functions.denorm(I_curr).detach()
#                 I_curr = I_curr[0,:,:,:].cpu().numpy()
#                 I_curr = I_curr.transpose(1, 2, 0)*255
#                 I_curr = I_curr.astype(np.uint8)
#
#             images_cur.append(I_curr)
#         count += 1
#     dir2save = functions.generate_dir2save(opt)
#     try:
#         os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
#     except OSError:
#         pass
#     imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
#     del images_cur
