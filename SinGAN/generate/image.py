import torch
import torch.nn as nn

import SinGAN.util as util


def generate_image(Gs, Zs, reals, NoiseAmp, cfg, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=50):
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=cfg.device)

    images_cur = []
    for n, (G, Z_opt, noise_amp) in enumerate(zip(Gs, Zs, NoiseAmp), start=n):
        pad1 = (cfg.ker_size-1)*cfg.num_layer/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        # save images from previous scale
        images_prev = images_cur
        images_cur = []

        for i in range(num_samples):
            if n == 0:
                z_curr = util.generate_noise([1, nzx, nzy], device=cfg.device)
                z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = util.generate_noise([cfg.noise_channels, nzx, nzy], device=cfg.device)
                z_curr = m(z_curr)

            if not images_prev:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = util.imresize(I_prev, 1/cfg.scale_factor, cfg)
                if cfg.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                    I_prev = util.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            try:
                z_in = noise_amp*z_curr + I_prev
            except RuntimeError as e:
                raise RuntimeError(f'Shape mismatch in generate ({z_curr.shape, I_prev.shape}). Try in/decreasing min/max_size in config.') from e

            I_curr = G(z_in.detach(), I_prev)

            # if n == len(reals)-1:
            #     assert cfg.mode == 'harmonisation'
            #     plt.imsave(f'{cfg.output_dir}/{i}.png', util.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            images_cur.append(I_curr)
    assert images_cur
    return images_cur[-1].detach()
