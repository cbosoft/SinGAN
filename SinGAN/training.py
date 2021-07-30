import os
import math

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import trange

import matplotlib.pyplot as plt

import SinGAN.models as models
import SinGAN.util as util


def train(Gs, Zs, reals, NoiseAmp, cfg):
    training_image = util.read_image(cfg.training.image, cfg)
    in_s = 0
    real = util.imresize(training_image, cfg.scale1, cfg)
    util.create_reals_pyramid(real, reals, cfg)
    nfc_prev = 0
    outp = cfg.training.models_dir

    for scale_num in trange(cfg.stop_scale+1, unit='scale'):
        cfg.nfc = min(cfg.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        cfg.min_nfc = min(cfg.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        outp_sdir = f'{outp}/{scale_num}'
        os.makedirs(outp_sdir, exist_ok=True)

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave(f'{outp_sdir}/real_scale.png', util.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr, G_curr = models.init(cfg)
        if nfc_prev == cfg.nfc:
            G_curr.load_state_dict(torch.load(f'{outp}/{scale_num-1}/netG.pth'))
            D_curr.load_state_dict(torch.load(f'{outp}/{scale_num-1}/netD.pth'))

        z_curr, in_s, G_curr, D_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, cfg)
        torch.save(G_curr.state_dict(), f'{outp_sdir}/netG.pth')
        torch.save(D_curr.state_dict(), f'{outp_sdir}/netD.pth')
        torch.save(z_curr, f'{outp_sdir}/z_opt.pth')

        G_curr = util.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = util.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(cfg.noise_amp)

        torch.save(Zs, f'{outp}/Zs.pth')
        torch.save(Gs, f'{outp}/Gs.pth')
        torch.save(reals, f'{outp}/reals.pth')
        torch.save(NoiseAmp, f'{outp}/NoiseAmp.pth')

        nfc_prev = cfg.nfc
        del D_curr, G_curr
    return



def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, cfg, centers=None):
    real = reals[len(Gs)]
    nzx = real.shape[2]
    nzy = real.shape[3]
    receptive_field = cfg.ker_size + ((cfg.ker_size-1)*(cfg.num_layer-1))*cfg.stride
    pad_noise = int(((cfg.ker_size - 1) * cfg.num_layer) / 2)
    pad_image = int(((cfg.ker_size - 1) * cfg.num_layer) / 2)

    # if cfg.mode == 'animation_train':
    #     cfg.nzx = real.shape[2]+(cfg.ker_size-1)*(cfg.num_layer)
    #     cfg.nzy = real.shape[3]+(cfg.ker_size-1)*(cfg.num_layer)
    #     pad_noise = 0

    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = cfg.alpha

    fixed_noise = util.generate_noise([cfg.noise_channels, nzx, nzy], device=cfg.device)
    z_cfg = torch.full(fixed_noise.shape, 0, device=cfg.device)
    z_cfg = m_noise(z_cfg)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.discriminator.lr, betas=(cfg.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.generator.lr, betas=(cfg.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=cfg.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=cfg.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_cfg2plot = []

    for epoch in trange(cfg.niter, unit=f'epoch'):
        if Gs == [] and cfg.mode != 'SR_train':
            z_cfg = util.generate_noise([1, nzx, nzy], device=cfg.device)
            z_cfg = m_noise(z_cfg.expand(1, 3, nzx, nzy))
            noise_ = util.generate_noise([1, nzx,  nzy], device=cfg.device)
            noise_ = m_noise(noise_.expand(1, 3, nzx, nzy))
        else:
            noise_ = util.generate_noise([cfg.noise_channels, nzx, nzy], device=cfg.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(cfg.discriminator.steps):
            # train with real
            netD.zero_grad()

            output = netD(real).to(cfg.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if j == 0 and epoch == 0:
                if Gs == [] and cfg.mode != 'SR_train':
                    prev = torch.full([1, cfg.noise_channels, nzx, nzy], 0, device=cfg.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1, cfg.noise_channels, nzx, nzy], 0, device=cfg.device)
                    z_prev = m_noise(z_prev)
                    cfg.noise_amp = 1
                elif cfg.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    cfg.noise_amp = float(cfg.noise_amp_init*RMSE)
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, cfg)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, cfg)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    cfg.noise_amp = float(cfg.noise_amp_init*RMSE)
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, cfg)
                prev = m_image(prev)

            # if cfg.mode == 'paint_train':
            #     prev = functions.quant2centers(prev,centers)
            #     plt.imsave('%s/prev.png' % (cfg.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if Gs == [] and cfg.mode != 'SR_train':
                noise = noise_
            else:
                noise = cfg.noise_amp*noise_+prev

            fake = netG(noise.detach(), prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = util.calc_gradient_penalty(netD, real, fake, cfg.lambda_grad, cfg.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(cfg.generator.steps):
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                loss = nn.MSELoss()
                # if cfg.mode == 'paint_train':
                #     z_prev = functions.quant2centers(z_prev, centers)
                #     plt.imsave('%s/z_prev.png' % (cfg.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_cfg = cfg.noise_amp*z_cfg+z_prev
                rec_loss = alpha*loss(netG(Z_cfg.detach(),z_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_cfg = z_cfg
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_cfg2plot.append(rec_loss)

        # if epoch % 25 == 0 or epoch == (cfg.niter-1):
        #     print('scale %d:[%d/%d]' % (len(Gs), epoch, cfg.niter))
        # if epoch % 500 == 0 or epoch == (cfg.niter-1):
        #     plt.imsave('%s/fake_sample.png' %  (cfg.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
        #     plt.imsave('%s/G(z_cfg).png'    % (cfg.outf),  functions.convert_image_np(netG(Z_cfg.detach(), z_prev).detach()), vmin=0, vmax=1)
        #     #plt.imsave('%s/D_fake.png'   % (cfg.outf), functions.convert_image_np(D_fake_map))
        #     #plt.imsave('%s/D_real.png'   % (cfg.outf), functions.convert_image_np(D_real_map))
        #     #plt.imsave('%s/z_cfg.png'    % (cfg.outf), functions.convert_image_np(z_cfg.detach()), vmin=0, vmax=1)
        #     #plt.imsave('%s/prev.png'     %  (cfg.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
        #     #plt.imsave('%s/noise.png'    %  (cfg.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
        #     #plt.imsave('%s/z_prev.png'   % (cfg.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
        #     torch.save(z_cfg, '%s/z_cfg.pth' % (cfg.outf))

        schedulerD.step()
        schedulerG.step()

    return z_cfg, in_s, netG, netD


def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,cfg):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((cfg.ker_size-1)*cfg.num_layer)/2)
            if cfg.mode == 'animation_train':
                pad_noise = 0
            for G,Z_cfg,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = util.generate_noise([1, Z_cfg.shape[2] - 2 * pad_noise, Z_cfg.shape[3] - 2 * pad_noise], device=cfg.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = util.generate_noise([cfg.noise_channels,Z_cfg.shape[2] - 2 * pad_noise, Z_cfg.shape[3] - 2 * pad_noise], device=cfg.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = util.imresize(G_z,1/cfg.scale_factor,cfg)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_cfg,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_cfg+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = util.imresize(G_z,1/cfg.scale_factor,cfg)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

# def train_paint(cfg,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
#     in_s = torch.full(reals[0].shape, 0, device=cfg.device)
#     scale_num = 0
#     nfc_prev = 0
# 
#     while scale_num<cfg.stop_scale+1:
#         if scale_num!=paint_inject_scale:
#             scale_num += 1
#             nfc_prev = cfg.nfc
#             continue
#         else:
#             cfg.nfc = min(cfg.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#             cfg.min_nfc = min(cfg.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
# 
#             cfg.out_ = functions.generate_dir2save(cfg)
#             cfg.outf = '%s/%d' % (cfg.out_,scale_num)
#             try:
#                 os.makedirs(cfg.outf)
#             except OSError:
#                     pass
# 
#             #plt.imsave('%s/in.png' %  (cfg.out_), functions.convert_image_np(real), vmin=0, vmax=1)
#             #plt.imsave('%s/original.png' %  (cfg.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
#             plt.imsave('%s/in_scale.png' %  (cfg.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
# 
#             D_curr,G_curr = init_models(cfg)
# 
#             z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],cfg,centers=centers)
# 
#             G_curr = functions.reset_grads(G_curr,False)
#             G_curr.eval()
#             D_curr = functions.reset_grads(D_curr,False)
#             D_curr.eval()
# 
#             Gs[scale_num] = G_curr
#             Zs[scale_num] = z_curr
#             NoiseAmp[scale_num] = cfg.noise_amp
# 
#             torch.save(Zs, '%s/Zs.pth' % (cfg.out_))
#             torch.save(Gs, '%s/Gs.pth' % (cfg.out_))
#             torch.save(reals, '%s/reals.pth' % (cfg.out_))
#             torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (cfg.out_))
# 
#             scale_num+=1
#             nfc_prev = cfg.nfc
#         del D_curr,G_curr
#     return
