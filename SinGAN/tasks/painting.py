from matplotlib import pyplot as plt

from SinGAN.util.imresize import imresize, imresize_to_shape
import SinGAN.util as util
from SinGAN.generate import generate_image


def run_task_painting(cfg):
    real = util.read_image(cfg.training.image, cfg)
    real = util.adjust_scales_to_image(real, cfg)
    Gs, Zs, reals, NoiseAmp = util.load_trained_pyramid(cfg.training.models_dir)
    assert 1 <= cfg.painting.start_scale < len(Gs), f'start scale should be in [1, {len(Gs)}) '

    ref = util.read_image(cfg.painting.reference_image, cfg)
    if cfg.painting.invert_image:
        print('inverting reference')
        ref = 1. - ref
    if ref.shape[3] != real.shape[3]:
        ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], cfg)
        ref = ref[:, :, :real.shape[2], :real.shape[3]]

    N = len(reals) - 1
    n = cfg.painting.start_scale
    in_s = imresize(ref, pow(cfg.scale_factor, (N - n + 1)), cfg)
    in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
    in_s = imresize(in_s, 1 / cfg.scale_factor, cfg)
    in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
    # if opt.quantization_flag:
    #     opt.mode = 'paint_train'
    #     dir2trained_model = functions.generate_dir2save(opt)
    #     # N = len(reals) - 1
    #     # n = opt.paint_start_scale
    #     real_s = imresize(real, pow(opt.scale_factor, (N - n)), opt)
    #     real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
    #     real_quant, centers = functions.quant(real_s, opt.device)
    #     plt.imsave('%s/real_quant.png' % dir2save, functions.convert_image_np(real_quant), vmin=0, vmax=1)
    #     plt.imsave('%s/in_paint.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
    #     in_s = functions.quant2centers(ref, centers)
    #     in_s = imresize(in_s, pow(opt.scale_factor, (N - n)), opt)
    #     # in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
    #     # in_s = imresize(in_s, 1 / opt.scale_factor, opt)
    #     in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
    #     plt.imsave('%s/in_paint_quant.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
    #     if (os.path.exists(dir2trained_model)):
    #         # print('Trained model does not exist, training SinGAN for SR')
    #         Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    #         opt.mode = 'paint2image'
    #     else:
    #         train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
    #         opt.mode = 'paint2image'
    out = generate_image(Gs[n:], Zs[n:], reals, NoiseAmp[n:], cfg, in_s, n=n, num_samples=1)
    plt.imsave(f'{cfg.output_dir}/start_scale={cfg.painting.start_scale}.png',
               util.convert_image_np(out.detach()), vmin=0, vmax=1)
