import argparse

from matplotlib import pyplot as plt

from config import cfg, finalise

from SinGAN.generate import generate_image
from SinGAN.util.imresize import imresize, imresize_to_shape
import SinGAN.util as util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config path', required=True)
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg = finalise(cfg)

    assert cfg.mode == 'harmonisation'

    real = util.read_image(cfg.harmonisation.background_image, cfg)
    real = util.adjust_scales_to_image(real, cfg)

    Gs, Zs, reals, NoiseAmp = util.load_trained_pyramid(cfg.training.models_dir)
    assert 1 <= cfg.harmonisation.start_scale < len(Gs), f'start scale should be in [1 and {len(Gs)}) '

    ref_name = cfg.harmonisation.reference_image
    ref = util.read_image(ref_name, cfg)
    mask_name = ref_name[:ref_name.rfind('.')] + '_mask' + ref_name[ref_name.rfind('.'):]
    mask = util.read_image(mask_name, cfg)
    if ref.shape[3] != real.shape[3]:
        mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], cfg)
        mask = mask[:, :, :real.shape[2], :real.shape[3]]
        ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], cfg)
        ref = ref[:, :, :real.shape[2], :real.shape[3]]
    mask = util.dilate_mask(mask, cfg)

    N = len(reals) - 1
    n = cfg.harmonisation.start_scale
    in_s = imresize(ref, pow(cfg.scale_factor, (N - n + 1)), cfg)
    in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
    in_s = imresize(in_s, 1 / cfg.scale_factor, cfg)
    in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
    out = generate_image(Gs[n:], Zs[n:], reals, NoiseAmp[n:], cfg, in_s, n=n, num_samples=cfg.harmonisation.num_samples)

    out = (1 - mask)*real + mask*out
    out = util.convert_image_np(out.detach())
    plt.imsave(f'{cfg.output_dir}/start_scale={cfg.harmonisation.start_scale}.png', out, vmin=0, vmax=1)
    plt.imsave('foo.png', out, vmin=0, vmax=1)




