import argparse

from config import cfg, finalise
from SinGAN.training import train
from SinGAN.util import read_image, adjust_scales_to_image


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path', type=str, required=True)
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    finalise(cfg)

    # Read training image
    real = read_image(cfg.training.image, cfg)
    adjusted = adjust_scales_to_image(real, cfg)

    # Data store for training params/weights?
    data = Gs, Zs, reals, noise_amp = [], [], [], []

    # Train!
    train(*data, cfg)
    print(cfg.training.date)

    # Generate? Check if the trained model works?
    # SinGAN_generate(*data, cfg)
