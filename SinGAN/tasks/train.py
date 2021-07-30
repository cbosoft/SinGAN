from SinGAN.training import train
from SinGAN.util import read_image, adjust_scales_to_image


def run_task_train(cfg):
    # Read training image
    real = read_image(cfg.training.image, cfg)
    adjust_scales_to_image(real, cfg)

    # Data store for training params/weights
    data = Gs, Zs, reals, noise_amp = [], [], [], []

    # Train!
    train(*data, cfg)
    print(cfg.training.date)

    # Generate? Check if the trained model works?
    # SinGAN_generate(*data, cfg)
