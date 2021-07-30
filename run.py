import argparse

from SinGAN.config import cfg, finalise
from SinGAN.tasks import task_list, run_task

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='<CONFIG FILE>', help='config file path', type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    finalise(cfg)
    run_task(cfg)
