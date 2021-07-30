import os
import random
from datetime import datetime

import torch
from yacs.config import CfgNode

cfg = CfgNode()

# set to 'True' to disable GPU
cfg.cpu = False

# Manual seed for random gen. Set to 'None' to get a new seed every time (default) or to an integer for a constant seed.
cfg.seed = None

cfg.noise_channels = 3
cfg.image_channels = 3
cfg.output_dir = './Output'

# networks hyper params
cfg.nfc = 32
cfg.min_nfc = 32
cfg.ker_size = 3
cfg.num_layer = 5
cfg.stride = 1
cfg.padd_size = 0

# Feature Pyramid parameters
cfg.scale_factor = 0.75
cfg.noise_amp = 0.1
cfg.min_size = 25
cfg.max_size = 250

# Optimisation hyper parameters
cfg.niter = 2000 # training epochs per scale
cfg.gamma = 0.1 # scheduler gamma
cfg.beta1 = 0.5
cfg.lambda_grad = 0.1
cfg.alpha = 10

cfg.generator = CfgNode()
cfg.generator.lr = 5e-4
cfg.generator.steps = 3
cfg.generator.path = None # path to previous checkpoint, e.g. to continue training

cfg.discriminator = CfgNode()
cfg.discriminator.lr = 5e-4
cfg.discriminator.steps = 3
cfg.discriminator.path = None # path to previous checkpoint, e.g. to continue training

cfg.mode = None

# Image used to train the network
cfg.training = CfgNode()
cfg.training.image = None
cfg.training.model_path = None
cfg.training.date = None # set to date of model training when doing inference

# Harmonisation
cfg.harmonisation = CfgNode()
cfg.harmonisation.reference_image = None
cfg.harmonisation.background_image = None
cfg.harmonisation.start_scale = 1
cfg.harmonisation.dilation_radius = 7

def finalise(c):
    '''called on `cfg` after merging in user config file; sets fixed parameters'''
    assert c.mode is not None, 'mode must be set in config file (train, harmonisation).'
    assert c.training.image is not None, 'path training.image must be set in config file.'

    c.device = 'cpu' if c.cpu else 'cuda:0'
    c.niter_init = c.niter
    c.noise_amp_init = c.noise_amp
    c.nfc_init = c.nfc
    c.min_nfc_init = c.min_nfc
    c.scale_factor_init = c.scale_factor

    if c.mode == 'train' and c.training.date is None:
        c.training.date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        assert c.training.date, 'training date must be set in config file and must be a valid training date.'
    image_name = os.path.basename(c.training.image)
    image_noext = image_name[:image_name.rfind('.')]
    c.run_key = f'{image_noext}_{c.training.date}'
    c.training.models_dir = f'TrainedModels/{c.run_key}/scalefactor={c.scale_factor}_alpha={c.alpha}'
    if c.mode == 'train':
        # c.training.models_dir = f'TrainedModels/{c.run_key}/scale_factor={c.scale_factor}/'
        c.output_dir = c.training.models_dir # f'TrainedModels/{c.run_key}/scalefactor={c.scale_factor}_alpha={c.alpha}'
        if os.path.exists(c.output_dir):
            raise Exception(f'Model target path "{c.output_dir}" already exists. Will not overwrite.')
        print(f'Trained models will be saved to "{c.output_dir}"')
    elif c.mode == 'harmonisation':
        ref_name = os.path.basename(c.harmonisation.reference_image)
        ref_noext = ref_name[:ref_name.rfind('.')]
        bg_name = os.path.basename(c.harmonisation.background_image)
        bg_noext = bg_name[:bg_name.rfind('.')]
        c.output_dir = f'{c.output_dir}/{c.mode}/{c.run_key}/ref={ref_noext},bg={bg_noext}'
        print(f'Trained models will be loaded from "{c.training.models_dir}"')
        print(f'Harmonisation results will be saved to "{c.output_dir}"')
    os.makedirs(c.output_dir, exist_ok=True)

    if c.seed is None:
        c.seed = random.randint(1, 10000)
    print(f'Random seed: {c.seed}')
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    # c.freeze() # ideally I'd like to make this object read only, but it is altered later on. TODO.

    return c
