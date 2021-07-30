import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super().__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class Discriminator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        N = cfg.nfc
        self.head = ConvBlock(cfg.image_channels, N, cfg.ker_size, cfg.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(cfg.num_layer - 2):
            N //= 2
            block = ConvBlock(max(2*N, cfg.min_nfc), max(N, cfg.min_nfc), cfg.ker_size, cfg.padd_size, 1)
            self.body.add_module(f'block{i+1}', block)
        self.tail = nn.Conv2d(max(N, cfg.min_nfc), 1, kernel_size=cfg.ker_size, stride=1, padding=cfg.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Generator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        N = cfg.nfc
        self.head = ConvBlock(cfg.image_channels, N, cfg.ker_size, cfg.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(cfg.num_layer - 2):
            N //= 2
            block = ConvBlock(max(2*N, cfg.min_nfc), max(N, cfg.min_nfc), cfg.ker_size, cfg.padd_size, 1)
            self.body.add_module(f'block{i+1}', block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, cfg.min_nfc), cfg.image_channels, kernel_size=cfg.ker_size, stride=1, padding=cfg.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init(cfg):

    # generator initialization
    netG = Generator(cfg).to(cfg.device)
    netG.apply(weights_init)
    if cfg.generator.path:
        netG.load_state_dict(torch.load(cfg.generator.path))

    # discriminator initialization
    netD = Discriminator(cfg).to(cfg.device)
    netD.apply(weights_init)
    if cfg.discriminator.path:
        netD.load_state_dict(torch.load(cfg.discriminator.path))

    return netD, netG

