import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .resnet import Resnet
from .patchgan import Patchgan

def init_weight(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # use kaiming initialization
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, init_gain = 0.02)
        init.constant_(m.bias.data, 0.0)

def Generator(opt, inc, outc):
    # instance norm
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = Resnet(inc, outc, opt.ngf, norm_layer=norm_layer, use_dropout=1 - opt.no_dropout, n_blocks=9)
    net.apply(init_weight)
    net.to(opt.gpu_ids[0])
    return net

def Discriminator(opt, inc):
    # instance norm
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

    net = Patchgan(inc, opt.ndf, n_layers=opt.n_layers_D, norm_layer=norm_layer)
    net.apply(init_weight)
    net.to(opt.gpu_ids[0])
    return net

  