# test load model pkl
import math
import random
import torch
import sys
import os
from torch import nn
from torch.nn import functional as F

# from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU,
#                                           StyleGAN2Generator)
# from basicsr.ops.fused_act import FusedLeakyReLU
# from basicsr.utils.registry import ARCH_REGISTRY

# print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../'))

from training.EG3d_v16 import Generator
# from archs.eg3d_arch import Generator, EG3dDiscriminator
# from archs import torch_utils
from torch_utils import misc
# from archs import dnnlib
import dnnlib
from archs.legacy import load_network_pkl
import copy



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)

decoder_load_path = '/home/czr/EG3D-pytorch/network-snapshot-012800.pkl'

save_path = '/home/czr/FR3D/experiments/pretrained_models'
new_decoder_load_path = os.path.join(save_path, 'eg3d_g.pth')

G = Generator(
    z_dim= 512,
    w_dim= 512, 
    mapping_kwargs=dnnlib.EasyDict(num_layers=8),
    use_noise=False,
    nerf_decoder_kwargs=dnnlib.EasyDict(
        in_c=32,
        mid_c=64,
        out_c=32, 
    ),
    fused_modconv_default = 'inference_only',
    conv_clamp = None,
    c_dim=16, #12, 
    img_resolution=512,
    img_channels= 96,
    backbone_resolution=None,
    rank=0,
)

state_dict = torch.load(new_decoder_load_path)


print(type(state_dict))
# print(state_dict[0])
# print(state_dict['nerf_decoder.fc1.bias'])
print(type(state_dict.keys()))
# print(list(state_dict.keys()))
keys = list(state_dict.keys())
flag, i = 1, 0
for k in state_dict.keys():
    if k != keys[i]:
        flag = 0
    i += 1
print(flag)

# print(type(state_dict.items().keys()))
# for k, v in state_dict:
#     print(k, v)
#     break

print(state_dict['nerf_decoder.fc1.bias'])

# print(state_dict)
# print(state_dict.items())
# print(state_dict.keys())

# G.load_state_dict(
#     # torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])
#     # torch.load(new_decoder_load_path, map_location=lambda storage, loc: storage)['G_ema'])
#     torch.load(new_decoder_load_path))

print("Success Load Model")

# torch.save(G.state_dict(), os.path.join(save_path, 'eg3d_g.pth'))

# print("Success Save Model at ", save_path)





