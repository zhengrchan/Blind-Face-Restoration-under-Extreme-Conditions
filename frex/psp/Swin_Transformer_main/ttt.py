
import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor



def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='/data/yzy/3dgan/yzy_flow/psp/Swin_Transformer_main/configs/swin_yzy/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')


    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config







class SwinEncoder(nn.Module):
    def __init__(self, config, mlp_layer=2) -> None:
        super(SwinEncoder,self).__init__()

        self.swin_coarse = build_model(config)
        self.swin_fine = build_model(config)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 7))
        module_list.append(nn.LeakyReLU())            
        self.mapper_coarse_spatial = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))
        module_list.append(nn.LeakyReLU())            
        self.mapper_coarse_channel = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 7))
        module_list.append(nn.LeakyReLU())            
        self.mapper_fine_spatial = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))
        module_list.append(nn.LeakyReLU())            
        self.mapper_fine_channel = nn.Sequential(*module_list)

        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 3))
        module_list.append(nn.LeakyReLU())            
        self.mapper_eyes_spatial = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))
        module_list.append(nn.LeakyReLU())            
        self.mapper_eyes_channel = nn.Sequential(*module_list)


    def forward(self, x):
        B = x.shape[0]

        x_coarse = self.swin_coarse(x)

        ws_coarse = self.mapper_coarse_spatial(x_coarse)
        ws_coarse = self.mapper_coarse_channel(ws_coarse.transpose(1, 2))


        x_fine= self.swin_fine(x)

        ws_fine = self.mapper_fine_spatial(x_fine)
        ws_fine = self.mapper_fine_channel(ws_fine.transpose(1, 2))

        ws_eyes = self.mapper_eyes_spatial(x_coarse) 
        ws_eyes = self.mapper_eyes_channel(ws_eyes.transpose(1, 2))


        ws_14 = torch.cat([ws_coarse, ws_fine ], dim=1)

        zeros_1 = torch.zeros(B,4,512).to(x.device)
        zeros_2 = torch.zeros(B,7,512).to(x.device)

        ws_eyes = torch.cat([zeros_1, ws_eyes, zeros_2 ], dim=1)

        rec_ws = ws_14 + ws_eyes

        return rec_ws




def main(config):






    # model = build_model(config)
    model =SwinEncoder(config)
    # print(model)

    # a = torch.randn(2,3,224,224)
    a = torch.randn(2,3,256,256)
    b = model(a)
    print(b.shape)



if __name__ == '__main__':
    args, config = parse_option()
    main(config)