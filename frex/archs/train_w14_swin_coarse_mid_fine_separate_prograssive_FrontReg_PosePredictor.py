import traceback
import logging
import time
from email.policy import default
from logging import raiseExceptions
from random import random
import sys
import os
from typing import OrderedDict
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import pickle
import glob
import math
import re
import json
from datetime import datetime, timezone, timedelta
from torch import nn, autograd, optim
from torch.nn import functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import dnnlib
from camera_utils import LookAtPoseSampler

from typing import Optional, List
from torch import nn, Tensor

from training.triplane import TriPlaneGenerator 
import argparse

from psp.encoders.ranger import Ranger
from psp.encoders.psp_encoders import ProgressiveStage_14, LatentCodesPool, LatentCodesDiscriminator, Encoder4Editing_256_simple
from psp.Swin_Transformer_main.config import get_config
from psp.Swin_Transformer_main.models import build_model
from psp.Swin_Transformer_main.data import build_loader
from psp.Swin_Transformer_main.lr_scheduler import build_scheduler
from psp.Swin_Transformer_main.optimizer import build_optimizer
from psp.Swin_Transformer_main.logger import create_logger
from psp.Swin_Transformer_main.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def launch_training(desc, outdir):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_progressive_steps_14(opts):
    # log_size = int(math.log(opts.stylegan_size, 2))
    num_style_layers = 14
    num_deltas = num_style_layers - 1
    if opts.progressive_start is not None:  # If progressive delta training
        progressive_steps = [0]
        next_progressive_step = opts.progressive_start
        for j in range(num_deltas):
            progressive_steps.append(next_progressive_step)
            next_progressive_step += opts.progressive_step_every
        return progressive_steps




def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss




def get_dims_to_discriminate(net_ddp):
    # deltas_starting_dimensions = net_ddp.module.encoder.get_deltas_starting_dimensions()

    stage =  net_ddp.module.encoder.stage

    if stage == 0:
        return [0]
    elif stage == 1:
        return list(range(4))
    elif stage == 2:
        return list(range(7))
    else:
        return list(range(14))



def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def get_latent(net_ddp, n_latent, device):
    latent_in = torch.randn(n_latent, 512, device=device)

    get_latent_pitch_range = 0.15
    get_latent_yaw_range = 0.4
    get_latent_cam_pivot = torch.tensor([0, 0, 0.2], device=device)
    get_latent_cam_radius = 2.7
    get_latent_intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device=device)
    pitch = np.random.uniform(-get_latent_pitch_range, get_latent_pitch_range)
    yaw = np.random.uniform(-get_latent_yaw_range, get_latent_yaw_range)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, get_latent_cam_pivot, radius=get_latent_cam_radius, device=device)
    label = torch.cat([cam2world_pose.reshape(-1, 16), get_latent_intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
    # label = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1)
    c = label.repeat(n_latent, 1)

    latent = net_ddp.module.decoder.mapping(latent_in,c).mean(1, keepdim=True)

    return latent

def sample_real_and_fake_latents( x, net_ddp, device, latent_avg, real_w_pool, fake_w_pool, opts, is_progressive_training=True):
    batch_size = opts.batch // opts.num_gpus
    # sample_z = torch.randn(batch_size, 512, device=self.device)
    real_w = get_latent(net_ddp, batch_size, device)
    fake_w, _ = net_ddp.module.encoder(x)
    if opts.start_from_latent_avg:
        fake_w = fake_w + latent_avg.repeat(fake_w.shape[0], 14, 1)
    if is_progressive_training:  # When progressive training, feed only unique w's
        dims_to_discriminate = get_dims_to_discriminate(net_ddp)
        fake_w = fake_w[:, dims_to_discriminate, :]
    if opts.use_w_pool:
        real_w = real_w_pool.query(real_w)
        fake_w = fake_w_pool.query(fake_w)
    if fake_w.ndim == 3:
        fake_w = fake_w[:, 0, :]
    return real_w, fake_w

def discriminator_loss(real_pred, fake_pred, loss_dict):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()

    loss_dict['d_real_loss'] = real_loss
    loss_dict['d_fake_loss'] = fake_loss

    return real_loss + fake_loss

def train_discriminator(x, D_latent_ddp, D_optimizer, i, net_ddp, device, latent_avg, real_w_pool, fake_w_pool, opts):
    loss_dict = {}
    # x = x.to(self.device).float()
    requires_grad(D_latent_ddp.module, True)

    with torch.no_grad():
        real_w, fake_w = sample_real_and_fake_latents(x, net_ddp, device, latent_avg, real_w_pool, fake_w_pool, opts,)
    real_pred = D_latent_ddp(real_w)
    fake_pred = D_latent_ddp(fake_w)
    loss = discriminator_loss(real_pred, fake_pred, loss_dict)
    loss_dict['discriminator_loss'] = loss

    D_optimizer.zero_grad()
    loss.backward()
    D_optimizer.step()

        # r1 regularization
    d_regularize = i % opts.d_reg_every == 0
    if d_regularize:
        real_w = real_w.detach()
        real_w.requires_grad = True
        real_pred = D_latent_ddp(real_w)
        r1_loss = discriminator_r1_loss(real_pred, real_w)
        D_latent_ddp.module.zero_grad()
        r1_final_loss = opts.r1 / 2 * r1_loss * opts.d_reg_every + 0 * real_pred[0]

        r1_final_loss.backward()
        D_optimizer.step()
        loss_dict['discriminator_r1_loss'] = r1_final_loss

        # Reset to previous state
    requires_grad(D_latent_ddp.module, False)

    return loss_dict








class CrossAttention_without_norm3(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)  
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


## ---------------------------------------------------- Swin ------------------------------------------------------

class SwinEncoder(nn.Module):
    def __init__(self, config, mlp_layer=2, ws_dim = 14, stage_list=[20000, 40000, 60000]) -> None:
        super(SwinEncoder, self).__init__()
        self.style_count = ws_dim       # 14
        self.stage_list = stage_list
        self.stage_dict =   {'base':0, 'coarse':1, 'mid':2, 'fine':3}        #   {0:'base', 1:'coarse', 2:'mid', 3:'fine'}
        self.stage = 3


## -------------------------------------------------- base w0 swin transformer -------------------------------------------
        self.swin_base = build_model(config)

        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 1))           
        self.mapper_base_spatial = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))          
        self.mapper_base_channel = nn.Sequential(*module_list)

        self.maxpool_base = nn.AdaptiveMaxPool1d(1)


## -------------------------------------------------- w Query mapper coarse mid fine  1024*64 -> (4-1)*512 3*512 7*512 -------------------------------------------
        self.maxpool_query = nn.AdaptiveMaxPool1d(1)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 3))           
        self.mapper_query_spatial_coarse = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))          
        self.mapper_query_channel_coarse = nn.Sequential(*module_list)




        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 3))           
        self.mapper_query_spatial_mid = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))          
        self.mapper_query_channel_mid = nn.Sequential(*module_list)



        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(64, 64))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(64, 7))           
        self.mapper_query_spatial_fine = nn.Sequential(*module_list)


        module_list = nn.ModuleList()
        for j in range(mlp_layer-1):
            module_list.append(nn.Linear(1024, 1024))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(1024, 512))          
        self.mapper_query_channel_fine = nn.Sequential(*module_list)





## -------------------------------------------------- w KQ coarse mid fine mapper to 512 -------------------------


        self.mapper_coarse_channel = nn.Sequential(nn.Linear(512,512), nn.LeakyReLU())      ##  B*256*512 -> B*256*512


        self.mapper_mid_channel = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU())          ##  B*1024*256 -> B*1024*256

        self.mapper_fine_channel = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(), nn.Linear(256,512), nn.LeakyReLU())    ##  B*4096*128 -> B*4096*512

        self.mapper_coarse_to_mid_spatial = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU(), nn.Linear(512,1024), nn.LeakyReLU())
        self.mapper_mid_to_fine_spatial = nn.Sequential(nn.Linear(1024,2048), nn.LeakyReLU(), nn.Linear(2048,4096), nn.LeakyReLU())


## -------------------------------------------------- w KQ coarse mid fine Cross Attention -------------------------

        self.cross_att_coarse = CrossAttention_without_norm3(512, 4, 1024, batch_first=True)
        self.cross_att_mid = CrossAttention_without_norm3(512, 4, 1024, batch_first=True)
        self.cross_att_fine = CrossAttention_without_norm3(512, 4, 1024, batch_first=True)


        self.progressive_stage = ProgressiveStage_14.Inference

            

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    # def set_progressive_stage(self, new_stage: ProgressiveStage_14):
    #     self.progressive_stage = new_stage
    #     print('Changed progressive stage to: ', new_stage)

    def set_stage(self, iter):
        if iter > self.stage_list[-1]:
            self.stage = 3
        else:
            for i, stage_iter in enumerate(self.stage_list):
                if iter<stage_iter:
                    break
            self.stage = i
        
        print(f"change training stage to {self.stage}")

        


    def forward(self, x):
        B = x.shape[0]
        # print(self.stage)

        x_base, x_query, x_coarse, x_mid, x_fine= self.swin_base(x)


## ----------------------  base 
        ws_base_max = self.maxpool_base(x_base).transpose(1, 2)
        ws_base_linear = self.mapper_base_spatial(x_base)
        ws_base = self.mapper_base_channel(ws_base_linear.transpose(1, 2)  + ws_base_max )

        ws_base = ws_base.repeat(1,14,1)

        if self.stage == self.stage_dict['base']:
            ws = ws_base
            return ws, ws_base, x_query

## ------------------------ query   coarse mid fine 

        ws_query_max = self.maxpool_query(x_query).transpose(1, 2)


        if self.stage >= self.stage_dict['coarse']:
            ws_query_linear_coarse = self.mapper_query_spatial_coarse(x_query)
            ws_query_coarse = self.mapper_query_channel_coarse(ws_query_linear_coarse.transpose(1, 2)  + ws_query_max )

            if self.stage >= self.stage_dict['mid']:
                ws_query_linear_mid = self.mapper_query_spatial_mid(x_query)
                ws_query_mid = self.mapper_query_channel_mid(ws_query_linear_mid.transpose(1, 2)  + ws_query_max )

                if self.stage >= self.stage_dict['fine']:
                    ws_query_linear_fine = self.mapper_query_spatial_fine(x_query)
                    ws_query_fine = self.mapper_query_channel_fine(ws_query_linear_fine.transpose(1, 2)  + ws_query_max )


## ------------------------- carse, mid, fine  mapper 
        if self.stage >= self.stage_dict['coarse']:
            kv_coarse = self.mapper_coarse_channel(x_coarse)
            if self.stage >= self.stage_dict['mid']:
                kv_mid = self.mapper_mid_channel(x_mid) + self.mapper_coarse_to_mid_spatial(kv_coarse.transpose(1, 2)).transpose(1, 2)
                if self.stage >= self.stage_dict['fine']:
                    kv_fine = self.mapper_fine_channel(x_fine) + self.mapper_mid_to_fine_spatial(kv_mid.transpose(1, 2)).transpose(1, 2)
        
        

## ------------------------- carse, mid, fine  Cross attention
        if self.stage >= self.stage_dict['coarse']:
            # print(self.stage, 'sssss')
            ws_coarse = self.cross_att_coarse(ws_query_coarse, kv_coarse )
            zero_1 = torch.zeros(B,1,512).to(ws_base.device)
            zero_2 = torch.zeros(B,10,512).to(ws_base.device)
            ws_delta = torch.cat([zero_1, ws_coarse, zero_2], dim=1)

            if self.stage >= self.stage_dict['mid']:
                # print(self.stage, 'sssss')
                ws_mid = self.cross_att_mid(ws_query_mid, kv_mid)
                zero_1 = torch.zeros(B,1,512).to(ws_base.device)
                zero_2 = torch.zeros(B,7,512).to(ws_base.device)
                ws_delta = torch.cat([zero_1, ws_coarse, ws_mid, zero_2], dim=1)


                if self.stage >= self.stage_dict['fine']:
                    # print(self.stage, 'sssss')
                    ws_fine = self.cross_att_fine(ws_query_fine, kv_fine)
        
                    zero = torch.zeros(B,1,512).to(ws_base.device)

                    ws_delta = torch.cat([zero, ws_coarse, ws_mid, ws_fine], dim=1)

        ws = ws_base + ws_delta
        return ws, ws_base, x_query



class PosePredictor(nn.Module):
    def __init__(self, ):
        super(PosePredictor, self).__init__()

        self.mapper_spatial = nn.Linear(64, 1)
        self.mapper_channel = nn.Linear(1024, 12)

    
    def forward(self, x):

        x = self.mapper_spatial(x)
        x = self.mapper_channel(x.transpose(1, 2))

        return x.squeeze(1)





##  ---------------------------------------------------  Set Model -------------------------------------------------------

class Net(nn.Module):
    def __init__(self, device, opts, config) -> None:
        super(Net, self).__init__()

        self.decoder = self.set_generator(device, opts)
        self.encoder = self.set_encoder_swin(device, opts, config)
        self.pose_pred = self.set_pose_pred(device, opts, config)

        self.start_from_latent_avg = opts.start_from_latent_avg
        if self.start_from_latent_avg:
            self.w_avg = self.decoder.backbone.mapping.w_avg[None, None, :].to(device)         # 1, 1, 512

        self.c_front = torch.tensor([1,0,0,0, 
                        0,-1,0,0,
                        0,0,-1,2.7, 
                        0,0,0,1, 
                        4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1)
        

        self.pose_fix = torch.tensor([
                        0,0,0,1, 
                        4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1)


    def set_generator(self, device, opts):
        with open(opts.G_init_args_kwargs, 'rb') as f:         # '/data/yzy/code/eg3d_encoder/pretrained/ffhqrebalanced512-128_G_init_args_kwargs.pkl'
            G_init_args_kwargs = pickle.load(f)
            # print(G_init_args_kwargs)
            G_init_args = G_init_args_kwargs['G_init_args']
            G_init_kwargs = G_init_args_kwargs['G_init_kwargs']
        G_init_kwargs['rendering_kwargs']['superresolution_module'] = 'training.superresolution.SuperresolutionHybrid8XDC'

        G = TriPlaneGenerator(*G_init_args, **G_init_kwargs).eval().requires_grad_(False).to(device)
        G.register_buffer('dataset_label_std', torch.randn((25),device=device))
        ### # G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
        G.neural_rendering_resolution = 128
        ckpt = torch.load(opts.G_ckpt, map_location=device)        # '/data/yzy/code/eg3d_encoder/pretrained/ffhqrebalanced512-128_generator.pt'
        G.load_state_dict(ckpt)
        del ckpt
        return G


    def set_encoder_swin(self, device, opts, config):
        E_swin = SwinEncoder(config, mlp_layer=opts.mlp_layer, stage_list=[10000, 20000, 30000]).to(device)

        print(f" mlp_layer = {opts.mlp_layer}")
        print(f" mlp_layer = {opts.mlp_layer}")
        print(f" mlp_layer = {opts.mlp_layer}")


        if opts.E_ckpt:
            E_swin_ckpt = torch.load(opts.E_ckpt, map_location=device)      # '/data/yzy/code/eg3d_encoder/pretrained/3D_pSp_ws_17_000330.pt'
            E_swin.load_state_dict(E_swin_ckpt)
            del E_swin_ckpt
        
        num_params = sum(param.numel() for param in E_swin.parameters())
        print("Encoder parmeters number is :    ", num_params)

        return E_swin


    def set_pose_pred(self, device, opts, P_ckpt=None):
        PosePred =  PosePredictor().to(device)
        if opts.P_ckpt:
            P_psp_ckpt = torch.load(opts.P_ckpt, map_location=device)      
            PosePred.load_state_dict(P_psp_ckpt)
            del P_psp_ckpt
        return PosePred


    def forward(self, x, label, use_pred_pose=True, front_depth=0, return_latents=False):  #, return_latents_offset=False):
        B = x.shape[0]
        rec_ws, _, x_query = self.encoder(x)
        # rec_ws = rec_ws.repeat(1,14,1)

        if self.start_from_latent_avg:
            # rec_ws_base = rec_ws_base + self.w_avg.repeat(B, 1, 1) 
            rec_ws = rec_ws + self.w_avg.repeat(B, 1, 1)

        
        pred_pose = self.pose_pred(x_query)
        pred_pose = torch.cat([pred_pose, self.pose_fix.repeat(B,1)], dim=1)
        
        if use_pred_pose:
            label = pred_pose


        rec_img_dict = self.decoder.synthesis(ws=rec_ws, c=label,  noise_mode='const')


        if front_depth > 0.0:
            rec_img_dict_front = self.decoder.synthesis(ws=rec_ws, c=self.c_front.repeat(B,1), noise_mode='const')

            rec_img_front = rec_img_dict_front['image']
            rec_img_depth_front = rec_img_dict_front['image_depth']

            rec_img_dict['image_front'] = rec_img_front
            rec_img_dict['image_depth_front'] = rec_img_depth_front


        rec_img_dict["pred_pose"] = pred_pose

        if return_latents:
            rec_img_dict['rec_ws'] = rec_ws
        
        return rec_img_dict
        

        











