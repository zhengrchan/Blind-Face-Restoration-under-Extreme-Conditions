import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import sys
sys.path.append("/diskE/yzy/code/eg3d_encoder/yzy_swin_psp/code/psp")
from psp.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from psp.stylegan2.model import EqualLinear, ScaledLeakyReLU, EqualConv2d
import torchvision.transforms as transforms
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor
# from psp.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
# from psp.stylegan2.model import EqualLinear
from enum import Enum

import random

class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18

class ProgressiveStage_14(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Inference = 14

class ProgressiveStage_17(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Inference = 17


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools-1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

# class GradualStyleBlock_512(Module):
#     def __init__(self, in_c, out_c, spatial):
#         super(GradualStyleBlock_256, self).__init__()
#         self.out_c = out_c
#         self.spatial = spatial
#         num_pools = int(np.log2(spatial))
#         modules = []
#         modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
#                     nn.LeakyReLU()]
#         for i in range(num_pools):
#             modules += [
#                 Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU()
#             ]
#         self.convs = nn.Sequential(*modules)
#         self.linear = EqualLinear(out_c, out_c, lr_mul=1)

#     def forward(self, x):
#         x = self.convs(x)
#         x = x.view(-1, self.out_c)
#         x = self.linear(x)
#         return x


class GradualStyleEncoder(Module):
    # 50, 'ir_se', self.opts
    #  G.img_resolution, G.mapping.num_ws, G.mapping.w_dim, input_dim=3
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_dim, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        # self.ws_avg = torch.zeros(1,18,512)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out



class Encoder4Editing(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x

class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)

class LatentCodesPool:
    """This class implements latent codes buffer that stores previously generated w latent codes.
    This buffer enables us to update discriminators using a history of generated w's
    rather than the ones produced by the latest encoder.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_ws = 0
            self.ws = []

    def query(self, ws):
        """Return w's from the pool.
        Parameters:
            ws: the latest generated w's from the generator
        Returns w's from the buffer.
        By 50/100, the buffer will return input w's.
        By 50/100, the buffer will return w's previously stored in the buffer,
        and insert the current w's to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return ws
        return_ws = []
        for w in ws:  # ws.shape: (batch, 512) or (batch, n_latent, 512)
            # w = torch.unsqueeze(image.data, 0)
            if w.ndim == 2:
                i = random.randint(0, len(w) - 1)  # apply a random latent index as a candidate
                w = w[i]
            self.handle_w(w, return_ws)
        return_ws = torch.stack(return_ws, 0)   # collect all the images and return
        return return_ws

    def handle_w(self, w, return_ws):
        if self.num_ws < self.pool_size:  # if the buffer is not full; keep inserting current codes to the buffer
            self.num_ws = self.num_ws + 1
            self.ws.append(w)
            return_ws.append(w)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored latent code, and insert the current code into the buffer
                random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = w
                return_ws.append(tmp)
            else:  # by another 50% chance, the buffer will return the current image
                return_ws.append(w)

                
class Encoder4Editing_withpose(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_withpose, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        pose = [EqualLinear(512, 256), EqualLinear(256, 25)]
        self.pose = nn.Sequential(*pose)

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        pose = self.pose(w0)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w, pose



class att_module(nn.Module):
    def __init__(self, latent_dim=512, if_pose=False):
        super(att_module, self).__init__()

        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)
        self.if_pose = if_pose
        if not if_pose:
            self.k_matrix = nn.Linear(512, 512, bias=False)
            self.q_matrix = nn.Linear(512, 512, bias=False)
            self.v_matrix = nn.Linear(512, 512, bias=False)
        else:
            self.k_matrix = nn.Linear(latent_dim*2, 512, bias=False)
            self.q_matrix = nn.Linear(512, 512, bias=False)
            self.v_matrix = nn.Linear(latent_dim*2, 512, bias=False)            



    def forward(self, w_2d, w_3d, c=None):
        if self.if_pose:
            w_2d = torch.cat([w_2d, c], dim = 2 )

        K = self.k_matrix(w_2d).transpose(-2, -1)
        V = self.v_matrix(w_2d)
        Q = self.q_matrix(w_3d)
        

        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        return V

class Encoder4Editing_withATT(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_withATT, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

        self.att_module = att_module(latent_dim=size, if_pose=True)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x, w_2d, c=None):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        w_att = self.att_module(w_2d, w, c)
        w_res = w + w_att

        return w_res

class Att_Mapping_Module(nn.Module):
    def __init__(self, latent_dim=512):
        super(Att_Mapping_Module, self).__init__()

        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)

        self.k_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU())
        self.q_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU())
        self.v_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU())
        self.pose_mapping =  nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU())        
    def forward(self, w_2d, w_3d, c):

        K = self.pose_mapping(c) 
        K = self.k_mapping(w_2d + K).transpose(-2, -1)
        V = self.v_mapping(w_2d)
        Q = self.q_mapping(w_3d)
        
        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        return V


class Encoder4Editing_withAttMapping(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_withAttMapping, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

        self.att_module = Att_Mapping_Module(latent_dim=size)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x, w_2d, c=None):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        w_att = self.att_module(w_2d, w, c)
        w_res = w + w_att

        return w_res

class Encoder4Editing_withAttMapping_256(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_withAttMapping_256, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

        self.att_module = Att_Mapping_Module(latent_dim=size)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x, w_2d, c=None):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        w_att = self.att_module(w_2d, w, c)
        w_res = w + w_att

        return w_res


class Encoder4Editing_256(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_14.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w


class Encoder4Editing_256_Ws17(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_Ws17, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w




class Att_Mapping_Module_14_To_18(nn.Module):
    def __init__(self, latent_dim=512, kqv_layers=2, c_layers=2, dim_layers=2):
        super(Att_Mapping_Module_14_To_18, self).__init__()
        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)

        k_modules = []
        for i in range(kqv_layers):
            k_modules.append(nn.Linear(latent_dim, latent_dim))
            k_modules.append(nn.LeakyReLU())
        self.k_mapping = Sequential(*k_modules)

        q_modules = []
        for i in range(kqv_layers):
            q_modules.append(nn.Linear(latent_dim, latent_dim))
            q_modules.append(nn.LeakyReLU())
        self.q_mapping = Sequential(*q_modules)

        v_modules = []
        for i in range(kqv_layers):
            v_modules.append(nn.Linear(latent_dim, latent_dim))
            v_modules.append(nn.LeakyReLU())
        self.v_mapping = Sequential(*v_modules)

        p_modules = []
        for i in range(c_layers):
            p_modules.append(nn.Linear(latent_dim, latent_dim))
            p_modules.append(nn.LeakyReLU())
        self.pose_mapping = Sequential(*p_modules)  

        dim_modules = []
        for i in range(dim_layers-1):
            dim_modules.append(nn.Linear(14, 14))
            dim_modules.append(nn.LeakyReLU())
        dim_modules.append(nn.Linear(14, 18))
        dim_modules.append(nn.LeakyReLU())
        self.mapping_14_To_18 = Sequential(*dim_modules) 
        # dim_modules = []
        # dim_modules.append(nn.Linear(14, 18))
        # dim_modules.append(nn.LeakyReLU())
        # for i in range(dim_layers-1):
        #     dim_modules.append(nn.Linear(18, 18))
        #     dim_modules.append(nn.LeakyReLU())
        # self.mapping_14_To_18 = Sequential(*dim_modules) 





        # self.k_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.q_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.v_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.pose_mapping =  nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())      

        # self.mapping_14_To_18 = nn.Sequential(nn.Linear(14, 14),
        #                                 nn.ReLU(),
        #                                 nn.Linear(14, 18),
        #                                 nn.ReLU()) 

    def forward(self, w_3d, c):
        C = self.pose_mapping(c) 
        K = self.k_mapping(w_3d + C).transpose(-2, -1)
        V = self.v_mapping(w_3d + C)
        Q = self.q_mapping(w_3d + C)
        
        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        V = self.mapping_14_To_18(V.transpose(-2, -1)).transpose(-2, -1)

        return V

class Att_Mapping_Module_18_To_14(nn.Module):
    def __init__(self, latent_dim=512, kqv_layers=2, c_layers=2, dim_layers=2):
        super(Att_Mapping_Module_18_To_14, self).__init__()

        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)

        k_modules = []
        for i in range(kqv_layers):
            k_modules.append(nn.Linear(latent_dim, latent_dim))
            k_modules.append(nn.LeakyReLU())
        self.k_mapping = Sequential(*k_modules)

        q_modules = []
        for i in range(kqv_layers):
            q_modules.append(nn.Linear(latent_dim, latent_dim))
            q_modules.append(nn.LeakyReLU())
        self.q_mapping = Sequential(*q_modules)

        v_modules = []
        for i in range(kqv_layers):
            v_modules.append(nn.Linear(latent_dim, latent_dim))
            v_modules.append(nn.LeakyReLU())
        self.v_mapping = Sequential(*v_modules)

        p_modules = []
        for i in range(c_layers):
            p_modules.append(nn.Linear(latent_dim, latent_dim))
            p_modules.append(nn.LeakyReLU())
        self.pose_mapping = Sequential(*p_modules)  

        dim_modules = []
        for i in range(dim_layers-1):
            dim_modules.append(nn.Linear(18, 18))
            dim_modules.append(nn.LeakyReLU())
        dim_modules.append(nn.Linear(18, 14))
        dim_modules.append(nn.LeakyReLU())
        self.mapping_18_To_14 = Sequential(*dim_modules) 
        # self.k_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.q_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.v_mapping = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU())
        # self.pose_mapping =  nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(latent_dim, latent_dim),
        #                                 nn.ReLU()) 

        # self.mapping_18_To_14 = nn.Sequential(nn.Linear(18, 18),
        #                                 nn.ReLU(),
        #                                 nn.Linear(18, 14),
        #                                 nn.ReLU()) 
   

    def forward(self, w_2d, c):
        C = self.pose_mapping(c) 
        K = self.k_mapping(w_2d + C).transpose(-2, -1)
        V = self.v_mapping(w_2d + C)
        # Q = self.q_mapping(w_2d[:,:14,:] + C[:,:14,:])
        Q = self.q_mapping(w_2d+ C)
        
        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        V = self.mapping_18_To_14(V.transpose(-2, -1)).transpose(-2, -1)

        return V

class Dual_Att_Mapping_Module(nn.Module):
    def __init__(self, latent_dim=512, 
                        kqv_layers_18_to_14=2, c_layers_18_to_14=2, dim_layers_18_to_14=2,
                        kqv_layers_14_to_18=2, c_layers_14_to_18=2, dim_layers_14_to_18=2,
                        ):
        super(Dual_Att_Mapping_Module, self).__init__()
        self.att_mapping_18_to_14 = Att_Mapping_Module_18_To_14(latent_dim, kqv_layers_18_to_14, c_layers_18_to_14, dim_layers_18_to_14)
        self.att_mapping_14_to_18 = Att_Mapping_Module_14_To_18(latent_dim, kqv_layers_14_to_18, c_layers_14_to_18, dim_layers_14_to_18)
    def forward(self, w_2d, c):
        rec_w_3d = self.att_mapping_18_to_14(w_2d, c)
        rec_w_2d = self.att_mapping_14_to_18(rec_w_3d, c[:,:14,:])

        return rec_w_2d, rec_w_3d

class Dual_Att_Mapping_Res_Module(nn.Module):
    def __init__(self, latent_dim=512, 
                        kqv_layers_18_to_14=2, c_layers_18_to_14=2, dim_layers_18_to_14=2,
                        kqv_layers_14_to_18=2, c_layers_14_to_18=2, dim_layers_14_to_18=2,
                        ):
        super(Dual_Att_Mapping_Res_Module, self).__init__()
        self.att_mapping_18_to_14 = Att_Mapping_Module_18_To_14(latent_dim, kqv_layers_18_to_14, c_layers_18_to_14, dim_layers_18_to_14)
        self.att_mapping_14_to_18 = Att_Mapping_Module_14_To_18(latent_dim, kqv_layers_14_to_18, c_layers_14_to_18, dim_layers_14_to_18)
    def forward(self, w_2d, c):
        rec_w_3d = self.att_mapping_18_to_14(w_2d, c)
        rec_w_2d = self.att_mapping_14_to_18(rec_w_3d, c[:,:14,:]) + w_2d


        return rec_w_2d, rec_w_3d

class Att_Mapping_Module_18_To_17(nn.Module):
    def __init__(self, latent_dim=512, kqv_layers=2, c_layers=2, dim_layers=2):
        super(Att_Mapping_Module_18_To_17, self).__init__()

        self.style_count = 17
        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)

        k_modules = []
        for i in range(kqv_layers):
            k_modules.append(nn.Linear(latent_dim, latent_dim))
            k_modules.append(nn.LeakyReLU())
            k_modules.append(nn.LayerNorm([512]))
        self.k_mapping = Sequential(*k_modules)

        q_modules = []
        for i in range(kqv_layers):
            q_modules.append(nn.Linear(latent_dim, latent_dim))
            q_modules.append(nn.LeakyReLU())
            q_modules.append(nn.LayerNorm([512]))
        self.q_mapping = Sequential(*q_modules)

        v_modules = []
        for i in range(kqv_layers):
            v_modules.append(nn.Linear(latent_dim, latent_dim))
            v_modules.append(nn.LeakyReLU())
            v_modules.append(nn.LayerNorm([512]))
        self.v_mapping = Sequential(*v_modules)


        self.pose_mapping_18 = Sequential(nn.Linear(1, 18), nn.LeakyReLU())
        p_modules = []
        for i in range(c_layers):
            p_modules.append(nn.Linear(latent_dim, latent_dim))
            p_modules.append(nn.LeakyReLU())
        self.pose_mapping_512 = Sequential(*p_modules)  




        dim_modules = []
        for i in range(dim_layers-1):
            dim_modules.append(nn.Linear(18, 18))
            dim_modules.append(nn.LeakyReLU())
        dim_modules.append(nn.Linear(18, 17))
        dim_modules.append(nn.LeakyReLU())
        self.mapping_18_To_17 = Sequential(*dim_modules) 

   
        self.progressive_stage = ProgressiveStage_17.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, w_2d, c):

        C = self.pose_mapping_18(c.transpose(-2, -1)).transpose(-2, -1)
        C = self.pose_mapping_512(C)
        # C = self.pose_mapping_512(c)
        

        K = self.k_mapping(w_2d + C).transpose(-2, -1)
        V = self.v_mapping(w_2d + C)
        # Q = self.q_mapping(w_2d[:,:14,:] + C[:,:14,:])
        Q = self.q_mapping(w_2d+ C)
        
        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        V = self.mapping_18_To_17(V.transpose(-2, -1)).transpose(-2, -1)

        return V



class Encoder4Editing_AttMapper_256(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se', 
                                    kqv_layers=2, c_layers=2, dim_layers=2, **unused):
        super(Encoder4Editing_AttMapper_256, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_14.Inference

        self.att_mapper = Att_Mapping_Module_18_To_14(latent_dim=size, kqv_layers=kqv_layers, c_layers=c_layers, dim_layers=dim_layers)



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x, ws_2d, c):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        w_delta = self.att_mapper(ws_2d, c)
        w = w + w_delta

        return w

class Att_Mapping_Module_18_To_17_return_C(nn.Module):
    def __init__(self, latent_dim=512, kqv_layers=2, c_layers=2, dim_layers=2):
        super(Att_Mapping_Module_18_To_17_return_C, self).__init__()

        self.style_count = 17
        self.latent_dim = latent_dim
        self.sqrt_dk = math.sqrt(self.latent_dim)

        k_modules = []
        for i in range(kqv_layers):
            k_modules.append(nn.Linear(latent_dim, latent_dim))
            k_modules.append(nn.LeakyReLU())
            k_modules.append(nn.LayerNorm([512]))
        self.k_mapping = Sequential(*k_modules)

        q_modules = []
        for i in range(kqv_layers):
            q_modules.append(nn.Linear(latent_dim, latent_dim))
            q_modules.append(nn.LeakyReLU())
            q_modules.append(nn.LayerNorm([512]))
        self.q_mapping = Sequential(*q_modules)

        v_modules = []
        for i in range(kqv_layers):
            v_modules.append(nn.Linear(latent_dim, latent_dim))
            v_modules.append(nn.LeakyReLU())
            v_modules.append(nn.LayerNorm([512]))
        self.v_mapping = Sequential(*v_modules)


        self.pose_mapping_18 = Sequential(nn.Linear(1, 18), nn.LeakyReLU())
        p_modules = []
        for i in range(c_layers):
            p_modules.append(nn.Linear(latent_dim, latent_dim))
            p_modules.append(nn.LeakyReLU())
        self.pose_mapping_512 = Sequential(*p_modules)  




        dim_modules = []
        for i in range(dim_layers-1):
            dim_modules.append(nn.Linear(18, 18))
            dim_modules.append(nn.LeakyReLU())
        dim_modules.append(nn.Linear(18, 17))
        dim_modules.append(nn.LeakyReLU())
        self.mapping_18_To_17 = Sequential(*dim_modules) 

   
        self.progressive_stage = ProgressiveStage_17.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, w_2d, c):

        C = self.pose_mapping_18(c.transpose(-2, -1)).transpose(-2, -1)
        C = self.pose_mapping_512(C)
        # C = self.pose_mapping_512(c)
        

        K = self.k_mapping(w_2d + C).transpose(-2, -1)
        V = self.v_mapping(w_2d + C)
        # Q = self.q_mapping(w_2d[:,:14,:] + C[:,:14,:])
        Q = self.q_mapping(w_2d+ C)
        
        score = torch.matmul(Q, K)  / self.sqrt_dk
        attention = F.softmax(score, dim=-1)  
        # print(V.shape)
        # print(attention.shape)
        V = torch.matmul(attention, V)
        # score * V 

        V = self.mapping_18_To_17(V.transpose(-2, -1)).transpose(-2, -1)

        return V, C




# Consultation encoder
class ResidualEncoder(Module):
    def __init__(self,  opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 =  Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])

        self.conv_layer3 =  Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.condition_scale3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))

        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))  



    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it



    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale, size=(64,64) , mode='bilinear')
        # scale = torch.nn.functional.interpolate(scale, size=(16,16) , mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift, size=(64,64) , mode='bilinear')
        # shift = torch.nn.functional.interpolate(shift, size=(16,16) , mode='bilinear')
        conditions.append(shift.clone())  
        return conditions


# ADA
class ResidualAligner(Module):
    def __init__(self,  opts=None):
        super(ResidualAligner, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(16),
                                      PReLU(16))

        self.conv_layer2 =  Sequential(*[bottleneck_IR(16,32,2), bottleneck_IR(32,32,1), bottleneck_IR(32,32,1)])
        self.conv_layer3 =  Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])
        self.conv_layer4 =  Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.dconv_layer1 =  Sequential(*[bottleneck_IR(112,64,1), bottleneck_IR(64,32,1), bottleneck_IR(32,32,1)])
        self.dconv_layer2 =  Sequential(*[bottleneck_IR(64,32,1), bottleneck_IR(32,16,1), bottleneck_IR(16,16,1)])
        self.dconv_layer3 =  Sequential(*[bottleneck_IR(32,16,1), bottleneck_IR(16,3,1), bottleneck_IR(3,3,1)])
 
    def forward(self, x):
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)
        feat4 = self.conv_layer4(feat3)

        feat4 = torch.nn.functional.interpolate(feat4, size=(64,64) , mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3),1))
        dfea1 = torch.nn.functional.interpolate(dfea1, size=(128,128) , mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat( (dfea1, feat2),1))
        dfea2 = torch.nn.functional.interpolate(dfea2, size=(256,256) , mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat( (dfea2, feat1),1))
 
        res_aligned = dfea3
 
        return res_aligned


class HFGI_Condition(Module):
    def __init__(self, device='cpu', distortion_scale=0.15, aug_rate=0.8):
        super(HFGI_Condition, self).__init__()

        self.grid_transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=aug_rate)
        self.grid_align = ResidualAligner() #ADA
        self.residue =  ResidualEncoder() #Ec
        self.device = device
    
    def forward(self, invert_img , real_img):
        res_gt = (invert_img - real_img ).detach() 
        res_unaligned = self.grid_transform(res_gt).detach() 
        res_aligned = self.grid_align(torch.cat((res_unaligned, invert_img ), 1))
        res = res_aligned.to(self.device)
        conditions = self.residue(res)

        return conditions






class Encoder4Editing_256_Ws17_AttTri(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_Ws17_AttTri, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w, c3


class Encoder4Editing_256_Ws17_AttTri_64(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_Ws17_AttTri_64, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w, p1





class Encoder4Editing_256_Ws17_return_p1(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_Ws17_return_p1, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        
        self.style_count = n_latents
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_256(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_256(512, 512, 32)
            else:
                style = GradualStyleBlock_256(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # print(w0.repeat(self.style_count, 1, 1).shape)
        # print(w0.repeat(self.style_count, 1, 1))
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i



        return w, p1



class Encoder4Editing_256_simple(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_simple, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        style = GradualStyleBlock_256(512, 512, 16)
        self.styles.append(style)
        # # log_size = int(math.log(opts.stylegan_size, 2))
        
        # self.style_count = n_latents
        # self.coarse_ind = 3
        # self.middle_ind = 7

        # for i in range(self.style_count):
        #     if i < self.coarse_ind:
        #         style = GradualStyleBlock_256(512, 512, 16)
        #     elif i < self.middle_ind:
        #         style = GradualStyleBlock_256(512, 512, 32)
        #     else:
        #         style = GradualStyleBlock_256(512, 512, 64)
        #     self.styles.append(style)

        # self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        # self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        # print(w0.shape)
        # print(w0)
        # w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # # print(w0.repeat(self.style_count, 1, 1).shape)
        # # print(w0.repeat(self.style_count, 1, 1))
        # stage = self.progressive_stage.value
        # features = c3
        # for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
        #     if i == self.coarse_ind:
        #         p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
        #         features = p2
        #     elif i == self.middle_ind:
        #         p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
        #         features = p1
        #     delta_i = self.styles[i](features)
        #     w[:, i] += delta_i

        return w0 # w



class Encoder4Editing_256_simple_2(Module):
    def __init__(self, size=512, n_latents=18, w_dim=512, add_dim=0, input_dim=3, num_layers=50, mode='ir_se',  **unused):
        super(Encoder4Editing_256_simple_2, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        style = GradualStyleBlock_256(512, 512, 16)
        self.styles.append(style)
        # # log_size = int(math.log(opts.stylegan_size, 2))
        
        # self.style_count = n_latents
        # self.coarse_ind = 3
        # self.middle_ind = 7

        # for i in range(self.style_count):
        #     if i < self.coarse_ind:
        #         style = GradualStyleBlock_256(512, 512, 16)
        #     elif i < self.middle_ind:
        #         style = GradualStyleBlock_256(512, 512, 32)
        #     else:
        #         style = GradualStyleBlock_256(512, 512, 64)
        #     self.styles.append(style)

        # self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        # self.progressive_stage = ProgressiveStage_17.Inference



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c128 = x
            elif i == 6:
                c64 = x
            elif i == 20:
                c32 = x
            elif i == 23:
                c16 = x

        # Infer main W and duplicate it



        return w0 # w



class CrossAttention(nn.Module):

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
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
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
        tgt = self.norm3(tgt)
        return tgt


class CrossAttention_2(nn.Module):

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



class Att_Triplane(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_Triplane, self).__init__()

        self.att_triplane_1 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
        self.att_triplane_2 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
        self.att_triplane_3 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)     



    def forward(self, triplane, c3_cond):
        B,_,M,_ = triplane.shape
        tri_1 =triplane[:,:32,:,:].view(B,32,-1)   # B 32 16 16 
        tri_2 =triplane[:,:32,:,:].view(B,32,-1)  # B 32 16 16 
        tri_3 =triplane[:,:32,:,:].view(B,32,-1)  # B 32 16 16 

        c3_cond = c3_cond.view(B,512,-1)
        tri_1_offset = self.att_triplane_1(tri_1, c3_cond)
        tri_2_offset = self.att_triplane_2(tri_2, c3_cond)
        tri_3_offset = self.att_triplane_3(tri_3, c3_cond) 

        tri_offset = torch.cat([tri_1_offset,tri_2_offset,tri_3_offset])  

        tri_offset = tri_offset.reshape(B,96,M,M)           # B 96 16 16 
        return tri_offset


class Att_Triplane_64(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_Triplane_64, self).__init__()

        self.att_triplane_1 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
        self.att_triplane_2 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
        self.att_triplane_3 = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)     



    def forward(self, triplane, c3_cond):
        B,_,M,_ = triplane.shape
        tri_1 =triplane[:,:32,:,:].view(B,32,-1)   # B 32 64 64 
        tri_2 =triplane[:,:32,:,:].view(B,32,-1)  # B 32 64 64  
        tri_3 =triplane[:,:32,:,:].view(B,32,-1)  # B 32 64 64  

        c3_cond = c3_cond.view(B,512,-1)
        tri_1_offset = self.att_triplane_1(tri_1, c3_cond)
        tri_2_offset = self.att_triplane_2(tri_2, c3_cond)
        tri_3_offset = self.att_triplane_3(tri_3, c3_cond) 

        tri_offset = torch.cat([tri_1_offset,tri_2_offset,tri_3_offset])  

        tri_offset = tri_offset.reshape(B,96,M,M)           # B 96 64 64  
        return tri_offset


class Att_FeatureMap_p1_32_mlp(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, mlp_layer=2, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_FeatureMap_p1_32_mlp, self).__init__()

        self.norm_p1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_x = nn.LayerNorm(d_model, elementwise_affine=False)

        module_list = nn.ModuleList()
        for i in range(mlp_layer):
            module_list.append(nn.Linear(d_model,d_model))
            module_list.append(nn.LeakyReLU())

        self.mapping = nn.Sequential(*module_list)



        self.att_feature_map = CrossAttention_2(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
 



    def forward(self, feature_map, p1_cond):
        B,L,M,_ = feature_map.shape
        
        p1_cond = p1_cond.flatten(2).permute(0,2,1)
        feature_map = feature_map.flatten(2).permute(0,2,1)
        feature_map = self.mapping(feature_map)

        p1_cond = self.norm_p1(p1_cond)
        feature_map = self.norm_x(feature_map)
        
        feature_map_offset = self.att_feature_map(feature_map, p1_cond)

        feature_map_offset = feature_map_offset.permute(0,2,1).reshape(B,L,M,M)           # B 96 64 64  
        return feature_map_offset



class Att_FeatureMap_p1_32(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_FeatureMap_p1_32, self).__init__()

        self.att_feature_map = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
 



    def forward(self, feature_map, p1_cond):
        B,L,M,_ = feature_map.shape
        p1_cond = p1_cond.flatten(2).permute(0,2,1)
        feature_map = feature_map.flatten(2).permute(0,2,1)
        
        feature_map_offset = self.att_feature_map(feature_map, p1_cond)

        feature_map_offset = feature_map_offset.permute(0,2,1).reshape(B,L,M,M)           # B 96 64 64  
        return feature_map_offset






class Att_FeatureMap_p1_64(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_FeatureMap_p1_64, self).__init__()

        self.att_feature_map = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
 



    def forward(self, feature_map, p1_cond):
        B,L,M,_ = feature_map.shape
        feature_map = feature_map.flatten(2).permute(0,2,1)
        p1_cond = p1_cond.flatten(2).permute(0,2,1)
        feature_map_offset = self.att_feature_map(feature_map, p1_cond)

        feature_map_offset = feature_map_offset.permute(0,2,1).reshape(B,L,M,M)           # B 96 64 64  
        return feature_map_offset




class Att_FeatureMap_p1_128(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_FeatureMap_p1_128, self).__init__()

        self.att_feature_map = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
 

        self.mapping = nn.Sequential(nn.Linear(512,256), nn.ReLU())


    def forward(self, feature_map, p1_cond):
        B,L,M,_ = feature_map.shape
        p1_cond = p1_cond.flatten(2).permute(0,2,1)
        p1_cond = self.mapping(p1_cond)
        feature_map = feature_map.flatten(2).permute(0,2,1)
        
        feature_map_offset = self.att_feature_map(feature_map, p1_cond)

        feature_map_offset = feature_map_offset.permute(0,2,1).reshape(B,L,M,M)           # B 96 64 64  
        return feature_map_offset



class Att_FeatureMap_ws(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super(Att_FeatureMap_ws, self).__init__()

        self.att_feature_map = CrossAttention(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, batch_first=batch_first)
 



    def forward(self, feature_map, ws):
        B,L,M,_ = feature_map.shape
        feature_map = feature_map.flatten(2).permute(0,2,1)
        feature_map_offset = self.att_feature_map(feature_map, ws)

        feature_map_offset = feature_map_offset.permute(0,2,1).reshape(B,L,M,M)           # B 96 64 64  
        return feature_map_offset









class Official_PSP(Module):
    def __init__(self, num_layers=50, mode='ir', input_nc=3, n_styles=14):
        super(Official_PSP, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out








class Official_e4e(Module):
    def __init__(self, num_layers, mode='ir', style_count=14):
        super(Official_e4e, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = style_count
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage_14.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage_14):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
        Returns:
        (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y 


    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w








if __name__ == "__main__":
    model = Encoder4Editing(size=512, n_latents=18)
    # import sys
    # sys.path.append("/diskE/yzy/3dgan/3dgan-inversion/eg3d")
    try:
        model.load_state_dict(torch.load("/diskE/yzy/3dgan/yzy_psp_encoder/pretrained/network-snapshot-000240.pkl"))
    except:
        a = torch.load("/diskE/yzy/3dgan/yzy_psp_encoder/pretrained/network-snapshot-000240.pkl")
        for key, value in a.items():
            print(key)
    # print(model)
    # model = HybridEncoder(size=512, n_latents_app=10, n_latents_geo=8, w_dim=512, add_dim=0, input_img_dim=3, input_seg_dim=19)
    img = torch.randn(2, 3, 512, 512)
    # seg = torch.randn(2, 19, 512, 512)
    # result = model(img, seg)
    # img = torch.randn(2, 3, 1024, 1024)
    result = model(img)    
    print(result.shape)
    print("hhh")