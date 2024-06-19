

import argparse
import math
import random
import copy
from wsgiref.simple_server import WSGIServer
import torch
import dnnlib
from torch import nn
from torch.nn import functional as F
import torchvision

import yaml
import yacs.config as cfg
from yacs.config import CfgNode as CfgNode

from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ModulatedConv2d, StyleConv, ResBlock, ScaledLeakyReLU,
                                          StyleGAN2Generator, StyleConv, ToRGB, StyleGAN2Discriminator)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY

from archs.train_w14_swin_coarse_mid_fine_separate_prograssive_FrontReg_PosePredictor \
    import Net as eg3d_inverter


@ARCH_REGISTRY.register()
class FR3D_Inv_v12_Arch(nn.Module):

    def __init__(
            self,
            out_size,
            sr_in_size=128,
            sr_in_channels=32,
            num_style_feat=512,
            num_c=25,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            encoder_3d_opt_path=None,
            encoder_3d_config_path=None,
            # eg3d_decoder_load_path=None,
            sr_decoder_load_path=None,
            encoder_load_path=None,
            # decoder_load_path=None,
            final_conv_load_path=None,
            final_linear_load_path=None,
            angle_linear_load_path=None,
            fix_eg3d_decoder=True,
            fix_sr_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False
    ):

        super(FR3D_Inv_v12_Arch, self).__init__()

        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat
        self.num_c = num_c
        self.sr_in_size = sr_in_size
        self.sr_in_channels = sr_in_channels

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        # channels_sft = {
        #     '4': int(512 * unet_narrow),
        #     '8': int(512 * unet_narrow),
        #     '16': int(256 * unet_narrow),
        #     '32': int(128 * unet_narrow),
        #     '64': int(64 * channel_multiplier * unet_narrow),
        #     '128': int(32 * channel_multiplier * unet_narrow),
        #     '256': int(16 * channel_multiplier * unet_narrow),
        #     '512': int(8 * channel_multiplier * unet_narrow)
        # }

        self.log_size = int(math.log(out_size, 2))
        self.log_sr_in_size = int(math.log(sr_in_size, 2))

        first_out_size = out_size
        self.conv_body_first = ConvLayer(
            3, channels[f'{first_out_size}'], 1, bias=True, activate=True)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(
                ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels

        self.final_conv = ConvLayer(
            in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(EqualConv2d(
                channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True, bias_init_val=0))

        # linear_out_channel = self.num_style_feat + self.num_angle

        # ws_3d, camera_params, ws_sr
        if different_w:
            sr_linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            sr_linear_out_channel = num_style_feat
            
        self.final_linear = EqualLinear(
            channels['4'] * 4 * 4, self.num_style_feat, bias=True, bias_init_val=0, lr_mul=1, activation=None)
        self.angle_linear = EqualLinear(
            channels['4'] * 4 * 4, self.num_c, bias=True, bias_init_val=0, lr_mul=1, activation=None)
        self.sr_linear = EqualLinear(
            channels['4'] * 4 * 4, sr_linear_out_channel, bias=True, bias_init_val=0, lr_mul=1, activation=None)

        # 3d feature encoder
        channels_3d = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(32),
        }

        self.feat_3d_encoder = nn.ModuleList()
        in_channels = channels_3d[f'{self.sr_in_size}']
        for i in range(self.log_sr_in_size, 2, -1): 
            out_channels = channels_3d[f'{2 ** (i - 1)}']
            self.feat_3d_encoder.append(
                ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels

        # camera_params embeddings
        self.embedding_camera_params = Embedding(self.num_c, 4)
        camera_params_emb_dim = 9 * 25

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        self.fuse_feat = nn.ModuleList()
        for i in range(3, self.log_sr_in_size):
            # in_channels = channels[f'{2**i}']
            out_channels = channels[f'{2**i}'] * 2
            if sft_half:
                sft_out_channels = int(out_channels / 2)
            else:
                raise('Only support sft_half(cs-sft) now')
            
            self.fuse_feat.append(StyleConv(out_channels, out_channels, 3, num_style_feat=camera_params_emb_dim))
            self.fuse_feat.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))

            # self.condition_scale.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=self.num_c))
            self.condition_scale.append(StyleConv(sft_out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))
            
            # self.condition_shift.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=self.num_c))
            self.condition_shift.append(StyleConv(sft_out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))


        for i in range(self.log_sr_in_size, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                raise('Only support sft_half(cs-sft) now')

            self.fuse_feat.append(StyleConv(out_channels, out_channels, 3, num_style_feat=camera_params_emb_dim))
            self.fuse_feat.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))

            # self.condition_scale.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=self.num_c))
            self.condition_scale.append(StyleConv(sft_out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))
            
            # self.condition_shift.append(StyleConv(out_channels, sft_out_channels, 3, num_style_feat=self.num_c))
            self.condition_shift.append(StyleConv(sft_out_channels, sft_out_channels, 3, num_style_feat=camera_params_emb_dim))

        out_size = 2 ** self.log_size

        # 3D encoder + eg3d
        with open(encoder_3d_config_path, "r", encoding="utf-8") as f:
            eg3d_config = yaml.safe_load(f)
        eg3d_config = cfg.CfgNode(eg3d_config)

        # eg3d_config = CfgNode()
        # eg3d_config.merge_from_file(encoder_3d_config_path)

        with open(encoder_3d_opt_path, "r", encoding="utf-8") as f:
            eg3d_opt = yaml.load(f, Loader=yaml.FullLoader)
        # parser = argparse.ArgumentParser()
        # for key, value in eg3d_opt.items():
        #     parser.add_argument('--{}'.format(key), type=type(value), default=value)
        # eg3d_opt = parser.parse_args()
        eg3d_opt = cfg.CfgNode(eg3d_opt)
        # del parser

        self.eg3d_inverter = eg3d_inverter(torch.device('cuda'), eg3d_opt, eg3d_config)

        # stylegan super resulution decoder
        self.sr_decoder = StyleganSR(
            out_size=out_size,
            num_style_feat=num_style_feat,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary
        # if eg3d_decoder_load_path:
        #     self.eg3d_decoder.load_state_dict(
        #         torch.load(eg3d_decoder_load_path))

        if sr_decoder_load_path:
            sr_decoder_dict = self.sr_decoder.state_dict()
            stylegan_dict = torch.load(sr_decoder_load_path, map_location=lambda storage, loc: storage)['params_ema']
            sr_decoder_dict.update({k: v for k, v in stylegan_dict.items()})
            self.sr_decoder.load_state_dict(sr_decoder_dict)
            del sr_decoder_dict, stylegan_dict
        
        if encoder_load_path:
            self.conv_body_down.load_state_dict(torch.load(encoder_load_path))
            self.final_conv.load_state_dict(torch.load(final_conv_load_path))
            self.final_linear.load_state_dict(torch.load(final_linear_load_path))
            self.angle_linear.load_state_dict(torch.load(angle_linear_load_path))

        # fix decoder without updating params
        if fix_eg3d_decoder:
            for _, param in self.eg3d_inverter.named_parameters():
                param.requires_grad = False
        if fix_sr_decoder:
            for _, param in self.sr_decoder.named_parameters():
                if param not in self.sr_decoder.fusion_3d:
                    param.requires_grad = False

        # torch.set_grad_enabled(True)

    def realign(self, feat, img, param, x=None):
        if param == None:
            return feat, img
        else:
            # print('Realigned')
            h, w, left, up, right, below = param['h'], param['w'], param['left'], param['up'], param['right'], param['below']
            if isinstance(h, int):
                h, w, left, up, right, below = [h], [w], [left], [up], [right], [below]
            # center_crop_size, target_size = param['center_crop_size'], param['target_size']
            realigned = F.interpolate(feat, size=(700, 700), mode='bicubic')
            # realigned = torchvision.transforms.functional.crop(realigned, -162, -162, 1024, 1024)
            # pad = nn.ReplicationPad2d(162)
            # realigned = pad(realigned)
            realigned = F.pad(realigned, (162, 162, 162, 162), 'reflect')
            cropped = []
            for ii, realigned_feat in enumerate(realigned):
                # if x != None:
                #     bg = F.interpolate(x.clone(), size=(h[ii], w[ii]), mode='bicubic')

                # print(realigned_feat.size())
                # print(-up[ii], -left[ii], h[ii], w[ii])
                realigned_feat = torchvision.transforms.functional.crop(realigned[ii:ii+1], -up[ii], -left[ii], h[ii], w[ii])
                # print(realigned_feat.size())
                # print('-------')
                realigned_feat = F.interpolate(realigned_feat, size=(128, 128), mode='bicubic')
                # print(realigned_feat.size())
                cropped.append(realigned_feat)
            realigned = torch.cat(cropped, dim=0)
            # print(realigned.size())
            return realigned, realigned[:, :3]

    def forward(self, x, x_256, return_rgb=True, randomize_noise=True, 
                return_generate_rows=False, 
                use_pred_pose=True, 
                camera_label=None,
                crop_param=None):
        """Forward function for 

        Args:
            x (Tensor): Input images.
        Return:
            image: B * C * H * W
        """
        conditions = []
        unet_skips = []
        out_rgbs = []
        feat_3d_skip = []

        # encoder
        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat)

        # style code
        # style_code = self.final_linear(feat.view(feat.size(0), -1))
        style_code = None
        camera_params = self.angle_linear(feat.view(feat.size(0), -1))
        style_code_sr = self.sr_linear(feat.view(feat.size(0), -1))

        pose_embeddings = self.embedding_camera_params(camera_params)

        # eg3d decodera
        output = self.eg3d_inverter(x_256, label=camera_label, use_pred_pose=use_pred_pose,)
        # lr_image: b * 3 * 128 * 128
        # feat: b * 32 * 128 * 128
        # conditions: 14list

        feat_3d, lr_image = self.realign(output['feature'], output['lr_image'], crop_param)
        
        # feat 3d decoder
        for i in range(self.log_sr_in_size - 3):
            feat_3d = self.feat_3d_encoder[i](feat_3d)
            feat_3d_skip.insert(0, feat_3d.clone())
        
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            
            # generate scale and shift for SFT layers
            # 3D feat guide 8 ~ 64 resolution
            if i < self.log_sr_in_size - 3:
                feat_concat = torch.cat((feat, feat_3d_skip[i]), 1)
            else:
                feat_concat = feat

            feat_concat = self.fuse_feat[i * 2](feat_concat, pose_embeddings)
            feat_concat = self.fuse_feat[i * 2 + 1](feat_concat, pose_embeddings)

            scale = self.condition_scale[i](feat_concat, pose_embeddings)
            # scale = self.condition_scale[i * 2 + 1](scale, camera_params)
            conditions.append(scale.clone())

            shift = self.condition_shift[i](feat_concat, pose_embeddings)
            # shift = self.condition_shift[i * 2 + 1](shift, camera_params)
            conditions.append(shift.clone())

                # feat_concat = self.fuse_feat[i * 2](feat, camera_params)
                # feat_concat = self.fuse_feat[i * 2 + 1](feat_concat, camera_params)

                # scale = self.condition_scale[i](feat, camera_params)
                # # scale = self.condition_scale[i * 2 + 1](scale, camera_params)
                # conditions.append(scale.clone())

                # shift = self.condition_shift[i](feat, camera_params)
                # # shift = self.condition_shift[i * 2 + 1](shift, camera_params)
                # conditions.append(shift.clone())
                
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat_concat))

            feat = feat_concat
            

        # decoder
        # image, _ = self.stylegan_decoder([style_code],
        #                                  conditions,
        #                                  return_latents=return_latents,
        #                                  input_is_latent=self.input_is_latent,
        #                                  randomize_noise=randomize_noise)

        # grdi_size, grid_c = self.setup_snapshot_image_grid()

        # feat_3d, lr_image = self.eg3d_decoder(
        #     z=style_code,
        #     angles=angle,
        #     cond=torch.zeros_like(angle),
        #     nerf_init_args={'img_size': 64},
        #     noise_mode='const',
        #     )


        if self.different_w:
            style_code_sr = style_code_sr.view(
                style_code_sr.size(0), -1, self.num_style_feat)

        image, generate_rows = self.sr_decoder([style_code_sr],
                                    conditions,
                                    feat_3d_skip,
                                    input_is_latent=self.input_is_latent,
                                    randomize_noise=randomize_noise,
                                    return_generate_rows=return_generate_rows)

        return {
            'image': image,
            'out_rgbs':out_rgbs, 
            'ws': style_code, 
            'c': camera_params,
            'lr_image': lr_image,
            'generate_rows': generate_rows if return_generate_rows else None,
        }
        # return image, out_rgbs, lr_image, style_code, camera_params

    def setup_snapshot_image_grid(num=1, random_seed=0):
        # rnd = np.random.RandomState(random_seed)
        gw = 5  # np.clip(7680 // training_set.image_shape[2], 7, 32)
        gh = num  # np.clip(4320 // training_set.image_shape[1], 4, 32)

        yaws = torch.linspace(-30, 30, gw)
        # yaws = torch.linspace(-1, 1, gw)
        conds = []
        for idx in range(gw):
            x = z = 0
            y = yaws[idx]
            angles = torch.tensor([x, y, z]).reshape(
                1, -1).expand(gh, -1)  # b, 3
            conds.append(angles)
        # Load data.
        # images, labels = zip(*[training_set[i] for i in grid_indices])
        # return (gw, gh),
        return [gh, gw], conds


class StyleganSR(StyleGAN2Generator):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.
    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, feat_3d_in_size=64, channel_multiplier=2, narrow=1, sft_half=False):
        super(StyleganSR, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow)
        
        log_3d_in_size = int(math.log(feat_3d_in_size, 2))
        unet_narrow = narrow * 0.5
        channels_3d = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(32),
        }
        
        self.fusion_3d = nn.ModuleList()
        for cur_log_size in range(3, log_3d_in_size + 1):
            self.fusion_3d.append(CrossAttention_without_norm(
                d_model=channels_3d[f'{2 ** cur_log_size}'],
                nhead=4,
                dim_feedforward=1024
            ))
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions_2d,
                conditions_3d,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False,
                return_generate_rows=False):
        """Forward function for StyleGAN2GeneratorCSFT.
        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        generate_rows = []

        i, j = 1, 0
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions_2d):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions_2d[i - 1] + conditions_2d[i]

                    # 3d condition fusion
                    if j < len(conditions_3d):
                        # print(out_sft.size(), conditions_3d[j].size())
                        out_sft = self.fusion_3d[j](out_sft, conditions_3d[j])
                        j += 1

                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions_2d[i - 1] + conditions_2d[i]

                

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            generate_rows.append(skip)
            i += 2
            

        image = skip

        if return_latents:
            return image, latent
        elif return_generate_rows:
            return image, generate_rows
        else:
            return image, None


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
 
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
 
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
 
        Inputs:
            x: (B, self.in_channels)
 
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
 
        return torch.cat(out, -1)
    

class ConvUpLayer(nn.Module):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.scale is used to scale the convolution weights, which is related to the common initializations.
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(
                out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2,
                            mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(torch.nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels,
                               3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(
            in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels,
                                1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class CrossAttention_without_norm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.mattn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)

        self.skip2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, query, memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None, ):
        
        b, c, *spatial = query.shape
        query = query.reshape(b, c, -1).permute(0, 2, 1)
        memory = memory.reshape(b, c, -1).permute(0, 2, 1)

        out = self.mattn(query=self.with_pos_embed(query, query_pos),
                         key=self.with_pos_embed(memory, pos),
                         value=memory,
                         attn_mask=memory_mask,
                         key_padding_mask=memory_key_padding_mask)[0]
        out = self.norm(query + self.dropout(out))   
        out = out + self.skip2(out)
        
        return out.reshape(b, c, *spatial)



class FeatureFusionBlock(nn.Module):
    '''
    borrowed from https://github.com/autonomousvision/projected-gan/blob/e1c246b8bdce4fac3c2bfcb69df309fc27df9b86/pg_modules/blocks.py#L221
    '''
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    '''
    borrowed from https://github.com/autonomousvision/projected-gan/blob/e1c246b8bdce4fac3c2bfcb69df309fc27df9b86/pg_modules/blocks.py#L221
    '''
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    '''
    borrowed from https://github.com/autonomousvision/projected-gan/blob/e1c246b8bdce4fac3c2bfcb69df309fc27df9b86/pg_modules/blocks.py#L221
    '''
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch