
import math
import cv2
import numpy as np
import os.path as osp
# from py import process
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# from basicsr.archs.lpips_arch import LPIPS
import lpips
from basicsr.metrics.fid import calculate_fid, load_patched_inception_v3
from basicsr.archs.inception import InceptionV3

@MODEL_REGISTRY.register()
class FR3D_Inv_v10_Model(BaseModel):
    def __init__(self, opt):
        super(FR3D_Inv_v10_Model, self).__init__(opt)
        self.idx = 0  # it is used for saving data for check

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.log_size = int(math.log(self.opt['network_g']['out_size'], 2))

        if self.is_train:
            self.init_training_settings()

        # lpips    
        self.lpips_fn = lpips.LPIPS(net='vgg')
        # self.lpips_fn = self.model_to_device(self.lpips_fn).eval()
        self.lpips_fn = self.lpips_fn.to(self.device).eval()

        # inception
        self.inception = InceptionV3([3], resize_input=True, normalize_input=False)
        self.inception = self.inception.to(self.device).eval()
        self.fid_stats = torch.load(self.opt['val']['metrics']['fid']['fid_stats'])
        self.num_sample = self.opt['val']['metrics']['fid']['num_sample']
        self.features = []

        # generate_rows
        self.return_generate_rows_images = self.opt['val']['return_generate_rows_images']

        self.save_training_results_interval = self.opt['train'].get('save_training_results_interval', -1)

    def init_training_settings(self):
        train_opt = self.opt['train']

        # ----------- define net_g with Exponential Moving Average (EMA) ----------- #
        # net_g_ema only used for testing on one GPU and saving. There is no need to wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'G_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_g_ema.eval()

        # # ----------- define net_d ----------- #
        self.use_d = True if 'network_d' in self.opt else False
        if self.use_d:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
            self.net_d.train()
        
        #  ----------- style code discriminator ----------- #
        self.use_d_ws = True if 'network_ws' in self.opt else False
        if self.use_d_ws:
            self.net_d_ws = build_network(self.opt['network_ws'])
            self.net_d_ws = self.model_to_device(self.net_d_ws)
            self.print_network(self.net_d_ws)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_d_ws', None)
            if load_path is not None:
                self.load_network(self.net_d_ws, load_path, self.opt['path'].get('strict_load_d', True))

            self.net_d_ws.train()

        #  ----------- 2d decoder rgb discriminator ----------- #
        self.use_d_2d_decoder_256_16 = True if 'network_d_2d_decoder_256_16' in self.opt else False
        if self.use_d_2d_decoder_256_16:
            self.net_d_2d_decoder_list = build_network(self.opt['network_d_2d_decoder_256_16'])
            self.net_d_2d_decoder_list = self.model_to_device(self.net_d_2d_decoder_list)
            self.print_network(self.net_d_2d_decoder_list)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_d_2d_decoder_256_16', None)
            if load_path is not None:
                self.load_network(self.net_d_2d_decoder_list, load_path, self.opt['path'].get('strict_load_d', True))

            self.net_d_2d_decoder_list.train()

        # ----------- facial component networks ----------- #
        if ('network_d_left_eye' in self.opt and 'network_d_right_eye' in self.opt and 'network_d_mouth' in self.opt):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        if self.use_facial_disc:
            # left eye
            self.net_d_left_eye = build_network(self.opt['network_d_left_eye'])
            self.net_d_left_eye = self.model_to_device(self.net_d_left_eye)
            self.print_network(self.net_d_left_eye)
            load_path = self.opt['path'].get('pretrain_network_d_left_eye')
            if load_path is not None:
                self.load_network(self.net_d_left_eye, load_path, True, 'params')
            # right eye
            self.net_d_right_eye = build_network(self.opt['network_d_right_eye'])
            self.net_d_right_eye = self.model_to_device(self.net_d_right_eye)
            self.print_network(self.net_d_right_eye)
            load_path = self.opt['path'].get('pretrain_network_d_right_eye')
            if load_path is not None:
                self.load_network(self.net_d_right_eye, load_path, True, 'params')
            # mouth
            self.net_d_mouth = build_network(self.opt['network_d_mouth'])
            self.net_d_mouth = self.model_to_device(self.net_d_mouth)
            self.print_network(self.net_d_mouth)
            load_path = self.opt['path'].get('pretrain_network_d_mouth')
            if load_path is not None:
                self.load_network(self.net_d_mouth, load_path, True, 'params')

            self.net_d_left_eye.train()
            self.net_d_right_eye.train()
            self.net_d_mouth.train()

            # ----------- define facial component gan loss ----------- #
            self.cri_component = build_loss(train_opt['gan_component_opt']).to(self.device)

        # ----------- define losses ----------- #
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_start_iter = train_opt.get('cri_pix_start_iter', 0)
        else:
            self.cri_pix = None
        # pixel loss for lr image
        if train_opt.get('pixel_lr_opt'):
            self.cri_pix_lr = build_loss(train_opt['pixel_lr_opt']).to(self.device)
            self.cri_pix_lr_start_iter = train_opt.get('cri_pix_lr_start_iter', 0)
        else:
            self.cri_pix_lr = None

        # perceptual loss for lr image
        if train_opt.get('perceptual_opt_lr'):
            self.cri_perceptual_lr = build_loss(train_opt['perceptual_opt_lr']).to(self.device)
            self.cri_perceptual_lr_start_iter = train_opt.get('cri_perceptual_lr_start_iter', 0)
        else:
            self.cri_perceptual_lr = None
        
        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.cri_perceptual_start_iter = train_opt.get('cri_perceptual_start_iter', 0)
        else:
            self.cri_perceptual = None
            
        # L1 loss is used in pyramid loss, component style loss and identity loss
        self.cri_l1 = build_loss(train_opt['L1_opt']).to(self.device)

        # gan loss (wgan)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # camera params loss
        self.cri_c = build_loss(train_opt['camera_params_opt']).to(self.device)

        # ws l1 loss
        if train_opt.get('ws_opt'):
            self.cri_ws_l1 = build_loss(train_opt['ws_opt']).to(self.device)
        else:
            self.cri_ws_l1 = None

        # ----------- define identity loss ----------- #
        if 'network_identity' in self.opt:
            self.use_identity = True
        else:
            self.use_identity = False

        if self.use_identity:
            # define identity network
            self.network_identity = build_network(self.opt['network_identity'])
            self.network_identity = self.model_to_device(self.network_identity)
            self.print_network(self.network_identity)
            load_path = self.opt['path'].get('pretrain_network_identity')
            if load_path is not None:
                self.load_network(self.network_identity, load_path, True, None)
            self.network_identity.eval()
            for param in self.network_identity.parameters():
                param.requires_grad = False
            self.network_identity_start_iter = train_opt.get('identity_start_iter', 0)

        # ----------- face structure loss ------------- #
        if train_opt.get('face_structure_opt'):

            self.network_structure = build_network(self.opt['network_structure'])
            process_group = torch.distributed.new_group()
            self.network_structure = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network_structure, process_group=process_group)
            self.network_structure = self.model_to_device(self.network_structure)
            self.print_network(self.network_structure)
            load_path = self.opt['path'].get('pretrain_network_structure')
            if load_path is not None:
                self.load_network(self.network_structure, load_path, True, None)
            self.network_structure.eval()
            for param in self.network_structure.parameters():
                param.requires_grad = False
            self.cri_structure = build_loss(train_opt['face_structure_opt']).to(self.device)
        else:
            self.cri_structure = None

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer 3d ---------- #
        net_3d_reg_ratio = 1
        params_3d = []
        params_3d_name = []
        for name, param in self.get_bare_model(self.net_g).eg3d_inverter.named_parameters():
            params_3d.append(param)
            params_3d_name.append(name)
        optim_params_3d = [{  # add normal params first
            'params': params_3d,
            'lr': train_opt['optim_3d']['lr']
        }]
        optim_type_3d = train_opt['optim_3d'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_3d_reg_ratio
        betas = (0**net_3d_reg_ratio, 0.99**net_3d_reg_ratio)
        self.optimizer_3d = self.get_optimizer(optim_type_3d, optim_params_3d, lr, betas=betas)
        self.optimizers.append(self.optimizer_3d)

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_g.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # init unfreeze eg3d option
        self.unfreeze_eg3d_iter = train_opt.get('unfreeze_eg3d', None)
        self.freeze_flag_eg3d = False

        # ----------- optimizer d ----------- #
        if self.use_d:
            net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
            normal_params = []
            for _, param in self.net_d.named_parameters():
                normal_params.append(param)
            optim_params_d = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_d']['lr']
            }]
            optim_type = train_opt['optim_d'].pop('type')
            lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
            betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
            self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
            self.optimizers.append(self.optimizer_d)

        # ----------- optimizers for facial component networks ----------- #
        if self.use_facial_disc:
            # setup optimizers for facial component discriminators
            optim_type = train_opt['optim_component'].pop('type')
            lr = train_opt['optim_component']['lr']
            # left eye
            self.optimizer_d_left_eye = self.get_optimizer(
                optim_type, self.net_d_left_eye.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_left_eye)
            # right eye
            self.optimizer_d_right_eye = self.get_optimizer(
                optim_type, self.net_d_right_eye.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_right_eye)
            # mouth
            self.optimizer_d_mouth = self.get_optimizer(
                optim_type, self.net_d_mouth.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_mouth)

            # weight
            self.mouth_weight = train_opt.get('comp_mouth_weight', 1)
            self.eye_weight = train_opt.get('comp_eye_weight', 1)

        # ----------- optimizer d of style code ----------- #
        if self.use_d_ws:
            net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
            normal_params = []
            for _, param in self.net_d_ws.named_parameters():
                normal_params.append(param)
            optim_params_d_ws = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_d_ws']['lr']
            }]
            optim_type = train_opt['optim_d_ws'].pop('type')
            lr = train_opt['optim_d_ws']['lr'] * net_d_reg_ratio
            betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
            self.optimizer_d_ws = self.get_optimizer(optim_type, optim_params_d_ws, lr, betas=betas)
            self.optimizers.append(self.optimizer_d_ws) 

        # ----------- optimizer d of 2d decoder ----------- #
        if self.use_d_2d_decoder_256_16:
            self.optimizer_d_2d_decoder_list = []
            net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
            optim_type = train_opt['optim_d_2d_decoder_list'].pop('type')
            lr = train_opt['optim_d_2d_decoder_list']['lr'] * net_d_reg_ratio
            for idx, disc in enumerate(self.net_d_2d_decoder_list.module.discriminators):
                optim = self.get_optimizer(
                    optim_type, disc.parameters(), lr, betas=(0.9 * net_d_reg_ratio, 0.99 * net_d_reg_ratio))
                self.optimizer_d_2d_decoder_list.append(optim)
                self.optimizers.append(optim)

    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        else:
            raise TypeError("Invalid type for move_to")

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq3d = data['img_for_3d'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'ws' in data:
            self.ws_gt = data['ws'].to(self.device)
            self.c_gt = data['c'].to(self.device)
        if 'loc_left_eye' in data:
            # get facial component locations, shape (batch, 4)
            self.loc_left_eyes = data['loc_left_eye']
            self.loc_right_eyes = data['loc_right_eye']
            self.loc_mouths = data['loc_mouth']
        if 'crop_param' in data:
            # self.crop_param = data['crop_param'].to(self.device)
            self.crop_param = self.move_to(data['crop_param'], self.device)
        else:
            self.crop_param = None


        # uncomment to check data
        # import torchvision
        # if self.opt['rank'] == 0:
        #     import os
        #     os.makedirs('tmp/gt', exist_ok=True)
        #     os.makedirs('tmp/lq', exist_ok=True)
        #     print(self.idx)
        #     torchvision.utils.save_image(
        #         self.gt, f'tmp/gt/gt_{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     torchvision.utils.save_image(
        #         self.lq, f'tmp/lq/lq{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     self.idx = self.idx + 1

    def construct_img_pyramid(self):
        """Construct image pyramid for intermediate restoration loss"""
        pyramid_gt = [self.gt]
        down_img = self.gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(self.opt['network_g']['out_size'] / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        # output
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def optimize_parameters(self, current_iter):
        # optimize net_g
        self.optimizer_g.zero_grad()
        if self.use_d:
            for p in self.net_d.parameters():
                p.requires_grad = False
        if self.use_d_ws:
            for p in self.net_d_ws.parameters():
                p.requires_grad = False
        if self.use_d_2d_decoder_256_16:
            for p in self.net_d_2d_decoder_list.parameters():
                p.required_grad = False

        # do not update facial component net_d
        if self.use_facial_disc:
            for p in self.net_d_left_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_right_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_mouth.parameters():
                p.requires_grad = False

        # check if unfreeze eg3d module
        # if current_iter >= self.unfreeze_eg3d_iter and self.freeze_flag_eg3d == False:
        #     self.freeze_flag_eg3d = True
        #     for parameters in self.get_bare_model(self.net_g).eg3d_decoder.parameters():
        #         parameters.requires_grad = True

        # image pyramid loss weight
        pyramid_loss_weight = self.opt['train'].get('pyramid_loss_weight', 0)
        if pyramid_loss_weight > 0 and current_iter > self.opt['train'].get('remove_pyramid_loss', float('inf')):
            pyramid_loss_weight = 1e-12  # very small weight to avoid unused param error
        if pyramid_loss_weight > 0:
            result = self.net_g(self.lq, self.lq3d, return_rgb=True, crop_param=self.crop_param)

            pyramid_gt = self.construct_img_pyramid()
        else:
            result = self.net_g(self.lq, self.lq3d, return_rgb=False, crop_param=self.crop_param)
        
        self.output, out_rgbs, ws, c, lr_image = result['image'], result['out_rgbs'], result['ws'], result['c'], result['lr_image']
        # get roi-align regions
        if self.use_facial_disc:
            self.get_roi_regions(eye_out_size=80, mouth_out_size=120)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix     

            if self.cri_pix_lr or self.cri_perceptual_lr or self.use_identity:
                scale_factor = float(self.opt['network_g']['sr_in_size']) / float(self.opt['network_g']['out_size'])
                gt_down_img = F.interpolate(self.gt, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                
            # lr pixel loss
            if self.cri_pix_lr and current_iter >= self.cri_pix_lr_start_iter:
                l_lr_pix = self.cri_pix_lr(lr_image, gt_down_img)
                l_g_total += l_lr_pix
                loss_dict['l_lr_pix'] = l_lr_pix

            # lr perceptual loss and style loss
            if self.cri_perceptual_lr and current_iter >= self.cri_perceptual_lr_start_iter:
                l_g_percep_lr, l_g_style_lr = self.cri_perceptual_lr(lr_image, gt_down_img)
                if l_g_percep_lr is not None:
                    l_g_total += l_g_percep_lr
                    loss_dict['l_g_percep_lr'] = l_g_percep_lr
                if l_g_style_lr is not None:
                    l_g_total += l_g_style_lr
                    loss_dict['l_g_style_lr'] = l_g_style_lr

            # perceptual loss
            if self.cri_perceptual and current_iter >= self.cri_perceptual_start_iter:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # face structure loss
            if self.cri_structure:
                out_gray = self.gray_resize_for_identity(self.output)
                gt_gray = self.gray_resize_for_identity(self.gt)
                l_g_structure, l_g_structure_style = self.cri_structure(out_gray, gt_gray, self.network_structure)
                if l_g_structure is not None:
                    l_g_total += l_g_structure
                    loss_dict['l_g_structure'] = l_g_structure
                if l_g_structure_style is not None:
                    l_g_total += l_g_structure_style
                    loss_dict['l_g_structure_style'] = l_g_structure_style

            # image pyramid loss
            if pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.cri_l1(out_rgbs[i], pyramid_gt[i]) * pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid

            # image pyramid gan loss
            if self.use_d_2d_decoder_256_16:
                j = 0
                for i in range(1, self.log_size - 3):
                    res = 2 ** (i + 3)
                    fake_g_pred = self.net_d_2d_decoder_list.module.discriminators[j](out_rgbs[i])
                    l_g_2d_decoder = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_2d_decoder
                    loss_dict[f'l_g_2d_decoder_{str(res)}'] = l_g_2d_decoder
                    j += 1

            # gan loss
            if self.use_d:
                dual_output = torch.cat((self.output, self.lq), 1)
                fake_g_pred = self.net_d(dual_output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

            # ws gan loss
            if self.use_d_ws:
                fake_g_pred = self.net_d_ws(ws)
                l_g_ws = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_ws
                loss_dict['l_g_ws'] = l_g_ws

            # ws l1 loss
            if self.cri_ws_l1:
                l_g_ws_l1 = self.cri_ws_l1(ws, self.ws_gt)
                l_g_total += l_g_ws_l1
                loss_dict['l_g_ws_l1'] = l_g_ws_l1

            # camera param loss
            if self.cri_c:
                l_g_c = self.cri_c(c, self.c_gt)
                l_g_total += l_g_c
                loss_dict['l_g_c'] = l_g_c

            # facial component loss
            if self.use_facial_disc:
                # left eye
                fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_left_eye, True, is_disc=False)
                l_g_total += l_g_gan * self.eye_weight
                loss_dict['l_g_gan_left_eye'] = l_g_gan * self.eye_weight
                # right eye
                fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_right_eye, True, is_disc=False)
                l_g_total += l_g_gan * self.eye_weight
                loss_dict['l_g_gan_right_eye'] = l_g_gan * self.eye_weight
                # mouth
                fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
                l_g_gan = self.cri_component(fake_mouth, True, is_disc=False)
                l_g_total += l_g_gan * self.mouth_weight
                loss_dict['l_g_gan_mouth'] = l_g_gan * self.mouth_weight

                if self.opt['train'].get('comp_style_weight', 0) > 0:
                    # get gt feat
                    _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
                    _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
                    _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(feat[0]), self._gram_mat(
                            feat_gt[0].detach())) * 0.5 + criterion(
                                self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

                    # facial component style loss
                    comp_style_loss = 0
                    comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
                    comp_style_loss = comp_style_loss * self.opt['train']['comp_style_weight']
                    l_g_total += comp_style_loss
                    loss_dict['l_g_comp_style_loss'] = comp_style_loss

            # identity loss
            if self.use_identity and current_iter >= self.network_identity_start_iter:
                identity_weight = self.opt['train']['identity_weight']
                # get gray images and resize
                out_gray = self.gray_resize_for_identity(self.output)
                gt_gray = self.gray_resize_for_identity(self.gt)

                identity_gt = self.network_identity(gt_gray).detach()
                identity_out = self.network_identity(out_gray)
                l_identity = self.cri_l1(identity_out, identity_gt) * identity_weight
                l_g_total += l_identity
                loss_dict['l_identity'] = l_identity

            loss_dict['l_g_total'] = l_g_total
            l_g_total.backward()
            self.optimizer_g.step()

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # ----------- optimize net_d ----------- #
        if self.use_d:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            if self.use_facial_disc:
                for p in self.net_d_left_eye.parameters():
                    p.requires_grad = True
                for p in self.net_d_right_eye.parameters():
                    p.requires_grad = True
                for p in self.net_d_mouth.parameters():
                    p.requires_grad = True
                self.optimizer_d_left_eye.zero_grad()
                self.optimizer_d_right_eye.zero_grad()
                self.optimizer_d_mouth.zero_grad()

            dual_gt = torch.cat((self.gt, self.lq), 1)
            fake_d_pred = self.net_d(dual_output.detach())
            real_d_pred = self.net_d(dual_gt)
            l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d
            # In WGAN, real_score should be positive and fake_score should be negative
            loss_dict['real_score'] = real_d_pred.detach().mean()
            loss_dict['fake_score'] = fake_d_pred.detach().mean()
            l_d.backward()

            # regularization loss
            if current_iter % self.net_d_reg_every == 0:
                # self.gt.requires_grad = True
                dual_gt.requires_grad = True
                real_pred = self.net_d(dual_gt)
                l_d_r1 = r1_penalty(real_pred, dual_gt)
                l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                loss_dict['l_d_r1'] = l_d_r1.detach().mean()
                l_d_r1.backward()

            self.optimizer_d.step()

        # optimize facial component discriminators
        if self.use_facial_disc:
            # left eye
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye
            l_d_left_eye.backward()
            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye
            l_d_right_eye.backward()
            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth
            l_d_mouth.backward()

            self.optimizer_d_left_eye.step()
            self.optimizer_d_right_eye.step()
            self.optimizer_d_mouth.step()

        if self.use_d_ws:
            for p in self.net_d_ws.parameters():
                p.requires_grad = True
            self.optimizer_d_ws.zero_grad()

            fake_d_pred = self.net_d_ws(ws.detach())
            real_d_pred = self.net_d_ws(self.ws_gt)
            l_d_ws = self.cri_gan(real_d_pred, True, is_disc=True) + \
                     self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_ws'] = l_d_ws
            loss_dict['real_score_ws'] = real_d_pred.detach().mean()
            loss_dict['fake_score_ws'] = fake_d_pred.detach().mean()
            l_d_ws.backward()

            # regularization loss
            if current_iter % self.net_d_reg_every == 0:
                self.ws_gt.requires_grad = True
                real_pred = self.net_d_ws(self.ws_gt)
                l_d_ws_r1 = r1_penalty(real_pred, self.ws_gt)
                l_d_ws_r1 = (self.r1_reg_weight / 2 * l_d_ws_r1 * self.net_d_reg_every + 0 * real_pred[0])
                loss_dict['l_d_ws_r1'] = l_d_ws_r1.detach().mean()
                l_d_ws_r1.backward()
                
            self.optimizer_d_ws.step()

        if self.use_d_2d_decoder_256_16:
            for p in self.net_d_2d_decoder_list.parameters():
                p.requires_grad = True
            for optim in self.optimizer_d_2d_decoder_list:
                optim.zero_grad()
            
            j = 0
            for i in range(1, self.log_size - 3):
                optim = self.optimizer_d_2d_decoder_list[j]
                optim.zero_grad()

                res = int(2 ** (i + 3))
                fake_d_pred = self.net_d_2d_decoder_list.module.discriminators[j](out_rgbs[i].detach())
                real_d_pred = self.net_d_2d_decoder_list.module.discriminators[j](pyramid_gt[i])
                l_d_pyramid = self.cri_gan(real_d_pred, True, is_disc=True) + \
                              self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict[f'l_d_2d_decoder_{str(res)}'] = l_d_pyramid
                l_d_pyramid.backward()

                if current_iter % self.net_d_reg_every == 0:
                    pyramid_gt[i].requires_grad = True
                    real_pred = self.net_d_2d_decoder_list.module.discriminators[j](pyramid_gt[i])
                    l_d_2d_decoder_r1 = r1_penalty(real_pred, pyramid_gt[i])
                    l_d_2d_decoder_r1 = (self.r1_reg_weight / 2 * l_d_2d_decoder_r1 * self.net_d_reg_every + 0 * real_pred[0])
                    loss_dict[f'l_d_2d_decoder_{str(res)}_r1'] = l_d_2d_decoder_r1.detach().mean()
                    l_d_2d_decoder_r1.backward()

                optim.step()
                j += 1

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.save_training_results_interval != -1 and current_iter % self.save_training_results_interval == 0:
            input_img = tensor2img(self.lq[0].detach().cpu(), min_max=(-1, 1))
            gt_img = tensor2img(self.gt[0].detach().cpu(), min_max=(-1, 1))
            sr_img = tensor2img(self.output[0].detach().cpu(), min_max=(-1, 1))
            eg3d_img = tensor2img(lr_image[0].detach().cpu(), min_max=(-1, 1))
            _, h, _ = eg3d_img.shape
            boarder = (512 - h) // 2
            eg3d_img = cv2.copyMakeBorder(eg3d_img, boarder, boarder, boarder, boarder, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = np.concatenate([input_img, sr_img, eg3d_img, gt_img], axis=1)
            save_path = osp.join(self.opt['path']['visualization'], f'training_{current_iter}.png')
            imwrite(img, save_path)


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                result = self.net_g_ema(self.lq, self.lq3d, return_generate_rows=True, return_rgb=True, crop_param=self.crop_param)
                self.output, _, self.ws, self.c, self.lr_image = result['image'], result['out_rgbs'], result['ws'], result['c'], result['lr_image']
                self.generate_rows = result['generate_rows']
                self.out_rgbs = result['out_rgbs']
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                result = self.net_g(self.lq, self.lq3d, return_generate_rows=True, return_rgb=True, crop_param=self.crop_param)
                self.output, _, self.ws, self.c, self.lr_image = result['image'], result['out_rgbs'], result['ws'], result['c'], result['lr_image']
                self.generate_rows = result['generate_rows']
                self.out_rgbs = result['out_rgbs']
                self.net_g.train()
    
    def calculate_lpips(self):
        # self.lpips_fn = LPIPS(net='alex',version='0.1', 
        #         pretrained_model_path='./experiments/pretrained_models/lpips/weights/v0.1/alex.pth')
        return np.mean(self.lpips_fn(self.output, self.gt).to('cpu').squeeze().tolist())

    def extract_fid_feature(self):
        feature = self.inception(self.output)[0].view(self.output.shape[0], -1)
        self.features.append(feature.to('cpu'))

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output.detach().cpu(), min_max=(-1, 1))
            lr_img = tensor2img(self.lr_image.detach().cpu(), min_max=(-1, 1))
            metric_data['img'] = sr_img
            if hasattr(self, 'gt'):
                gt_img = tensor2img(self.gt.detach().cpu(), min_max=(-1, 1))
                metric_data['img2'] = gt_img

            if hasattr(self, 'generate_rows') and img_name in self.return_generate_rows_images:
                imgs = []
                for sr_img in self.generate_rows:
                    sr_img = tensor2img(sr_img.detach().cpu(), rgb2bgr=True, min_max=(-1, 1))
                    _, h, _ = sr_img.shape
                    boarder = (512 - h) // 2
                    sr_img = cv2.copyMakeBorder(sr_img, boarder, boarder, boarder, boarder, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    imgs.append(sr_img)
                generate_rows = np.concatenate(imgs, axis=1)
            else:
                generate_rows = None
                # del self.gt

            # tentative for out of GPU memory
            # del self.lq
            # del self.lr_image
            # del self.generate_rows
            # # del self.output
            # torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')        
                imwrite(sr_img, save_img_path)

                if self.opt['val']['save_lr_img']:
                    if self.opt['is_train']:
                        save_lr_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_lr_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_lr_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_lr_{self.opt["name"]}.png')
                    imwrite(lr_img, save_lr_img_path)

                if img_name in self.return_generate_rows_images and self.opt['is_train']:
                    save_generate_rows_path = osp.join(self.opt['path']['visualization'], dataset_name, 
                                                       f'{img_name}_generate_rows_{current_iter}.png')
                    imwrite(generate_rows, save_generate_rows_path)
                    print('save generate_rows in ', save_generate_rows_path)

                    rgbs = []
                    for ii, rgb in enumerate(self.out_rgbs):
                        rgb = tensor2img(rgb.detach().cpu(), min_max=(-1, 1)).astype('uint8')
                        _, h, _ = rgb.shape
                        boarder = (512 - h) // 2
                        rgb = cv2.copyMakeBorder(rgb, boarder, boarder, boarder, boarder, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        rgbs.append(rgb)
                    rgbs = np.concatenate(rgbs, axis=1)
                    save_out_rgbs_path = osp.join(self.opt['path']['visualization'], dataset_name, 
                                                  f'{img_name}_decoder_{current_iter}.png')
                    imwrite(rgbs, save_out_rgbs_path)
                    
                        

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips':
                        # if not hasattr(self, 'lpips_fn'):
                        #     self.lpips_fn = lpips.LPIPS(net='vgg')
                        #     # self.lpips_fn = self.model_to_device(self.lpips_fn).eval()
                        #     self.lpips_fn = self.lpips_fn.to(self.device).eval()
                        self.metric_results[name] += self.calculate_lpips()
                    elif name == 'fid':
                        # if not hasattr(self, 'inception'):
                        #     self.inception = InceptionV3([3], resize_input=True, normalize_input=False)
                        #     # self.inception = self.model_to_device(self.inception).eval()
                        #     self.inception = self.inception.to(self.device).eval()
                        #     self.fid_stats = torch.load(self.opt['val']['metrics']['fid']['fid_stats'])
                        #     self.num_sample = self.opt['val']['metrics']['fid']['num_sample']
                        #     self.features = []
                        #     self.metric_results[name] = 0
                        # self.metric_results[name] += self.calculate_fid()
                        self.extract_fid_feature()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if metric == 'fid':
                    self.features = torch.cat(self.features, 0).numpy()
                    self.features = self.features[:self.num_sample]
                    sample_mean = np.mean(self.features, 0)
                    sample_cov = np.cov(self.features, rowvar=False)
                    self.features = []
                    real_mean = self.fid_stats['mean']
                    real_cov = self.fid_stats['cov']
                    self.metric_results[metric] = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            # if hasattr(self, 'lpips_fn'):
            #     del self.lpips_fn
            # if hasattr(self, 'inception'):
            #     del self.inception
            #     del self.fid_stats

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        # save net_g and net_d
        self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        if self.use_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        # save component discriminators
        if self.use_facial_disc:
            self.save_network(self.net_d_left_eye, 'net_d_left_eye', current_iter)
            self.save_network(self.net_d_right_eye, 'net_d_right_eye', current_iter)
            self.save_network(self.net_d_mouth, 'net_d_mouth', current_iter)
        if self.use_d_2d_decoder_256_16:
            self.save_network(self.net_d_2d_decoder_list, 'net_d_2d_decoder_256_16', current_iter)
        if self.use_d_ws:
            self.save_network(self.net_d_ws, 'net_d_ws', current_iter)
        # save training state
        self.save_training_state(epoch, current_iter)

