# v20: add ground truth of ws, camara_params for GAN loss

# from asyncio import SafeChildWatcher
# from msilib.schema import File

# pretrain
#   v15:using eg3d-generated imgs and ws / camera_params instead of original ffhq
#   v16:fix bug of horizontal img 
#   v27:no need for ws
#   nodeg: no degradation, input = gt

import cv2
import numpy as np
import torch
import os.path as osp
import math
import json

import torch.utils.data as data
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_gamma, adjust_hue, adjust_saturation, normalize

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FFHQ_Pretrain_Nodeg_Dataset(data.Dataset):
    """
    FFHQ_Pretrain_v15_Dataset
    """

    def __init__(self, opt):
        super(FFHQ_Pretrain_Nodeg_Dataset, self).__init__()
        self.opt = opt

        # general config
        self.gt_folder = opt["dataroot_gt"]
        self.paths = paths_from_folder(self.gt_folder)
        self.io_backend_opt = opt["io_backend"]
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # component list
        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        if self.crop_components:
            # load component list from a pre-process pth files
            self.components_list = torch.load(opt.get('component_path'))

        # degradation config
        self.mean = opt["mean"]
        self.std = opt["std"]
        self.out_size = opt["out_size"]

        ## first order degradation config    
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.blur_kernel_list = opt["blur_kernel_list"]
        self.blur_kernel_prob = opt["blur_kernel_prob"]
        self.blur_sigma       = opt["blur_sigma"]

        self.downsample_range = opt["downsample_range"]
        self.noise_range      = opt["noise_range"]
        self.jpeg_range       = opt["jpeg_range"]

        ## second order degradation config
        self.second_deg_prob   = opt["second_deg_prob"]
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.blur_kernel_list2 = opt["blur_kernel_list2"]
        self.blur_kernel_prob2 = opt["blur_kernel_prob2"]
        self.blur_sigma2       = opt["blur_sigma2"]

        self.downsample_range2 = opt["downsample_range2"]
        self.noise_range2      = opt["noise_range2"]
        self.jpeg_range2       = opt["jpeg_range2"]

        self.color_jitter_shift = opt["color_jitter_shift"] / 255.
        
        self.color_jitter_prob = opt["color_jitter_prob"]
        self.color_jitter_pt_prob = opt["color_jitter_pt_prob"]

        self.gray_prob = opt["gray_prob"]

        self._load_labels(opt['ws_gt'], opt['camera_params_gt'])

        # write log
        logger = get_root_logger()
        logger.info(f'blur_kernel_size: {str(self.blur_kernel_size)}')
        logger.info(f'blur_kernel_prob: {str(self.blur_kernel_prob)}')
        logger.info(f'blur_sigma: {str(self.blur_sigma)}')
        logger.info(f'downsample_range: {str(self.downsample_range)}')
        logger.info(f'noise_range: {str(self.noise_range)}')
        logger.info(f'jpeg_range: {str(self.jpeg_range)}')
        logger.info(f'color_jitter_shift: {str(self.color_jitter_shift)}')
        logger.info(f'color_jitter_prob: {str(self.color_jitter_prob)}')
        logger.info(f'color_jitter_pt_prob: {str(self.color_jitter_pt_prob)}')
        logger.info(f'gray_prob: {str(self.gray_prob)}')
    
    
    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img


    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def _load_labels(self, ws_path, c_path):
        # with open(ws_path, 'r') as f:
        #     ws = json.load(f)
        # self.real_ws = dict(ws)
        
        with open(c_path, 'r') as f:
            c = json.load(f)['labels']
        self.real_c = dict(c)
        # c = [c[k] for k in c.keys()]
        # self.real_c = np.array(c).astype(np.float32)
        # assert self.real_c.ndim == 2


    def get_component_coordinates(self, index, status):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations
    
    def __getitem__(self, index):
        

        # get original image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation: only horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt["use_hflip"], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        img_name = osp.basename(gt_path)

        # get real ws and real camera_params c
        if status[0] == False:
            c = torch.from_numpy(np.array(self.real_c[img_name], dtype=np.float32))
        else:
            name = img_name.split('.')[0] + '_mirror.png'
            c = torch.from_numpy(np.array(self.real_c[name], dtype=np.float32))
        
        ws = 1

        
        # get facial component coordinates
        if self.crop_components:
            img_idx_cleaned = osp.splitext(img_name)[0]
            locations = self.get_component_coordinates(int(img_idx_cleaned), status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ------------------- first order degradation -----------------------------
        ## blur
        # kernel = degradations.random_mixed_kernels(
        #     self.blur_kernel_list, 
        #     self.blur_kernel_prob, 
        #     self.blur_kernel_size,
        #     self.blur_sigma, 
        #     self.blur_sigma, [-math.pi, math.pi],
        #     noise_range=None)
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        
        # ## downsample
        # scale_factor = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale_factor), int(h // scale_factor)), interpolation=cv2.INTER_LINEAR)

        # ## noise
        # img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)

        # ## jpeg compression
        # img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # ------------------- second order degradation -----------------------------
        # if np.random.uniform() < self.second_deg_prob:
        #      ## resize to original
        #     img_lq = cv2.resize(img_lq, (h, w), interpolation=cv2.INTER_LINEAR)
            
        #     ## blur
        #     kernel = degradations.random_mixed_kernels(
        #         self.blur_kernel_list2, 
        #         self.blur_kernel_prob2, 
        #         self.blur_kernel_size2,
        #         self.blur_sigma2, 
        #         self.blur_sigma2, [-math.pi, math.pi],
        #         noise_range=None)
        #     img_lq = cv2.filter2D(img_lq, -1, kernel)
            
        #     ## downsample
        #     scale_factor = np.random.uniform(self.downsample_range2[0], self.downsample_range2[1])
        #     img_lq = cv2.resize(img_lq, (int(w // scale_factor), int(h // scale_factor)), interpolation=cv2.INTER_LINEAR)

        #     ## noise
        #     img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range2)

        #     ## jpeg compression
        #     img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range2)

        ## resize to original
        # img_lq = cv2.resize(img_lq, (h, w), interpolation=cv2.INTER_LINEAR)

        ## random color jitter
        # img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        
        ## random to gray
        # if self.gray_prob and np.random.uniform() < self.gray_prob:
        #     img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
        #     img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        #     if self.opt.get('gt_gray', None):  # whether convert GT to gray images
        #         img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        #         img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        ## BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        
        ## random color jitter (pytorch version) (only for lq)
        # if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
        #     brightness = self.opt.get('brightness', (0.5, 1.5))
        #     contrast = self.opt.get('contrast', (0.5, 1.5))
        #     saturation = self.opt.get('saturation', (0, 1.5))
        #     hue = self.opt.get('hue', (-0.1, 0.1))
        #     img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        ## round and clip
        # img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        ## normalize
        normalize(img_gt, self.mean, self.std, inplace=True)

        # img_gt = img_gt.permute(1, 2, 0)

        # normalize(img_lq, self.mean, self.std, inplace=True)

        if self.crop_components:
            return_dict = {
                'lq': img_gt.detach().clone(),
                'gt': img_gt,
                'gt_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth,
                'ws': ws,
                'c': c
            }
            return return_dict
        else:
            return {'lq': img_gt.detach().clone(), 'gt': img_gt, 'gt_path': gt_path, 'ws':ws, 'c':c}

    def __len__(self):
        return len(self.paths)