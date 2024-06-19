import os
import os.path as osp
import glob
import math
import argparse
from tqdm import tqdm
import warnings
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import data
from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from basicsr.metrics import calculate_niqe, calculate_psnr, calculate_ssim
from basicsr.utils.registry import METRIC_REGISTRY

import lpips


def fid_metric(data_loader, total_batch, num_sample, inception, device, fid_stats):

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, device=device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    real_mean = fid_stats['mean']
    real_cov = fid_stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)

    return fid


def lpips_metric(lpips_vgg, data_loader, device):
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    lpips_all = []

    for data in data_loader:
        img_restored = data['lq'].to(device)
        img_gt = data['gt'].to(device)

        lpips_val = lpips_vgg(img_restored, img_gt).to('cpu').squeeze().tolist()
        lpips_all.extend(lpips_val)
    #     print(lpips_val)
    # print(lpips_all)

    return sum(lpips_all) / len(lpips_all)


if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', type=str, default='', help='Path to the experiment version folder.')
    parser.add_argument('--restored_img_folder', type=str, default='', help='if not use result images in exp_folder')
    parser.add_argument('--gt', type=str, default='../datasets/validation/reference', help='Path to the validation gt folder')
    parser.add_argument('--fid_stats', type=str, default='./metrics/inception_CelebAHQ_test_512.pth', help='Path to the dataset fid statistics.')
    parser.add_argument('--iter', type=int, default=300000, help='iters used to generate validation images')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=3000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    parser.add_argument('--cpu', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    
    start = time.time()
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset
    opt = {}
    opt['name'] = 'test_restored_imgs'
    opt['type'] = 'Wider_Metrics_Dataset'
    if not args.restored_img_folder:
        opt['dataroot_gt'] = os.path.join(args.exp_folder, 'visualization')
        opt['validation'] = True
        opt['validation_gt_path'] = args.gt
        opt['iter'] = args.iter
    else:
        opt['use_inference_img_folder'] = True
        opt['dataroot_gt'] = args.restored_img_folder
        opt['validation'] = True
        opt['validation_gt_path'] = args.gt
    opt['io_backend'] = dict(type=args.backend)
    opt['use_hflip'] = False
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]
    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=False)
    num_sample = min(args.num_sample, len(dataset))
    total_batch = math.ceil(num_sample / args.batch_size)
    print(f'-> create dataloader, size:{len(data_loader)}, use {(time.time() - start)/60.:.2f}min')

    # ------------------------- calculate fid -------------------------
    # inception model
    inception = load_patched_inception_v3(device)
    stats = torch.load(args.fid_stats)
    
    fid = fid_metric(data_loader, total_batch, num_sample, inception, device, stats)
    print(f'-> calculate fid={fid:.6f}, use {(time.time() - start)/60.:.2f}min')
    del inception, stats

    
    # ------------------------- calculate niqe, ssim, psnr ----------------------
    niqe_all, psnr_all, ssim_all = 0, 0, 0
    crop_border = 0
    # if args.restored_img_folder: 
    #     img_list = list(map(osp.realpath, glob.glob(osp.join(args.restored_img_folder, 'restored_faces/*.png'))))
    # else:
    #     img_list = list(map(osp.realpath, glob.glob(osp.join(args.exp_folder, 'visualization/**', f'*[0-9]_{str(args.iter)}.png'), recursive=True)))
    img_list = list(glob.glob(os.path.join(args.restored_img_folder, '*.png')))
    pbar = tqdm(total=len(img_list), unit='img', desc='Extract')
    for img_path in img_list:
        pbar.update(1)

        basename = osp.splitext(osp.basename(img_path))[0].split('_')[0]
        # val_gt_path = osp.join(args.gt, f'{basename}.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # gt_img = cv2.imread(val_gt_path, cv2.IMREAD_UNCHANGED)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, crop_border, input_order='HWC', convert_to='y')
        # print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all += niqe_score


    niqe = niqe_all / len(img_list)

    print(f'-> calculate niqe, psnr, ssim, use {(time.time() - start)/60.:.2f}min')

    if args.restored_img_folder:
        print(f'Version: {osp.basename(args.restored_img_folder)}')
    else:
        print(f'Version: {osp.basename(args.exp_folder)}')
    
    print(f'| fid: {fid:.6f} | niqe: {niqe:.6f} |')
    # print('| fid: 18.282884 | lpips: 0.380232 | niqe: 4.119213 | psnr: 24.172518 | ssim:  0.639233 | <- gfpgan')


# def calculate_fid_from_validation(data_path, gt_path, iters, stats, num_sample=3000, batch_size=3, backend='disk', num_workers=4, ):
#     # create dataset
#     opt = {}
#     opt['name'] = 'test_restored_imgs'
#     opt['type'] = 'FFHQ_GT_Dataset'
#     opt['dataroot_gt'] = data_path
#     opt['validation'] = True
#     opt['validation_gt_path'] = gt_path
#     opt['iter'] = iters
#     opt['io_backend'] = dict(type=backend)
#     opt['use_hflip'] = False
#     opt['mean'] = [0.5, 0.5, 0.5]
#     opt['std'] = [0.5, 0.5, 0.5]
#     dataset = build_dataset(opt)

#     # create dataloader
#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         sampler=None,
#         drop_last=False)
#     num_sample = min(num_sample, len(dataset))
#     total_batch = math.ceil(num_sample / batch_size)

#     # inception model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inception = load_patched_inception_v3(device)

#     return fid_metric(data_loader, total_batch, num_sample, inception, device, stats)

