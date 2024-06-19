import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from .. import data
from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def fid_metric(img, img2, inception, **kwargs):
    feature = inception(img)

def calculate_fid_folder(inception, data_path, args, device):

    # create dataset
    opt = {}
    opt['name'] = 'test_restored_imgs'
    opt['type'] = 'FFHQ_GT_Dataset'
    opt['dataroot_gt'] = data_path
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
    args.num_sample = min(args.num_sample, len(dataset))
    total_batch = math.ceil(args.num_sample / args.batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['gt']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:args.num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')
    
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(args.fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid:', fid)
    return fid


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='Path to the folder.')
    parser.add_argument('--fid_stats', type=str, help='Path to the dataset fid statistics.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=3000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()

    # inception model
    inception = load_patched_inception_v3(device)

    gfpgan_restored_path = os.path.join(args.folder, 'CelebAHQ_test_restored_gfpgan', 'restored_faces')
    gfpgan_nojitter_restored_path = os.path.join(args.folder, 'CelebAHQ_test_restored_gfpgan_nojitter', 'restored_faces')
    sgpn_restored_path = os.path.join(args.folder, 'CelebAHQ_test_restored_sgpn', 'restored_faces')
    fr3d_restored_path = os.path.join(args.folder, 'CelebAHQ_test_restored_fr3d', 'restored_faces')
     
    gfpgan_fid = calculate_fid_folder(inception, gfpgan_restored_path, args, device)
    gfpgan_nojitter_fid = calculate_fid_folder(inception, gfpgan_nojitter_restored_path, args, device)
    sgpn_fid = calculate_fid_folder(inception, sgpn_restored_path, args, device)
    fr3d_fid = calculate_fid_folder(inception, fr3d_restored_path, args, device)

    print(f'fid: ')
    print(f'gfpgan: {gfpgan_fid:.6f}')
    print(f'gfp_nojitter: {gfpgan_nojitter_fid:.6f}')
    print(f'sgpn: {sgpn_fid:.6f}')
    print(f'fr3d: {fr3d_fid:.6f}')