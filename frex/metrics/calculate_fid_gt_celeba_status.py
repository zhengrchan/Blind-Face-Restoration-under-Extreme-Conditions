import argparse
import math
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import data
from basicsr.data import build_dataset
from basicsr.metrics.fid import extract_inception_features, load_patched_inception_v3
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../fr3d'))

def calculate_stats_from_dataset():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', type=int, default=27000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--dataroot', type=str, default='datasets/ffhq')
    args = parser.parse_args()

    # inception model
    inception = load_patched_inception_v3(device)

    # create dataset
    opt = {}
    opt['name'] = 'CelebAHQ_test'
    opt['type'] = 'FFHQ_GT_128_Dataset'
    # opt['type'] = 'MiniFFHQDataset'
    opt['dataroot_gt'] = args.dataroot
    opt['io_backend'] = {'type': 'disk'}
    opt['use_hflip'] = False
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]
    dataset = build_dataset(opt)

    # return 
    # create dataloader
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=None, drop_last=False)
    total_batch = math.ceil(args.num_sample / args.batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:args.num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')
    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    save_path = f'inception_{opt["name"]}_{args.size}.pth'
    torch.save(
        dict(name=opt['name'], size=args.size, mean=mean, cov=cov), save_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    # move this file to fr3d folder to run
    calculate_stats_from_dataset()