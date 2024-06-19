import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch

import archs 
from copy import deepcopy

from basicsr.archs import build_network
from basicsr.losses import build_loss
# from arcface.config.config import Config
# from arcface.models.resnet import resnet_face18
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
# from vqfr.utils import img2tensor


def load_image(img_path):
    image = cv2.imread(img_path, 0)  # only on gray images
    # resise
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image)
    return image


def load_image_torch(img_path):
    image = cv2.imread(img_path) / 255.
    image = image.astype(np.float32)
    image = img2tensor(image, bgr2rgb=True, float32=True)
    normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    image.unsqueeze_(0)
    image = (0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :])
    image = image.unsqueeze(1)
    image = F.interpolate(image, (128, 128), mode='bilinear', align_corners=False)
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def calculate_cos_dist(args):

    test_model_path = '/home/zrch/FR3D/experiments/pretrained_models/arcface_resnet18.pth'

    restored_list = sorted(glob.glob(os.path.join(args.restored_folder, '*')))
    gt_list = sorted(glob.glob(os.path.join(args.gt_folder, '*')))

    # opt = Config()
    # if opt.backbone == 'resnet18':
    #     model = resnet_face18(opt.use_se)
    # else:
    #     raise NotImplementedError
    # elif opt.backbone == 'resnet34':
    #    model = resnet34()
    # elif opt.backbone == 'resnet50':
    #    model = resnet50()

    # model = DataParallel(model)
    # model.load_state_dict(torch.load(test_model_path))
    # model.to(torch.device('cuda'))
    # model.eval()
    arcface_opt = {
        'type': 'ArcFaceIDFeatureExtractor',
        'block': 'IRBlock',
        'layers': [2, 2, 2, 2],
        'use_se': False
    }
    model = build_network(arcface_opt)
    model = model.to(torch.device('cuda'))
    load_net = torch.load(test_model_path)
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net)


    dist_list = []
    identical_count = 0
    for idx, restored_path in enumerate(restored_list):
        # print(restored_path)
        # img_name = os.path.basename(restored_path).split('_')[0] + '.png'
        img_name = os.path.basename(restored_path).split('_')[0]
        gt_path = os.path.join(args.gt_folder, img_name)
        basename, ext = os.path.splitext(img_name)
        img = load_image(gt_path)
        img2 = load_image(restored_path)
        # img = load_image_torch(img_path)
        # img2 = load_image_torch(img_path2)
        data = torch.stack([img, img2], dim=0)
        data = data.to(torch.device('cuda'))
        output = model(data)
        output = output.data.cpu().numpy()
        dist = cosin_metric(output[0], output[1])
        dist = np.arccos(dist) / math.pi * 180
        print(f'{idx} - {dist} o : {basename}')
        if dist < 1:
            print(f'{basename} is almost identical to original.')
            identical_count += 1
            # dist_list.append(dist)
        else:
            dist_list.append(dist)

    print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
    print(f'identical count: {identical_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('--gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()
    calculate_cos_dist(args)