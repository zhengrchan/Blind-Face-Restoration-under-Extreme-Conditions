import cv2
import glob
import numpy as np
import os.path as osp
import argparse
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')

def calculate_lpips(folder_gt, folder_restored, suffix=''):
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).to('cpu').item()

        # print(f'{i+1:3d}: {basename:25} \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    average_lpips = sum(lpips_all) / len(lpips_all)
    
    
    return average_lpips


def main():
    # Configurations
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='')
    parser.add_argument('--restored', type=str, default='')
    args = parser.parse_args()

    folder_gt = args.gt
    folder_restored = args.restored
    gfpgan_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_gfpgan', 'restored_faces')
    sgpn_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_sgpn', 'restored_faces')
    fr3d_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_fr3d', 'restored_faces')
    gfpgan_nojitter_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_gfpgan_nojitter', 'restored_faces')

    # -------------------------------------------------------------------------
    
    gfpgan_lpips = calculate_lpips(folder_gt, gfpgan_folder_restored, suffix='_gfp_00')
    sgpn_lpips = calculate_lpips(folder_gt, sgpn_folder_restored, suffix='_sgpn_00')
    fr3d_lpips = calculate_lpips(folder_gt, fr3d_folder_restored, suffix='_fr3d_00')
    gfpgan_nojitter_lpips = calculate_lpips(folder_gt, gfpgan_nojitter_restored, suffix='_gfp_nojitter_00')

    print(f'Average: LPIPS:\n gfp: {gfpgan_lpips:.6f}\n sgpn:{sgpn_lpips:.6f}\n fr3d:{fr3d_lpips:.6f}\n {gfpgan_nojitter_lpips:.6f}')


if __name__ == '__main__':
    main()