import argparse
import cv2
import os
import os.path as osp
import glob
import warnings
import time

from basicsr.metrics import calculate_niqe
from basicsr.utils import scandir

def calculate_dataset_niqe(restored_path, crop_border):
    niqe_all = []
    img_list = sorted(glob.glob(osp.join(restored_path, '*')))

    for i, img_path in enumerate(img_list):
        # basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, crop_border, input_order='HWC', convert_to='y')
        # print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)
    
    niqe_score = sum(niqe_all) / len(niqe_all)
    return niqe_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input path')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()

    folder_restored = args.input
    gfpgan_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_gfpgan', 'restored_faces')
    sgpn_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_sgpn', 'restored_faces')
    fr3d_folder_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_fr3d', 'restored_faces')
    gfpgan_nojitter_restored = osp.join(folder_restored, 'CelebAHQ_test_restored_gfpgan_nojitter', 'restored_faces')
    
    start = time.time()
    niqe_gfpgan = calculate_dataset_niqe(gfpgan_folder_restored, args.crop_border)
    print(f'gfpgan: {niqe_gfpgan:.6f}\tusing {(time.time() - start)/60.:.2f}min')
    niqe_gfpgan_nojitter = calculate_dataset_niqe(gfpgan_nojitter_restored, args.crop_border)
    print(f'gfpgan_nojitter: {niqe_gfpgan_nojitter:.6f}\tusing {(time.time() - start)/60.:.2f}min')
    niqe_sgpn = calculate_dataset_niqe(sgpn_folder_restored, args.crop_border)
    print(f'sgpn: {niqe_sgpn:.6f}\tusing {(time.time() - start)/60.:.2f}min')
    niqe_fr3d = calculate_dataset_niqe(fr3d_folder_restored, args.crop_border)
    print(f'fr3d: {niqe_fr3d:.6f}\tusing {(time.time() - start)/60.:.2f}min')


    
    
    


