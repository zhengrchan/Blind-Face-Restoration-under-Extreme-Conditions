import random
import time
import os
from os import path as osp
import glob
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

def paths_from_validation(visualization_folder, img_iter):
    paths = list(map(os.path.realpath, glob.glob(osp.join(visualization_folder, '**', f'*[0-9]_{str(img_iter)}.png'), recursive=True)))
    return paths

@DATASET_REGISTRY.register()
class Wider_Metrics_Dataset(data.Dataset):
    """FFHQ dataset for StyleGAN.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
    """

    def __init__(self, opt):
        super(Wider_Metrics_Dataset, self).__init__()
        self.opt = opt

        # file client (io backend)
        self.folder = opt["dataroot_gt"]
        self.is_val = self.opt.get('validation', False)
        self.use_inf = self.opt.get('use_inference_img_folder', False)

        self.paths = list(glob.glob(os.path.join(self.folder, '*.png')))
        self.io_backend_opt = opt['io_backend']
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # self.folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']

    def __getitem__(self, index):
       
        index = index % len(self.paths)
        img_path = self.paths[index]

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(img_path)
            except Exception as error:
                print(error)
                # change another file to read
                index = random.randint(0, self.__len__())
                img_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img = augment(img, hflip=self.opt['use_hflip'], rotation=False)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img = img2tensor(img, bgr2rgb=True, float32=True)
        # normalize
        normalize(img, self.mean, self.std, inplace=True)

        return {'lq': img, 'img_path': img_path}

    def __len__(self):
        return len(self.paths) * 200