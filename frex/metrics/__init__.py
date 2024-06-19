from copy import deepcopy
import importlib
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.registry import METRIC_REGISTRY
# from .niqe import calculate_niqe
# from .psnr_ssim import calculate_psnr, calculate_ssim

# __all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate']

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'metrics.{file_name}') for file_name in metric_filenames]

def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric