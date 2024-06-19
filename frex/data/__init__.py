import importlib
from basicsr.utils import scandir
from os import path as osp

# project_name = 'fr3d'

# automatically scan and import dataset modules for registry
# scan all the files that end with '_dataset.py' under the data folder
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(file))[0] for file in scandir(data_folder) if file.endswith('_dataset.py')]
# print('----test-------\n\n')
# print(dataset_filenames)
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]
# print(_dataset_modules)