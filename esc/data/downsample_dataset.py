import torch
import random
from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import os.path as osp
from basicsr.utils import scandir
import math


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))
    

def single_random_crop(img, crop_size):
    input_type = 'Tensor' if torch.is_tensor(img) else 'Numpy'
    
    if input_type == 'Tensor':
        h, w = img.size()[-2:]
    else:
        h, w = img.shape[0:2]
        
    if h < crop_size or w < crop_size:
        raise ValueError(f'GT ({h}, {w}) is smaller than crop size '
                         f'({crop_size}, {crop_size}).')
        
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    
    if input_type == 'Tensor':
        img = img[:, top:top + crop_size, left:left + crop_size]
    else:
        img = img[top:top + crop_size, left:left + crop_size, ...]
    return img


@DATASET_REGISTRY.register()
class DownsampleImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(DownsampleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.scale = opt['scale']
        self.gt_size = opt['gt_size']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = opt['dataroot_gt']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        scale = self.scale
        
        # random crop
        img_gt = single_random_crop(img_gt, self.gt_size)
        
        # flip, rotation
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        
        # downsample
        img_lq = resize_fn(img_gt, (img_gt.shape[1] // scale, img_gt.shape[2] // scale))
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'gt': img_gt, 'lq': img_lq, 'gt_path': gt_path, 'lq_path': gt_path}

    def __len__(self):
        return len(self.paths)
    