import argparse
import os
from os import path as osp
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

from tqdm import tqdm


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', default='input.png')
#     parser.add_argument('--model')
#     parser.add_argument('--scale')
#     parser.add_argument('--output', default='output.png')
#     parser.add_argument('--gpu', default='0')
#     args = parser.parse_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     scale_max = 4 # Maximum scale factor during training
    
#     img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
#     model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
#     h = int(img.shape[-2] * int(args.scale))
#     w = int(img.shape[-1] * int(args.scale))
#     scale = h / img.shape[-2]
#     coord = make_coord((h, w)).cuda()
#     cell = torch.ones_like(coord)
#     cell[:, 0] *= 2 / h
#     cell[:, 1] *= 2 / w
    
#     cell_factor = max(scale/scale_max, 1)
#     pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
#         coord.unsqueeze(0), cell_factor*cell.unsqueeze(0), bsize=30000)[0]
#     pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
#     transforms.ToPILImage()(pred).save(args.output)

def inference(model, scale, is_lr, input_path, image_name, output_path):
    scale_max = 4 # Maximum scale factor during training

    img = transforms.ToTensor()(Image.open(osp.join(input_path, image_name)).convert('RGB'))
    if is_lr:
        h = int(img.shape[-2] * int(scale))
        w = int(img.shape[-1] * int(scale))
    else:
        h = img.shape[-2]
        w = img.shape[-1]
        scale = int(scale)
        img = resize_fn(img, (h//scale, w//scale))

    scale = h / img.shape[-2]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    cell_factor = max(scale/scale_max, 1)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell_factor*cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(osp.join(output_path, image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    # parser.add_argument('--is_lr', type=bool)
    parser.add_argument('--is_lr', action='store_true')
    parser.add_argument('--output_path')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    imgs_list = os.listdir(args.input_path)
    os.makedirs(args.output_path, exist_ok=True)
    for img in tqdm(imgs_list):
        inference(model, args.scale, args.is_lr, args.input_path, img, args.output_path)
