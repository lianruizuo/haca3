#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from modules.model import HACA3
import torch
import nibabel as nib
from PIL import Image
import os
import numpy as np
from torchvision.transforms import ToTensor, CenterCrop, Compose, ToPILImage

def normalize_intensity(image):
    p99 = np.percentile(image.flatten(), 99)
    image = np.clip(image, a_min=0.0, a_max=p99)
    image = image / p99
    return image, p99

def obtain_single_image(img_path, is_target=False):
    img_file = nib.load(img_path)
    img_vol = np.array(img_file.get_fdata().astype(np.float32))
    img_vol, norm_val = normalize_intensity(img_vol)
    n_row, n_col, n_slc = img_vol.shape
    # get images with proper zero padding
    img_padded = np.zeros((224, 224, 224)).astype(np.float32)
    img_padded[112 - n_row // 2:112 + n_row // 2 + n_row % 2,
               112 - n_col // 2:112 + n_col // 2 + n_col % 2,
               112 - n_slc // 2:112 + n_slc // 2 + n_slc % 2] = img_vol
    if is_target:
        return ToTensor()(img_padded), img_file.header, img_file.affine, norm_val
    else:
        return ToTensor()(img_padded), img_file.header, img_file.affine


def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Harmonization with HACA3')
    parser.add_argument('--t1', type=str, default='')
    parser.add_argument('--t2', type=str, default='')
    parser.add_argument('--pd', type=str, default='')
    parser.add_argument('--flair', type=str, default='')
    parser.add_argument('--target-image', type=str, nargs='+', required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='subject1')
    parser.add_argument('--pretrained-harmonization', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--eta-dim', type=int, default=2)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=4)
    parser.add_argument('--save-intermediate', action='store_true', default=False)
    parser.add_argument('--pretrained-fusion', type=str, default=None)
    args = parser.parse_args(args)

    contrast_names = ["T1", "T2", "PD", "FLAIR"]
    contrast_dropout = np.ones((4,)).astype(np.float32) * 1e5
    # initialize model
    harmonization_model = HACA3(beta_dim=args.beta_dim,
                                theta_dim=args.theta_dim,
                                eta_dim=args.eta_dim,
                                pretrained_harmonization=args.pretrained_harmonization,
                                gpu=args.gpu)

    # load source images
    source_imgs = []
    # t1
    if args.t1 != 'none':
        img_vol, img_header, img_affine = obtain_single_image(args.t1)
        contrast_dropout[0] = 0.0
    else:
        img_vol = ToTensor()(np.ones((224, 224, 224)).astype(np.float32))
    source_imgs.append(img_vol.float().permute(2, 1, 0))
    
    # t2
    if args.t2 != 'none':
        img_vol, img_header, img_affine = obtain_single_image(args.t2)
        contrast_dropout[1] = 0.0
    else:
        img_vol = ToTensor()(np.ones((224, 224, 224)).astype(np.float32))
    source_imgs.append(img_vol.float().permute(2, 1, 0))
    
    # pd
    if args.pd != 'none':
        img_vol, img_header, img_affine = obtain_single_image(args.pd)
        contrast_dropout[2] = 0.0
    else:
        img_vol = ToTensor()(np.ones((224, 224, 224)).astype(np.float32))
    source_imgs.append(img_vol.float().permute(2, 1, 0))
    
    # flair
    if args.flair != 'none':
        img_vol, img_header, img_affine = obtain_single_image(args.flair)
        contrast_dropout[3] = 0.0
    else:
        img_vol = ToTensor()(np.ones((224, 224, 224)).astype(np.float32))
    source_imgs.append(img_vol.float().permute(2, 1, 0))

    # load target images
    target_imgs, target_contrasts = [], []
    for target_img_path in args.target_image:
        target_contrasts.append([t for t in contrast_names if t in target_img_path][0])
        target_img_tmp, _, _, norm_val = obtain_single_image(target_img_path, is_target=True)
        target_imgs.append(target_img_tmp.permute(2, 1, 0).permute(0, 2, 1).flip(1)[100:120, ...])

    # axial
    harmonization_model.harmonization(source_imgs=[img.permute(2, 0, 1) for img in source_imgs],
                                      target_imgs=target_imgs,
                                      target_contrasts=target_contrasts,
                                      contrast_dropout=torch.from_numpy(contrast_dropout),
                                      out_dir=args.out_dir,
                                      prefix=args.prefix,
                                      recon_orientation='axial',
                                      header=img_header,
                                      affine=img_affine,
                                      num_batches=args.num_batches,
                                      save_intermediate=args.save_intermediate,
                                      norm_val=norm_val)
    # coronal
    harmonization_model.harmonization(source_imgs=[img.permute(0, 2, 1).flip(1) for img in source_imgs],
                                      target_imgs=target_imgs,
                                      target_contrasts=target_contrasts,
                                      contrast_dropout=torch.from_numpy(contrast_dropout),
                                      out_dir=args.out_dir,
                                      prefix=args.prefix,
                                      recon_orientation='coronal',
                                      header=img_header,
                                      affine=img_affine,
                                      num_batches=args.num_batches,
                                      save_intermediate=args.save_intermediate,
                                      norm_val=norm_val)
                                      
    # sagittal
    harmonization_model.harmonization(source_imgs=[img.permute(1, 2, 0).flip(1) for img in source_imgs],
                                      target_imgs=target_imgs,
                                      target_contrasts=target_contrasts,
                                      contrast_dropout=torch.from_numpy(contrast_dropout),
                                      out_dir=args.out_dir,
                                      prefix=args.prefix,
                                      recon_orientation='sagittal',
                                      header=img_header,
                                      affine=img_affine,
                                      num_batches=args.num_batches,
                                      save_intermediate=args.save_intermediate,
                                      norm_val=norm_val)
            
    for target_contrast in target_contrasts:
        orientations = ['axial', 'coronal', 'sagittal']
        decode_img_dirs = []
        for orientation in orientations:
            decode_img_dirs.append(os.path.join(args.out_dir,
                                                f'{args.prefix}_harmonized_to_{target_contrast}_{orientation}.nii.gz'))
        harmonization_model.combine_images(decode_img_dirs, args.out_dir, args.prefix, target_contrast, args.pretrained_fusion)

if __name__ == '__main__':
    main()
