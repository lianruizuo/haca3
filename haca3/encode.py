import sys
import argparse
from modules.model import HACA3
import torch
import nibabel as nib
from PIL import Image
import os
import numpy as np
from torchvision.transforms import ToTensor, CenterCrop, Compose, ToPILImage


def obtain_single_image(img_path, normalization_val=1000.0):
    img_file = nib.load(img_path)
    img_vol = np.array(img_file.get_fdata().astype(np.float32))
    img_vol = img_vol / normalization_val * 0.25
    n_row, n_col, n_slc = img_vol.shape
    # get images with proper zero padding
    img_padded = np.zeros((288, 288, 288)).astype(np.float32)
    img_padded[144 - n_row // 2:144 + n_row // 2 + n_row % 2,
    144 - n_col // 2:144 + n_col // 2 + n_col % 2,
    144 - n_slc // 2:144 + n_slc // 2 + n_slc % 2] = img_vol

    return ToTensor()(img_padded), img_file.header, img_file.affine


def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Learn anatomy, artifact, and contrast with HACA3')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='subject1')
    parser.add_argument('--pretrained-harmonization', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--eta-dim', type=int, default=2)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--norm', default=1000.0, type=float)
    args = parser.parse_args(args)

    harmonization_model = HACA3(beta_dim=args.beta_dim,
                                theta_dim=args.theta_dim,
                                eta_dim=args.eta_dim,
                                pretrained_harmonization=args.pretrained_harmonization,
                                gpu=args.gpu)

    # load image
    img_vol, img_header, img_affine = obtain_single_image(args.image, args.norm)

    # Encoding
    harmonization_model.encode(img=img_vol.float().permute(2, 1, 0).permute(2, 0, 1),
                               out_dir=args.out_dir,
                               prefix=args.prefix,
                               affine=img_affine,
                               header=img_header)


if __name__ == '__main__':
    main()
