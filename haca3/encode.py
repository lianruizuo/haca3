import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from .modules.model import HACA3
from .modules.utils import mkdir_p
from .test import obtain_single_image


def axial_slices(image_path, bg_removal=True):
    """Load an image and arrange it as a stack of axial slices (num_slices, 224, 224).

    Uses the same preprocessing (95th-percentile normalization, background removal,
    zero-padding to 224^3) and axial orientation as ``haca3-test``.
    """
    image_vol, header, _ = obtain_single_image(image_path, bg_removal)
    # Match the source-image axial arrangement used in test.main / HACA3.harmonize.
    image_vol = image_vol.float().permute(2, 1, 0).permute(2, 0, 1)
    return image_vol, header


def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(
        description='Encode the contrast representation (theta) of MR images with HACA3. '
                    'Inference only: no harmonization is performed.')
    parser.add_argument('--in-path', type=Path, action='append', required=True,
                        help='Input image. May be repeated to encode multiple images in one run.')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='Directory where theta CSV files are written.')
    parser.add_argument('--harmonization-model', type=Path, required=True,
                        help='Pretrained HACA3 weights (the theta encoder is loaded from here).')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Output file prefix. Defaults to the first input file name.')
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--foreground-thresh', type=float, default=0.01,
                        help='Minimum non-background fraction for a slice to count toward the '
                             'per-volume mean theta.')
    parser.add_argument('--no-bg-removal', dest='bg_removal', action='store_false', default=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=4)
    args = parser.parse_args(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HACA3 THETA ENCODING {text_div}')

    # ==== RESOLVE PATHS ====
    args.in_path = [path.resolve() for path in args.in_path]
    args.out_dir = args.out_dir.resolve()
    args.harmonization_model = args.harmonization_model.resolve()
    mkdir_p(args.out_dir)
    prefix = args.prefix if args.prefix is not None else args.in_path[0].name.replace('.nii.gz', '')

    # ==== INITIALIZE MODEL (only the theta encoder is used) ====
    haca3 = HACA3(beta_dim=args.beta_dim,
                  theta_dim=args.theta_dim,
                  eta_dim=2,
                  pretrained_haca3=args.harmonization_model,
                  gpu_id=args.gpu_id)

    # ==== ENCODE THETA FOR EACH IMAGE ====
    per_slice_rows, mean_rows = [], []
    theta_cols = [f'theta{i}' for i in range(args.theta_dim)]
    for image_id, image_path in enumerate(args.in_path):
        image_vol, _ = axial_slices(image_path, args.bg_removal)
        theta, foreground = haca3.encode_theta(image_vol, num_batches=args.num_batches)
        theta = theta.numpy()
        foreground = foreground.numpy()

        for slice_id in range(theta.shape[0]):
            per_slice_rows.append([f'image{image_id}', f'slice{slice_id:03d}',
                                   f'{foreground[slice_id]:.6f}'] +
                                  ['%.6f' % v for v in theta[slice_id]])

        keep = foreground >= args.foreground_thresh
        if not keep.any():
            keep = np.ones_like(foreground, dtype=bool)
        theta_mean = theta[keep].mean(axis=0)
        mean_rows.append([f'image{image_id}', image_path.name] + ['%.6f' % v for v in theta_mean])
        print(f'  {image_path.name}: mean theta = '
              f'[{", ".join("%.4f" % v for v in theta_mean)}] '
              f'over {int(keep.sum())} foreground slices')

    # ==== WRITE OUTPUTS ====
    per_slice_path = args.out_dir / f'{prefix}_theta.csv'
    with open(per_slice_path, 'w') as fp:
        fp.write(','.join(['image', 'slice', 'foreground'] + theta_cols) + '\n')
        for row in per_slice_rows:
            fp.write(','.join(row) + '\n')

    mean_path = args.out_dir / f'{prefix}_theta_mean.csv'
    with open(mean_path, 'w') as fp:
        fp.write(','.join(['image', 'file'] + theta_cols) + '\n')
        for row in mean_rows:
            fp.write(','.join(row) + '\n')

    print(f'{text_div} DONE {text_div}')
    print(f'Per-slice theta : {per_slice_path}')
    print(f'Per-volume theta: {mean_path}')


if __name__ == '__main__':
    main()
