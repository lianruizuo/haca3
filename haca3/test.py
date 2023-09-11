import sys
import argparse
import torch
import nibabel as nib
import numpy as np
import os
from torchvision.transforms import ToTensor, CenterCrop, Compose, ToPILImage
from modules.model import HACA3


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 99)
    image = np.clip(image, a_min=0.0, a_max=thresh)
    image = image / thresh
    return image, thresh


def obtain_single_image(image_path):
    image_obj = nib.load(image_path)
    image_vol = np.array(image_obj.get_fdata().astype(np.float32))
    image_vol, norm_val = normalize_intensity(image_vol)

    n_row, n_col, n_slc = image_vol.shape
    # zero padding
    image_padded = np.zeros((224, 224, 224)).astype(np.float32)
    image_padded[112 - n_row // 2:112 + n_row // 2 + n_row % 2,
    112 - n_col // 2:112 + n_col // 2 + n_col % 2,
    112 - n_slc // 2:112 + n_slc // 2 + n_slc % 2] = image_vol
    return ToTensor()(image_padded), image_obj.affine, image_obj.header, norm_val


def load_source_images(image_paths):
    source_images = []
    contrast_dropout = np.ones((4,)).astype(np.float32) * 1e5
    for contrast_id, image_path in enumerate(image_paths):
        if image_path is not None:
            image_vol, image_affine, image_header, _ = obtain_single_image(image_path)
            contrast_dropout[contrast_id] = 0.0
        else:
            image_vol = ToTensor()(np.ones((224, 224, 224)).astype(np.float32))
        source_images.append(image_vol.float().permute(2, 1, 0))
    return source_images, contrast_dropout, image_affine, image_header


def parse_array(arg_str):
    #return np.array(arg_str)
    return np.array([float(x) for x in arg_str.split(',')])


def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Harmonization with HACA3.')
    parser.add_argument('--t1', type=str, default=None)
    parser.add_argument('--t2', type=str, default=None)
    parser.add_argument('--pd', type=str, default=None)
    parser.add_argument('--flair', type=str, default=None)
    parser.add_argument('--target-image', type=str, nargs='+', default=None)
    parser.add_argument('--target-theta', type=float, nargs='+', default=None)
    parser.add_argument('--target-eta', type=str, default='0.0,0.0')
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--file-name', type=str, default='testing_subject.nii.gz')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=4)
    parser.add_argument('--save-intermediate', action='store_true', default=False)
    parser.add_argument('--pretrained-harmonization', type=str, default=None)
    parser.add_argument('--pretrained-fusion', type=str, default=None)
    args = parser.parse_args(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HACA3 HARMONIZATION {text_div}')

    # ==== CHECK CONDITIONS OF INPUT ARGUMENTS ====
    if args.t1 is None and args.t2 is None and args.pd is None and args.flair is None:
        parser.error("At least one source image must be provided.")

    if args.target_image is None and args.target_theta is None:
        parser.error('"target_image" OR "target_theta" should be provided.')

    if args.target_image is not None and args.target_theta is not None:
        print('Warning: Both "target_image" and "target_theta" are provided. Only "target_image" will be used...')

    # ==== INITIALIZE MODEL ====
    haca3 = HACA3(beta_dim=5,
                  theta_dim=2,
                  eta_dim=2,
                  pretrained_harmonization=args.pretrained_harmonization,
                  gpu=args.gpu_id)

    # ==== LOAD SOURCE IMAGES ====
    source_images, contrast_dropout, image_affine, image_header = load_source_images(
        [args.t1, args.t2, args.pd, args.flair])

    # ==== LOAD TARGET IMAGES IF PROVIDED ====
    if args.target_image is not None:
        contrast_names = ["T1", "T2", "PD", "FLAIR"]
        target_images, target_contrasts, norm_vals = [], [], []
        for target_image_path in args.target_image:
            target_contrasts.append([t for t in contrast_names if t in target_image_path.upper()][0])
            target_image_tmp, _, _, norm_val = obtain_single_image(target_image_path)
            target_images.append(target_image_tmp.permute(2, 1, 0).permute(0, 2, 1).flip(1)[100:120, ...])
            norm_vals.append(norm_val)
        target_theta = np.zeros((2,))
        target_eta = np.zeros((2,))
    else:
        target_images = None
        target_contrasts = None
        target_theta = np.array(args.target_theta)
        #target_theta = parse_array(args.target_theta)
        target_eta = parse_array(args.target_eta)
        norm_vals = [1.0]

    # ===== BEGIN HARMONIZATION WITH HACA3 =====
    # Axial
    haca3.harmonize(source_images=[image.permute(2, 0, 1) for image in source_images],
                    target_images=target_images,
                    target_theta=torch.from_numpy(target_theta),
                    target_eta=torch.from_numpy(target_eta),
                    target_contrasts=target_contrasts,
                    contrast_dropout=torch.from_numpy(contrast_dropout),
                    out_dir=args.out_dir,
                    file_name=args.file_name,
                    recon_orientation='axial',
                    affine=image_affine,
                    header=image_header,
                    num_batches=args.num_batches,
                    save_intermediate=args.save_intermediate,
                    norm_val=norm_vals)

    # Coronal
    haca3.harmonize(source_images=[image.permute(0, 2, 1).flip(1) for image in source_images],
                    target_images=target_images,
                    target_theta=torch.from_numpy(target_theta),
                    target_eta=torch.from_numpy(target_eta),
                    target_contrasts=target_contrasts,
                    contrast_dropout=torch.from_numpy(contrast_dropout),
                    out_dir=args.out_dir,
                    file_name=args.file_name,
                    recon_orientation='coronal',
                    affine=image_affine,
                    header=image_header,
                    num_batches=args.num_batches,
                    save_intermediate=args.save_intermediate,
                    norm_val=norm_vals)

    # Sagittal
    haca3.harmonize(source_images=[image.permute(1, 2, 0).flip(1) for image in source_images],
                    target_images=target_images,
                    target_theta=torch.from_numpy(target_theta),
                    target_eta=torch.from_numpy(target_eta),
                    target_contrasts=target_contrasts,
                    contrast_dropout=torch.from_numpy(contrast_dropout),
                    out_dir=args.out_dir,
                    file_name=args.file_name,
                    recon_orientation='sagittal',
                    affine=image_affine,
                    header=image_header,
                    num_batches=args.num_batches,
                    save_intermediate=args.save_intermediate,
                    norm_val=norm_vals)

    print(f'{text_div} BEGIN FUSION {text_div}')
    prefix = args.file_name.replace(".nii.gz", "")
    if target_contrasts is None:
        theta_value_array = [str(x) for x in target_theta.flatten()]
        target_contrasts = [f'theta{"_".join(theta_value_array)}']
    for target_contrast in target_contrasts:
        orientations = ['axial', 'coronal', 'sagittal']
        decode_img_dirs = []
        for orientation in orientations:
            decode_img_dirs.append(os.path.join(args.out_dir,
                                                f'{prefix}_harmonized_to_{target_contrast}_{orientation}.nii.gz'))
        haca3.combine_images(decode_img_dirs, args.out_dir, prefix, target_contrast,
                             args.pretrained_fusion)

if __name__ == '__main__':
    main()