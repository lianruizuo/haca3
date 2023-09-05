#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, Resize, ToPILImage, ColorJitter
import torchio as tio
import nibabel as nib
import fnmatch
import random

default_transform = Compose([ToPILImage(), Pad(40), CenterCrop((224, 224))])
intensity_transform = Compose([ToPILImage(), Pad(40), CenterCrop((224, 224)),
                               ColorJitter(contrast=(0, 0.7))])
transform_dict = {
    tio.RandomMotion(degrees=(15, 30), translation=(10, 20)): 0.25,
    tio.RandomNoise(std=(0.01, 0.1)): 0.25,
    tio.RandomGhosting(num_ghosts=(4, 10)): 0.25,
    tio.RandomBiasField(): 0.25
}
artifact_transform = tio.OneOf(transform_dict)

contrast_names = ['T1PRE', 'T2', 'PD', 'FLAIR']


def get_tensor_from_path(img_path):
    if os.path.exists(img_path):
        img = np.squeeze(nib.load(img_path).get_fdata().astype(np.float32)).transpose([1, 0])
        img = np.array(default_transform(img))
        img = ToTensor()(img)
    else:
        img = torch.ones((1, 224, 224))
    return img


def masking_imgs(img_dicts):
    """
    Masking multi-contrast MRIs to make sure they have the same FOV.
    :param img_dicts:
    :return:
    """
    num_contrasts = len(contrast_names)
    mask = torch.ones((1, 224, 224))
    for img_dict in img_dicts:
        mask = mask * img_dict['img'].ge(1e-8)
    for i in range(num_contrasts):
        img_dicts[i]["img"] = img_dicts[i]["img"] * mask
        img_dicts[i]["img_artifact_augmented"] = img_dicts[i]["img_artifact_augmented"] * mask

    return img_dicts


class HACA3Dataset(Dataset):
    def __init__(self, dataset_dirs, contrasts, orientations, mode='train'):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.contrasts = contrasts
        self.orientations = orientations
        self.t1_paths, self.site_ids = self._get_files()

    def _get_files(self):
        img_paths = []
        site_ids = []
        for site_id, dataset_dir in enumerate(self.dataset_dirs):
            for orientation in self.orientations:
                t1_niis = os.path.join(dataset_dir, self.mode, f'*T1PRE*{orientation.upper()}*.nii.gz')
                t1_niis = sorted(glob(t1_niis))
                for img_path in t1_niis:
                    img_paths.append(img_path)
                    site_ids.append(site_id)
        return img_paths, site_ids

    def __len__(self):
        return len(self.t1_paths)

    def same_volume_different_slice(self, img_path, contrast_id):
        candidate_slices = ['SLICE10', 'SLICE11', 'SLICE12', 'SLICE13']
        candidate_img_paths = []
        path_pool = []
        for p in self.t1_paths:
            new_p = p.replace('T1PRE', contrast_names[contrast_id])
            path_pool.append(new_p)
        same_volume_different_slice_path = img_path
        if os.path.exists(img_path):
            for candidate_slice in candidate_slices:
                if 'AXIAL' in img_path:
                    str_id = img_path.find('_AXIAL')
                    candidate_img_paths += fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_CORONAL_{candidate_slice}*.nii.gz') + \
                                           fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_SAGITTAL_{candidate_slice}*.nii.gz')
                elif 'CORONAL' in img_path:
                    str_id = img_path.find('_CORONAL')
                    candidate_img_paths += fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_AXIAL_{candidate_slice}*.nii.gz') + \
                                           fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_SAGITTAL_{candidate_slice}*.nii.gz')
                else:
                    str_id = img_path.find('_SAGITTAL')
                    candidate_img_paths += fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_AXIAL_{candidate_slice}*.nii.gz') + \
                                           fnmatch.filter(path_pool,
                                                          f'{img_path[:str_id]}_CORONAL_{candidate_slice}*.nii.gz')
            same_volume_different_slice_path = random.choice(candidate_img_paths)

        return get_tensor_from_path(same_volume_different_slice_path)

    def __getitem__(self, idx: int):
        img_dicts = []
        for contrast_id, contrast_name in enumerate(contrast_names):
            t1_path = self.t1_paths[idx]
            img_path = t1_path.replace('T1PRE', contrast_name)
            img = get_tensor_from_path(img_path)  # load MRI if exists, otherwise return constant image.
            img_prime = self.same_volume_different_slice(img_path, contrast_id)
            img_artifact_augmented = artifact_transform(img.unsqueeze(1)).squeeze(1)
            site_id = self.site_ids[idx]

            img_dict = {'img': img,
                        'img_artifact_augmented': img_artifact_augmented,
                        'img_prime': img_prime,
                        'site_id': site_id,
                        'contrast_id': contrast_id,
                        'exists': 0.0 if img.mean() > 0.999 else 1.0}
            img_dicts.append(img_dict)

        return masking_imgs(img_dicts)

