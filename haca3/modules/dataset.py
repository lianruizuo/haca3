import os
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, ToPILImage
import torchio as tio
import nibabel as nib

default_transform = Compose([ToPILImage(), Pad(40), CenterCrop([224, 224])])
transform_dict = {
    tio.RandomMotion(degrees=(15, 30), translation=(10, 20)): 0.25,
    tio.RandomNoise(std=(0.01, 0.1)): 0.25,
    tio.RandomGhosting(num_ghosts=(4, 10)): 0.25,
    tio.RandomBiasField(): 0.25
}
degradation_transform = tio.OneOf(transform_dict)
contrast_names = ['T1PRE', 'T2', 'PD', 'FLAIR']


def get_tensor_from_fpath(fpath, normalization_method):
    if os.path.exists(fpath):
        image = np.squeeze(nib.load(fpath).get_fdata().astype(np.float32)).transpose([1, 0])

        if normalization_method == 'wm':
            image = image / 2.0
        else:
            p99 = np.percentile(image.flatten(), 95)
            image = image / (p99 + 1e-5)
            image = np.clip(image, a_min=0.0, a_max=5.0)

        image = np.array(default_transform(image))
        image = ToTensor()(image)
    else:
        image = torch.ones([1, 224, 224])
    return image


def background_removal(image_dicts):
    num_contrasts = len(contrast_names)
    mask = torch.ones((1, 224, 224))
    for image_dict in image_dicts:
        mask = mask * image_dict['image'].ge(1e-8)
    for i in range(num_contrasts):
        image_dicts[i]['image'] = image_dicts[i]['image'] * mask
        image_dicts[i]['image_degrade'] = image_dicts[i]['image_degrade'] * mask
        image_dicts[i]['mask'] = mask.bool()
    return image_dicts


class HACA3Dataset(Dataset):
    def __init__(self, dataset_dirs, contrasts, orientations, mode='train', normalization_method='01'):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.contrasts = contrasts
        self.orientations = orientations
        self.t1_paths, self.site_ids = self._get_file_paths()
        self.normalization_method = normalization_method

    def _get_file_paths(self):
        fpaths, site_ids = [], []
        for site_id, dataset_dir in enumerate(self.dataset_dirs):
            for orientation in self.orientations:
                t1_paths = sorted(glob(os.path.join(dataset_dir, self.mode, f'*T1PRE*{orientation.upper()}*nii.gz')))
                fpaths += t1_paths
                site_ids += [site_id] * len(t1_paths)
        return fpaths, site_ids

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx: int):
        image_dicts = []
        for contrast_id, contrast_name in enumerate(contrast_names):
            image_path = self.t1_paths[idx].replace('T1PRE', contrast_name)
            image = get_tensor_from_fpath(image_path, self.normalization_method)
            image_degrade = degradation_transform(image.unsqueeze(1)).squeeze(1)
            site_id = self.site_ids[idx]
            image_dict = {'image': image,
                          'image_degrade': image_degrade,
                          'site_id': site_id,
                          'contrast_id': contrast_id,
                          'exists': 0 if image[0, 0, 0] > 0.9999 else 1}
            image_dicts.append(image_dict)
        return background_removal(image_dicts)
