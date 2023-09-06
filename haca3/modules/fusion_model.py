import os
from tqdm import tqdm
import numpy as np
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
import torch.nn.functional as F
import nibabel as nib
from .utils import mkdir_p

class Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch+16, 16, 3, 1, 1),
            #nn.InstanceNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        #return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))

class MultiOrientationDataset(Dataset):
    def __init__(self, dataset_dirs):
        self.dataset_dirs = dataset_dirs
        self.imgs = self._get_files()

    def _get_files(self):
        img_paths = []
        contrast_names = ['T1', 'T2', 'PD', 'FLAIR']
        for dataset_dir in self.dataset_dirs:
            for contrast_name in contrast_names:
                img_path_tmp = os.path.join(dataset_dir, f'*harmonized_to_{contrast_name}_ori.nii.gz')
                img_path_tmp = sorted(glob(img_path_tmp))
                img_paths = img_paths + img_path_tmp
        return img_paths

    def __len__(self):
        return len(self.imgs)

    def get_tensor_from_path(self, img_path, if_norm_val=False):
        img = nib.load(img_path).get_fdata().astype(np.float32)
        img = ToTensor()(img)
        img = img.permute(2,1,0).permute(2,0,1).unsqueeze(0)
        img, norm_val = self.normalize_intensity(img)
        if if_norm_val:
            return img, norm_val
        else:
            return img
        
    def normalize_intensity(self, image):
        p99 = np.percentile(image.flatten(), 99)
        image = np.clip(image, a_min=0.0, a_max=p99)
        image = image / p99
        return image, p99
    
    def __getitem__(self, idx:int):
        img_path = self.imgs[idx]
        str_id = img_path.find('_ori')
        axial_img_path = img_path[:str_id] + '_axial.nii.gz'
        coronal_img_path = img_path[:str_id] + '_coronal.nii.gz'
        sagittal_img_path = img_path[:str_id] + '_sagittal.nii.gz'
        ori_image, norm_val = self.get_tensor_from_path(img_path, if_norm_val=True)
        img_dict = {'ori_img' : ori_image,
                    'axial_img' : self.get_tensor_from_path(axial_img_path),
                    'coronal_img' : self.get_tensor_from_path(coronal_img_path),
                    'sagittal_img' : self.get_tensor_from_path(sagittal_img_path),
                    'norm_val' : norm_val
                    }

        return img_dict


class FusionNet:
    def __init__(self, pretrained_model=None, gpu=0):
        self.device = torch.device('cuda:0' if gpu==0 else'cuda:1')

        # define networks
        self.fusion_net = Net(in_ch=3, out_ch=1)

        # initialize training variables
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optim_fusion_net = None

        # load pretrained models
        self.checkpoint = None
        if pretrained_model is not None:
            self.checkpoint = torch.load(pretrained_model, map_location=self.device)
            self.fusion_net.load_state_dict(self.checkpoint['fusion_net'])
        self.fusion_net.to(self.device)
        self.start_epoch = 0

    def load_dataset(self, dataset_dirs, batch_size):
        all_dataset = MultiOrientationDataset(dataset_dirs)
        num_instances = all_dataset.__len__()
        num_train = int(0.8 * num_instances)
        num_valid = num_instances - num_train
        train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset,
                                                                     [num_train, num_valid])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def initialize_training(self, out_dir, lr):
        self.out_dir = out_dir
        mkdir_p(out_dir)
        mkdir_p(os.path.join(out_dir, 'results'))
        mkdir_p(os.path.join(out_dir, 'models'))

        # define loss
        self.l1_loss = nn.L1Loss(reduction='none')

        self.optim_fusion_net = torch.optim.Adam(self.fusion_net.parameters(), lr=lr)

        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optim_fusion_net.load_state_dict(self.checkpoint['optim_fusion_net'])
        self.start_epoch += 1
        
    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs+1):
            self.train_loader = tqdm(self.train_loader)
            self.fusion_net.train()
            train_loss_sum = 0.0
            num_train_imgs = 0
            for batch_id, img_dict in enumerate(self.train_loader):
                syn_img = torch.cat([
                    img_dict['axial_img'],
                    img_dict['coronal_img'],
                    img_dict['sagittal_img']
                ], dim=1).to(self.device)
                ori_img = img_dict['ori_img'].to(self.device)
                batch_size = ori_img.shape[0]
                
                ori_img = ori_img * (syn_img[:,[0],:,:,:] > 1e-8).detach()

                fusion_img = self.fusion_net(syn_img)

                rec_loss = self.l1_loss(fusion_img, ori_img).mean()
                self.optim_fusion_net.zero_grad()
                rec_loss.backward()
                self.optim_fusion_net.step()

                train_loss_sum += rec_loss.item() * batch_size
                num_train_imgs += batch_size
                self.train_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {rec_loss.item():.3f}; '
                                                   f'avg_trn: {train_loss_sum / num_train_imgs:.3f}; '))
                if batch_id % 50 - 1 == 0:
                    img_affine = [[-1, 0, 0, 96], [0, -1, 0, 96], [0, 0, 1, -78], [0, 0, 0, 1]]
                    img_save = np.array(fusion_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                    img_save = nib.Nifti1Image(img_save * np.array(img_dict['norm_val']), img_affine)
                    file_name = os.path.join(self.out_dir, 'results', f'train_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_syn.nii.gz')
                    nib.save(img_save, file_name)

                    img_save = np.array(ori_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                    img_save = nib.Nifti1Image(img_save * np.array(img_dict['norm_val']), img_affine)
                    file_name = os.path.join(self.out_dir, 'results', f'train_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_ori.nii.gz')
                    nib.save(img_save, file_name)

                # save models
                if batch_id % 100 == 0:
                     file_name = os.path.join(self.out_dir, 'models',
                                             f'epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}.pt')
                     self.save_model(file_name, epoch)
            # VALIDATION
            self.valid_loader = tqdm(self.valid_loader)
            valid_loss_sum = 0.0
            num_valid_imgs = 0
            self.fusion_net.eval()
            with torch.set_grad_enabled(False):
                for batch_id, img_dict in enumerate(self.valid_loader):
                    syn_img = torch.cat([
                    img_dict['axial_img'],
                    img_dict['coronal_img'],
                    img_dict['sagittal_img']
                    ], dim=1).to(self.device)
                    ori_img = img_dict['ori_img'].to(self.device)
                    batch_size = ori_img.shape[0]
                    
                    #mask = syn_img[:,[0],:,:,:] > 1e-8
                    ori_img = ori_img * (syn_img[:,[0],:,:,:] > 1e-8).detach()

                    fusion_img = self.fusion_net(syn_img)

                    rec_loss = self.l1_loss(fusion_img, ori_img).mean()

                    valid_loss_sum += rec_loss.item() * batch_size
                    num_valid_imgs += batch_size
                    self.valid_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {rec_loss.item():.3f}; '
                                                       f'avg_trn: {valid_loss_sum / num_valid_imgs:.3f}; '))
                    if batch_id % 50 - 1 == 0:
                        img_affine = [[-1, 0, 0, 96], [0, -1, 0, 96], [0, 0, 1, -78], [0, 0, 0, 1]]
                        img_save = np.array(fusion_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                        img_save = nib.Nifti1Image(img_save * np.array(img_dict['norm_val']), img_affine)
                        file_name = os.path.join(self.out_dir, 'results', f'valid_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_syn.nii.gz')
                        nib.save(img_save, file_name)

                        img_save = np.array(ori_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                        img_save = nib.Nifti1Image(img_save * np.array(img_dict['norm_val']), img_affine)
                        file_name = os.path.join(self.out_dir, 'results', f'valid_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_ori.nii.gz')
                        nib.save(img_save, file_name)


    def save_model(self, file_name, epoch):
        state = {'epoch': epoch,
                 'fusion_net': self.fusion_net.state_dict(),
                 'optim_fusion_net': self.optim_fusion_net.state_dict()}
        torch.save(obj=state, f=file_name)
                    
                    
                
