import os
from tqdm import tqdm
import numpy as np
import random

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
import nibabel as nib
from torchvision.transforms import ToTensor

from .dataset import HACA3Dataset
from .network import UNet, ThetaEncoder, EtaEncoder, Vgg16, AttentionModule, Patchifier, Discriminator, FusionNet
from .utils import mkdir_p, PatchNCELoss, KLDivergenceLoss, reparameterize_logit, dropout_contrasts, apply_beta_mask, \
    selecting_available_contrasts, divide_into_batches


class HACA3:
    def __init__(self, beta_dim, theta_dim, eta_dim, pretrained_harmonization=None, pretrained_eta_encoder=None,
                 gpu=0):
        self.beta_dim = beta_dim
        self.theta_dim = theta_dim
        self.eta_dim = eta_dim
        self.device = torch.device("cuda:0" if gpu == 0 else "cuda:1")

        # define networks
        self.beta_encoder = UNet(in_ch=1, out_ch=self.beta_dim, base_ch=4, final_act='noact')
        self.theta_encoder = ThetaEncoder(in_ch=1, out_ch=self.theta_dim)
        self.eta_encoder = EtaEncoder(in_ch=1, out_ch=self.eta_dim)
        self.decoder = UNet(in_ch=1 + self.theta_dim, out_ch=1, conditional_ch=0, base_ch=16,
                            final_act='sigmoid')
        self.patchifier = Patchifier(in_ch=1, out_ch=128)
        self.vgg = Vgg16(requires_grad=False)
        self.attn_module = AttentionModule(self.theta_dim + self.eta_dim, v_dim=self.beta_dim)

        # initialize training variables
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optim_beta_encoder, self.optim_theta_encoder = None, None
        self.optim_decoder = None
        self.optim_patchifier = None
        self.optim_attn_module = None

        # load pretrained models
        self.checkpoint = None
        self.checkpoint_eta_encoder = None
        if pretrained_harmonization is not None:
            self.checkpoint = torch.load(pretrained_harmonization, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.theta_encoder.load_state_dict(self.checkpoint["theta_encoder"])
            self.eta_encoder.load_state_dict(self.checkpoint["eta_encoder"])
            self.decoder.load_state_dict(self.checkpoint["decoder"])
            self.patchifier.load_state_dict(self.checkpoint["patchifier"])
            self.attn_module.load_state_dict(self.checkpoint["attn_module"])
        if pretrained_eta_encoder is not None:
            self.checkpoint_eta_encoder = torch.load(pretrained_eta_encoder, map_location=self.device)
            self.eta_encoder.load_state_dict(self.checkpoint_eta_encoder["eta_encoder"])

        # save models to device
        self.beta_encoder.to(self.device)
        self.theta_encoder.to(self.device)
        self.eta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.patchifier.to(self.device)
        self.attn_module.to(self.device)
        self.vgg.to(self.device)
        self.start_epoch = 0

    def load_dataset(self, dataset_dirs, contrasts, orientations, batch_size):
        train_dataset = HACA3Dataset(dataset_dirs, contrasts, orientations, 'train')
        valid_dataset = HACA3Dataset(dataset_dirs, contrasts, orientations, "valid")
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def initialize_training(self, out_dir, lr):
        self.out_dir = out_dir
        mkdir_p(self.out_dir)
        mkdir_p(os.path.join(out_dir, 'training_results'))
        mkdir_p(os.path.join(out_dir, 'training_models'))

        # define losses
        self.kld_loss = KLDivergenceLoss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.patch_nce_loss = PatchNCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.optim_beta_encoder = torch.optim.Adam(self.beta_encoder.parameters(), lr=lr)
        self.optim_theta_encoder = torch.optim.Adam(self.theta_encoder.parameters(), lr=lr)
        self.optim_eta_encoder = torch.optim.Adam(self.eta_encoder.parameters(), lr=lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.optim_patchifier = torch.optim.Adam(self.patchifier.parameters(), lr=lr)
        self.optim_attn_module = torch.optim.Adam(self.attn_module.parameters(), lr=lr)

        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optim_beta_encoder.load_state_dict(self.checkpoint['optim_beta_encoder'])
            self.optim_theta_encoder.load_state_dict(self.checkpoint['optim_theta_encoder'])
            self.optim_decoder.load_state_dict(self.checkpoint['optim_decoder'])
            self.optim_patchifier.load_state_dict(self.checkpoint['optim_patchifier'])
            self.optim_attn_module.load_state_dict(self.checkpoint['optim_attn_module'])
        self.start_epoch += 1
        if self.checkpoint_eta_encoder is not None:
            self.optim_eta_encoder.load_state_dict(self.checkpoint_eta_encoder['optim_eta_encoder'])

    def channel_aggregation(self, beta):
        """Combine multiple channels of one-hot encoded beta into one.
        * INPUTS:
             - beta (Tensor): (batch_size, self.beta_dim, img_dim, img_dim)
        * OUTPUTS:
            - beta (Tensor): (batch_size, 1, img_dim, img_dim)
        """
        batch_size = beta.shape[0]
        img_dim = beta.shape[3]
        value_tensor = (torch.arange(0, self.beta_dim) * 1.0).to(self.device)
        value_tensor = value_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, img_dim, img_dim)
        beta = beta * value_tensor.detach()
        beta = beta.sum(1, keepdim=True)
        return beta

    def cal_beta(self, imgs):
        """Calculate betas from input images.
        * INPUTS
            - imgs (list): (num_mri_contrasts, ). Element shape (batch_size, 1, img_dim, img_dim).
        * OUTPUTS
            - logits (list): (num_mri_contrasts, ). Element shape (batch_size, self.beta_dim, img_dim, img_dim).
            - betas (list): Same shape as logits. One-hot encoded variables for anatomical features.
        """
        logits, betas = [], []
        for img in imgs:
            logit = self.beta_encoder(img)
            beta = self.channel_aggregation(reparameterize_logit(logit))
            mask = ((img > 1e-6) * 1.0).detach()
            beta = beta * mask
            logit = logit * mask
            betas.append(beta)
            logits.append(logit)
        return logits, betas

    def cal_theta(self, imgs):
        """Calculate thetas from input images.
        * INPUTS
            - imgs (list): (num_mri_contrasts, ). Element shape (batch_size, 1, img_dim, img_dim).
        * OUTPUTS
            - thetas (list): (num_mri_contrast, ). Element shape (batch_size, self.theta_dim, 1, 1).
            - mus (list): same as thetas.
            - logvars (list): same as thetas.
        """
        if isinstance(imgs, list):
            thetas, mus, logvars = [], [], []
            for img in imgs:
                mu, logvar = self.theta_encoder(img)
                theta = torch.randn(mu.size()).to(self.device) * torch.sqrt(torch.exp(logvar)) + mu
                thetas.append(theta)
                mus.append(mu)
                logvars.append(logvar)
            return thetas, mus, logvars
        else:
            mu, logvar = self.theta_encoder(imgs)
            theta = torch.randn(mu.size()).to(self.device) * torch.sqrt(torch.exp(logvar)) + mu
            return theta, mu, logvar

    def cal_eta(self, imgs):
        """
        Calculate etas from input images
        :param imgs: list
                    (num_mri_contrasts, ). Element shape (batch_size, 1, img_dim, img_dim)
        :return etas: list
                    (num_mri_contrasts, ). Element shape (batch_size, self.eta_dim, 1, 1)
        """
        if isinstance(imgs, list):
            batch_size = imgs[0].shape[0]
            etas = []
            for img in imgs:
                eta = self.eta_encoder(img).view(batch_size, self.eta_dim, 1, 1)
                etas.append(eta)
            return etas
        else:
            batch_size = imgs.shape[0]
            eta = self.eta_encoder(imgs).view(batch_size, self.eta_dim, 1, 1)
            return eta

    def decode(self, logits, target_theta, query, keys, available_contrast_label, contrast_dropout=True,
               selected_contrast_ids=None):
        """
        :param contrast_dropout:
        :param logits:
        :param target_theta:
        :param query:
        :param keys:
        :param available_contrast_label: torch.Tensor
            Shape (batch_size, num_contrasts).
        :return:
        """
        num_contrasts = len(logits)
        batch_size = logits[0].shape[0]
        img_dim = logits[0].shape[-1]

        # (batch_size, self.beta_dim, num_contrasts, img_dim, img_dim)
        logits_combined = torch.stack(logits, dim=-1).permute(0, 1, 4, 2, 3)
        logits_combined = logits_combined.view(batch_size, self.beta_dim, num_contrasts, img_dim * img_dim)
        contrast_dropout_val = None
        # v: (batch_size, self.beta_dim, img_dim*img_dim, num_contrasts)
        v = logits_combined.permute(0, 1, 3, 2)
        # k: (batch_size, self.theta_dim+self.eta_dim, 1, num_contrasts)
        k = torch.cat(keys, dim=-1)
        # q: (batch_size, self.theta_dim+self.eta_dim, 1)
        q = query.view(batch_size, self.theta_dim + self.eta_dim, 1)
        if contrast_dropout:
            contrast_dropout_val = dropout_contrasts(available_contrast_label, selected_contrast_ids)
        else:
            contrast_dropout_val = (available_contrast_label == 0) * 1e5
        logit_fusion, attn = self.attn_module(q, k, v, contrast_dropout_val.to(self.device))
        beta_fusion = self.channel_aggregation(reparameterize_logit(logit_fusion))
        beta_fusion = beta_fusion * ((logits[0][:, [0], ...] != 0) * 1.0)
        combined_img = torch.cat([beta_fusion, target_theta.repeat(1, 1, img_dim, img_dim)], dim=1)
        rec_img = self.decoder(combined_img, condition=None)

        return rec_img, attn, beta_fusion

    def cal_cycle_loss(self, theta_rec, theta_ref, eta_rec, eta_ref, beta_rec, beta_ref, is_train=True):
        theta_loss = self.l1_loss(theta_rec, theta_ref).mean()
        eta_loss = self.l1_loss(eta_rec, eta_ref).mean()
        beta_loss = self.l1_loss(beta_rec, beta_ref).mean()

        cycle_loss = theta_loss + eta_loss + 8e-2 * beta_loss
        if is_train:
            self.optim_beta_encoder.zero_grad()
            self.optim_theta_encoder.zero_grad()
            self.optim_patchifier.zero_grad()
            self.optim_attn_module.zero_grad()
            self.optim_decoder.zero_grad()
            (1e-2 * cycle_loss).backward()
            self.optim_beta_encoder.step()
            self.optim_theta_encoder.step()
            self.optim_decoder.step()
            self.optim_patchifier.step()
            self.optim_attn_module.step()
        loss = {
            'theta_cyc': theta_loss.item(),
            'eta_cyc': eta_loss.item(),
            'beta_cyc': beta_loss.item()
        }

        return loss

    def cal_loss(self, rec_img, ref_img, mu, logvar, betas, source_imgs, available_contrast_label,
                 is_train=True):
        """
        Calculate losses for intra-site paired image-to-image translation.
        :param rec_img:
        :param ref_img:
        :param mu:
        :param logvar:
        :param betas:
        :param source_imgs:
        :param available_contrast_label:
        :param is_train:
        :return:
        """
        batch_size = rec_img.shape[0]
        num_contrasts = available_contrast_label.shape[1]
        img_dim = rec_img.shape[-1]
        nce_loss = 0.0

        # 1. reconstruction loss and perceptual loss
        rec_loss = self.l1_loss(rec_img, ref_img).mean()
        rec_features = self.vgg(rec_img.repeat(1, 3, 1, 1))
        ref_features = self.vgg(ref_img.repeat(1, 3, 1, 1))
        per_loss = self.l1_loss(rec_features, ref_features).mean()

        # 2.3 KLD loss
        kld_loss = self.kld_loss(mu, logvar).mean()

        # 2.4 beta nce loss
        # availabel_contrast_per_batch: list. Length: batch_size.
        betas_combined = torch.stack(betas, dim=-1)  # (batch_size, self.beta_dim, img_dim, img_dim, num_contrasts)
        source_imgs_combined = torch.stack(source_imgs, dim=-1)  # (batch_size, 1, img_dim, img_dim, num_contrasts)
        query_contrast_ids, different_contrast_ids = [], []
        available_contrast_per_batch = [list(available_contrast_label)[i].nonzero(as_tuple=True)[0] for i in
                                        range(batch_size)]
        for subj_id in range(batch_size):
            current_contrast_ids = random.sample(set(available_contrast_per_batch[subj_id]), 2)
            query_contrast_ids.append(current_contrast_ids[0])
            different_contrast_ids.append(current_contrast_ids[1])
        patches_query = self.patchifier(
            torch.cat(
                [betas_combined[[subj_id], :, :, :, query_contrast_ids[subj_id]] for subj_id in range(batch_size)],
                dim=0)
        ).view(batch_size, 128, -1)
        patches_positive = self.patchifier(
            torch.cat(
                [betas_combined[[subj_id], :, :, :, different_contrast_ids[subj_id]] for subj_id in range(batch_size)],
                dim=0)
        ).view(batch_size, 128, -1)
        patches_negative = torch.cat([
            self.patchifier(
                torch.cat([source_imgs_combined[[subj_id], :, :, :, query_contrast_ids[subj_id]] for subj_id in
                           range(batch_size)], dim=0)
            ).view(batch_size, 128, -1),
            self.patchifier(
                torch.cat([source_imgs_combined[[subj_id], :, :, :, different_contrast_ids[subj_id]] for subj_id in
                           range(batch_size)], dim=0)
            ).view(batch_size, 128, -1),
            self.patchifier(
                torch.cat(
                    [betas_combined[[subj_id], :, :, :, query_contrast_ids[subj_id]] for subj_id in range(batch_size)],
                    dim=0)
            ).view(batch_size, 128, -1)[:, :, torch.randperm(36)],
            self.patchifier(
                torch.cat(
                    [betas_combined[[subj_id], :, :, :, query_contrast_ids[subj_id]] for subj_id in range(batch_size)],
                    dim=0)
            ).view(batch_size, 128, -1)[torch.randperm(batch_size), :, :],
        ], dim=-1)
        nce_loss = self.patch_nce_loss(patches_query, patches_positive.detach(),
                                       patches_negative.detach()).mean()

        # combine losses
        total_loss = 10 * rec_loss + per_loss + 2e-6 * kld_loss + 5e-2 * nce_loss
        if is_train:
            self.optim_beta_encoder.zero_grad()
            self.optim_theta_encoder.zero_grad()
            self.optim_patchifier.zero_grad()
            self.optim_attn_module.zero_grad()
            self.optim_decoder.zero_grad()
            total_loss.backward()
            self.optim_beta_encoder.step()
            self.optim_theta_encoder.step()
            self.optim_decoder.step()
            self.optim_patchifier.step()
            self.optim_attn_module.step()
        loss = {
            'rec_loss': rec_loss.item(),
            'per_loss': per_loss.item(),
            'kld_loss': kld_loss.item(),
            'nce_loss': nce_loss.item(),
            'total_loss': total_loss.item()
        }
        return loss

    def mixing_augmented_images(self, img_dicts):
        """
        Mixing original images and augmented images for encoding.
        :param img_dicts:
        :return:
        """
        num_contrasts = len(img_dicts)
        num_contrasts_with_artifacts = np.random.permutation(num_contrasts)[0]
        artifact_contrast_ids = sorted(
            np.random.choice(range(num_contrasts), num_contrasts_with_artifacts, replace=False))
        source_images = []
        for i in range(num_contrasts):
            if i in artifact_contrast_ids:
                source_images.append(img_dicts[i]['img_artifact_augmented'].to(self.device))
            else:
                source_images.append(img_dicts[i]['img'].to(self.device))

        return source_images

    def train_harmonization(self, epochs):
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_loader = tqdm(self.train_loader)
            self.eta_encoder.eval()
            self.beta_encoder.train()
            self.theta_encoder.train()
            self.decoder.train()
            self.patchifier.train()
            self.attn_module.train()
            train_loss_sum = 0.0
            num_train_imgs = 0
            for batch_id, img_dicts in enumerate(self.train_loader):
                source_imgs = self.mixing_augmented_images(img_dicts)
                target_imgs = [img_dict['img_prime'] for img_dict in img_dicts]
                reference_imgs = [img_dict['img'] for img_dict in img_dicts]
                available_contrast_label = torch.stack([img_dict['exists'] for img_dict in img_dicts],
                                                       dim=-1)  # (batch_size, num_contrasts)
                target_img_selected, reference_img_selected, selected_contrast_ids = selecting_available_contrasts(
                    target_imgs, reference_imgs, available_contrast_label)
                target_img_selected = target_img_selected.to(self.device)
                reference_img_selected = reference_img_selected.to(self.device)
                batch_size = source_imgs[0].shape[0]
                # print(target_img_selected.shape, reference_img_selected.shape)

                num_contrasts = len(source_imgs)
                # 1. intra-site image-to-image translation
                logits, betas = self.cal_beta(source_imgs)
                thetas_source, _, _ = self.cal_theta(source_imgs)
                etas_source = self.cal_eta(source_imgs)
                theta_target, mu_target, logvar_target = self.cal_theta(target_img_selected)
                eta_target = self.cal_eta(target_img_selected)
                query = torch.cat([theta_target, eta_target], dim=1)
                keys = [torch.cat([theta, eta], dim=1) for (theta, eta) in zip(thetas_source, etas_source)]
                rec_img, attn, beta_fusion = self.decode(logits, theta_target, query, keys,
                                                         available_contrast_label,
                                                         selected_contrast_ids=None if torch.rand(
                                                             (1,)) > 0.8 else selected_contrast_ids)
                loss = self.cal_loss(rec_img, reference_img_selected, mu_target, logvar_target, betas,
                                     source_imgs, available_contrast_label)

                if batch_id % 100 - 1 == 0:
                    # save images for intra-site I2I
                    # reference images and synthetic images
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_source_image'
                    self.save_image(source_imgs + [target_img_selected], file_prefix)
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_syn_image'
                    self.save_image([rec_img] + [reference_img_selected], file_prefix)
                    # save logits and betas
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_beta'
                    self.save_image(source_imgs + betas + [beta_fusion], file_prefix)
                    # save attns
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_attention'
                    self.save_image(source_imgs + [target_img_selected] + \
                                    [attn[:, [i], ...] for i in range(num_contrasts)], file_prefix)

                if batch_id > 1000 or epoch > 1:
                    # 2. inter-site image-to-image translation
                    random_idx = torch.randperm(batch_size)
                    target_img_shuffled = target_img_selected[random_idx, ...]
                    reference_img_shuffled = reference_img_selected[random_idx, ...]
                    logits, betas = self.cal_beta(source_imgs)
                    thetas_source, _, _ = self.cal_theta(source_imgs)
                    etas_source = self.cal_eta(source_imgs)
                    theta_target, mu_target, logvar_target = self.cal_theta(target_img_shuffled)
                    eta_target = self.cal_eta(target_img_shuffled)
                    query = torch.cat([theta_target, eta_target], dim=1)
                    keys = [torch.cat([theta, eta], dim=1) for (theta, eta) in zip(thetas_source, etas_source)]
                    rec_img, attn, beta_fusion = self.decode(logits, theta_target, query, keys,
                                                             available_contrast_label,
                                                             contrast_dropout=False)
                    theta_recon, _ = self.theta_encoder(rec_img)
                    eta_recon = self.eta_encoder(rec_img)
                    beta_recon = self.channel_aggregation(reparameterize_logit(rec_img))
                    loss_cyc = self.cal_cycle_loss(theta_recon, theta_target.detach(),
                                                   eta_recon, eta_target.detach(),
                                                   beta_recon, beta_fusion.detach(),
                                                   is_train=True)

                    # 3. save training results for inter-site I2I
                    if batch_id % 100 - 1 == 0:
                        # reference images and synthetic images
                        file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_source_image_inter'
                        self.save_image(source_imgs + [target_img_shuffled], file_prefix)
                        file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_syn_image_inter'
                        self.save_image([rec_img] + [target_img_shuffled], file_prefix)
                        # save logits and betas
                        file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_beta_inter'
                        self.save_image(source_imgs + betas + [beta_fusion], file_prefix)
                        # save attns
                        file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_attention_inter'
                        self.save_image(source_imgs + [target_img_shuffled] + \
                                        [attn[:, [i], ...] for i in range(num_contrasts)], file_prefix)

                train_loss_sum += loss['total_loss'] * batch_size
                num_train_imgs += batch_size
                if batch_id > 1000 or epoch > 1:
                    self.train_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {loss["rec_loss"]:.3f}; '
                                                       f'per: {loss["per_loss"]:.3f}; '
                                                       f'kld: {loss["kld_loss"]:.3f}; '
                                                       f'nce: {loss["nce_loss"]:.3f}; '
                                                       f'theta_c: {loss_cyc["theta_cyc"]:.3f}; '
                                                       f'eta_c: {loss_cyc["eta_cyc"]:.3f}; '
                                                       f'beta_c: {loss_cyc["beta_cyc"]:.3f}; '
                                                       f'avg_trn: {train_loss_sum / num_train_imgs:.3f}; '))
                else:
                    self.train_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {loss["rec_loss"]:.3f}; '
                                                       f'per: {loss["per_loss"]:.3f}; '
                                                       f'kld: {loss["kld_loss"]:.3f}; '
                                                       f'nce: {loss["nce_loss"]:.3f}; '
                                                       f'avg_trn: {train_loss_sum / num_train_imgs:.3f}; '))

                # save models
                if batch_id % 2000 == 0:
                    file_name = os.path.join(self.out_dir, 'training_models',
                                             f'epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}.pt')
                    self.save_model(file_name, epoch)

    def harmonization(self, source_imgs, target_imgs, target_contrasts, contrast_dropout, out_dir, prefix,
                      recon_orientation, header, affine, num_batches=4, save_intermediate=False, norm_val=1000.0):
        """
        Harmonize source images to target images.
        :param num_batches:
        :param source_imgs:
        :param target_imgs:
        :param contrast_dropout:
        :param out_dir:
        :param prefix:
        :param recon_orientation:
        :param header:
        :param affine:
        :return:
        """
        contrast_names = ['T1', 'T2', 'PD', 'FLAIR']
        mkdir_p(out_dir)
        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()
            # 1. calculate beta, theta, eta from source images
            logits, betas, keys, masks = [], [], [], []
            for source_img in source_imgs:
                source_img = source_img.unsqueeze(1)
                source_img_batches = divide_into_batches(source_img, num_batches)
                mask_tmp, logit_tmp, beta_tmp, key_tmp = [], [], [], []
                for source_img_batch in source_img_batches:
                    curr_batch_size = source_img_batch.shape[0]
                    source_img_batch = source_img_batch.to(self.device)
                    mask = (source_img_batch > 1e-6) * 1.0
                    logit = self.beta_encoder(source_img_batch)
                    beta = self.channel_aggregation(reparameterize_logit(logit))
                    theta_source, _ = self.theta_encoder(source_img_batch)
                    eta_source = self.eta_encoder(source_img_batch).view(curr_batch_size, self.eta_dim, 1, 1)
                    mask_tmp.append(mask)
                    logit_tmp.append(logit)
                    beta_tmp.append(beta)
                    key_tmp.append(torch.cat([theta_source, eta_source], dim=1))
                masks.append(torch.cat(mask_tmp, dim=0))
                logits.append(torch.cat(logit_tmp, dim=0))
                betas.append(torch.cat(beta_tmp, dim=0))
                keys.append(torch.cat(key_tmp, dim=0))

            betas = apply_beta_mask(masks, betas)
            logits = apply_beta_mask(masks, logits)

            # 2. calculate theta, eta for target images
            queries, thetas_target, etas_target = [], [], []
            for target_img in target_imgs:
                target_img = target_img.to(self.device).unsqueeze(1)  # (num_slices, 1, 288, 288)
                theta_target, _ = self.theta_encoder(target_img)
                theta_target = theta_target.mean(dim=0, keepdim=True)
                eta_target = self.eta_encoder(target_img).mean(dim=0, keepdim=True).view(1, self.eta_dim, 1, 1)
                thetas_target.append(theta_target)
                etas_target.append(etas_target)
                queries.append(torch.cat([theta_target, eta_target], dim=1))

            # 3. save encoded variables
            if save_intermediate:
                if recon_orientation == 'axial':
                    # 3a. source images
                    for contrast_id, source_img in enumerate(source_imgs):
                        if contrast_dropout[contrast_id] <= 1.0:
                            img_save = np.array(source_img.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2)).astype(
                                np.float32)
                            img_save = img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]
                            img_save = nib.Nifti1Image(img_save, affine, header)
                            file_name = os.path.join(out_dir, f'{prefix}_source_{contrast_names[contrast_id]}.nii.gz')
                            nib.save(img_save, file_name)
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1).cpu().squeeze().permute(1, 2, 0, 3).permute(1, 0, 2, 3)
                    img_save = nib.Nifti1Image(np.array(beta)[112 - 96:112 + 96, :, 112 - 96:112 + 96, :],
                                               affine, header)
                    file_name = os.path.join(out_dir, f'{prefix}_beta_axial.nii.gz')
                    nib.save(img_save, file_name)

            # 4. decoding
            for theta_target, query, target_contrast in zip(thetas_target, queries, target_contrasts):
                theta_target = theta_target.to(self.device)
                rec_img, beta_fusion, attn = [], [], []
                for batch_id in range(num_batches):
                    keys_tmp = [divide_into_batches(ks, num_batches)[batch_id] for ks in keys]
                    logits_tmp = [divide_into_batches(ls, num_batches)[batch_id] for ls in logits]
                    curr_batch_size = keys_tmp[0].shape[0]
                    curr_query = query.view(1, self.theta_dim + self.eta_dim, 1).repeat(curr_batch_size, 1, 1)
                    k = torch.cat(keys_tmp, dim=-1).view(curr_batch_size, self.theta_dim + self.eta_dim, 1, 4)
                    v = torch.stack(logits_tmp, dim=-1).view(curr_batch_size, self.beta_dim, 224 * 224, 4)
                    logit_fusion_tmp, attn_tmp = self.attn_module(curr_query, k, v, contrast_dropout.to(self.device))
                    beta_fusion_tmp = self.channel_aggregation(reparameterize_logit(logit_fusion_tmp))
                    beta_fusion_tmp = ((beta_fusion_tmp * (logits_tmp[0][:, [0], ...] != 0)) * 1.0)
                    combined_map = torch.cat([beta_fusion_tmp, theta_target.repeat(curr_batch_size, 1, 224, 224)],
                                             dim=1)
                    rec_img_tmp = self.decoder(combined_map)
                    rec_img.append(rec_img_tmp)
                    beta_fusion.append(beta_fusion_tmp)
                    attn.append(attn_tmp)

                rec_img = apply_beta_mask(masks, [torch.cat(rec_img, dim=0)])[0]
                beta_fusion = torch.cat(beta_fusion, dim=0)
                attn = torch.cat(attn, dim=0)

                # 5. save synthetic images
                if save_intermediate:
                    # 5a. beta fusion
                    if recon_orientation == 'axial':
                        img_save = np.array(beta_fusion.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96],
                                                   affine, header)
                        file_name = os.path.join(out_dir, f'{prefix}_beta_fusion_to_{target_contrast}.nii.gz')
                        nib.save(img_save, file_name)

                    # 5b. attention
                    if recon_orientation == 'axial':
                        img_save = np.array(attn.cpu().permute(2, 3, 0, 1).permute(1, 0, 2, 3))
                        img_save = nib.Nifti1Image(
                            img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96],
                            affine, header)
                        file_name = os.path.join(out_dir, f'{prefix}_attention_to_{target_contrast}.nii.gz')
                        nib.save(img_save, file_name)

                # 5c. synthetic image
                if recon_orientation == "axial":
                    img_save = np.array(rec_img.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                elif recon_orientation == "coronal":
                    img_save = np.array(rec_img.cpu().squeeze().permute(0, 2, 1).flip(2).permute(1, 0, 2))
                else:
                    img_save = np.array(rec_img.cpu().squeeze().permute(2, 0, 1).flip(2).permute(1, 0, 2))
                img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96] * norm_val,
                                           affine, header)
                file_name = os.path.join(out_dir,
                                         f'{prefix}_harmonized_to_{target_contrast}_{recon_orientation}.nii.gz')
                nib.save(img_save, file_name)

    def encode(self, img, out_dir, prefix, affine, header):
        """
        Learn anatomy, artifact, and contrast representations using HACA3.
        :param img:
        :param out_dir:
        :param prefix:
        :param affine:
        :param header:
        :return:
        """
        mkdir_p(out_dir)
        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()

            # 1. anatomical representation: beta
            beta = []
            num_slices = img.shape[0]
            for s in range(num_slices):
                img_slice = img[[s], ...].to(self.device).unsqueeze(1)  # (1, 1, 288, 288)
                beta_slice = self.channel_aggregation(reparameterize_logit(self.beta_encoder(img_slice)))
                mask = (img_slice > 1e-6) * 1.0
                beta.append((beta_slice * mask).cpu())
            beta = torch.cat(beta, dim=0)

            # 2. contrast: theta, artifact: eta
            img_center = img[120:145, ...].unsqueeze(1)
            img_center = img_center.to(self.device)
            theta, _ = self.theta_encoder(img_center)
            theta = theta.mean(dim=0, keepdim=True)
            eta = self.eta_encoder(img_center).mean(dim=0, keepdim=True).view(1, self.eta_dim, 1, 1)

        # 3. save encoded variables
        # 3a. source image
        img_save = np.array(img.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2)).astype(np.float32)
        img_save = img_save[144 - 120:144 + 121, 144 - 143:144 + 143, 144 - 120:144 + 121]
        img_save = nib.Nifti1Image(img_save * 4000.0, affine, header)
        file_name = os.path.join(out_dir, f'{prefix}_image.nii.gz')
        nib.save(img_save, file_name)
        # 3b. beta images
        beta = beta.squeeze().permute(1, 2, 0).permute(1, 0, 2)
        img_save = nib.Nifti1Image(np.array(beta)[144 - 120:144 + 121, 144 - 143:144 + 143, 144 - 120:144 + 121],
                                   affine, header)
        file_name = os.path.join(out_dir, f'{prefix}_beta_axial.nii.gz')
        nib.save(img_save, file_name)
        # 3c. theta
        file_name = os.path.join(out_dir, f'{prefix}_theta.txt')
        theta_save = theta[0, :, 0, 0].cpu().numpy()
        np.savetxt(file_name, np.expand_dims(theta_save, axis=0), delimiter=",", fmt="%5f")
        # 3d. eta
        file_name = os.path.join(out_dir, f'{prefix}_eta.txt')
        eta_save = eta[0, :, 0, 0].cpu().numpy()
        np.savetxt(file_name, np.expand_dims(eta_save, axis=0), delimiter=",", fmt="%5f")

    def normalize_intensity(self, image):
        p99 = np.percentile(image.flatten(), 99)
        image = np.clip(image, a_min=0.0, a_max=p99)
        image = image / p99
        return image, p99

    def combine_images(self, img_dirs, out_dir, prefix, target_contrast, pretrained_fusion=None):
        # obtain images
        imgs = []
        for img_dir in img_dirs:
            img_pad = torch.zeros((224, 224, 224))
            img_file = nib.load(img_dir)
            img_vol, norm_val = self.normalize_intensity(torch.from_numpy(img_file.get_fdata().astype(np.float32)))
            img_pad[112 - 96:112 + 96, :, 112 - 96:112 + 96] = img_vol
            img_hdr = img_file.header
            img_affine = img_file.affine
            imgs.append(img_pad.numpy())
        if pretrained_fusion is not None:
            self.checkpoint = torch.load(pretrained_fusion, map_location=self.device)
            self.fusion_net = FusionNet(in_ch=3, out_ch=1)
            self.fusion_net.load_state_dict(self.checkpoint['fusion_net'])
            self.fusion_net.to(self.device)
            img = torch.cat([ToTensor()(im).permute(2, 1, 0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) for im in imgs],
                            dim=1)
            img = img.to(self.device)
            img_fusion = self.fusion_net(img).squeeze().detach().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
        else:
            # calculate median
            img_cat = np.stack(imgs, axis=-1)
            img_fusion = np.median(img_cat, axis=-1)

        # save fusion_image
        img_save = img_fusion[112 - 96:112 + 96, :, 112 - 96:112 + 96] * norm_val
        img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
        file_name = os.path.join(out_dir, f'{prefix}_harmonized_to_{target_contrast}_fusion.nii.gz')
        nib.save(img_save, file_name)

    def save_image(self, imgs, file_prefix):
        num_channels = imgs[0].shape[1]
        if num_channels == 1:
            img_save = torch.cat([img[:4, [0], :, :].cpu() for img in imgs], dim=0)
            grid = utils.make_grid(tensor=img_save, nrow=4, normalize=False, range=(0, 1))
            grid = grid.detach().numpy()[0, ...]
            affine = np.eye(4)
            img_save = nib.Nifti1Image(grid.transpose(1, 0), affine)
        else:
            imgs_save = []
            for ch in range(num_channels):
                img_save = torch.cat([img[:4, [ch], :, :].cpu() for img in imgs], dim=0)
                grid = utils.make_grid(tensor=img_save, nrow=4, normalize=False, range=(0, 1))
                grid = grid.detach().numpy()[0, ...]
                imgs_save.append(np.expand_dims(grid, axis=1))
            grid = np.concatenate(imgs_save, axis=1)
            affine = np.eye(4)
            img_save = nib.Nifti1Image(grid.transpose(2, 0, 1), affine)
        file_name = os.path.join(self.out_dir, 'training_results', f'{file_prefix}.nii.gz')
        nib.save(img_save, file_name)

    def save_model(self, file_name, epoch):
        state = {'epoch': epoch,
                 'beta_encoder': self.beta_encoder.state_dict(),
                 'theta_encoder': self.theta_encoder.state_dict(),
                 'eta_encoder': self.eta_encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'patchifier': self.patchifier.state_dict(),
                 'attn_module': self.attn_module.state_dict(),
                 'optim_beta_encoder': self.optim_beta_encoder.state_dict(),
                 'optim_theta_encoder': self.optim_theta_encoder.state_dict(),
                 'optim_decoder': self.optim_decoder.state_dict(),
                 'optim_patchifier': self.optim_patchifier.state_dict(),
                 'optim_attn_module': self.optim_attn_module.state_dict()}
        torch.save(obj=state, f=file_name)
