import os
import torch
from torch import nn
from torch.nn import functional as F
import errno
import nibabel as nib
from torchvision import utils
import torchvision.models as models
import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def reparameterize_logit(logit):
    import warnings
    warnings.filterwarnings('ignore', message='.*Mixed memory format inputs detected.*')
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta


def save_image(images, file_name):
    image_save = torch.cat([image[:4, [0], ...].cpu() for image in images], dim=0)
    image_save = utils.make_grid(tensor=image_save, nrow=4, normalize=False, range=(0, 1)).detach().numpy()[0, ...]
    image_save = nib.Nifti1Image(image_save.transpose(1, 0), np.eye(4))
    nib.save(image_save, file_name)


def dropout_contrasts(available_contrast_id, contrast_id_to_drop=None):
    """
    Randomly dropout contrasts for HACA3 training.

    ==INPUTS==j
    * available_contrast_id: torch.Tensor (batch_size, num_contrasts)
        Indicates the availability of each MR contrast. 1: if available, 0: if unavailable.

    * contrast_id_to_drop: torch.Tensor (batch_size, num_contrasts)
        If provided, indicates the contrast indexes forced to drop. Default: None

    ==OUTPUTS==
    * contrast_id_after_dropout: torch.Tensor (batch_size, num_contrasts)
        Some available contrasts will be randomly dropped out (as if they are unavailable).
        However, each sample will have at least one contrast available.
    """
    batch_size = available_contrast_id.shape[0]
    if contrast_id_to_drop is not None:
        available_contrast_id = available_contrast_id - contrast_id_to_drop
    contrast_id_after_dropout = available_contrast_id.clone()
    for i in range(batch_size):
        available_contrast_ids_per_subject = (available_contrast_id[i] == 1).nonzero(as_tuple=False).squeeze(1)
        num_available_contrasts = available_contrast_ids_per_subject.numel()
        if num_available_contrasts > 1:
            num_contrast_to_drop = torch.randperm(num_available_contrasts - 1)[0]
            contrast_ids_to_drop = torch.randperm(num_available_contrasts)[:num_contrast_to_drop]
            contrast_ids_to_drop = available_contrast_ids_per_subject[contrast_ids_to_drop]
            contrast_id_after_dropout[i, contrast_ids_to_drop] = 0.0
    return contrast_id_after_dropout


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.children())[:13]).eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, query_feature, positive_feature, negative_feature):
        B, C, N = query_feature.shape

        l_positive = (query_feature * positive_feature).sum(dim=1)[:, :, None]
        l_negative = torch.bmm(query_feature.permute(0, 2, 1), negative_feature)

        logits = torch.cat((l_positive, l_negative), dim=2) / self.temperature

        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * N, dtype=torch.long).to(query_feature.device)
        return self.ce_loss(predictions, targets).mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        return kld_loss


def divide_into_batches(in_tensor, num_batches):
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image, image_dim=256):
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded

def zero_pad2d(image, image_dim=256):
    [n_row, n_col] = image.shape
    image_padded = np.zeros((image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2] = image
    return image_padded


def crop(image, n_row, n_col, n_slc):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]

def crop2d(image, n_row, n_col):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2]
