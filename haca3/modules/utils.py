import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import random
from skimage.morphology import isotropic_closing


def selecting_available_contrasts(target_imgs, reference_imgs, available_contrasts_label):
    """
    Selecting from available contrasts
    :param reference_imgs:
    :param target_imgs:
    :param available_contrasts_label:
    :return:
    """
    subj_ids = available_contrasts_label.nonzero(as_tuple=True)[0]
    contrast_ids = available_contrasts_label.nonzero(as_tuple=True)[1]
    unique_subj_ids = list(torch.unique(subj_ids))
    selected_contrast_ids = []
    for subj_id in unique_subj_ids:
        selected_contrast_ids.append(random.choice(contrast_ids[subj_ids == subj_id]))
    target_img_combined = torch.cat(target_imgs, dim=1)
    reference_img_combined = torch.cat(reference_imgs, dim=1)

    return target_img_combined[unique_subj_ids, selected_contrast_ids, ...].unsqueeze(1), \
        reference_img_combined[unique_subj_ids, selected_contrast_ids, ...].unsqueeze(1), \
        selected_contrast_ids


def apply_beta_mask(masks, betas):
    mask = masks[0]
    for mask_tmp in masks:
        mask = mask * mask_tmp
    mask = isotropic_closing(np.array(mask.cpu()).astype(bool), radius=5)
    mask = torch.from_numpy(mask).to(betas[0].get_device())
    betas = [beta * mask for beta in betas]
    return betas


def dropout_contrasts(available_contrast_label, selected_contrast_ids=None):
    """
    Randomly dropout MRI contrasts for HACA3 training.
    :param selected_contrast_ids:
    :param available_contrast_label: torch.Tensor
        Shape (batch_size, num_contrasts)
    :return:
    """
    batch_size = available_contrast_label.shape[0]
    contrast_dropout_val = (available_contrast_label == 0) * 1e5
    for subj_id in range(batch_size):
        num_available_contrasts = int(available_contrast_label[subj_id, :].sum())
        num_dropped_out_contrasts = np.random.permutation(num_available_contrasts)[0]
        available_contrast_label_tmp = available_contrast_label[subj_id, :].nonzero(as_tuple=False)
        dropped_out_contrast_ids = sorted(
            np.random.choice([s.squeeze() for s in available_contrast_label_tmp],
                             num_dropped_out_contrasts,
                             replace=False))
        for i in dropped_out_contrast_ids:
            contrast_dropout_val[subj_id, i] = 1e5
        if selected_contrast_ids is not None:
            contrast_dropout_val[subj_id, selected_contrast_ids[subj_id]] = 1e5
    return contrast_dropout_val


def reparameterize_logit(logit):
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta


class TemperatureAnneal:
    def __init__(self, initial_temp=1.0, anneal_rate=0.0, min_temp=0.5, device=torch.device('cuda')):
        self.initial_temp = initial_temp
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp
        self.device = device

        self._temperature = self.initial_temp
        self.last_epoch = 0

    def get_temp(self):
        return self._temperature

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        current_temp = self.initial_temp * np.exp(-self.anneal_rate * self.last_epoch)
        # noinspection PyArgumentList
        self._temperature = torch.max(torch.FloatTensor([current_temp, self.min_temp]).to(self.device))

    def reset(self):
        self._temperature = self.initial_temp
        self.last_epoch = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def softmax(logits, temperature=1.0, dim=1):
    exps = torch.exp(logits / temperature)
    return exps / torch.sum(exps, dim=dim)


def gumbel(size, device=torch.device('cuda:1'), eps=1e-8):
    return -torch.log(-torch.log(torch.rand(size, device=device) + eps) + eps)


def sample_gumbel_softmax(logits, temperature=1.0, dim=1, device=torch.device('cuda:1')):
    return F.softmax((logits.to(device) + gumbel(logits.size(), device).to(device)) / temperature, dim=dim)


def gumbel_softmax(logits, temperature=1.0, dim=1, hard=True, is_prob=True, device=torch.device('cuda:1')):
    if is_prob:
        logits = torch.log(logits + 1e-8)
    soft_sample = sample_gumbel_softmax(logits, temperature, dim, device)
    if hard:
        hard_sample = create_one_hot(soft_sample, dim=dim)
        return (hard_sample - soft_sample).detach() + soft_sample
    else:
        return soft_sample


def entropy(in_tensor, marginalize=False):
    if not marginalize:
        b = F.softmax(in_tensor, dim=1) * F.log_softmax(in_tensor, dim=1)
        h = -1.0 * b.sum()
    else:
        b = F.softmax(in_tensor, dim=1).mean(0)
        h = -torch.sum(b * torch.log(b + 1e-6))
    return h


def marginal_cross_entropy(x, y):
    x = F.softmax(x, dim=1).mean(0)
    y = F.softmax(y, dim=1).mean(0)
    h = -torch.sum(x * torch.log(y + 1e-6))
    return h


def create_one_hot(soft_prob, dim):
    indices = torch.argmax(soft_prob, dim=dim)
    hard = F.one_hot(indices, soft_prob.size()[dim])
    new_axes = tuple(range(dim)) + (len(hard.shape) - 1,) + tuple(range(dim, len(hard.shape) - 1))
    return hard.permute(new_axes).float()


class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):
        kld = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        if self.reduction == 'mean':
            kld = kld.mean()
        return kld


class CosineDissimilarityLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean', eps=1e-6, scale=1.0):
        super(CosineDissimilarityLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.reduction = reduction
        self.scale = scale

    def forward(self, rec, trg):
        # noinspection PyTypeChecker
        loss: torch.Tensor = torch.max(F.cosine_similarity(rec.view(rec.size()[0], rec.size()[1], -1),
                                                           trg.view(trg.size()[0], trg.size()[1], -1),
                                                           dim=-1, eps=self.eps) - self.margin,
                                       torch.as_tensor(0.0, device=rec.device)) * self.scale
        return loss if self.reduction == 'none' else (torch.mean(loss) if self.reduction == 'mean'
                                                      else torch.sum(loss))


class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, patch_query, patch_positive, patch_negative, tau=0.1):
        """
        patch_query : torch.Tensor
            Shape(batch_size, feature_dim, num_pathes)
        """
        B, C, N = patch_query.shape
        N_negative = patch_negative.shape[2]

        # B * N * 1
        l_pos = (patch_query * patch_positive).sum(dim=1)[:, :, None]

        # B * N * N_negative
        l_neg = torch.bmm(patch_query.permute(0, 2, 1), patch_negative)

        # B * N * (N_negative + 1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau

        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * N, dtype=torch.long).to(patch_query.device)
        return self.ce_loss(predictions, targets)


# http://stackoverflow.com/a/22718321
def mkdir_p(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def divide_into_batches(x, num_batches):
    batch_size = x.shape[0] // num_batches
    remainder = x.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(x[current_start:current_end, ...])
        current_start = current_end

    return batches
