import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG16Weights
import math


class FusionNet(nn.Module):
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
            nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        # return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights=VGG16Weights.IMAGENET1K_V1)
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.module = nn.Sequential()

        for x in range(9):
            self.module.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.module(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))

    def forward(self, in_tensor, condition=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)


class EtaEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, 1, 2),  # (*, 16, 224, 224)
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, 3, 1, 1),  # (*, 64, 224, 224)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.seq = nn.Sequential(
            nn.Conv2d(64 + in_ch, 32, 32, 32, 0),  # (*, 32, 7, 7)
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 7, 7, 0))

    def forward(self, x):
        return self.seq(torch.cat([self.in_conv(x), x], dim=1))


class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 17, 9, 4),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),  # (*, 32, 28, 28)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),  # (*, 64, 14, 14)
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1))  # (* 64, 7, 7)
        self.mean_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))

    def forward(self, x):
        M = self.conv(x)
        mu = self.mean_conv(M)
        logvar = self.logvar_conv(M)
        return mu, logvar


class Patchifier(nn.Module):
    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, 64, 32, 32, 0),  # (*, 64, 7, 7)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_ch, 1, 1, 0))

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, 16, 4, 2, 1),  # (*, 16, 112, 112)
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, 4, 2, 1),  # (*, 64, 56, 56)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 16, 4, 2, 1),  # (*, 16, 28, 28)
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, out_ch, 4, 2, 1)  # (*, out_ch, 14, 14)
        )

    def forward(self, in_tensor):
        return self.seq(in_tensor)


class AttentionModule(nn.Module):
    def __init__(self, dim, v_dim=5):
        super().__init__()
        self.dim = dim
        self.v_dim = v_dim
        self.q_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))
        self.k_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))

        self.scale = self.dim ** (-0.5)

    def forward(self, q, k, v, modality_dropout=None):
        """
        * INPUTS
            - q : torch.Tensor
                  (batch_size, feature_dim_q, num_q_patches=1)
            - k : torch.Tensor
                  (batch_size, feature_dim_k, num_k_patches=1, num_modalities)
            - v : torch.Tensor
                  (batch_size, v_dim, 288*288, num_modalities)
        """
        batch_size, feature_dim_q, num_q_patches = q.shape
        _, feature_dim_k, _, num_modalities = k.shape
        num_v_patches = v.shape[2]
        H = int(math.sqrt(num_v_patches))

        if feature_dim_q != self.dim or feature_dim_k != self.dim:
            raise ValueError

        q = q.unsqueeze(-1).permute(0, 2, 3, 1)  # (batch_size, num_q_patches, 1, feature_dim_q)
        k = k.permute(0, 2, 3, 1)  # (batch_size, num_k_patches, num_modalities, feature_dim_k)
        v = v.permute(0, 2, 3, 1)  # (batch_size, num_v_patches=288*288, num_modalities, v_dim)
        q = self.q_fc(q)
        k = self.k_fc(k).permute(0, 1, 3, 2)  # (batch_size, num_k_patches=1, feature_dim_k, num_modalities)

        dot_prod = (q @ k) * self.scale  # (batch_size, num_q_patches=1, 1, num_modalities)
        if num_v_patches % num_q_patches:
            raise ValueError
        else:
            interp_factor = int(math.sqrt(num_v_patches // num_q_patches))

        # (batch_size, H, W, num_modalities)
        H = int(math.sqrt(num_q_patches))
        dot_prod = dot_prod.view(batch_size, H, H, num_modalities)
        dot_prod = dot_prod.repeat(1, interp_factor, interp_factor, 1)
        if modality_dropout is not None:
            img_dim = dot_prod.shape[1]
            if len(modality_dropout.shape) == 1:
                dot_prod = dot_prod - modality_dropout.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, img_dim, img_dim, 1).detach()
            else:
                dot_prod = dot_prod - modality_dropout.unsqueeze(1).unsqueeze(1).repeat(1, img_dim, img_dim, 1).detach()
        attn = (dot_prod / 20.0).softmax(dim=-1)  # (batch_size, H, H, num_modalities)
        v = attn.view(batch_size, num_v_patches, 1, num_modalities) @ v  # (batch_size, num_v_patches, 1, v_dim)
        H = int(math.sqrt(num_v_patches))
        v = v.view(batch_size, H, H, self.v_dim).permute(0, 3, 1, 2)  # (batch_size, v_dim, H, H)
        attn = attn.view(batch_size, H, H, num_modalities).permute(0, 3, 1, 2)  # (batch_size, num_modalities, H, H)
        return v, attn
