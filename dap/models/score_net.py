"""Score network for MNIST VP-SDE: Fourier time embedding, UNet backbone, ScoreNet wrapper."""

import numpy as np
import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier time embedding.

    Input:  t  (B,)
    Output: embedding  (B, embed_dim)
    """

    def __init__(self, embed_dim, scale=30.0, generator=None):
        super().__init__()
        assert embed_dim % 2 == 0
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2, generator=generator) * scale,
            requires_grad=False,
        )

    def forward(self, t):
        proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class _Dense(nn.Module):
    """Projects time embedding to (B, out_ch, 1, 1) spatial bias."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)[..., None, None]


class UNet(nn.Module):
    """4-level UNet score network for 1-channel 28×28 images (MNIST).

    Encoder: 4 conv layers with stride-2 downsampling.
    Decoder: 4 transposed-conv layers with skip connections.
    Time conditioning: additive Dense bias after each layer.
    Normalization: GroupNorm. Activation: Swish.

    Args:
        channels: feature map widths at each level. Default: [32, 64, 128, 256].
        temb_dim: Fourier time embedding dimension. Default: 256.
    """

    def __init__(self, channels=None, temb_dim=256):
        super().__init__()
        ch = channels or [32, 64, 128, 256]
        ted = temb_dim
        self.act = lambda x: x * torch.sigmoid(x)  # Swish

        # Encoder
        self.conv1 = nn.Conv2d(1, ch[0], 3, stride=1, bias=False)
        self.d1 = _Dense(ted, ch[0]);  self.gn1 = nn.GroupNorm(4,  ch[0])

        self.conv2 = nn.Conv2d(ch[0], ch[1], 3, stride=2, bias=False)
        self.d2 = _Dense(ted, ch[1]);  self.gn2 = nn.GroupNorm(32, ch[1])

        self.conv3 = nn.Conv2d(ch[1], ch[2], 3, stride=2, bias=False)
        self.d3 = _Dense(ted, ch[2]);  self.gn3 = nn.GroupNorm(32, ch[2])

        self.conv4 = nn.Conv2d(ch[2], ch[3], 3, stride=2, bias=False)
        self.d4 = _Dense(ted, ch[3]);  self.gn4 = nn.GroupNorm(32, ch[3])

        # Decoder
        self.tc4 = nn.ConvTranspose2d(ch[3],      ch[2], 3, stride=2, bias=False)
        self.d5 = _Dense(ted, ch[2]);  self.tgn4 = nn.GroupNorm(32, ch[2])

        self.tc3 = nn.ConvTranspose2d(ch[2]*2, ch[1], 3, stride=2, bias=False, output_padding=1)
        self.d6 = _Dense(ted, ch[1]);  self.tgn3 = nn.GroupNorm(32, ch[1])

        self.tc2 = nn.ConvTranspose2d(ch[1]*2, ch[0], 3, stride=2, bias=False, output_padding=1)
        self.d7 = _Dense(ted, ch[0]);  self.tgn2 = nn.GroupNorm(32, ch[0])

        self.tc1 = nn.ConvTranspose2d(ch[0]*2, 1, 3, stride=1)

    def forward(self, x, tembed):
        a = self.act
        h1 = a(self.gn1(self.conv1(x)  + self.d1(tembed)))
        h2 = a(self.gn2(self.conv2(h1) + self.d2(tembed)))
        h3 = a(self.gn3(self.conv3(h2) + self.d3(tembed)))
        h4 = a(self.gn4(self.conv4(h3) + self.d4(tembed)))

        h = a(self.tgn4(self.tc4(h4)                        + self.d5(tembed)))
        h = a(self.tgn3(self.tc3(torch.cat([h, h3], dim=1)) + self.d6(tembed)))
        h = a(self.tgn2(self.tc2(torch.cat([h, h2], dim=1)) + self.d7(tembed)))
        return self.tc1(torch.cat([h, h1], dim=1))


class ScoreNet(nn.Module):
    """UNet wrapped with Fourier time embedding; exposes forward(x, t).

    Output is normalised by σ_t following yang-song/score_sde convention:
        score(x, t) = -z / σ_t

    Args:
        sde:      VPSDE_DDPM instance (for σ_t normalisation).
        model:    UNet instance.
        temb_dim: Fourier embedding dimension. Default: 256.
    """

    def __init__(self, sde, model, temb_dim=256):
        super().__init__()
        self.sde = sde
        self.model = model
        self.temb = nn.Sequential(
            GaussianFourierProjection(embed_dim=temb_dim),
            nn.Linear(temb_dim, temb_dim),
        )
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        tembed = self.act(self.temb(t))
        out = self.model(x, tembed)
        _, std = self.sde.marginal_prob(x, t)
        return out / std[:, None, None, None]
