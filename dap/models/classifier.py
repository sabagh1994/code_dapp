"""CNN classifiers for MNIST: static backbone (F_uni / F_ref) and DAP (time-conditioned)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .score_net import GaussianFourierProjection


class MN_CNN_CLS(nn.Module):
    """2-conv CNN classifier for MNIST (no time conditioning).

    Used as F_uni (universal classifier) and F_ref (reference classifier trained on S_ref).

    Args:
        in_ch:   input channels. Default: 1.
        chs:     conv channel sizes [ch0, ch1]. Default: [32, 64].
        hdims:   MLP head hidden dims. Default: [256].
        outdim:  number of classes. Default: 10.
        dropout: dropout probability. Default: 0.0.

    Input:  x  (B, 1, 28, 28)
    Output: logits  (B, outdim)
    """

    def __init__(self, in_ch=1, chs=None, hdims=None, outdim=10, dropout=0.0):
        super().__init__()
        chs   = chs   or [32, 64]
        hdims = hdims or [256]
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  chs[0], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(chs[0], chs[1], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat_dim = chs[1] * 7 * 7
        layers, prev = [], flat_dim
        for h in hdims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, outdim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class DAP(nn.Module):
    """Dynamics Aware Predictor: time-conditioned CNN classifier for MNIST.

    Predicts p(y | x_t, σ_t) at any noise level σ_t along a diffusion trajectory.
    Uses the same conv backbone as MN_CNN_CLS, enabling weight transfer from a
    pretrained static classifier.

    Two time-conditioning modes:
      'addbias'  — σ_t embedding added as spatial bias after each conv (default).
      'concat'   — σ_t embedding concatenated with flattened conv features.

    Args:
        in_ch:         input channels. Default: 1.
        chs:           conv channel sizes [ch0, ch1]. Default: [32, 64].
        hdims:         MLP head hidden dims. Default: [].
        outdim:        number of output classes. Default: 10.
        temb_dim:      time embedding dimension (≥2 for Fourier, 1 for raw scalar).
        fourier_scale: scale for GaussianFourierProjection.
        dropout:       dropout probability.
        t_mode:        'addbias' or 'concat'. Default: 'addbias'.

    Input:  x (B, 1, 28, 28),  log_sigma (B,)
    Output: logits (B, outdim)
    """

    def __init__(self, in_ch=1, chs=None, hdims=None, outdim=10,
                 temb_dim=256, fourier_scale=30.0, dropout=0.0, t_mode='addbias'):
        super().__init__()
        chs   = chs   or [32, 64]
        hdims = hdims or []
        assert t_mode in ('addbias', 'concat'), \
            f"t_mode must be 'addbias' or 'concat', got '{t_mode}'"
        self.t_mode   = t_mode
        self.temb_dim = temb_dim

        # Conv backbone — same key names as MN_CNN_CLS for weight transfer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  chs[0], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(chs[0], chs[1], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat_dim = chs[1] * 7 * 7

        if temb_dim >= 2:
            assert temb_dim % 2 == 0
            self.temb = GaussianFourierProjection(embed_dim=temb_dim,
                                                  scale=fourier_scale)
        else:
            self.temb = None  # raw log_sigma scalar

        if t_mode == 'addbias':
            self.td1 = nn.Linear(temb_dim, chs[0])
            self.td2 = nn.Linear(temb_dim, chs[1])
            head_in = flat_dim
        else:
            head_in = flat_dim + temb_dim

        layers, prev = [], head_in
        for h in hdims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, outdim))
        self.head = nn.Sequential(*layers)

    def _t_feat(self, log_sigma):
        return self.temb(log_sigma) if self.temb is not None else log_sigma.unsqueeze(-1)

    def forward(self, x, log_sigma):
        bsz    = x.shape[0]
        t_feat = self._t_feat(log_sigma)   # (B, temb_dim)
        if self.t_mode == 'addbias':
            # conv[0]=Conv2d, conv[1]=ReLU, conv[2]=MaxPool2d  (block 1)
            # conv[3]=Conv2d, conv[4]=ReLU, conv[5]=MaxPool2d  (block 2)
            h = self.conv[0](x) + self.td1(t_feat)[:, :, None, None]
            h = self.conv[2](self.conv[1](h))
            h = self.conv[3](h) + self.td2(t_feat)[:, :, None, None]
            h = self.conv[5](self.conv[4](h))
            return self.head(h.reshape(bsz, -1))
        else:
            feat = self.conv(x).reshape(bsz, -1)
            return self.head(torch.cat([feat, t_feat], dim=-1))


class NoisyClsOracle(nn.Module):
    """Noisy wrapper for classifier oracles.

    With probability p uses the oracle's softmax output; with probability (1-p)
    substitutes noise according to mode. p=1.0 / mode='none' gives clean oracle.

    Args:
        mdl:       wrapped classifier (outputs logits).
        p:         probability of using clean oracle output.
        num_cls:   number of output classes.
        mode:      'none' | 'uniform' | 'shuffle'. Default: 'uniform'.
        generator: torch.Generator for reproducibility.
    """

    def __init__(self, mdl, p, num_cls, mode='uniform', generator=None):
        super().__init__()
        assert mode in ('none', 'uniform', 'shuffle')
        self.mdl     = mdl
        self.p       = p
        self.num_cls = num_cls
        self.mode    = mode
        self.gen     = generator

    def forward(self, x):
        with torch.no_grad():
            prb = self.mdl(x).softmax(dim=-1)   # (B, num_cls)
        if self.p >= 1.0 or self.mode == 'none':
            return prb
        bs   = x.shape[0]
        mask = torch.bernoulli(
            torch.full((bs,), self.p, device=x.device, dtype=x.dtype),
            generator=self.gen,
        ).bool()
        if self.mode == 'uniform':
            noisy = torch.full((bs, self.num_cls), 1.0 / self.num_cls,
                               device=x.device, dtype=prb.dtype)
        else:  # shuffle
            perm  = torch.argsort(
                torch.rand(bs, self.num_cls, device=x.device, generator=self.gen),
                dim=-1,
            )
            noisy = prb.gather(1, perm)
        return torch.where(mask.unsqueeze(-1), prb, noisy)
