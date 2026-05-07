"""VP-SDE (DDPM parameterization) for continuous diffusion."""

import torch


class VPSDE_DDPM:
    """Variance Preserving SDE: dx = -0.5*β(t)*x*dt + √β(t)*dW.

    β(t) = beta_min + t*(beta_max - beta_min),  T = 1,  prior = N(0, I).

    Args:
        N:        discretization steps (kept for checkpoint compatibility).
        dtype:    torch dtype.
        device:   torch device.
        beta_min: β(0). Default: 0.1.
        beta_max: β(1). Default: 20.0.
    """

    def __init__(self, N=1000, dtype=torch.float32, device='cpu',
                 beta_min=0.1, beta_max=20.0):
        self.N = N
        self.dtype = dtype
        self.device = device
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self):
        return 1

    def prior_sampling(self, shape, generator=None):
        return torch.randn(shape, device=self.device, dtype=self.dtype,
                           generator=generator)

    def sde(self, x, t):
        """Drift and diffusion of the forward SDE."""
        nb, *xd = x.shape
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t.reshape(nb, *([1] * len(xd))) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Mean and std of p_{0t}(x_t | x_0)."""
        nb, *xd = x.shape
        log_mean_coeff = (-0.25 * t**2 * (self.beta_1 - self.beta_0)
                          - 0.5 * t * self.beta_0)
        mean = torch.exp(log_mean_coeff.reshape(nb, *([1] * len(xd)))) * x
        std = torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def direct_infer(self, x, t, score_fn):
        """Recover x_0 from x_t in one step using the score."""
        nb, *xd = x.shape
        dm = [1] * len(xd)
        mean, std = self.marginal_prob(x, t)
        score = score_fn(x, t)
        neg_lmc = (0.25 * t**2 * (self.beta_1 - self.beta_0)
                   + 0.5 * t * self.beta_0)
        return torch.exp(neg_lmc.reshape(nb, *dm)) * (
            x + std.reshape(nb, *dm)**2 * score
        )

    def reverse(self, score_fn, prob_flow=False):
        """Return a reverse-time SDE object with a .sde(x, t) method."""
        sde_fn = self.sde

        class RSDE:
            def sde(self, x, t):
                nb, *xd = x.shape
                dm = [1] * len(xd)
                drift, g = sde_fn(x, t)
                score = score_fn(x, t)
                score_m = score * 0.5 if prob_flow else score
                drift_post = drift - g.reshape(nb, *dm)**2 * score_m
                g_out = torch.zeros_like(g) if prob_flow else g
                return drift_post, g_out

        return RSDE()
