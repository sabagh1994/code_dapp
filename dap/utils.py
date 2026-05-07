"""Trajectory generation and evaluation utilities for DAP training."""

import torch


def gen_trj(sde, scr_mdl, n_trj, xd, nstps, eps, dev, dtype, gen):
    """Generate reverse SDE trajectories with full state logging.

    Runs Euler-Maruyama reverse sampling and records all quantities needed for
    DAP training: states, posterior means, diffusion coefficients, posterior
    drifts, and timesteps.

    Args:
        sde:     VPSDE_DDPM instance.
        scr_mdl: trained score network.
        n_trj:   number of trajectories.
        xd:      data shape tuple, e.g. (1, 28, 28).
        nstps:   number of reverse SDE steps.
        eps:     minimum time (numerical stability).
        dev:     torch.device.
        dtype:   torch.dtype.
        gen:     torch.Generator for reproducibility.

    Returns:
        dict with keys:
            'xt':     (n_t, n_trj, *xd) — noisy states at each step.
            'mn_x':   (n_t, n_trj, *xd) — posterior means.
            'g':      (n_t, n_trj)       — diffusion coefficients.
            'tm':     (n_t,)             — time values (T → eps).
            'stp_sz': scalar             — step size.
    """
    dm_ext = [1] * len(xd)
    rsde   = sde.reverse(scr_mdl, prob_flow=False)
    tstps  = torch.linspace(sde.T, eps, nstps, device=dev)
    stp_sz = tstps[0] - tstps[1]
    x      = sde.prior_sampling(shape=(n_trj, *xd), generator=gen).to(dtype=dtype)

    lg_x, lg_g, lg_mn, lg_tm = [x.detach()], [], [], []
    with torch.no_grad():
        for tstp in tstps:
            btch_t      = torch.ones(n_trj, device=dev, dtype=dtype) * tstp
            drft_pst, g = rsde.sde(x, btch_t)
            gr          = g.reshape(n_trj, *dm_ext)
            mn_x        = x + drft_pst * (-stp_sz)
            z           = torch.randn(x.shape, device=dev, dtype=dtype, generator=gen)
            x           = mn_x + torch.sqrt(stp_sz) * gr * z
            lg_x.append(x.detach())
            lg_g.append(g.detach())
            lg_mn.append(mn_x.detach())
            lg_tm.append(tstp.item())

    return {
        'xt':     torch.stack(lg_x[:-1], dim=0),   # (n_t, n_trj, *xd)
        'mn_x':   torch.stack(lg_mn,     dim=0),
        'g':      torch.stack(lg_g,      dim=0),   # (n_t, n_trj)
        'tm':     torch.tensor(lg_tm, device=dev, dtype=dtype),
        'stp_sz': stp_sz,
    }


def calc_ce_crv_t(mdl, xt, tm, sde, y0, num_cls, bsz=512, dev=None):
    """CE curve across timesteps for a time-conditioned classifier.

    Computes CE(y_0, DAP(x_t, log_sigma_t)) at each timestep, averaged over
    trajectories. Processes in mini-batches to avoid OOM.

    Args:
        mdl:     time-conditioned classifier: (x, log_sigma) → logits.
        xt:      (n_t, n_trj, *xd) — noisy states (may be on CPU).
        tm:      (n_t,) — time values.
        sde:     VPSDE_DDPM instance.
        y0:      (n_trj, num_cls) — ground-truth class probabilities.
        num_cls: number of classes.
        bsz:     mini-batch size for evaluation.
        dev:     device to move batches to before model call.

    Returns:
        ce_t: (n_t,) — mean CE at each timestep.
    """
    n_t, n_trj = xt.shape[0], xt.shape[1]
    xd         = xt.shape[2:]
    n_flat     = n_t * n_trj
    xt_flat    = xt.reshape(n_flat, *xd)

    # log_sigma from time values only (std doesn't depend on x in VP-SDE)
    x_dummy = torch.zeros(n_t, *xd, device=tm.device, dtype=xt.dtype)
    _, sig_t   = sde.marginal_prob(x_dummy, tm)
    log_sig    = sig_t.log().reshape(n_t, 1).expand(n_t, n_trj).reshape(n_flat)

    lgts_lst = []
    with torch.no_grad():
        for i0 in range(0, n_flat, bsz):
            i1  = min(i0 + bsz, n_flat)
            xb  = xt_flat[i0:i1].to(dev) if dev is not None else xt_flat[i0:i1]
            lgts_lst.append(mdl(xb, log_sig[i0:i1]))

    lgts   = torch.cat(lgts_lst).reshape(n_t, n_trj, num_cls)
    lg_prd = lgts - lgts.logsumexp(-1, keepdim=True)
    ce_ttrj = -torch.einsum('tjc,jc->tj', lg_prd, y0)
    return ce_ttrj.mean(dim=-1)   # (n_t,)


def calc_acc_crv_t(mdl, xt, tm, sde, y0, num_cls, bsz=512, dev=None):
    """Accuracy curve across timesteps for a time-conditioned classifier.

    Args: same as calc_ce_crv_t.

    Returns:
        acc_t: (n_t,) — mean accuracy at each timestep.
    """
    n_t, n_trj = xt.shape[0], xt.shape[1]
    xd         = xt.shape[2:]
    n_flat     = n_t * n_trj
    xt_flat    = xt.reshape(n_flat, *xd)

    x_dummy = torch.zeros(n_t, *xd, device=tm.device, dtype=xt.dtype)
    _, sig_t   = sde.marginal_prob(x_dummy, tm)
    log_sig    = sig_t.log().reshape(n_t, 1).expand(n_t, n_trj).reshape(n_flat)

    lgts_lst = []
    with torch.no_grad():
        for i0 in range(0, n_flat, bsz):
            i1  = min(i0 + bsz, n_flat)
            xb  = xt_flat[i0:i1].to(dev) if dev is not None else xt_flat[i0:i1]
            lgts_lst.append(mdl(xb, log_sig[i0:i1]))

    lgts    = torch.cat(lgts_lst).reshape(n_t, n_trj, num_cls)
    prd_cls = lgts.argmax(-1)                          # (n_t, n_trj)
    gt_cls  = y0.argmax(-1)                            # (n_trj,)
    return (prd_cls == gt_cls.unsqueeze(0)).float().mean(-1)   # (n_t,)
