"""Train DAP (Dynamics Aware Predictor) on MNIST.

DAP is a time-conditioned classifier p(y | x_t, σ_t) trained along diffusion
trajectories via an intermediate loss constraining the predictions to adhere 
to reverse generative dynamics:

  L = w_bnd * CE( DAP(x_ds, σ_min),  y_ds )
    + w_intr * CE( p_target(x_{t-1}),  DAP(x_t, σ_t) )

where p_target is produced by an EMA copy of DAP (detached target network).

Usage:
    python train_dap.py --config configs/dap_mnist.yml
    python train_dap.py --config configs/dap_mnist.yml --seed 123
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dap.sde import VPSDE_DDPM
from dap.models.score_net import UNet, ScoreNet
from dap.models.classifier import MN_CNN_CLS, DAP
from dap.utils import gen_trj, calc_ce_crv_t, calc_acc_crv_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dap_mnist.yml')
    parser.add_argument('--seed',   type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic     = True
    torch.backends.cudnn.benchmark         = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev   = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if cfg['dtype'] == 'float32' else torch.float64
    tch_gen = torch.Generator(device=dev).manual_seed(seed)
    print(f"Device: {dev}  dtype: {dtype}  seed: {seed}")

    # ── Load pretrained diffusion model ───────────────────────────────────────
    cdiff_ckpt     = torch.load(cfg['pretrained']['cdiff_ckpt'], map_location=dev)
    cdiff_full_cfg = cdiff_ckpt['cfg']
    cdiff_mcfg     = cdiff_full_cfg['model']
    sde            = VPSDE_DDPM(dtype=dtype, device=dev, **cdiff_full_cfg['sde']['args'])
    xd             = (1, 28, 28)

    inner_cfg = cdiff_mcfg['inner_model']
    inner_mdl = UNet(**inner_cfg['args'])
    scr_mdl   = ScoreNet(sde=sde, model=inner_mdl,
                         temb_dim=cdiff_mcfg['args']['temb_dim'])
    scr_mdl.load_state_dict(cdiff_ckpt['mdl_sd'])
    scr_mdl = scr_mdl.to(device=dev, dtype=dtype).eval()
    print("Loaded score network.")

    # ── Load pretrained universal classifier F_uni ────────────────────────────
    uni_ckpt = torch.load(cfg['pretrained']['f_uni_ckpt'], map_location=dev)
    f_uni    = MN_CNN_CLS(**uni_ckpt['mdl_cfg']['args'])
    f_uni.load_state_dict(uni_ckpt['mdl_sd'])
    f_uni = f_uni.to(device=dev, dtype=dtype).eval()
    print("Loaded F_uni.")

    # ── Config ────────────────────────────────────────────────────────────────
    nstps       = cfg['sde']['n_steps']
    eps         = cfg['sde']['eps']
    n_trj_trn   = cfg['trajectories']['n_train']
    n_trj_eval  = cfg['trajectories']['n_eval']
    n_ds        = cfg['d_s']['n_samples']
    trn_stps    = cfg['train']['n_steps']
    bsz         = cfg['train']['batch_size']
    lr          = cfg['train']['lr']
    lg_intv     = cfg['train']['log_interval']
    w_bnd       = cfg['train']['w_boundary']
    w_intr      = cfg['train']['w_interior']
    do_dch      = cfg['train']['use_ema_target']
    tau         = cfg['train']['ema_tau']
    num_cls     = cfg['data']['n_classes']
    dap_cfg     = cfg['dap']

    # ── Load dataset ──────────────────────────────────────────────────────────
    with h5py.File(cfg['data']['h5_path'], 'r') as hf:
        x_all   = hf['data/x'][:]
        y_all   = hf['data/y'][:]
        trn_idx = hf['split/trn_idx'][:]

    x_trn = torch.tensor(x_all[trn_idx], device=dev, dtype=dtype)
    y_trn = torch.tensor(y_all[trn_idx], device=dev, dtype=torch.long)
    print(f"Data: train={tuple(x_trn.shape)}")

    # ── Boundary log-sigma ────────────────────────────────────────────────────
    with torch.no_grad():
        _, sig_eps = sde.marginal_prob(
            torch.zeros(1, *xd, device=dev, dtype=dtype),
            torch.tensor([eps], device=dev, dtype=dtype),
        )
    log_sig_bnd = sig_eps.log().squeeze()    # scalar tensor (value at t=eps)
    print(f"log_sigma(t=eps={eps}): {log_sig_bnd:.4f}")

    # ── Build D_s: stratified reference subset ────────────────────────────────
    n_per_c   = n_ds // num_cls
    sub_gen   = torch.Generator().manual_seed(seed)
    sub_idx   = []
    for c in range(num_cls):
        c_idx = (y_trn == c).nonzero(as_tuple=True)[0]
        perm  = torch.randperm(len(c_idx), generator=sub_gen)[:n_per_c]
        sub_idx.append(c_idx[perm])
    sub_idx   = torch.cat(sub_idx)
    x_ds      = x_trn[sub_idx]
    y_ds      = y_trn[sub_idx]
    y_ds_prb  = F.one_hot(y_ds, num_cls).to(dtype=dtype)   # (n_ds, num_cls)
    n_ds      = x_ds.shape[0]
    print(f"D_s: {tuple(x_ds.shape)}")

    # ── Generate evaluation trajectories ─────────────────────────────────────
    trj_eval = gen_trj(sde, scr_mdl, n_trj_eval, xd, nstps, eps, dev, dtype, tch_gen)
    xt_eval  = trj_eval['xt'].cpu()    # keep on CPU to save GPU memory
    tm_eval  = trj_eval['tm']
    with torch.no_grad():
        y0_eval = f_uni(trj_eval['mn_x'][-1]).softmax(-1)   # (n_trj_eval, num_cls)
    print(f"Eval trajectories ready: {tuple(xt_eval.shape)}")

    # ── Generate training trajectories ────────────────────────────────────────
    trj       = gen_trj(sde, scr_mdl, n_trj_trn, xd, nstps, eps, dev, dtype, tch_gen)
    xt_trn    = trj['xt'].cpu()
    mn_x_trn  = trj['mn_x'].cpu()
    g_trn     = trj['g'].cpu()          # (n_t, n_trj)
    step_size = trj['stp_sz']
    n_t_trn   = xt_trn.shape[0]

    # log_sigma for each training timestep
    x_dummy   = torch.zeros(n_t_trn, *xd, device=dev, dtype=dtype)
    with torch.no_grad():
        _, sig_t = sde.marginal_prob(x_dummy, trj['tm'])
    log_sig_per_t = sig_t.log()   # (n_t_trn,)  on dev

    # Flatten interior points: (n_t-1) timesteps × n_trj trajectories
    dm_ext     = [1] * len(xd)
    n_intr     = (n_t_trn - 1) * n_trj_trn
    xt_intr    = xt_trn[:-1].reshape(n_intr, *xd)          # CPU
    mn_x_intr  = mn_x_trn[:-1].reshape(n_intr, *xd)        # CPU
    g_intr     = g_trn[:-1].reshape(n_intr, *dm_ext)        # CPU

    log_sig_all      = log_sig_per_t.reshape(n_t_trn, 1).expand(n_t_trn, n_trj_trn)
    log_sig_t_intr   = log_sig_all[:-1].reshape(n_intr)    # log_sigma at step t
    log_sig_tm1_intr = log_sig_all[1:].reshape(n_intr)     # log_sigma at step t-1

    print(f"Training data: {n_intr} interior points  {n_ds} boundary points")

    # ── Train DAP ─────────────────────────────────────────────────────────────
    dap_args = dict(
        in_ch         = dap_cfg.get('in_ch', 1),
        chs           = dap_cfg.get('chs', [32, 64]),
        hdims         = dap_cfg.get('hdims', []),
        outdim        = dap_cfg.get('n_classes', num_cls),
        temb_dim      = dap_cfg.get('temb_dim', 256),
        fourier_scale = dap_cfg.get('fourier_scale', 30.0),
        dropout       = dap_cfg.get('dropout', 0.0),
        t_mode        = dap_cfg.get('t_mode', 'addbias'),
    )
    dap_mdl  = DAP(**dap_args).to(device=dev, dtype=dtype)
    dap_trgt = DAP(**dap_args).to(device=dev, dtype=dtype) if do_dch else dap_mdl
    if do_dch:
        dap_trgt.load_state_dict(dap_mdl.state_dict())
    optm = torch.optim.Adam(dap_mdl.parameters(), lr=lr)
    dap_mdl.train();  dap_trgt.train()

    print(f"\nTraining DAP for {trn_stps} steps ...")
    for it in range(trn_stps):
        optm.zero_grad()

        # ── Boundary loss: CE( DAP(x_ds, σ_min),  y_ds ) ───────────────────
        idx_bnd  = torch.randint(0, n_ds, (bsz,), device=dev, generator=tch_gen)
        lgts_bnd = dap_mdl(x_ds[idx_bnd], log_sig_bnd.expand(bsz))
        lg_bnd   = lgts_bnd - lgts_bnd.logsumexp(-1, keepdim=True)
        loss_bnd = -torch.einsum('bc,bc->b', y_ds_prb[idx_bnd], lg_bnd).mean()

        # ── Intermediate loss ─────────────────────────────────────────────────
        # CE( p_target(x_{t-1}),  DAP(x_t, σ_t) )
        idx_intr     = torch.randint(0, n_intr, (bsz,), device=dev, generator=tch_gen)
        idx_cpu      = idx_intr.cpu()

        lgts_t = dap_mdl(xt_intr[idx_cpu].to(dev), log_sig_t_intr[idx_intr])
        lg_t   = lgts_t - lgts_t.logsumexp(-1, keepdim=True)

        # Sample x_{t-1} ~ N(mean_x, step_size * g_t^2 * I)
        z    = torch.randn(bsz, *xd, device=dev, dtype=dtype, generator=tch_gen)
        xtm1 = (mn_x_intr[idx_cpu].to(dev)
                + torch.sqrt(step_size) * g_intr[idx_cpu].to(dev) * z)

        with torch.no_grad():
            lgts_tm1 = dap_trgt(xtm1, log_sig_tm1_intr[idx_intr])
        prb_tm1  = lgts_tm1.softmax(-1)
        ce_intr  = -torch.einsum('bc,bc->b', prb_tm1, lg_t)   # (bsz,)

        loss = w_bnd * loss_bnd + w_intr * ce_intr.mean()
        loss.backward()
        if it > 0:
            optm.step()

        # EMA update of target network
        if do_dch and it > 0:
            with torch.no_grad():
                for pm, pt in zip(dap_mdl.parameters(), dap_trgt.parameters()):
                    pt.lerp_(pm, 1.0 - tau)

        if (it + 1) % lg_intv == 0:
            print(f"  it={it+1:>6d}/{trn_stps}"
                  f"  loss={loss.item():.5f}"
                  f"  bnd={loss_bnd.item():.5f}"
                  f"  intr={ce_intr.mean().item():.5f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    dap_mdl.eval()
    ce_t  = calc_ce_crv_t(dap_mdl, xt_eval, tm_eval, sde, y0_eval, num_cls, dev=dev)
    acc_t = calc_acc_crv_t(dap_mdl, xt_eval, tm_eval, sde, y0_eval, num_cls, dev=dev)
    print(f"\nEvaluation over {n_trj_eval} trajectories:")
    print(f"  Mean CE  (all t): {ce_t.mean():.4f}")
    print(f"  Mean Acc (all t): {acc_t.mean():.4f}")
    print(f"  CE  at t=eps (clean): {ce_t[-1]:.4f}")
    print(f"  Acc at t=eps (clean): {acc_t[-1]:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    if cfg.get('save_checkpoint', True):
        out_dir = Path(cfg.get('output_dir', 'results/dap'))
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / 'dap_ckpt.pt'
        torch.save({
            'dap_cfg': dap_args,
            'mdl_sd':  dap_mdl.state_dict(),
            'seed':    seed,
            'cfg':     cfg,
            'ce_crv':  ce_t.cpu().numpy(),
            'acc_crv': acc_t.cpu().numpy(),
        }, ckpt_path)
        print(f"\nCheckpoint saved → {ckpt_path}")


if __name__ == '__main__':
    main()
