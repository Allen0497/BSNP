"""
BSNP Training Script
Implements Paper Algorithm 1.

Supports:
  - Single-GPU training via CUDA_VISIBLE_DEVICES
  - DistributedDataParallel training via torchrun
  - Mixed-precision (AMP) training
  - Gradient accumulation
  - Persistent DataLoader workers

Usage:
    # Single-GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/poisson.yaml

    # Multi-GPU DDP (if supported in the environment)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/poisson.yaml
"""
import os
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered stdout

import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bsnp import BSNP
from models.losses import bsnp_total_loss
from data.poisson import Poisson1DDataset
from data.burgers import BurgersDataset
from data.collate import collate_fn
from utils.metrics import mnse, nll, ecp
from utils.pde_residuals import (poisson_residual, poisson_bc_residual,
                                  burgers_residual, burgers_ic_residual)


# ---- DDP helpers ------------------------------------------------------------

def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def is_main(rank):
    return rank == 0

def cleanup():
    dist.destroy_process_group()


# ---- Dataset / model factories ---------------------------------------------

def build_datasets(cfg):
    """Build train/val/test datasets from config."""
    pde = cfg["pde"]["type"]
    d = cfg["data"]
    common = dict(
        n_context_min=d["n_context_min"], n_context_max=d["n_context_max"],
        n_target=d["n_target"], noise_std=d["noise_std"],
        param_range=cfg["pde"]["param_range"],
    )
    if pde == "poisson_1d":
        Cls = Poisson1DDataset
        extra = dict(n_grid=d["n_grid"])
    else:
        Cls = BurgersDataset
        extra = dict(n_grid_x=d["n_grid_x"], n_grid_t=d["n_grid_t"])

    train_ds = Cls(n_tasks=d["n_train_tasks"], seed=42,   **common, **extra)
    val_ds   = Cls(n_tasks=d["n_val_tasks"],   seed=1000, **common, **extra)
    test_ds  = Cls(n_tasks=d["n_test_tasks"],  seed=2000, **common, **extra)
    return train_ds, val_ds, test_ds


def build_loaders(train_ds, val_ds, test_ds, cfg, rank, world_size, use_ddp):
    bs = cfg["training"]["batch_size"]
    loader_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
                     collate_fn=collate_fn)
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=bs,
                                  sampler=train_sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kw)
    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader


def build_model(cfg):
    """Instantiate BSNP from config."""
    m, p = cfg["model"], cfg["pde"]
    x_dim = 1 if p["type"] == "poisson_1d" else 2
    return BSNP(
        x_dim=x_dim, y_dim=1, param_dim=p["param_dim"],
        grid_size=m["grid_size"], latent_dim=m["latent_dim"],
        cnn_channels=m["cnn_channels"], cnn_kernel_size=m["cnn_kernel_size"],
        param_embed_dim=m["param_embed_dim"], head_hidden_dim=m["head_hidden_dim"],
        lengthscale=m["kernel_lengthscale"], epsilon=m["epsilon"],
        sigma_min=m["sigma_min"],
    )


# ---- Physics residual computation ------------------------------------------

def compute_physics(model, batch, cfg, device, grid, g, z_phys):
    """
    Paper Algorithm 1, lines 11-12.
    Evaluate mean-field u_bar(x) = mu(x; C, lam, z_phys) at stochastic
    collocation points, then compute PDE + boundary residuals.
    """
    t = cfg["training"]
    pde_type = cfg["pde"]["type"]
    lam = batch["lam"].to(device)
    B = lam.size(0)

    # u_bar from the mean-field predictor (Paper Eq. 15) using z_phys
    # (context-only latent -- no target leakage)
    def u_bar(x_in):
        decoder = model.module.decoder if hasattr(model, 'module') else model.decoder
        mu, _ = decoder(grid, g, z_phys, x_in)
        return mu

    if pde_type == "poisson_1d":
        # Interior collocation: sample uniformly in [0, 1]
        x_r = torch.rand(B, t["n_collocation"], 1, device=device)
        res_int = poisson_residual(u_bar, x_r, lam)
        # Dirichlet BCs at x=0 and x=1
        x_bc = torch.cat([
            torch.zeros(B, t["n_boundary"] // 2, 1, device=device),
            torch.ones(B,  t["n_boundary"] // 2, 1, device=device),
        ], dim=1)
        res_bc = poisson_bc_residual(u_bar, x_bc)
    else:
        # Burgers: sample (x, t) uniformly in [0,1]^2
        x_r = torch.rand(B, t["n_collocation"], 2, device=device)
        res_int = burgers_residual(u_bar, x_r, lam)
        # Initial condition at t=0
        x_ic = torch.cat([
            torch.rand(B, t["n_boundary"], 1, device=device),
            torch.zeros(B, t["n_boundary"], 1, device=device),
        ], dim=-1)
        res_bc = burgers_ic_residual(u_bar, x_ic, lam)

    return res_int, res_bc


# ---- Single training step --------------------------------------------------

def train_step(model, batch, cfg, device, scaler, optimizer, step):
    t = cfg["training"]
    lam      = batch["lam"].to(device)
    x_ctx    = batch["x_ctx"].to(device)
    y_ctx    = batch["y_ctx"].to(device)
    x_tgt    = batch["x_tgt"].to(device)
    y_tgt    = batch["y_tgt"].to(device)
    ctx_mask = batch.get("ctx_mask")
    if ctx_mask is not None:
        ctx_mask = ctx_mask.to(device)

    with autocast('cuda'):
        (mu_pred, sigma_pred,
         mu_z_ctx, logvar_z_ctx,
         mu_z_full, logvar_z_full,
         g, z_phys, grid) = model(x_ctx, y_ctx, x_tgt, lam, x_tgt, y_tgt,
                                   ctx_mask=ctx_mask)

        res_int, res_bc = compute_physics(model, batch, cfg, device, grid, g, z_phys)

        losses = bsnp_total_loss(
            mu_pred, sigma_pred, y_tgt,
            mu_z_ctx, logvar_z_ctx,
            mu_z_full, logvar_z_full,
            res_int, res_bc,
            beta=t["beta"], beta_0=t["beta_0"], kl_weight=t["kl_weight"],
        )

    loss = losses["total"] / t["grad_accum_steps"]
    scaler.scale(loss).backward()

    if (step + 1) % t["grad_accum_steps"] == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {k: v.item() for k, v in losses.items()}


# ---- Evaluation ------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, cfg, n_samples=50):
    """Compute mean MNSE / NLL / ECP over a loader."""
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()
    all_mnse, all_nll, all_ecp = [], [], []
    for batch in loader:
        lam      = batch["lam"].to(device)
        x_ctx    = batch["x_ctx"].to(device)
        y_ctx    = batch["y_ctx"].to(device)
        x_tgt    = batch["x_tgt"].to(device)
        y_tgt    = batch["y_tgt"].to(device)
        ctx_mask = batch.get("ctx_mask")
        if ctx_mask is not None:
            ctx_mask = ctx_mask.to(device)
        mu, sigma = raw.predict(x_ctx, y_ctx, x_tgt, lam,
                                n_samples=n_samples, ctx_mask=ctx_mask)
        all_mnse.append(mnse(mu, y_tgt).item())
        all_nll.append(nll(mu, sigma, y_tgt).item())
        all_ecp.append(ecp(mu, sigma, y_tgt, level=0.9))
    raw.train()
    return {
        "mnse": float(torch.tensor(all_mnse).mean()),
        "nll":  float(torch.tensor(all_nll).mean()),
        "ecp":  float(torch.tensor(all_ecp).mean()),
    }


# ---- Main training loop ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True
    use_ddp = "RANK" in os.environ
    if use_ddp:
        rank, local_rank = setup_ddp()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg["experiment"]["seed"] + rank)

    # Data
    train_ds, val_ds, test_ds = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, cfg, rank, world_size, use_ddp)

    # Model
    model = build_model(cfg).to(device)
    # Note: torch.compile is disabled because it conflicts with autograd-based
    # physics residual computation in this version of PyTorch.
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer + MultiStep LR schedule
    t = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(t["lr"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=t["lr_decay_steps"], gamma=t["lr_decay"])
    scaler = GradScaler('cuda')

    # Optional resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if use_ddp else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]

    os.makedirs("experiments", exist_ok=True)
    exp_name = cfg["experiment"]["name"]
    log_path = f"experiments/{exp_name}_log.txt"

    step = start_step
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < t["n_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            if use_ddp:
                train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        losses = train_step(model, batch, cfg, device, scaler, optimizer, step)
        step += 1
        scheduler.step()

        if is_main(rank) and step % 200 == 0:
            elapsed = time.time() - t0
            msg = (f"step={step}/{t['n_steps']} "
                   f"loss={losses['total']:.4f} elbo={losses['elbo']:.4f} "
                   f"phys={losses['phys']:.4f} bc={losses['bc']:.4f} "
                   f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.1f}s")
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        if is_main(rank) and step % 2000 == 0:
            metrics = evaluate(model, val_loader, device, cfg)
            msg = (f"[VAL] step={step} mnse={metrics['mnse']:.4e} "
                   f"nll={metrics['nll']:.4f} ecp={metrics['ecp']:.4f}")
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            raw = model.module if use_ddp else model
            torch.save({
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step, "metrics": metrics,
            }, f"experiments/{exp_name}_ckpt_{step}.pt")

    # Final test
    if is_main(rank):
        metrics = evaluate(model, test_loader, device, cfg)
        msg = (f"[TEST] mnse={metrics['mnse']:.4e} "
               f"nll={metrics['nll']:.4f} ecp={metrics['ecp']:.4f}")
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        raw = model.module if use_ddp else model
        torch.save({"model": raw.state_dict(), "metrics": metrics},
                   f"experiments/{exp_name}_final.pt")

    if use_ddp:
        cleanup()


if __name__ == "__main__":
    main()
