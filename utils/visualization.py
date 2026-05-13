"""
Visualization utilities for BSNP experiments.

Produces figures in the style of the paper (Figure 2-style predictions,
noise-robustness curves, training curves, etc.).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def plot_prediction(x_grid, u_true, x_ctx, y_ctx, mu_pred, sigma_pred,
                    title="BSNP Prediction", save_path=None):
    """Single-task prediction plot: true vs mean vs 90% CI with context markers."""
    z90 = stats.norm.ppf(0.95)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, u_true, 'k-', lw=2, label='True solution')
    ax.plot(x_grid, mu_pred, 'b-', lw=1.5, label='BSNP mean')
    ax.fill_between(x_grid,
                    mu_pred - z90 * sigma_pred,
                    mu_pred + z90 * sigma_pred,
                    alpha=0.3, color='blue', label='90% CI')
    ax.scatter(x_ctx, y_ctx, c='red', s=20, zorder=5,
               label=f'Context (N={len(x_ctx)})')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_noise_robustness(noise_levels, mnse_means, mnse_stds=None,
                          paper_values=None, save_path=None):
    """Paper Table 2: MNSE vs noise level, overlaid with reported paper values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(noise_levels, mnse_means, yerr=mnse_stds,
                marker='o', label='BSNP (ours)', capsize=4)
    if paper_values is not None:
        ax.plot(noise_levels, paper_values, 's--', color='gray',
                label='Paper reported', alpha=0.7)
    ax.set_xlabel('Noise level (%)'); ax.set_ylabel('MNSE')
    ax.set_yscale('log')
    ax.set_title('Noise Robustness (Paper Table 2)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(log_path, save_path=None):
    """Parse a training log and plot total loss / physics residual / val MNSE."""
    steps, losses, phys, val_steps, val_mnse = [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            if line.startswith('step='):
                parts = line.split()
                s = int(parts[0].split('=')[1].split('/')[0])
                l = float([p for p in parts if p.startswith('loss=')][0].split('=')[1])
                p = float([p for p in parts if p.startswith('phys=')][0].split('=')[1])
                steps.append(s); losses.append(l); phys.append(p)
            elif '[VAL]' in line and 'mnse=' in line:
                s = int(line.split('step=')[1].split()[0])
                m = float(line.split('mnse=')[1].split()[0])
                val_steps.append(s); val_mnse.append(m)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, losses); axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Step')
    axes[1].plot(steps, phys); axes[1].set_title('Physics Residual')
    axes[1].set_xlabel('Step'); axes[1].set_yscale('log')
    if val_mnse:
        axes[2].plot(val_steps, val_mnse, 'o-')
        axes[2].set_yscale('log'); axes[2].set_title('Val MNSE')
        axes[2].set_xlabel('Step')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_multiple_predictions(model, dataset, device, n_examples=4, save_path=None):
    """Paper Figure 2 style: grid of prediction plots across multiple tasks."""
    from data.collate import collate_fn
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=n_examples,
                        collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        lam      = batch['lam'].to(device)
        x_ctx    = batch['x_ctx'].to(device)
        y_ctx    = batch['y_ctx'].to(device)
        x_grid   = batch['x_grid'].to(device)
        u_grid   = batch['u_grid'].to(device)
        ctx_mask = batch['ctx_mask'].to(device)
        mu, sigma = model.predict(x_ctx, y_ctx, x_grid, lam,
                                  n_samples=50, ctx_mask=ctx_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    z90 = stats.norm.ppf(0.95)
    for i, ax in enumerate(axes.flat):
        if i >= n_examples:
            break
        xg = x_grid[i, :, 0].cpu().numpy()
        ug = u_grid[i, :, 0].cpu().numpy()
        mp = mu[i, :, 0].cpu().numpy()
        sp = sigma[i, :, 0].cpu().numpy()
        xc = x_ctx[i][ctx_mask[i]].cpu().numpy()
        yc = y_ctx[i][ctx_mask[i]].cpu().numpy()
        mnse_i = ((mp - ug) ** 2).sum() / (ug ** 2).sum()

        ax.plot(xg, ug, 'k-', lw=2, label='True')
        ax.plot(xg, mp, 'b-', lw=1.5, label='BSNP')
        ax.fill_between(xg, mp - z90 * sp, mp + z90 * sp,
                        alpha=0.3, color='blue', label='90% CI')
        ax.scatter(xc, yc, c='red', s=15, zorder=5)
        ax.set_title(f'Task {i+1} | Nc={ctx_mask[i].sum().item()} '
                     f'| MNSE={mnse_i:.2e}')
        ax.legend(fontsize=7)

    plt.suptitle('BSNP Predictions (Paper Figure 2 style)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
