"""
BSNP Evaluation Script
Load a trained checkpoint and report final metrics on the test set.

Usage:
    python scripts/evaluate.py --config configs/poisson.yaml \\
        --ckpt experiments/poisson_1d_final.pt
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from scripts.train import build_datasets, build_model
from data.collate import collate_fn
from utils.metrics import mnse, nll, ecp


@torch.no_grad()
def evaluate_full(model, loader, device, n_samples=50):
    """Per-batch metrics aggregated to mean/std over the test set."""
    model.eval()
    results = {"mnse": [], "nll": [], "ecp": []}
    for batch in loader:
        lam      = batch["lam"].to(device)
        x_ctx    = batch["x_ctx"].to(device)
        y_ctx    = batch["y_ctx"].to(device)
        x_tgt    = batch["x_tgt"].to(device)
        y_tgt    = batch["y_tgt"].to(device)
        ctx_mask = batch.get("ctx_mask")
        if ctx_mask is not None:
            ctx_mask = ctx_mask.to(device)
        mu, sigma = model.predict(x_ctx, y_ctx, x_tgt, lam,
                                  n_samples=n_samples, ctx_mask=ctx_mask)
        results["mnse"].append(mnse(mu, y_tgt).item())
        results["nll"].append(nll(mu, sigma, y_tgt).item())
        results["ecp"].append(ecp(mu, sigma, y_tgt, 0.9))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds = build_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate_full(model, test_loader, device, args.n_samples)
    print("\n=== Test Results ===")
    for k, (mean, std) in metrics.items():
        print(f"  {k}: {mean:.4e} +- {std:.4e}")


if __name__ == "__main__":
    main()
