"""
Visualization Script - generate paper-style figures from a trained checkpoint.

Usage:
    python scripts/visualize.py --config configs/poisson.yaml \\
        --ckpt experiments/poisson_1d_final.pt
"""
import os
import sys
import argparse
import yaml
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from scripts.train import build_datasets, build_model
from data.collate import collate_fn
from utils.visualization import (plot_multiple_predictions, plot_training_curves,
                                  plot_noise_robustness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", default="experiments/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds = build_datasets(cfg)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Paper Figure 2 style multi-task predictions (Poisson only -- needs full grid)
    if cfg["pde"]["type"] == "poisson_1d":
        print("Generating prediction plots...")
        plot_multiple_predictions(model, test_ds, device, n_examples=4,
                                   save_path=f"{args.outdir}/fig2_predictions.png")

    # Training curves from log file
    log_path = args.ckpt.replace("_final.pt", "_log.txt")
    if os.path.exists(log_path):
        print("Generating training curves...")
        plot_training_curves(log_path, f"{args.outdir}/training_curves.png")

    # Noise robustness sweep (Paper Table 2)
    print("Generating noise robustness plot...")
    from utils.metrics import mnse as mnse_fn
    noise_levels = [0, 5, 10, 15, 30]
    mnse_means, mnse_stds = [], []
    for noise_pct in noise_levels:
        cfg_copy = yaml.safe_load(yaml.dump(cfg))
        cfg_copy["data"]["noise_std"] = noise_pct / 100.0
        _, _, ds = build_datasets(cfg_copy)
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
        vals = []
        with torch.no_grad():
            for batch in loader:
                mu, _ = model.predict(
                    batch["x_ctx"].to(device), batch["y_ctx"].to(device),
                    batch["x_tgt"].to(device), batch["lam"].to(device),
                    n_samples=20, ctx_mask=batch["ctx_mask"].to(device))
                vals.append(mnse_fn(mu, batch["y_tgt"].to(device)).item())
        mnse_means.append(float(torch.tensor(vals).mean()))
        mnse_stds.append(float(torch.tensor(vals).std()))
        print(f"  noise={noise_pct}%: MNSE={mnse_means[-1]:.4e}")

    plot_noise_robustness(noise_levels, mnse_means, mnse_stds, None,
                          f"{args.outdir}/noise_robustness.png")

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
