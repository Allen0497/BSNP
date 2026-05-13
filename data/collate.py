"""
Custom collate function for variable-length context sets.

Context size N_c is randomly sampled per task in [N_ctx_min, N_ctx_max],
so batching requires zero-padding + a boolean mask that the encoder
uses to exclude padded positions from kernel aggregation.
"""
import torch


def collate_fn(batch):
    """
    Pack a list of task dicts into a batched dict with:
      - fixed-size tensors for lam, x_tgt, y_tgt
      - padded tensors for x_ctx, y_ctx plus a boolean ctx_mask
    """
    lam   = torch.stack([b["lam"]   for b in batch])
    x_tgt = torch.stack([b["x_tgt"] for b in batch])
    y_tgt = torch.stack([b["y_tgt"] for b in batch])

    max_nc = max(b["x_ctx"].size(0) for b in batch)
    x_dim  = batch[0]["x_ctx"].size(-1)

    x_ctx_pad = torch.zeros(len(batch), max_nc, x_dim)
    y_ctx_pad = torch.zeros(len(batch), max_nc, 1)
    mask      = torch.zeros(len(batch), max_nc, dtype=torch.bool)

    for i, b in enumerate(batch):
        nc = b["x_ctx"].size(0)
        x_ctx_pad[i, :nc] = b["x_ctx"]
        y_ctx_pad[i, :nc] = b["y_ctx"]
        mask[i, :nc]      = True

    out = {"lam": lam, "x_ctx": x_ctx_pad, "y_ctx": y_ctx_pad,
           "x_tgt": x_tgt, "y_tgt": y_tgt, "ctx_mask": mask}

    # Optional full-grid tensors (only present for Poisson, used for visualization)
    if "x_grid" in batch[0]:
        out["x_grid"] = torch.stack([b["x_grid"] for b in batch])
        out["u_grid"] = torch.stack([b["u_grid"] for b in batch])
    return out
