"""
Burgers Equation Dataset
Paper Section 5.1.

PDE:    u_t + u * u_x = nu * u_xx
Domain: x in [0,1), t in [0,1] -- treated as 2D meta-learning problem
Param:  lambda = (nu, A) ~ U[0.005, 0.1] x U[0.5, 2.0]

Note: nu lower bound is 0.005 (paper says 0.001), as smaller nu produces
near-shock solutions that the spectral solver cannot handle stably.
NaN tasks (rare) are filtered and resampled automatically.
"""
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from multiprocessing import Pool

from data.burgers_solver import _solve_burgers_task


class BurgersDataset(Dataset):
    """
    Burgers meta-learning dataset (2D space-time).
    Disk caching analogous to Poisson1DDataset.
    """
    def __init__(self, n_tasks: int, n_context_min: int = 10,
                 n_context_max: int = 50, n_target: int = 100,
                 n_grid_x: int = 64, n_grid_t: int = 64,
                 noise_std: float = 0.05,
                 param_range: list = None, seed: int = 42,
                 cache_dir: str = "data/cache"):
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max
        self.n_target = n_target
        self.noise_std = noise_std
        self.param_range = param_range or [[0.005, 0.1], [0.5, 2.0]]

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir,
            f"burgers_n{n_tasks}_x{n_grid_x}_t{n_grid_t}_s{seed}.npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            self.tasks = list(zip(data["lams"], data["coords"], data["vals"]))
        else:
            rng = np.random.default_rng(seed)
            lams_all, coords_all, vals_all = [], [], []
            attempts = 0
            # Resample to handle rare NaN tasks (spectral solver may fail at extremes)
            while len(lams_all) < n_tasks:
                need = (n_tasks - len(lams_all)) * 2
                lams = np.array([
                    [rng.uniform(*self.param_range[0]),
                     rng.uniform(*self.param_range[1])]
                    for _ in range(need)
                ], dtype=np.float32)
                with Pool() as pool:
                    results = pool.map(_solve_burgers_task,
                                       [(lam, n_grid_x, n_grid_t) for lam in lams])
                for lam, coords, vals in results:
                    if not np.isnan(vals).any() and len(lams_all) < n_tasks:
                        lams_all.append(lam)
                        coords_all.append(coords)
                        vals_all.append(vals)
                attempts += 1
                if attempts > 10:
                    raise RuntimeError("Too many NaN tasks in Burgers solver")
            lams_out   = np.stack(lams_all[:n_tasks])
            coords_out = np.stack(coords_all[:n_tasks])
            vals_out   = np.stack(vals_all[:n_tasks])
            np.savez(cache_path, lams=lams_out, coords=coords_out, vals=vals_out)
            self.tasks = list(zip(lams_out, coords_out, vals_out))

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        """Random context/target sampling from the full (x, t) grid."""
        lam, coords, vals = self.tasks[idx]
        n_pts = len(vals)
        rng = np.random.default_rng(idx)

        n_ctx = rng.integers(self.n_context_min, self.n_context_max + 1)
        ctx_idx = rng.choice(n_pts, n_ctx, replace=False)
        tgt_idx = rng.choice(n_pts, self.n_target, replace=False)

        y_ctx = vals[ctx_idx, None]
        if self.noise_std > 0:
            y_ctx = y_ctx + rng.normal(0, self.noise_std * np.abs(vals).max(),
                                        y_ctx.shape).astype(np.float32)
        return {
            "lam":   torch.from_numpy(lam),
            "x_ctx": torch.from_numpy(coords[ctx_idx]),
            "y_ctx": torch.from_numpy(y_ctx),
            "x_tgt": torch.from_numpy(coords[tgt_idx]),
            "y_tgt": torch.from_numpy(vals[tgt_idx, None]),
        }
