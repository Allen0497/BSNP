"""
1D Nonlinear Poisson Equation Data Generator
Paper Section 5.1.

PDE:    -d/dx [ a(x; lambda) du/dx ] = f(x),   x in [0, 1]
BC:     u(0) = u(1) = 0
Coeff:  a(x; lam) = exp(a0 * sin(pi*x) + a1 * cos(pi*x))
Source: f(x) = sin(pi*x)
Param:  lambda = (a0, a1) ~ U[-2, 2]^2

Solver: 2nd-order finite difference + scipy banded solver (CPU-fast, exact).
Solutions are pre-generated and cached on disk to avoid recomputation.
"""
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from scipy.linalg import solve_banded
from multiprocessing import Pool


def solve_poisson_1d(lam: np.ndarray, n_grid: int = 128) -> tuple:
    """
    Finite-difference solver for the 1D nonlinear Poisson equation.
    Returns (x_interior, u_interior), each shape (n_grid,).
    """
    a0, a1 = lam
    x = np.linspace(0, 1, n_grid + 2)[1:-1]   # interior points (Dirichlet BCs at endpoints)
    h = 1.0 / (n_grid + 1)

    # Half-grid coefficients a_{i+/-1/2} (conservative discretization)
    x_full = np.linspace(0, 1, n_grid + 2)
    a_full = np.exp(a0 * np.sin(np.pi * x_full) + a1 * np.cos(np.pi * x_full))
    a_plus  = 0.5 * (a_full[1:-1] + a_full[2:])
    a_minus = 0.5 * (a_full[:-2]  + a_full[1:-1])

    # Tridiagonal system in scipy banded format
    diag  = (a_plus + a_minus) / h**2
    upper = -a_plus[:-1]  / h**2
    lower = -a_minus[1:]  / h**2
    ab = np.zeros((3, n_grid))
    ab[0, 1:]  = upper
    ab[1, :]   = diag
    ab[2, :-1] = lower

    f = np.sin(np.pi * x)
    u = solve_banded((1, 1), ab, f)
    return x, u


def _solve_poisson_task(args):
    """Worker function for multiprocessing pool."""
    lam, n_grid = args
    x, u = solve_poisson_1d(lam, n_grid)
    return lam.astype(np.float32), x.astype(np.float32), u.astype(np.float32)


class Poisson1DDataset(Dataset):
    """
    Meta-learning dataset for 1D nonlinear Poisson.
    Each item is one task instance (one lambda) with random context/target split.

    Disk caching: solutions for each (n_tasks, n_grid, seed) combo are saved
    in <cache_dir>/poisson_n*.npz to avoid recomputation across runs.
    """
    def __init__(self, n_tasks: int, n_context_min: int = 10,
                 n_context_max: int = 50, n_target: int = 100,
                 n_grid: int = 128, noise_std: float = 0.05,
                 param_range: list = None, seed: int = 42,
                 cache_dir: str = "data/cache"):
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max
        self.n_target = n_target
        self.noise_std = noise_std
        self.param_range = param_range or [[-2.0, 2.0], [-2.0, 2.0]]

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir,
            f"poisson_n{n_tasks}_g{n_grid}_s{seed}.npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            self.tasks = list(zip(data["lams"], data["xs"], data["us"]))
        else:
            rng = np.random.default_rng(seed)
            lams = np.array([
                [rng.uniform(*self.param_range[0]),
                 rng.uniform(*self.param_range[1])]
                for _ in range(n_tasks)
            ], dtype=np.float32)
            with Pool() as pool:
                results = pool.map(_solve_poisson_task,
                                   [(lam, n_grid) for lam in lams])
            lams_out = np.stack([r[0] for r in results])
            xs_out   = np.stack([r[1] for r in results])
            us_out   = np.stack([r[2] for r in results])
            np.savez(cache_path, lams=lams_out, xs=xs_out, us=us_out)
            self.tasks = list(zip(lams_out, xs_out, us_out))

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        """
        Returns one task as a dict containing context/target tensors.
        Variable n_ctx is sampled per item; padding is handled by collate_fn.
        Heteroscedastic Gaussian noise (relative to peak |u|) is added to context.
        """
        lam, x_grid, u_grid = self.tasks[idx]
        n_grid = len(x_grid)
        rng = np.random.default_rng(idx)   # deterministic per-item sampling

        n_ctx = rng.integers(self.n_context_min, self.n_context_max + 1)
        ctx_idx = rng.choice(n_grid, n_ctx, replace=False)
        tgt_idx = rng.choice(n_grid, self.n_target, replace=False)

        x_ctx = x_grid[ctx_idx, None]
        y_ctx = u_grid[ctx_idx, None]
        if self.noise_std > 0:
            y_ctx = y_ctx + rng.normal(0, self.noise_std * np.abs(u_grid).max(),
                                        y_ctx.shape).astype(np.float32)
        return {
            "lam":    torch.from_numpy(lam),
            "x_ctx":  torch.from_numpy(x_ctx),
            "y_ctx":  torch.from_numpy(y_ctx),
            "x_tgt":  torch.from_numpy(x_grid[tgt_idx, None]),
            "y_tgt":  torch.from_numpy(u_grid[tgt_idx, None]),
            "x_grid": torch.from_numpy(x_grid[:, None]),
            "u_grid": torch.from_numpy(u_grid[:, None]),
        }
