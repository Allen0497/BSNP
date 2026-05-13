"""
ConvCNP-based Encoder for BSNP
Paper Section 4.1 (Weak Bias Architecture), Eq. 7-11

Implements the ConvCNP encoder: irregular context observations are deposited
onto a regular grid via RKHS kernel interpolation, processed by a translation-
equivariant CNN, and pooled to produce latent distribution parameters.
"""
import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    """
    RBF kernel: kappa_rho(s, x) = exp(-||s-x||^2 / (2*rho^2))
    Used in Eq. 7 for RKHS-based grid interpolation.
    Lengthscale rho is learnable (log-parameterized for positivity).
    """
    def __init__(self, lengthscale: float = 0.1):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale).log())

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # s: (..., M, d) - grid points; x: (..., N, d) - context points
        rho = self.log_lengthscale.exp()
        diff = s.unsqueeze(-2) - x.unsqueeze(-3)  # (..., M, N, d)
        return torch.exp(-0.5 * (diff ** 2).sum(-1) / rho ** 2)


class GridDeposition(nn.Module):
    """
    Stage 1: Deposit irregular context C onto regular grid S.
    Paper Eq. 7-8.

    h_m       = sum_i kappa(s_m, x_i) phi_y(y_i) / (sum_i kappa(s_m, x_i) + eps)
    d_m       = sum_i kappa(s_m, x_i)                          (density channel)
    h_m^(lam) = concat(h_m, d_m, phi_lambda(lambda))           (Eq. 8)

    Density channel d_m provides the model with information about
    where context observations are dense vs sparse.
    """
    def __init__(self, y_dim: int, param_dim: int, param_embed_dim: int,
                 lengthscale: float = 0.1, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.kernel = RBFKernel(lengthscale)
        # Value embedding (identity-like for scalar y)
        self.phi_y = nn.Linear(y_dim, y_dim)
        # PDE parameter embedding lambda -> R^{param_embed_dim}
        self.phi_lambda = nn.Sequential(
            nn.Linear(param_dim, param_embed_dim),
            nn.SiLU(),
            nn.Linear(param_embed_dim, param_embed_dim),
        )
        self.out_dim = y_dim + 1 + param_embed_dim

    def forward(self, grid: torch.Tensor, x_ctx: torch.Tensor,
                y_ctx: torch.Tensor, lam: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        grid:  (B, M, d_x)        regular grid points
        x_ctx: (B, N_c, d_x)      context coordinates
        y_ctx: (B, N_c, d_y)      context observations
        lam:   (B, p)             PDE parameters
        mask:  (B, N_c) bool      True = valid (for padded variable-length contexts)
        returns: (B, M, out_dim)
        """
        kappa = self.kernel(grid, x_ctx)                      # (B, M, N_c)
        if mask is not None:
            kappa = kappa * mask.unsqueeze(1).float()         # zero out padded positions
        phi_y = self.phi_y(y_ctx)                             # (B, N_c, d_y)

        # Eq. 7: weighted mean (RKHS kernel interpolation)
        denom = kappa.sum(-1, keepdim=True) + float(self.epsilon)
        h_m = torch.bmm(kappa, phi_y) / denom                 # (B, M, d_y)

        # Density channel (Eq. 8)
        d_m = kappa.sum(-1, keepdim=True)                     # (B, M, 1)

        # Broadcast PDE parameter embedding to all grid points
        phi_lam = self.phi_lambda(lam).unsqueeze(1).expand(-1, grid.size(1), -1)
        return torch.cat([h_m, d_m, phi_lam], dim=-1)


class CNNBackbone(nn.Module):
    """
    Stage 2: Translation-equivariant CNN processing.
    Paper Eq. 9: {g_m}, r = f_G({h_m^(lambda)})

    Supports both 1D (e.g., Poisson) and 2D (e.g., Burgers) grids.
    Uses GroupNorm + SiLU activation; weight sharing enforces translation
    equivariance, a key inductive bias for spatial PDE problems.
    """
    def __init__(self, in_channels: int, channels: list, kernel_size: int = 5, dim: int = 1):
        super().__init__()
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                Conv(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.SiLU(),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1D: (B, C, M);  2D: (B, C, H, W)
        return self.net(x)


class ConvCNPEncoder(nn.Module):
    """
    Full ConvCNP-based encoder.
    Paper Section 4.1, Eq. 7-11.

    Outputs:
      g        : grid features {g_m}              (B, M, feat_dim)
      mu_z     : latent mean (Eq. 10)             (B, latent_dim)
      logvar_z : latent log-variance (Eq. 11)     (B, latent_dim)
    """
    def __init__(self, x_dim: int, y_dim: int, param_dim: int,
                 grid_size, latent_dim: int,
                 cnn_channels: list, cnn_kernel_size: int = 5,
                 param_embed_dim: int = 16, lengthscale: float = 0.1,
                 epsilon: float = 1e-6):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.deposition = GridDeposition(y_dim, param_dim, param_embed_dim,
                                         lengthscale, epsilon)
        in_ch = self.deposition.out_dim
        self.cnn = CNNBackbone(in_ch, cnn_channels, cnn_kernel_size, dim=x_dim)
        feat_dim = cnn_channels[-1]

        # Global pooling psi (Eq. 9) -> latent distribution params (Eq. 10-11)
        self.psi = nn.Linear(feat_dim, feat_dim)
        self.to_mean = nn.Linear(feat_dim, latent_dim)        # W_m r + b_m
        self.to_logvar = nn.Linear(feat_dim, latent_dim)      # W_s r + b_s

    def forward(self, grid: torch.Tensor, x_ctx: torch.Tensor,
                y_ctx: torch.Tensor, lam: torch.Tensor,
                mask: torch.Tensor = None):
        # Stage 1: kernel-based deposition onto grid
        h = self.deposition(grid, x_ctx, y_ctx, lam, mask)    # (B, M, in_ch)

        # Stage 2: CNN processing (1D or 2D)
        if self.cnn.dim == 1:
            g = self.cnn(h.transpose(1, 2)).transpose(1, 2)   # (B, M, feat_dim)
        else:
            # 2D: reshape (B, M, C) -> (B, C, H, W) for Conv2d, then back
            gs = self.grid_size if isinstance(self.grid_size, (list, tuple)) \
                                 else [self.grid_size, self.grid_size]
            H, W = gs[0], gs[1]
            g = self.cnn(h.transpose(1, 2).reshape(h.size(0), -1, H, W))
            g = g.reshape(h.size(0), -1, H * W).transpose(1, 2)

        # Global pooling: r = M^{-1} sum_m psi(g_m)  (Eq. 9 sufficient statistic)
        r = self.psi(g).mean(dim=1)                           # (B, feat_dim)

        # Latent distribution params (diagonal Gaussian)
        mu_z = self.to_mean(r)
        logvar_z = self.to_logvar(r)
        return g, mu_z, logvar_z
