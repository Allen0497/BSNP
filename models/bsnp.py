"""
BSNP: Bias-Spectrum Neural Processes
Full model assembly.
Paper Section 4, Eq. 4-21.

Key design (Paper Section 4.3):
  - TWO latent samples per training step:
      * z_data  ~ q(z | C ∪ T)  -- target-conditioned, for ELBO data term
      * z_phys  ~ q(z | C)      -- context-only, for physics residual (no target leakage)
  - Decoder shares the same grid features g between data and physics evaluations.
"""
import torch
import torch.nn as nn
from .encoder import ConvCNPEncoder
from .decoder import BSNPDecoder


class BSNP(nn.Module):
    """
    Full BSNP model.

    The encoder is shared for both the context-only posterior q(z|C,lam)
    and the target-augmented posterior q(z|C ∪ T, lam) -- the latter is
    obtained by feeding context ∪ target points through the same network.
    """
    def __init__(self, x_dim: int, y_dim: int, param_dim: int,
                 grid_size, latent_dim: int,
                 cnn_channels: list, cnn_kernel_size: int = 5,
                 param_embed_dim: int = 16, head_hidden_dim: int = 64,
                 lengthscale: float = 0.1, epsilon: float = 1e-6,
                 sigma_min: float = 1e-4):
        super().__init__()
        self.grid_size = grid_size
        self.x_dim = x_dim

        self.encoder = ConvCNPEncoder(
            x_dim=x_dim, y_dim=y_dim, param_dim=param_dim,
            grid_size=grid_size, latent_dim=latent_dim,
            cnn_channels=cnn_channels, cnn_kernel_size=cnn_kernel_size,
            param_embed_dim=param_embed_dim,
            lengthscale=lengthscale, epsilon=epsilon,
        )
        feat_dim = cnn_channels[-1]
        self.decoder = BSNPDecoder(
            feat_dim=feat_dim, latent_dim=latent_dim, x_dim=x_dim,
            hidden_dim=head_hidden_dim, sigma_min=sigma_min,
            lengthscale=lengthscale, epsilon=epsilon,
        )

    def _make_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build a uniform grid S over [0,1]^{x_dim} shared across the batch."""
        gs = self.grid_size if isinstance(self.grid_size, (list, tuple)) \
                             else [self.grid_size]
        if self.x_dim == 1:
            g = torch.linspace(0, 1, gs[0], device=device)
            return g.unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, -1)
        elif self.x_dim == 2:
            gx = torch.linspace(0, 1, gs[0], device=device)
            gt = torch.linspace(0, 1, gs[1] if len(gs) > 1 else gs[0], device=device)
            gx2, gt2 = torch.meshgrid(gx, gt, indexing='ij')
            grid = torch.stack([gx2.flatten(), gt2.flatten()], dim=-1)
            return grid.unsqueeze(0).expand(batch_size, -1, -1)
        raise ValueError(f"Unsupported x_dim={self.x_dim}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Gaussian reparameterization trick: z = mu + std * eps, eps ~ N(0, I)."""
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x_ctx, y_ctx, x_tgt, lam,
                x_tgt_full=None, y_tgt_full=None, ctx_mask=None):
        """
        Training forward pass.
        Paper Algorithm 1 lines 5-9.

        Returns (tuple for downstream loss computation):
            mu_pred, sigma_pred   -- predictive distribution at x_tgt (uses z_data)
            mu_z_ctx, logvar_z_ctx    -- context-only posterior q(z|C, lam)
            mu_z_full, logvar_z_full  -- target-augmented posterior q(z|C ∪ T, lam)
            g_ctx                 -- grid features for decoder reuse
            z_phys                -- context-only latent sample for physics loss
            grid                  -- shared grid for physics residual evaluation
        """
        B = x_ctx.size(0)
        grid = self._make_grid(B, x_ctx.device)

        # q(z | C, lam)  -- context-only posterior  (Eq. 5)
        (g_ctx, mu_z_ctx, logvar_z_ctx) = self.encoder(grid, x_ctx, y_ctx, lam, ctx_mask)

        # q(z | C ∪ T, lam)  -- target-augmented posterior  (Eq. 6)
        if x_tgt_full is not None and y_tgt_full is not None:
            x_full = torch.cat([x_ctx, x_tgt_full], dim=1)
            y_full = torch.cat([y_ctx, y_tgt_full], dim=1)
            if ctx_mask is not None:
                tgt_mask = torch.ones(B, x_tgt_full.size(1),
                                      dtype=torch.bool, device=x_ctx.device)
                full_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
            else:
                full_mask = None
            (_, mu_z_full, logvar_z_full) = self.encoder(grid, x_full, y_full, lam, full_mask)
        else:
            mu_z_full, logvar_z_full = mu_z_ctx, logvar_z_ctx

        # z_data: for ELBO (target-aware);  z_phys: for physics (context-only, no leakage)
        z_data = self.reparameterize(mu_z_full, logvar_z_full)
        z_phys = self.reparameterize(mu_z_ctx, logvar_z_ctx)

        mu_pred, sigma_pred = self.decoder(grid, g_ctx, z_data, x_tgt)

        return (mu_pred, sigma_pred,
                mu_z_ctx, logvar_z_ctx,
                mu_z_full, logvar_z_full,
                g_ctx, z_phys, grid)

    @torch.no_grad()
    def predict(self, x_ctx, y_ctx, x_query, lam, n_samples: int = 1, ctx_mask=None):
        """
        Inference: marginalize predictive distribution over latent samples.
        Returns predictive mean and total-uncertainty std.
        """
        B = x_ctx.size(0)
        grid = self._make_grid(B, x_ctx.device)
        (g, mu_z, logvar_z) = self.encoder(grid, x_ctx, y_ctx, lam, ctx_mask)

        mus, sigmas = [], []
        for _ in range(n_samples):
            z = self.reparameterize(mu_z, logvar_z)
            mu, sigma = self.decoder(grid, g, z, x_query)
            mus.append(mu); sigmas.append(sigma)

        mu_mean = torch.stack(mus).mean(0)
        # Total variance = E[Var(y|z)] + Var(E[y|z])  (law of total variance)
        sigma_total = (torch.stack([s**2 for s in sigmas]).mean(0) +
                       torch.stack(mus).var(0)).sqrt()
        return mu_mean, sigma_total
