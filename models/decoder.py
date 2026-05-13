"""
BSNP Decoder
Paper Section 4.1, Eq. 12-13

Stage 4: Condition grid features on a sampled latent z, then interpolate
to arbitrary query points x* to produce the predictive distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSNPDecoder(nn.Module):
    """
    Query-point predictor.

    Pipeline (Paper Eq. 12-13):
        q_m(z)   = Head_0(concat(g_m, z))                    # condition on latent
        alpha_m  = kappa(s_m, x*) / sum_m' kappa(s_m', x*)   # interpolation weights
        q(x*)    = sum_m alpha_m * q_m(z)                    # soft lookup
        mu(x*)   = mu_head(q(x*))
        sigma(x*)= softplus(sigma_head(q(x*))) + sigma_min   # heteroscedastic std
    """
    def __init__(self, feat_dim: int, latent_dim: int, x_dim: int,
                 hidden_dim: int = 64, sigma_min: float = 1e-4,
                 lengthscale: float = 0.1, epsilon: float = 1e-6):
        super().__init__()
        self.sigma_min = sigma_min
        self.epsilon = epsilon
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale).log())

        # Head_0: conditions grid features on z (Eq. 12)
        self.head0 = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Predictive mean and log-std output heads
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def _kernel(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """RBF kernel (same form as encoder). s:(B,M,d), x:(B,N,d) -> (B,M,N)"""
        rho = self.log_lengthscale.exp()
        diff = s.unsqueeze(-2) - x.unsqueeze(-3)
        return torch.exp(-0.5 * (diff ** 2).sum(-1) / rho ** 2)

    def forward(self, grid: torch.Tensor, g: torch.Tensor,
                z: torch.Tensor, x_query: torch.Tensor):
        """
        grid:    (B, M, d_x)       grid points
        g:       (B, M, feat_dim)  encoder grid features
        z:       (B, latent_dim)   sampled latent code
        x_query: (B, N_q, d_x)     query locations

        Returns:
            mu    : (B, N_q, 1) predictive mean
            sigma : (B, N_q, 1) predictive std (>= sigma_min)
        """
        B, M, _ = g.shape

        # Condition grid features on latent z (Eq. 12)
        z_exp = z.unsqueeze(1).expand(-1, M, -1)
        q_m = self.head0(torch.cat([g, z_exp], dim=-1))        # (B, M, hidden_dim)

        # Normalized kernel interpolation weights alpha_m(x*)
        kappa = self._kernel(grid, x_query)                    # (B, M, N_q)
        alpha = kappa / (kappa.sum(1, keepdim=True) + float(self.epsilon))

        # Weighted aggregation to query points
        q_x = torch.einsum('bmn,bmd->bnd', alpha, q_m)         # (B, N_q, hidden_dim)

        mu = self.mu_head(q_x)
        sigma = F.softplus(self.sigma_head(q_x)) + float(self.sigma_min)
        return mu, sigma
