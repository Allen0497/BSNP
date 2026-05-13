"""
BSNP Loss Functions
Paper Section 4.2, Eq. 14-21.

Total objective (Eq. 21):  L = L_data + beta * J_phys + beta_0 * J_bc

where
  L_data : ELBO data term (Eq. 19)  -- reconstruction + KL
  J_phys : interior PDE residual (Eq. 16)
  J_bc   : boundary residual (Eq. 16)

Note we minimize -L_data + beta * J_phys + beta_0 * J_bc
(the sign convention in code is "smaller is better").
"""
import torch


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian log-likelihood per element (Paper Eq. 20)."""
    return -0.5 * torch.log(torch.tensor(2 * torch.pi)) \
           - sigma.log() - 0.5 * ((y - mu) / sigma) ** 2


def kl_diagonal_gaussians(mu1, logvar1, mu2, logvar2) -> torch.Tensor:
    """
    KL( N(mu1, diag exp(logvar1)) || N(mu2, diag exp(logvar2)) )
    Used as the KL term in the data ELBO (Eq. 19).
    """
    return 0.5 * (logvar2 - logvar1
                  + (logvar1.exp() + (mu1 - mu2) ** 2) / logvar2.exp() - 1).sum(-1)


def elbo_loss(mu_pred, sigma_pred, y_tgt,
              mu_z_ctx, logvar_z_ctx,
              mu_z_full, logvar_z_full,
              kl_weight: float = 1.0) -> torch.Tensor:
    """
    Negative data ELBO (Paper Eq. 19):
        L_data = E[log p(Y_T | C, T, z_data)] - KL( q(z|C∪T) || q(z|C) )
    Returns:  -L_data  (so that we minimize)
    """
    # Reconstruction term (Eq. 20)
    recon = gaussian_nll(mu_pred, sigma_pred, y_tgt).sum(-1).mean(-1).mean()
    # KL between target-augmented and context-only posteriors
    kl = kl_diagonal_gaussians(mu_z_full, logvar_z_full,
                               mu_z_ctx, logvar_z_ctx).mean()
    return -recon + kl_weight * kl


def physics_loss(residuals: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared PDE residual (Paper Eq. 16):
        J_phys = (1/N_r) sum_k || r_lambda(x_k; u_bar) ||^2
    """
    return (residuals ** 2).mean()


def bsnp_total_loss(mu_pred, sigma_pred, y_tgt,
                    mu_z_ctx, logvar_z_ctx,
                    mu_z_full, logvar_z_full,
                    res_interior, res_boundary,
                    beta: float = 0.1, beta_0: float = 0.05,
                    kl_weight: float = 1.0) -> dict:
    """
    Complete training objective (Paper Eq. 21):
        L_total = L_data + beta * J_phys + beta_0 * J_bc

    Returns a dict of named loss components for logging.
    """
    l_data = elbo_loss(mu_pred, sigma_pred, y_tgt,
                       mu_z_ctx, logvar_z_ctx,
                       mu_z_full, logvar_z_full, kl_weight)
    j_phys = physics_loss(res_interior)
    j_bc = physics_loss(res_boundary) if res_boundary is not None \
                                       else torch.zeros(1, device=l_data.device)

    total = l_data + beta * j_phys + beta_0 * j_bc
    return {"total": total, "elbo": l_data, "phys": j_phys, "bc": j_bc}
