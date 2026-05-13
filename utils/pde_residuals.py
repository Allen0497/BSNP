"""
PDE physics residual operators (Paper Section 4.2, Eq. 14).

Each function takes a callable u_fn that maps coordinates to predicted
solution values, and uses PyTorch autograd to compute the differential
operator G_lambda[u] at given collocation points.

The residual is then squared-and-averaged in models.losses.physics_loss
to form J_phys (Paper Eq. 16).
"""
import torch


# ---- 1D Nonlinear Poisson ---------------------------------------------------

def poisson_residual(u_fn, x_col: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Residual of   -(a(x;lam) * u'(x))' - f(x) = 0
    with         a(x;lam) = exp(a0*sin(pi*x) + a1*cos(pi*x))
                 f(x)     = sin(pi*x)

    u_fn:  callable taking (B, N, 1) -> (B, N, 1) (must be differentiable in x)
    x_col: (B, N_r, 1) interior collocation points
    lam:   (B, 2)
    """
    x = x_col.requires_grad_(True)
    u = u_fn(x)

    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    a0 = lam[:, 0:1].unsqueeze(1)
    a1 = lam[:, 1:2].unsqueeze(1)
    a = torch.exp(a0 * torch.sin(torch.pi * x) + a1 * torch.cos(torch.pi * x))

    flux = a * du_dx
    d_flux_dx = torch.autograd.grad(flux.sum(), x, create_graph=True)[0]

    f = torch.sin(torch.pi * x)
    return -d_flux_dx - f


def poisson_bc_residual(u_fn, x_bc: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet BC residual: u(0) = u(1) = 0.
    x_bc: (B, N_0, 1) -- typically half zeros, half ones.
    """
    return u_fn(x_bc)


# ---- Burgers (space-time) ---------------------------------------------------

def burgers_residual(u_fn, x_col: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Residual of   u_t + u * u_x - nu * u_xx = 0

    x_col: (B, N_r, 2) -- columns are (x, t)
    lam:   (B, 2)      -- (nu, A); only nu is used here
    """
    x = x_col.requires_grad_(True)
    u = u_fn(x)

    grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]   # (B, N, 2)
    du_dx = grads[..., 0:1]
    du_dt = grads[..., 1:2]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0][..., 0:1]

    nu = lam[:, 0:1].unsqueeze(1)
    return du_dt + u * du_dx - nu * d2u_dx2


def burgers_ic_residual(u_fn, x_ic: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Initial-condition residual for Burgers:  u(x, 0) = A * sin(2*pi*x)
    x_ic: (B, N_0, 2) where the t-column is 0
    """
    u = u_fn(x_ic)
    A = lam[:, 1:2].unsqueeze(1)
    x_only = x_ic[..., 0:1]
    return u - A * torch.sin(2 * torch.pi * x_only)
