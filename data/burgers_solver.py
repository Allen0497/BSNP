"""
Stable Burgers Equation Solver via Integrating Factor Method

PDE: u_t + u * u_x = nu * u_xx,   x in [0,1), t in [0,1]
IC:  u(x, 0) = A * sin(2*pi*x)
BC:  periodic in x

The integrating factor (ETD-like) scheme treats the stiff diffusion term
*exactly* in Fourier space via multiplication by exp(-nu*k^2*t), avoiding
the CFL restriction of explicit RK4 that caused overflow/NaN in earlier
versions for low-viscosity parameters.
"""
import numpy as np
from scipy.fft import fft, ifft


def solve_burgers_stable(lam: np.ndarray, n_x: int = 64, n_t: int = 64) -> tuple:
    """
    Pseudo-spectral Burgers solver with integrating-factor time stepping.

    Args:
        lam: (nu, A) -- viscosity and initial-condition amplitude
        n_x: number of spatial points (FFT grid)
        n_t: number of output time snapshots

    Returns:
        x: (n_x,)  spatial grid
        t: (n_t,)  output time grid
        u: (n_t, n_x) solution snapshots
    """
    nu, A = lam
    x = np.linspace(0, 1, n_x, endpoint=False)
    t_end = 1.0

    # Adaptive inner step count (satisfies advective CFL for any |u| in [-A, A])
    dx = 1.0 / n_x
    dt_cfl = 0.4 * dx / max(abs(A), 0.1)
    n_steps = max(int(t_end / dt_cfl) + 1, 200)
    dt = t_end / n_steps

    # Fourier wavenumbers
    k = np.fft.fftfreq(n_x, d=1.0/n_x)
    k2 = (2 * np.pi * k) ** 2
    ik = 1j * 2 * np.pi * k

    # Integrating factors for diffusion over one and half steps
    E  = np.exp(-nu * k2 * dt)
    E2 = np.exp(-nu * k2 * dt / 2)

    u = A * np.sin(2 * np.pi * x)
    u_hat = fft(u)

    # Save snapshots at evenly spaced output times
    save_idx = set(np.round(np.linspace(0, n_steps, n_t)).astype(int))
    u_all = []

    def nonlinear(u_hat_):
        """Conservative form of nonlinear term:  -(1/2) d/dx (u^2)."""
        u_ = np.real(ifft(u_hat_))
        return -0.5 * ik * fft(u_ ** 2)

    # Integrating-factor RK4 time stepping
    for i in range(n_steps + 1):
        if i in save_idx:
            u_all.append(np.real(ifft(u_hat)).copy())
        if i == n_steps:
            break
        N1 = nonlinear(u_hat)
        k1 = dt * N1
        N2 = nonlinear(E2 * u_hat + E2 * k1 / 2)
        k2_ = dt * N2
        N3 = nonlinear(E2 * u_hat + k2_ / 2)
        k3 = dt * N3
        N4 = nonlinear(E * u_hat + E * k3)
        k4 = dt * N4
        u_hat = E * u_hat + (E * k1 + 2 * E2 * k2_ + 2 * E2 * k3 + k4) / 6

    t = np.linspace(0, t_end, n_t)
    u_arr = np.stack(u_all[:n_t], axis=0)
    return x, t, u_arr


def _solve_burgers_task(args):
    """Worker for multiprocessing data generation."""
    lam, n_x, n_t = args
    x, t, u = solve_burgers_stable(lam, n_x, n_t)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    coords = np.stack([xx.flatten(), tt.flatten()], axis=-1).astype(np.float32)
    vals = u.T.flatten().astype(np.float32)
    return lam.astype(np.float32), coords, vals
