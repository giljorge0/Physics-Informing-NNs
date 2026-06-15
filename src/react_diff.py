"""
1D Reaction-Diffusion PDE extension.
======================================

True PDE (ground truth):
    u_t = D * u_xx + r * u * (1 - u) + eps * sin(pi*x) * u²

"Known physics" given to the PINN:
    u_t = D * u_xx + r * u * (1 - u)

Hidden term:  eps * sin(pi*x) * u²

This is the Fisher-KPP equation (logistic reaction + diffusion) with
a spatially-varying quadratic source term hidden from the model.

The per-term Pareto decomposition now has FOUR spatial operators:
  - L_diffusion:  D * u_xx   (usually well-known)
  - L_linear_rxn: r * u      (linear part, trusted)
  - L_nonlin_rxn: -r * u²    (nonlinear part, less certain)
  - L_temporal:   u_t        (always exact)

The "missing" eps*sin(pi*x)*u² should be recovered by distillation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded


# ── True parameters ────────────────────────────────────────────────────
TRUE_PARAMS_PDE = dict(
    D   = 0.01,   # diffusion coefficient
    r   = 1.0,    # reaction rate
    eps = 0.3,    # hidden spatially-modulated source coefficient
    L   = 1.0,    # spatial domain [0, L]
)


def make_grid(Nx=64, params=None):
    p = params or TRUE_PARAMS_PDE
    x = np.linspace(0, p["L"], Nx)
    dx = x[1] - x[0]
    return x, dx


def true_rhs_pde(t, u, x, dx, p=None):
    """Full RHS including hidden term."""
    p = p or TRUE_PARAMS_PDE
    N = len(u)
    # Diffusion: second-order central differences with Neumann BC
    uxx = np.zeros(N)
    uxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    uxx[0]    = (u[1] - u[0]) / dx**2      # zero-flux BC
    uxx[-1]   = (u[-2] - u[-1]) / dx**2

    reaction = p["r"] * u * (1 - u)
    hidden   = p["eps"] * np.sin(np.pi * x) * u**2
    return p["D"] * uxx + reaction + hidden


def incomplete_rhs_pde(t, u, x, dx, p=None):
    """Known physics only — missing the hidden source."""
    p = p or TRUE_PARAMS_PDE
    N = len(u)
    uxx = np.zeros(N)
    uxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    uxx[0]    = (u[1] - u[0]) / dx**2
    uxx[-1]   = (u[-2] - u[-1]) / dx**2
    reaction = p["r"] * u * (1 - u)
    return p["D"] * uxx + reaction


def hidden_term_pde(x, u, p=None):
    """The exact missing term for evaluation."""
    p = p or TRUE_PARAMS_PDE
    return p["eps"] * np.sin(np.pi * x) * u**2


def generate_data_pde(t_max=2.0, Nt=50, Nx=64, noise_std=0.005,
                       seed=0, params=None):
    """
    Integrate the true PDE and return noisy (t, x, u) data.

    Returns:
        t_eval:  (Nt,)
        x_grid:  (Nx,)
        u_noisy: (Nt, Nx)  — noisy observations
        u_clean: (Nt, Nx)  — clean truth
    """
    p = params or TRUE_PARAMS_PDE
    x, dx = make_grid(Nx, p)
    t_eval = np.linspace(0, t_max, Nt)

    # Initial condition: small smooth perturbation
    rng = np.random.default_rng(seed)
    u0 = 0.5 * np.exp(-50 * (x - 0.5)**2) + 0.02 * rng.standard_normal(Nx)
    u0 = np.clip(u0, 0, 1)

    sol = solve_ivp(
        true_rhs_pde,
        [0, t_max], u0,
        t_eval=t_eval,
        args=(x, dx, p),
        method="RK45", rtol=1e-7, atol=1e-8,
        dense_output=False,
    )
    u_clean = sol.y.T  # (Nt, Nx)
    u_noisy = u_clean + rng.normal(0, noise_std, u_clean.shape)
    return t_eval, x, u_noisy, u_clean


# ── SINDy library for PDE case ─────────────────────────────────────────
# The distillation library now needs spatial basis functions too.
# We build a (Nt*Nx, n_terms) feature matrix from (x, u).

PDE_CANDIDATE_LIBRARY = {
    "u":             lambda x, u: u,
    "u^2":           lambda x, u: u**2,
    "u^3":           lambda x, u: u**3,
    "sin(pi*x)":     lambda x, u: np.sin(np.pi * x),
    "sin(pi*x)*u":   lambda x, u: np.sin(np.pi * x) * u,
    "sin(pi*x)*u^2": lambda x, u: np.sin(np.pi * x) * u**2,
    "cos(pi*x)*u^2": lambda x, u: np.cos(np.pi * x) * u**2,
    "x*(1-x)*u^2":   lambda x, u: x * (1-x) * u**2,
    "u*(1-u)":        lambda x, u: u * (1 - u),
    "1":              lambda x, u: np.ones_like(u),
}


def build_pde_library(x_flat, u_flat):
    """x_flat, u_flat: 1D arrays of flattened (t,x) grid values."""
    cols, names = [], []
    for name, fn in PDE_CANDIDATE_LIBRARY.items():
        cols.append(fn(x_flat, u_flat))
        names.append(name)
    return np.stack(cols, axis=1), names


if __name__ == "__main__":
    t, x, u_noisy, u_clean = generate_data_pde()
    print(f"PDE data: t={t.shape}, x={x.shape}, u={u_clean.shape}")
    print(f"u range: [{u_clean.min():.3f}, {u_clean.max():.3f}]")
    print(f"Hidden term magnitude: "
          f"{np.abs(hidden_term_pde(x[None,:], u_clean)).mean():.4f}")
