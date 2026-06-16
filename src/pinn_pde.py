"""
PDE Physics-Informing Neural Network
======================================

Extension of the ODE framework to 1D PDEs (Fisher-KPP / reaction-diffusion).

Key differences from the ODE case:
  - Network takes (t, x) as input, outputs u(t, x)
  - Physics residual involves spatial derivatives (u_xx) computed via autograd
  - Per-term decomposition: temporal, diffusion, linear reaction, nonlinear reaction
  - Collocation points are a 2D grid (t, x) not just t
  - Distillation library depends on (x, u) not (x, v)

The same three ideas apply:
  1. Physics-informing gap: network violates incomplete PDE, residual → SINDy
  2. Multidimensional Pareto: per-term λ_i for each PDE operator
  3. AdaptiveWeights: same relative-progress scheme, now for PDE terms

Architecture: u_net(t, x) → u
  Inputs normalised to [-1,1]. Tanh activation throughout.
  Spatial and temporal derivatives via torch.autograd.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────
class PDEMLP(nn.Module):
    """u(t, x) — solution network for 1D PDE."""
    def __init__(self, hidden=64, layers=5, activation=nn.Tanh):
        super().__init__()
        mods = [nn.Linear(2, hidden), activation()]  # 2 inputs: t, x
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), activation()]
        mods += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=-1)
        return self.net(inp)


class PDEInformingNet(nn.Module):
    """Adversarial: (x, u) -> proposed missing source term.
    The missing term in this PDE is a function of position and solution value,
    so we condition on both x (spatial location) and u (current solution)."""
    def __init__(self, hidden=32, layers=3, activation=nn.Tanh):
        super().__init__()
        mods = [nn.Linear(2, hidden), activation()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), activation()]
        mods += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=-1))


# ──────────────────────────────────────────────────────────────────────
# PDE physics terms
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PDEPhysicsTerms:
    """
    Decomposed residual for:  u_t = D*u_xx + r*u*(1-u)   [incomplete known]

    True PDE:  u_t = D*u_xx + r*u*(1-u) + eps*sin(pi*x)*u²

    Residual form (everything to one side):
        u_t - D*u_xx - r*u + r*u² = 0   [known, should be zero]
    
    Components:
        temporal:     u_t
        diffusion:   -D * u_xx
        lin_rxn:     -r * u
        nonlin_rxn:  +r * u²
    
    Sum = 0 is the incomplete physics constraint.
    Hidden term eps*sin(pi*x)*u² will appear as residual.
    """
    D: float  # diffusion coefficient
    r: float  # reaction rate

    def residual_components(self, t, x, u, u_t, u_x, u_xx):
        return {
            "temporal":   u_t,
            "diffusion":  -self.D * u_xx,
            "lin_rxn":    -self.r * u,
            "nonlin_rxn": self.r * u ** 2,
        }


# ──────────────────────────────────────────────────────────────────────
# Adaptive weights (same as ODE version, reused)
# ──────────────────────────────────────────────────────────────────────
class AdaptiveWeightsPDE:
    """Relative-progress-based per-term lambda_i, PDE version."""
    def __init__(self, term_names, init_lambda=0.1, alpha=0.7,
                 min_lambda=1e-4, max_lambda=5.0, ema_beta=0.7):
        self.term_names  = term_names
        self.init_lambda = init_lambda
        self.alpha       = alpha
        self.min_lambda  = min_lambda
        self.max_lambda  = max_lambda
        self.ema_beta    = ema_beta
        self.lambdas     = {n: init_lambda for n in term_names}
        self.baselines: Optional[Dict[str, float]] = None
        self.ema_losses  = {n: None for n in term_names}
        self.history     = {n: [] for n in term_names}

    def set_baselines(self, per_term_losses):
        self.baselines  = {n: max(v, 1e-8) for n, v in per_term_losses.items()}
        self.ema_losses = dict(self.baselines)
        print(f"[AdaptiveWeights PDE] baselines: "
              f"{', '.join(f'{k}={v:.3e}' for k,v in self.baselines.items())}")

    def update(self, per_term_losses):
        if self.baselines is None:
            return
        for name in self.term_names:
            L_now = per_term_losses.get(name, 1e-8)
            if self.ema_losses[name] is None:
                self.ema_losses[name] = L_now
            else:
                self.ema_losses[name] = (self.ema_beta * self.ema_losses[name]
                                         + (1 - self.ema_beta) * L_now)
            ratio   = self.baselines[name] / (self.ema_losses[name] + 1e-8)
            new_lam = self.init_lambda * (ratio ** self.alpha)
            self.lambdas[name] = float(np.clip(new_lam, self.min_lambda, self.max_lambda))
            self.history[name].append(self.lambdas[name])

    def as_dict(self):
        return dict(self.lambdas)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PDETrainConfig:
    epochs:        int   = 8000
    lr:            float = 5e-4
    warmup_epochs: int   = 3000

    adaptive:       bool  = True
    adaptive_alpha: float = 0.7
    adaptive_ema:   float = 0.7
    lambda_init:    dict  = field(default_factory=lambda: {
        "temporal": 0.1, "diffusion": 0.1, "lin_rxn": 0.1, "nonlin_rxn": 0.1,
    })
    lambda_fixed:   dict  = None

    use_informing_net: bool  = False
    adversarial:       bool  = False
    informing_lr:      float = 2e-4
    informing_steps:   int   = 2
    informing_reg:     float = 0.1

    # Normalisation of t and x inputs to [-1, 1]
    t_max: float = 2.0
    x_max: float = 1.0

    print_every: int = 1000


# ──────────────────────────────────────────────────────────────────────
# Helpers: autograd derivatives for PDE
# ──────────────────────────────────────────────────────────────────────
def _grad(y, x, create=True):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=create, retain_graph=True)[0]


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_pde(t_data_np, x_data_np, u_data_np,
              pde_terms: PDEPhysicsTerms,
              t_col_np, x_col_np,
              config: PDETrainConfig = None,
              device="cpu"):
    """
    Train PDE MP-PINN.

    t_data_np, x_data_np, u_data_np: flat arrays of (t,x,u) data points
    t_col_np, x_col_np: collocation points for physics residual
    """
    if config is None:
        config = PDETrainConfig()

    term_names = ["temporal", "diffusion", "lin_rxn", "nonlin_rxn"]

    # ── normalise inputs to [-1, 1] for better NN conditioning ───────
    def norm_t(t): return 2 * t / config.t_max - 1
    def norm_x(x): return 2 * x / config.x_max - 1

    def to_t(arr, req_grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
        if req_grad: t.requires_grad_(True)
        return t

    # Data tensors (normalised)
    t_d  = to_t(norm_t(t_data_np))
    x_d  = to_t(norm_x(x_data_np))
    u_d  = to_t(u_data_np)

    # Collocation tensors (normalised, require grad for autograd)
    t_c = to_t(norm_t(t_col_np), req_grad=True)
    x_c = to_t(norm_x(x_col_np), req_grad=True)

    # ── networks ─────────────────────────────────────────────────────
    u_net    = PDEMLP().to(device)
    inf_net  = PDEInformingNet().to(device) if config.use_informing_net else None

    # ── optimisers ───────────────────────────────────────────────────
    if config.adversarial and inf_net is not None:
        opt_u   = torch.optim.Adam(u_net.parameters(),   lr=config.lr)
        opt_inf = torch.optim.Adam(inf_net.parameters(), lr=config.informing_lr)
    else:
        params  = list(u_net.parameters())
        if inf_net is not None: params += list(inf_net.parameters())
        opt_u   = torch.optim.Adam(params, lr=config.lr)
        opt_inf = None

    # ── weights ──────────────────────────────────────────────────────
    if config.adaptive:
        weights      = AdaptiveWeightsPDE(term_names, alpha=config.adaptive_alpha,
                                           ema_beta=config.adaptive_ema,
                                           init_lambda=list(config.lambda_init.values())[0])
        baselines_set = False
    else:
        weights       = None
        baselines_set = True
        fixed         = config.lambda_fixed or config.lambda_init

    # ── history ──────────────────────────────────────────────────────
    loss_history   = {"data": [], "physics": [], "total": []}
    lambda_history = {n: [] for n in term_names}

    # ── training loop ────────────────────────────────────────────────
    for epoch in range(config.epochs):
        warm = epoch < config.warmup_epochs

        # ── data loss ────────────────────────────────────────────────
        u_pred_data = u_net(t_d, x_d)
        loss_data   = torch.mean((u_pred_data - u_d) ** 2)

        # ── PDE residual at collocation points ───────────────────────
        u_c    = u_net(t_c, x_c)
        u_t    = _grad(u_c,  t_c)
        u_x    = _grad(u_c,  x_c)
        u_xx   = _grad(u_x,  x_c)

        comps = pde_terms.residual_components(t_c, x_c, u_c, u_t, u_x, u_xx)
        per_term_loss = {n: torch.mean(comps[n] ** 2) for n in term_names}

        # ── informing net ─────────────────────────────────────────────
        inf_term = None
        if inf_net is not None:
            inf_term = inf_net(x_c.detach(), u_c.detach())

        # ── warmup ───────────────────────────────────────────────────
        if warm:
            opt_u.zero_grad()
            loss_data.backward()
            opt_u.step()
            loss_history["data"].append(loss_data.item())
            loss_history["physics"].append(0.0)
            loss_history["total"].append(loss_data.item())
            for n in term_names: lambda_history[n].append(0.0)
            if epoch % config.print_every == 0:
                print(f"[epoch {epoch:5d}] WARMUP  data={loss_data.item():.3e}")
            continue

        # ── set baselines ─────────────────────────────────────────────
        if config.adaptive and not baselines_set:
            weights.set_baselines({n: per_term_loss[n].item() for n in term_names})
            baselines_set = True

        # ── lambdas ──────────────────────────────────────────────────
        if config.adaptive:
            weights.update({n: per_term_loss[n].item() for n in term_names})
            lambdas_now = weights.as_dict()
        else:
            lambdas_now = fixed

        # ── physics loss ─────────────────────────────────────────────
        loss_physics = sum(lambdas_now[n] * per_term_loss[n] for n in term_names)
        full_residual = sum(comps.values())
        if inf_term is not None:
            full_residual = full_residual + inf_term
        loss_closure = torch.mean(full_residual ** 2)

        # ── adversarial update ───────────────────────────────────────
        if config.adversarial and inf_net is not None:
            loss_u = (loss_data + loss_physics + 0.1 * loss_closure
                      + config.informing_reg * torch.mean(inf_term ** 2))
            opt_u.zero_grad()
            loss_u.backward(retain_graph=True)
            opt_u.step()

            raw_res   = sum(comps.values()).detach()
            x_c_d     = x_c.detach()
            u_c_d     = u_c.detach()
            for _ in range(config.informing_steps):
                opt_inf.zero_grad()
                corr        = inf_net(x_c_d, u_c_d)
                res_after   = raw_res + corr   # correction should cancel residual
                loss_inf    = (torch.mean(res_after ** 2)
                               + config.informing_reg * torch.mean(corr ** 2))
                loss_inf.backward()
                opt_inf.step()
            loss_total = loss_u.detach()
        else:
            loss_total = loss_data + loss_physics + 0.1 * loss_closure
            if inf_net is not None:
                loss_total = loss_total + config.informing_reg * torch.mean(inf_term ** 2)
            opt_u.zero_grad()
            loss_total.backward()
            opt_u.step()

        # ── history ──────────────────────────────────────────────────
        loss_history["data"].append(loss_data.item())
        loss_history["physics"].append(loss_physics.item())
        loss_history["total"].append(float(loss_total.item() if hasattr(loss_total,"item") else loss_total))
        for n in term_names: lambda_history[n].append(lambdas_now[n])

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            lam_str = ", ".join(f"{k}:{v:.4f}" for k,v in lambdas_now.items())
            print(f"[epoch {epoch:5d}] data={loss_data.item():.3e} "
                  f"phys={loss_physics.item():.3e} λ={{{lam_str}}}")

    return {
        "u_net":          u_net,
        "inf_net":        inf_net,
        "loss_history":   loss_history,
        "lambda_history": lambda_history,
        "weights":        weights,
        "config":         config,
    }


# ──────────────────────────────────────────────────────────────────────
# Residual evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_pde_residual(u_net, pde_terms: PDEPhysicsTerms,
                           t_np, x_np, config: PDETrainConfig,
                           device="cpu"):
    """
    Evaluate the supraphysical gap at arbitrary (t,x) points.
    Returns u_pred, residual (≈ -hidden_term), and true sign residual.

    residual = u_t - D*u_xx - r*u + r*u²   [should ≈ hidden term]
    """
    def norm_t(t): return 2 * t / config.t_max - 1
    def norm_x(x): return 2 * x / config.x_max - 1

    t = torch.tensor(norm_t(t_np), dtype=torch.float32, device=device).reshape(-1,1)
    x = torch.tensor(norm_x(x_np), dtype=torch.float32, device=device).reshape(-1,1)
    t.requires_grad_(True); x.requires_grad_(True)

    u   = u_net(t, x)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx= _grad(u_x, x, create=False)

    # Known physics residual (should be zero if physics were complete)
    physics_res = (u_t
                   - pde_terms.D   * u_xx
                   - pde_terms.r   * u
                   + pde_terms.r   * u ** 2)

    # Supraphysical gap = what must be added to satisfy the true PDE
    # = -physics_res  (since true PDE = 0 means known_physics + hidden = 0)
    gap = -physics_res

    return (
        x.detach().cpu().numpy().flatten(),
        u.detach().cpu().numpy().flatten(),
        gap.detach().cpu().numpy().flatten(),
    )
