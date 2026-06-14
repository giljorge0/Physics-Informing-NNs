"""
Multidimensional Pareto PINN (MP-PINN)
=======================================

Core idea (from the conceptual proposal):

Standard PINN loss:
    L_total = L_data + lambda * L_physics

This module decomposes L_physics into a VECTOR of per-term residuals,
each with its own weight lambda_i:

    L_total = L_data
              + lambda_inertia   * L_inertia      (x'' term, always trusted)
              + lambda_damping   * L_damping       (c*x' term)
              + lambda_stiffness * L_stiffness     (k*x term)

Each lambda_i can be:
  - fixed (user-specified prior confidence in that physics term), or
  - adaptive, updated via gradient-norm balancing (a simplified
    GradNorm/PCGrad-style scheme) so that no single term's gradient
    dominates training and "uncertain" terms are automatically
    down-weighted if they conflict with the data.

The NETWORK is allowed to produce "supraphysical" predictions: nothing
forces the total physics residual to zero. Instead, the residual that
remains (data fits well, physics doesn't fully) becomes the SIGNAL we
hand to the symbolic-regression distillation step (sindy_distill.py)
to discover the missing term.

We additionally train a small "informing" head (the adversarial /
minimax counterpart) whose job is to predict the *physics residual*
directly from (x, v, t) -- i.e. it tries to "inform" what physics is
missing, while the main PINN ("informed" network) tries to fit data
+ known physics. Both are trained jointly; the informing head's output
is exactly the supraphysical gap, ready for symbolic distillation.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Network architectures
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, layers=4, activation=nn.Tanh):
        super().__init__()
        mods = [nn.Linear(in_dim, hidden), activation()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), activation()]
        mods += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class InformedNet(nn.Module):
    """The 'informed' network: x(t) -> predicted state. This is the
    standard PINN solution network."""

    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.mlp = MLP(1, 1, hidden, layers)

    def forward(self, t):
        return self.mlp(t)


class InformingNet(nn.Module):
    """The 'informing' network: (x, v) -> predicted missing-physics
    residual. This is the adversarial/minimax counterpart that tries
    to explain away whatever the InformedNet + known physics cannot."""

    def __init__(self, hidden=32, layers=3):
        super().__init__()
        self.mlp = MLP(2, 1, hidden, layers)

    def forward(self, x, v):
        return self.mlp(torch.cat([x, v], dim=-1))


# ----------------------------------------------------------------------
# Decomposed physics residual terms
# ----------------------------------------------------------------------
@dataclass
class PhysicsTerms:
    """Container for individually-weighted residual components of
    m*x'' + c*x' + k*x - F(t) = 0  (incomplete known physics)."""
    c: float
    k: float
    F0: float
    w_f: float

    def residual_components(self, t, x, x_t, x_tt):
        """Return a dict of named residual tensors (NOT yet squared/weighted)."""
        forcing = self.F0 * torch.cos(self.w_f * t)
        return {
            "inertia": x_tt,
            "damping": self.c * x_t,
            "stiffness": self.k * x,
            "forcing": -forcing,
        }

    def total_residual(self, t, x, x_t, x_tt, informing_term=None):
        comps = self.residual_components(t, x, x_t, x_tt)
        r = comps["inertia"] + comps["damping"] + comps["stiffness"] + comps["forcing"]
        if informing_term is not None:
            r = r + informing_term  # the informing net's proposed correction
        return r, comps


# ----------------------------------------------------------------------
# Adaptive lambda (per-term weight) manager
# ----------------------------------------------------------------------
class AdaptiveWeights:
    """Tracks per-term lambda_i, updated by gradient-norm balancing.

    Simplified GradNorm: lambda_i is rescaled so that the gradient norm
    of lambda_i * L_i (w.r.t. shared params) tracks a target ratio
    relative to L_data's gradient norm. Terms the network "fights"
    (high uncertainty / conflicting with data) get automatically
    down-weighted.
    """

    def __init__(self, term_names, init_lambda=1.0, alpha=0.02, min_lambda=1e-3, max_lambda=10.0):
        self.term_names = term_names
        self.lambdas = {name: init_lambda for name in term_names}
        self.alpha = alpha
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.history = {name: [] for name in term_names}

    def update(self, grad_norms: dict, data_grad_norm: float):
        """grad_norms: {term_name: ||grad(lambda_i * L_i)||}"""
        eps = 1e-8
        for name in self.term_names:
            target = data_grad_norm
            current = grad_norms.get(name, eps) + eps
            ratio = target / current
            new_lambda = self.lambdas[name] * (1 - self.alpha) + self.lambdas[name] * ratio * self.alpha
            self.lambdas[name] = float(np.clip(new_lambda, self.min_lambda, self.max_lambda))
            self.history[name].append(self.lambdas[name])

    def as_dict(self):
        return dict(self.lambdas)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 4000
    lr: float = 1e-3
    adaptive: bool = True
    lambda_init: dict = field(default_factory=lambda: {
        "inertia": 1.0, "damping": 1.0, "stiffness": 1.0, "forcing": 1.0,
    })
    lambda_fixed: dict = None  # if set (and adaptive=False), use these fixed values
    use_informing_net: bool = True
    informing_weight: float = 0.01  # regularizer keeping informing net small
    print_every: int = 500
    warmup_epochs: int = 500


def train(t_data, x_data, physics_terms: PhysicsTerms, t_collocation,
          config: TrainConfig = TrainConfig(), device="cpu"):

    t_data_t = torch.tensor(t_data, dtype=torch.float32, device=device).reshape(-1, 1)
    x_data_t = torch.tensor(x_data, dtype=torch.float32, device=device).reshape(-1, 1)
    t_col = torch.tensor(t_collocation, dtype=torch.float32, device=device).reshape(-1, 1)
    t_col.requires_grad_(True)

    informed = InformedNet().to(device)
    informing = InformingNet().to(device) if config.use_informing_net else None

    params = list(informed.parameters())
    if informing is not None:
        params += list(informing.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    term_names = ["inertia", "damping", "stiffness", "forcing"]
    if config.adaptive:
        weights = AdaptiveWeights(term_names, init_lambda=1.0)
    else:
        weights = None
        fixed = config.lambda_fixed or config.lambda_init

    loss_history = {"data": [], "physics": [], "total": []}
    lambda_history = {name: [] for name in term_names}

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        # --- data loss ---
        x_pred_data = informed(t_data_t)
        loss_data = torch.mean((x_pred_data - x_data_t) ** 2)

        # --- physics loss (collocation points) ---
        x_pred = informed(t_col)
        x_t = torch.autograd.grad(x_pred, t_col, grad_outputs=torch.ones_like(x_pred),
                                    create_graph=True)[0]
        x_tt = torch.autograd.grad(x_t, t_col, grad_outputs=torch.ones_like(x_t),
                                     create_graph=True)[0]

        informing_term = None
        if informing is not None:
            informing_term = informing(x_pred, x_t)

        comps = physics_terms.residual_components(t_col, x_pred, x_t, x_tt)

        per_term_loss = {}
        for name in term_names:
            per_term_loss[name] = torch.mean(comps[name] ** 2)

        # full residual including informing-net correction
        full_residual = sum(comps.values())
        if informing_term is not None:
            full_residual = full_residual + informing_term
        loss_residual = torch.mean(full_residual ** 2)

        warm = epoch < config.warmup_epochs
        if warm:
            loss_total = loss_data
            loss_physics = loss_residual.detach() * 0.0  # placeholder, zero
            lambdas_now = {n: 0.0 for n in term_names}
        else:
            # --- adaptive weight update via gradient norms ---
            if config.adaptive:
                grad_norms = {}
                data_grad_norm = _grad_norm(loss_data, params)
                for name in term_names:
                    lam = weights.lambdas[name]
                    grad_norms[name] = _grad_norm(lam * per_term_loss[name], params)
                weights.update(grad_norms, data_grad_norm)
                lambdas_now = weights.as_dict()
            else:
                lambdas_now = fixed

            loss_physics = sum(lambdas_now[name] * per_term_loss[name] for name in term_names)
            loss_physics = loss_physics + 0.1 * loss_residual  # light residual-closure regularizer

            loss_total = loss_data + loss_physics
            if informing is not None:
                loss_total = loss_total + config.informing_weight * torch.mean(informing_term ** 2)

        loss_total.backward()
        optimizer.step()

        loss_history["data"].append(loss_data.item())
        loss_history["physics"].append(loss_physics.item())
        loss_history["total"].append(loss_total.item())
        for name in term_names:
            lambda_history[name].append(lambdas_now[name])

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            print(f"[epoch {epoch:5d}] data={loss_data.item():.3e} "
                  f"physics={loss_physics.item():.3e} "
                  f"lambdas={{{', '.join(f'{k}:{v:.3f}' for k,v in lambdas_now.items())}}}")

    return {
        "informed": informed,
        "informing": informing,
        "loss_history": loss_history,
        "lambda_history": lambda_history,
        "weights": weights,
    }


def _grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(torch.sum(g ** 2))
    return total ** 0.5


# ----------------------------------------------------------------------
# Convenience: extract residuals at arbitrary points for distillation
# ----------------------------------------------------------------------
def evaluate_supraphysical_residual(informed, physics_terms: PhysicsTerms, t_array, device="cpu"):
    """Compute the 'supraphysical gap': what the network's solution implies
    is missing from the known physics, i.e.
        residual(t) = x'' + c*x' + k*x - F(t)
    evaluated using the trained network's x(t). This residual should
    approximate eps*x^3 + delta*sign(v)*v^2 from the true system."""
    t = torch.tensor(t_array, dtype=torch.float32, device=device).reshape(-1, 1)
    t.requires_grad_(True)
    x = informed(t)
    x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]

    forcing = physics_terms.F0 * torch.cos(physics_terms.w_f * t)
    residual = x_tt + physics_terms.c * x_t + physics_terms.k * x - forcing

    return (
        x.detach().cpu().numpy().flatten(),
        x_t.detach().cpu().numpy().flatten(),
        x_tt.detach().cpu().numpy().flatten(),
        (-residual).detach().cpu().numpy().flatten(),  # missing term ≈ -residual
    )
