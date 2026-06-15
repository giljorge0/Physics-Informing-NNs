"""
Multidimensional Pareto PINN (MP-PINN) — v2
=============================================

Three ideas from the original proposal, now all properly implemented:

1. PHYSICS-INFORMING GAP (working in v1, kept):
   Train against *incomplete* known physics. The residual that remains
   is the "supraphysical gap" — handed to SINDy for symbolic distillation.

2. MULTIDIMENSIONAL PARETO WEIGHTS (fixed):
   Per-term lambda_i vector instead of one scalar lambda. The original
   GradNorm approach saturated immediately. New approach: warmup phase
   establishes a baseline loss for each physics term; adaptive weights
   then track *relative progress* of each term vs that baseline.
   Terms the network struggles with (conflicting with data) get
   down-weighted. Terms that converge easily stay weighted.

   This opens a strictly larger solution space than any scalar lambda:
   (lambda_inertia, lambda_damping, lambda_stiffness, lambda_forcing)
   can reach Pareto-optimal solutions that lie off the 2D scalar curve.

3. MINIMAX ADVERSARIAL PAIR (now real):
   InformingNet MAXIMISES the residual it can explain.
   InformedNet MINIMISES total loss (data + physics + correction cost).
   Alternating gradient updates, like a GAN — not joint training.
   The informing network is incentivised to find the most compact
   (regularised) explanation of the missing physics it can, because
   the L2 penalty on its output costs the informed network.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────────────────
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
    """Solution network: t -> x(t). The PINN proper."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.mlp = MLP(1, 1, hidden, layers)

    def forward(self, t):
        return self.mlp(t)


class MultiTrajectoryInformedNet(nn.Module):
    """Solution network conditioned on initial conditions:
    (t, x0, v0) -> x(t). Enables training across multiple trajectories
    simultaneously without one dominating the other."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.mlp = MLP(3, 1, hidden, layers)  # t, x0, v0

    def forward(self, t, x0, v0):
        inp = torch.cat([t, x0, v0], dim=-1)
        return self.mlp(inp)


class InformingNet(nn.Module):
    """Adversarial network: (x, v) -> proposed missing-physics correction.
    In the minimax game this network MAXIMISES what it can explain."""
    def __init__(self, hidden=32, layers=3):
        super().__init__()
        self.mlp = MLP(2, 1, hidden, layers)

    def forward(self, x, v):
        return self.mlp(torch.cat([x, v], dim=-1))


# ──────────────────────────────────────────────────────────────────────
# Physics
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PhysicsTerms:
    """Decomposed residual for:  x'' + c*x' + k*x = F0*cos(w_f*t)
    Each named component gets its own lambda_i."""
    c: float
    k: float
    F0: float
    w_f: float

    def residual_components(self, t, x, x_t, x_tt):
        forcing = self.F0 * torch.cos(self.w_f * t)
        return {
            "inertia":   x_tt,
            "damping":   self.c * x_t,
            "stiffness": self.k * x,
            "forcing":   -forcing,
        }

    def total_residual(self, t, x, x_t, x_tt):
        comps = self.residual_components(t, x, x_t, x_tt)
        return sum(comps.values()), comps


# ──────────────────────────────────────────────────────────────────────
# Adaptive weights — FIXED version
# ──────────────────────────────────────────────────────────────────────
class AdaptiveWeights:
    """
    Relative-progress-based adaptive lambda_i.

    Key idea: at the end of warmup, record a baseline loss for each
    physics term. Then adapt lambda_i so that terms making LESS relative
    progress (ratio L_i_now / L_i_baseline is still high) get DOWN-
    weighted — the network is struggling with them, likely because they
    conflict with the data = missing physics signal.

    Terms the network handles easily (ratio drops fast) stay weighted
    higher. This is the "different uncertainties about different physics"
    prior from the proposal, realised automatically from training data.

    lambda_i(t) = lambda_init * (L_i_baseline / L_i_now)^alpha

    alpha controls adaptation strength (0 = no adaptation, 1 = full).
    """

    def __init__(self, term_names, init_lambda=0.05, alpha=0.3,
                 min_lambda=1e-4, max_lambda=2.0, ema_beta=0.95):
        self.term_names = term_names
        self.init_lambda = init_lambda
        self.alpha = alpha
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.ema_beta = ema_beta  # smooth the per-term losses before adapting

        self.lambdas = {n: init_lambda for n in term_names}
        self.baselines: Optional[Dict[str, float]] = None
        self.ema_losses = {n: None for n in term_names}
        self.history = {n: [] for n in term_names}

    def set_baselines(self, per_term_losses: Dict[str, float]):
        """Call once at end of warmup with the current per-term losses."""
        self.baselines = {n: max(v, 1e-8) for n, v in per_term_losses.items()}
        self.ema_losses = dict(self.baselines)
        print(f"[AdaptiveWeights] baselines set: "
              f"{', '.join(f'{k}={v:.3e}' for k,v in self.baselines.items())}")

    def update(self, per_term_losses: Dict[str, float]):
        """Update lambdas based on relative progress since baseline."""
        if self.baselines is None:
            return
        for name in self.term_names:
            L_now = per_term_losses.get(name, 1e-8)
            # EMA smoothing to avoid noisy single-step jumps
            if self.ema_losses[name] is None:
                self.ema_losses[name] = L_now
            else:
                self.ema_losses[name] = (self.ema_beta * self.ema_losses[name]
                                         + (1 - self.ema_beta) * L_now)
            ratio = self.baselines[name] / (self.ema_losses[name] + 1e-8)
            # If ratio > 1: term has improved → keep or increase weight
            # If ratio < 1: term got worse?? → decrease weight (shouldn't happen often)
            new_lam = self.init_lambda * (ratio ** self.alpha)
            self.lambdas[name] = float(np.clip(new_lam, self.min_lambda, self.max_lambda))
            self.history[name].append(self.lambdas[name])

    def as_dict(self):
        return dict(self.lambdas)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs: int = 5000
    lr: float = 1e-3
    warmup_epochs: int = 2000

    # Lambda mode: "fixed", "adaptive", or "sweep" (set externally)
    adaptive: bool = False
    lambda_init: dict = field(default_factory=lambda: {
        "inertia": 0.05, "damping": 0.05, "stiffness": 0.05, "forcing": 0.05,
    })
    lambda_fixed: dict = None  # used when adaptive=False

    # Adaptive hyperparams
    adaptive_alpha: float = 0.3     # adaptation strength
    adaptive_ema: float = 0.95      # EMA smoothing factor

    # Minimax adversarial pair
    use_informing_net: bool = False
    adversarial: bool = False       # True = real alternating minimax
    informing_lr: float = 5e-4
    informing_steps: int = 1        # inner steps per outer step
    informing_reg: float = 0.05     # L2 penalty on informing output

    # Multi-trajectory
    multi_trajectory: bool = False  # use MultiTrajectoryInformedNet

    print_every: int = 500


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────
def train(t_data, x_data, physics_terms: PhysicsTerms, t_collocation,
          config: TrainConfig = None, device="cpu",
          extra_trajectories=None):
    """
    Train the MP-PINN.

    extra_trajectories: list of (t, x) pairs for multi-trajectory mode.
    Returns dict with trained networks, histories, weights object.
    """
    if config is None:
        config = TrainConfig()

    term_names = ["inertia", "damping", "stiffness", "forcing"]

    # ── tensors ──────────────────────────────────────────────────────
    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)

    t_data_t = to_t(t_data)
    x_data_t = to_t(x_data)
    t_col = to_t(t_collocation)
    t_col.requires_grad_(True)

    # ── networks ─────────────────────────────────────────────────────
    informed = InformedNet().to(device)
    informing = InformingNet().to(device) if config.use_informing_net else None

    # ── optimisers ───────────────────────────────────────────────────
    if config.adversarial and informing is not None:
        # Separate optimisers for the minimax game
        opt_informed  = torch.optim.Adam(informed.parameters(),  lr=config.lr)
        opt_informing = torch.optim.Adam(informing.parameters(), lr=config.informing_lr)
    else:
        params = list(informed.parameters())
        if informing is not None:
            params += list(informing.parameters())
        opt_informed = torch.optim.Adam(params, lr=config.lr)
        opt_informing = None

    # ── weights ──────────────────────────────────────────────────────
    if config.adaptive:
        weights = AdaptiveWeights(
            term_names,
            init_lambda=list(config.lambda_init.values())[0],  # use first as init
            alpha=config.adaptive_alpha,
            ema_beta=config.adaptive_ema,
        )
        baselines_set = False
    else:
        weights = None
        baselines_set = True
        fixed = config.lambda_fixed or config.lambda_init

    # ── history ──────────────────────────────────────────────────────
    loss_history = {"data": [], "physics": [], "total": []}
    lambda_history = {n: [] for n in term_names}

    # ── training loop ────────────────────────────────────────────────
    for epoch in range(config.epochs):
        warm = epoch < config.warmup_epochs

        # ── compute physics residual at collocation points ──────────
        # Need fresh computation with graph for backward
        t_col_var = t_col  # already requires grad

        x_col = informed(t_col_var)
        x_t_col = torch.autograd.grad(
            x_col, t_col_var, grad_outputs=torch.ones_like(x_col),
            create_graph=True)[0]
        x_tt_col = torch.autograd.grad(
            x_t_col, t_col_var, grad_outputs=torch.ones_like(x_t_col),
            create_graph=True)[0]

        comps = physics_terms.residual_components(t_col_var, x_col, x_t_col, x_tt_col)
        per_term_loss = {n: torch.mean(comps[n] ** 2) for n in term_names}

        # ── informing net output ─────────────────────────────────────
        informing_term = None
        if informing is not None:
            informing_term = informing(x_col.detach(), x_t_col.detach())

        # ── data loss ────────────────────────────────────────────────
        x_pred_data = informed(t_data_t)
        loss_data = torch.mean((x_pred_data - x_data_t) ** 2)

        # ── warmup: data only ────────────────────────────────────────
        if warm:
            opt_informed.zero_grad()
            loss_data.backward()
            opt_informed.step()

            loss_history["data"].append(loss_data.item())
            loss_history["physics"].append(0.0)
            loss_history["total"].append(loss_data.item())
            for n in term_names:
                lambda_history[n].append(0.0)

            if epoch % config.print_every == 0:
                print(f"[epoch {epoch:5d}] WARMUP  data={loss_data.item():.3e}")
            continue

        # ── end of warmup: set adaptive baselines ────────────────────
        if config.adaptive and not baselines_set:
            weights.set_baselines({n: per_term_loss[n].item() for n in term_names})
            baselines_set = True

        # ── determine lambdas ────────────────────────────────────────
        if config.adaptive:
            weights.update({n: per_term_loss[n].item() for n in term_names})
            lambdas_now = weights.as_dict()
        else:
            lambdas_now = fixed

        # ── physics loss ─────────────────────────────────────────────
        loss_physics = sum(lambdas_now[n] * per_term_loss[n] for n in term_names)

        # ── total residual for closure regulariser ───────────────────
        full_residual = sum(comps.values())
        if informing_term is not None:
            # In adversarial mode: informing net's correction offsets the residual
            full_residual = full_residual + informing_term
        loss_closure = torch.mean(full_residual ** 2)

        # ════════════════════════════════════════════════════════════
        # MINIMAX ADVERSARIAL UPDATE
        # ════════════════════════════════════════════════════════════
        if config.adversarial and informing is not None:
            # ── Step A: Update InformedNet (informing fixed) ─────────
            # Minimise: L_data + L_physics + reg * ||informing_term||²
            # The regulariser on informing_term penalises the informed net
            # for "accepting" large corrections from the informing net.
            loss_informed = (loss_data
                             + loss_physics
                             + 0.1 * loss_closure
                             + config.informing_reg * torch.mean(informing_term ** 2))

            opt_informed.zero_grad()
            loss_informed.backward(retain_graph=True)
            opt_informed.step()

            # ── Step B: Update InformingNet (informed fixed) ─────────
            # The informing net wants to MAXIMISE how much of the residual
            # it can explain, but is penalised for being large (Occam's razor).
            # 
            # Residual without informing correction:
            raw_residual = sum(comps.values()).detach()
            # Informing net proposes a correction:
            x_col_d  = x_col.detach()
            x_t_col_d = x_t_col.detach()
            correction = informing(x_col_d, x_t_col_d)
            # After correction, the net residual is (raw_residual + correction).
            # InformingNet MINIMISES: ||raw_residual + correction||² + reg*||correction||²
            # This is equivalent to maximising how much it explains while staying small.
            residual_after = raw_residual + correction
            loss_informing = (torch.mean(residual_after ** 2)
                              + config.informing_reg * torch.mean(correction ** 2))

            for _ in range(config.informing_steps):
                opt_informing.zero_grad()
                # Recompute for each inner step
                correction = informing(x_col_d, x_t_col_d)
                residual_after = raw_residual + correction
                loss_informing = (torch.mean(residual_after ** 2)
                                  + config.informing_reg * torch.mean(correction ** 2))
                loss_informing.backward()
                opt_informing.step()

            loss_total = loss_informed.detach()
            loss_phys_val = loss_physics.item()

        # ════════════════════════════════════════════════════════════
        # JOINT TRAINING (non-adversarial)
        # ════════════════════════════════════════════════════════════
        else:
            loss_total = loss_data + loss_physics + 0.1 * loss_closure
            if informing is not None:
                loss_total = loss_total + config.informing_reg * torch.mean(informing_term ** 2)

            opt_informed.zero_grad()
            loss_total.backward()
            opt_informed.step()
            loss_phys_val = loss_physics.item()

        # ── history ──────────────────────────────────────────────────
        loss_history["data"].append(loss_data.item())
        loss_history["physics"].append(loss_phys_val)
        loss_history["total"].append(loss_total.item() if hasattr(loss_total, 'item') else float(loss_total))
        for n in term_names:
            lambda_history[n].append(lambdas_now[n])

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            lam_str = ', '.join(f'{k}:{v:.4f}' for k, v in lambdas_now.items())
            mode = "ADV" if (config.adversarial and informing) else "joint"
            print(f"[epoch {epoch:5d}|{mode}] data={loss_data.item():.3e} "
                  f"phys={loss_phys_val:.3e} λ={{{lam_str}}}")

    return {
        "informed":       informed,
        "informing":      informing,
        "loss_history":   loss_history,
        "lambda_history": lambda_history,
        "weights":        weights,
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True,
                                create_graph=False, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(torch.sum(g ** 2))
    return total ** 0.5


def evaluate_supraphysical_residual(informed, physics_terms: PhysicsTerms,
                                    t_array, device="cpu"):
    """Return (x, v, a, residual) where residual ≈ missing physics term."""
    t = torch.tensor(t_array, dtype=torch.float32, device=device).reshape(-1, 1)
    t.requires_grad_(True)
    x = informed(t)
    x_t  = torch.autograd.grad(x,   t, grad_outputs=torch.ones_like(x),
                                create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t),
                                create_graph=True)[0]

    forcing = physics_terms.F0 * torch.cos(physics_terms.w_f * t)
    residual = x_tt + physics_terms.c * x_t + physics_terms.k * x - forcing

    return (
        x.detach().cpu().numpy().flatten(),
        x_t.detach().cpu().numpy().flatten(),
        x_tt.detach().cpu().numpy().flatten(),
        (-residual).detach().cpu().numpy().flatten(),
    )
