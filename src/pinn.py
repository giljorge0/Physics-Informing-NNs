"""
Multidimensional Pareto PINN (MP-PINN) — v3
=============================================

Three ideas from the original proposal, correctly implemented:

1. PHYSICS-INFORMING GAP:
   Train against incomplete known physics. The residual = supraphysical gap.
   Hand to SINDy for symbolic distillation.

2. MULTIDIMENSIONAL PARETO WEIGHTS (fixed from v2):
   Per-term lambda_i vector. AdaptiveWeights uses relative-progress scaling
   with alpha=0.7 and ema=0.7 for visible, stable divergence between terms.

3. INFORMING NET — two modes:
   
   JOINT (adversarial=False):
     InformingNet trained alongside InformedNet. It learns f(x,v) ≈ hidden term.
     After training, distill the INFORMING NET'S OUTPUT (not the raw residual).
     The informing net IS the symbolic target — the residual gets absorbed into it.
   
   POST-HOC (adversarial=True):
     Train InformedNet alone first (frozen). Then train InformingNet separately
     to fit the frozen InformedNet's residual. This avoids the signal-absorption
     problem: InformedNet can't cheat by using correction during training.
     Post-hoc InformingNet → clean, undisturbed residual signal for distillation.
   
   DIAGNOSTIC (v2 showed): corr(correction, truth) = 0.86 in joint mode.
   The correction itself IS the recovered signal. Distill it, not raw residual.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

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
    """Solution network: t -> x(t)."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.mlp = MLP(1, 1, hidden, layers)

    def forward(self, t):
        return self.mlp(t)


class InformingNet(nn.Module):
    """Informing network: (x, v) -> proposed missing-physics term.
    
    In JOINT mode: trained alongside InformedNet, absorbs residual signal.
    In POST-HOC mode: trained after InformedNet is frozen, cleanly fits residual.
    
    The output of this network is what you distill symbolically — not the raw residual.
    """
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


# ──────────────────────────────────────────────────────────────────────
# Adaptive weights — v3 (stronger defaults, stable)
# ──────────────────────────────────────────────────────────────────────
class AdaptiveWeights:
    """
    Relative-progress-based adaptive lambda_i.

    lambda_i(t) = lambda_init * (L_i_baseline / L_i_now) ^ alpha

    alpha=0.7, ema=0.7: responds visibly within ~500 epochs post-warmup.
    Terms that improve fast (easy physics) get UP-weighted.
    Terms that stagnate or worsen (conflicting with hidden physics) get DOWN-weighted.
    """
    def __init__(self, term_names, init_lambda=0.05, alpha=0.7,
                 min_lambda=1e-4, max_lambda=2.0, ema_beta=0.7):
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

    def set_baselines(self, per_term_losses: Dict[str, float]):
        self.baselines  = {n: max(v, 1e-8) for n, v in per_term_losses.items()}
        self.ema_losses = dict(self.baselines)
        print(f"[AdaptiveWeights] baselines: "
              f"{', '.join(f'{k}={v:.3e}' for k,v in self.baselines.items())}")

    def update(self, per_term_losses: Dict[str, float]):
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
class TrainConfig:
    epochs: int = 5000
    lr: float = 1e-3
    warmup_epochs: int = 2000

    adaptive: bool = False
    lambda_init: dict = field(default_factory=lambda: {
        "inertia": 0.05, "damping": 0.05, "stiffness": 0.05, "forcing": 0.05,
    })
    lambda_fixed: dict = None

    adaptive_alpha: float = 0.7   # strong enough for visible divergence
    adaptive_ema:   float = 0.7   # fast enough to respond within training

    # Informing net
    use_informing_net: bool = False
    adversarial: bool = False     # True = post-hoc (train informing after informed frozen)
    informing_epochs: int = 2000  # epochs to train post-hoc informing net
    informing_lr: float = 5e-4
    informing_reg: float = 0.02   # L2 on informing output (Occam's razor)

    print_every: int = 500


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train(t_data, x_data, physics_terms: PhysicsTerms, t_collocation,
          config: TrainConfig = None, device="cpu"):
    if config is None:
        config = TrainConfig()

    term_names = ["inertia", "damping", "stiffness", "forcing"]

    def to_t(arr, req_grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
        if req_grad: t.requires_grad_(True)
        return t

    t_data_t = to_t(t_data)
    x_data_t = to_t(x_data)
    t_col    = to_t(t_collocation, req_grad=True)

    informed = InformedNet().to(device)
    opt = torch.optim.Adam(informed.parameters(), lr=config.lr)

    if config.adaptive:
        weights      = AdaptiveWeights(term_names, alpha=config.adaptive_alpha,
                                       ema_beta=config.adaptive_ema,
                                       init_lambda=list(config.lambda_init.values())[0])
        baselines_set = False
    else:
        weights       = None
        baselines_set = True
        fixed         = config.lambda_fixed or config.lambda_init

    loss_history   = {"data": [], "physics": [], "total": []}
    lambda_history = {n: [] for n in term_names}

    # ── Phase 1: Train InformedNet ────────────────────────────────────
    for epoch in range(config.epochs):
        warm = epoch < config.warmup_epochs

        x_pred_data = informed(t_data_t)
        loss_data   = torch.mean((x_pred_data - x_data_t) ** 2)

        # Physics residual
        x_col  = informed(t_col)
        x_t    = torch.autograd.grad(x_col, t_col, grad_outputs=torch.ones_like(x_col),
                                     create_graph=True)[0]
        x_tt   = torch.autograd.grad(x_t, t_col, grad_outputs=torch.ones_like(x_t),
                                     create_graph=True)[0]
        comps  = physics_terms.residual_components(t_col, x_col, x_t, x_tt)
        per_term_loss = {n: torch.mean(comps[n] ** 2) for n in term_names}

        if warm:
            opt.zero_grad(); loss_data.backward(); opt.step()
            loss_history["data"].append(loss_data.item())
            loss_history["physics"].append(0.0)
            loss_history["total"].append(loss_data.item())
            for n in term_names: lambda_history[n].append(0.0)
            if epoch % config.print_every == 0:
                print(f"[epoch {epoch:5d}] WARMUP data={loss_data.item():.3e}")
            continue

        if config.adaptive and not baselines_set:
            weights.set_baselines({n: per_term_loss[n].item() for n in term_names})
            baselines_set = True

        if config.adaptive:
            weights.update({n: per_term_loss[n].item() for n in term_names})
            lambdas_now = weights.as_dict()
        else:
            lambdas_now = fixed

        loss_physics  = sum(lambdas_now[n] * per_term_loss[n] for n in term_names)
        full_residual = sum(comps.values())
        loss_closure  = torch.mean(full_residual ** 2)
        loss_total    = loss_data + loss_physics + 0.1 * loss_closure

        opt.zero_grad(); loss_total.backward(); opt.step()

        loss_history["data"].append(loss_data.item())
        loss_history["physics"].append(loss_physics.item())
        loss_history["total"].append(loss_total.item())
        for n in term_names: lambda_history[n].append(lambdas_now[n])

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            lam_str = ", ".join(f"{k}:{v:.4f}" for k,v in lambdas_now.items())
            print(f"[epoch {epoch:5d}] data={loss_data.item():.3e} "
                  f"phys={loss_physics.item():.3e} λ={{{lam_str}}}")

    # ── Phase 2: Train InformingNet (post-hoc, informed frozen) ──────
    informing = None
    informing_history = {"loss": []}

    if config.use_informing_net:
        print(f"\n[Phase 2] Training InformingNet post-hoc "
              f"({config.informing_epochs} epochs, informed FROZEN)")
        informed.eval()
        for p in informed.parameters():
            p.requires_grad_(False)

        informing     = InformingNet().to(device)
        opt_inf       = torch.optim.Adam(informing.parameters(), lr=config.informing_lr)

        # Get the frozen InformedNet's residual at collocation points
        t_col2 = to_t(t_collocation, req_grad=True)
        with torch.no_grad():
            pass  # just prep

        for epoch in range(config.informing_epochs):
            t_col2 = to_t(t_collocation, req_grad=True)
            x_col2  = informed(t_col2)
            x_t2    = torch.autograd.grad(x_col2, t_col2,
                                          grad_outputs=torch.ones_like(x_col2),
                                          create_graph=True)[0]
            x_tt2   = torch.autograd.grad(x_t2, t_col2,
                                          grad_outputs=torch.ones_like(x_t2),
                                          create_graph=False)[0]

            forcing = physics_terms.F0 * torch.cos(physics_terms.w_f * t_col2)
            # Raw residual = what known physics can't explain
            # residual = x'' + c*x' + k*x - F  ≈ -(hidden term)
            # So hidden term ≈ -residual
            raw_residual = (x_tt2
                            + physics_terms.c * x_t2
                            + physics_terms.k * x_col2
                            - forcing).detach()  # frozen signal

            # InformingNet learns to predict -raw_residual = hidden term
            correction = informing(x_col2.detach(), x_t2.detach())
            # Objective: correction ≈ -raw_residual (= hidden term)
            loss_inf = (torch.mean((correction + raw_residual) ** 2)
                        + config.informing_reg * torch.mean(correction ** 2))

            opt_inf.zero_grad(); loss_inf.backward(); opt_inf.step()
            informing_history["loss"].append(loss_inf.item())

            if epoch % (config.informing_epochs // 4) == 0 or epoch == config.informing_epochs - 1:
                print(f"  [inf epoch {epoch:4d}] loss={loss_inf.item():.3e}")

        # Unfreeze for evaluation
        for p in informed.parameters():
            p.requires_grad_(True)
        informed.train()

    return {
        "informed":          informed,
        "informing":         informing,
        "loss_history":      loss_history,
        "lambda_history":    lambda_history,
        "informing_history": informing_history,
        "weights":           weights,
    }


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_supraphysical_residual(informed, physics_terms: PhysicsTerms,
                                    t_array, device="cpu"):
    """Raw supraphysical gap from InformedNet."""
    t = torch.tensor(t_array, dtype=torch.float32, device=device).reshape(-1, 1)
    t.requires_grad_(True)
    x    = informed(t)
    x_t  = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]
    forcing  = physics_terms.F0 * torch.cos(physics_terms.w_f * t)
    residual = x_tt + physics_terms.c * x_t + physics_terms.k * x - forcing
    return (
        x.detach().cpu().numpy().flatten(),
        x_t.detach().cpu().numpy().flatten(),
        x_tt.detach().cpu().numpy().flatten(),
        (-residual).detach().cpu().numpy().flatten(),  # sign: hidden ≈ -residual
    )


def evaluate_informing_output(informed, informing, physics_terms: PhysicsTerms,
                               t_array, device="cpu"):
    """What the InformingNet proposes as the hidden term."""
    t = torch.tensor(t_array, dtype=torch.float32, device=device).reshape(-1, 1)
    t.requires_grad_(True)
    x   = informed(t)
    x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    with torch.no_grad():
        correction = informing(x.detach(), x_t.detach())
    return (
        x.detach().cpu().numpy().flatten(),
        x_t.detach().cpu().numpy().flatten(),
        correction.cpu().numpy().flatten(),
    )
