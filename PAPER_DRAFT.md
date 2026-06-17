# Physics-Informing Neural Networks: Supraphysical Gap Discovery via Multidimensional Pareto Loss Decomposition

**Draft — based on experimental results, June 2025**

---

## Abstract

We propose *physics-informing* neural networks (PI-NNs), a reversal of the standard physics-informed approach. Rather than constraining a network to satisfy known physics exactly, we allow it to violate incomplete physics where data demands it, decompose the violation into a vector of per-term residuals with individually-tunable confidence weights, and distill the remaining "supraphysical gap" into a symbolic mathematical expression. We demonstrate three findings on a driven nonlinear oscillator and a 1D reaction-diffusion PDE: (1) decomposing the physics loss into per-term weights (λ_inertia, λ_damping, λ_stiffness, λ_forcing) reaches solutions unreachable by any scalar λ, with the best configuration lying off the equal-weight diagonal (R²=0.935 vs 0.913); (2) relative-progress-based adaptive weighting automatically discovers these asymmetric weights from data without manual tuning; and (3) a post-hoc "informing network" trained on the frozen PINN's residual improves symbolic recovery from R²=0.900 to R²=0.947. The PDE extension demonstrates that adaptive weights reveal physically meaningful differential convergence across spatial operators.

---

## 1. Introduction

Physics-informed neural networks (PINNs) [Raissi et al., 2019] encode known physics as a soft constraint:

$$\mathcal{L} = \mathcal{L}_\text{data} + \lambda \mathcal{L}_\text{physics}$$

This formulation assumes the known physics is *complete*. When it is not — when the model encodes an approximation of the true system — the scalar λ forces the network to simultaneously fit data and satisfy incomplete physics, producing a biased solution that satisfies neither well.

We ask a different question: **what if we let the network violate the known physics, and read off that violation as a discovery?**

The idea has three components:

**Physics-informing (not physics-informed).** We train against *incomplete* known physics with a deliberately permissive loss. The network is allowed to make "supraphysical" predictions — solutions that fit the data better than the known physics would allow. The gap between what the known physics implies and what the data requires is the *supraphysical residual*, the signal we distill.

**Multidimensional Pareto decomposition.** Rather than one scalar λ, we decompose the physics loss into named per-term components, each with its own weight λᵢ:

$$\mathcal{L} = \mathcal{L}_\text{data} + \sum_i \lambda_i \mathcal{L}_i$$

For a damped oscillator: λ_inertia, λ_damping, λ_stiffness, λ_forcing. This opens a higher-dimensional tradeoff space. The optimal (λ_inertia, λ_damping, λ_stiffness, λ_forcing) point need not lie on the scalar-λ Pareto curve — and empirically, it does not.

**Post-hoc symbolic distillation via informing network.** After training the solution network (InformedNet), we train a separate lightweight network (InformingNet) on the *frozen* PINN's residual. This post-hoc design avoids signal absorption: the PINN can't "cheat" by using the correction during training. The InformingNet output is then distilled symbolically via SINDy-style sparse regression.

---

## 2. Method

### 2.1 System and incomplete physics

We study a driven, damped oscillator with two hidden nonlinear terms:

$$\underbrace{x'' + cx' + kx = F_0\cos(\omega_f t)}_{\text{known (incomplete)}} + \underbrace{\varepsilon x^3 + \delta \,\text{sign}(x') x'^2}_{\text{hidden}}$$

Parameters: c=0.3, k=1.0, ε=0.4, δ=0.15, F₀=0.5, ωf=1.2. The PINN sees only the left side. The hidden terms are a Duffing-type cubic stiffness term and quadratic velocity drag — physically distinct, with different functional forms, motivating separate λᵢ weights.

We extend to a 1D reaction-diffusion PDE:

$$\underbrace{u_t = D u_{xx} + r u(1-u)}_{\text{known}} + \underbrace{\varepsilon \sin(\pi x) u^2}_{\text{hidden}}$$

with D=0.01, r=1.0, ε=0.3. The hidden term is a spatially-modulated quadratic source.

### 2.2 Multidimensional Pareto weights

The per-term decomposition creates a (n_terms + 1)-dimensional objective space (data + each physics term). The standard scalar PINN traces a 2D Pareto curve (data vs. total physics). Our decomposition traces a family of curves, one per (λ₁,...,λₙ) combination, whose union covers a strictly larger region of the objective space.

**Adaptive reweighting.** We initialise all λᵢ = λ₀ at the end of warmup and then adapt:

$$\lambda_i(t) = \lambda_0 \cdot \left(\frac{\mathcal{L}_i^{\text{baseline}}}{\tilde{\mathcal{L}}_i(t)}\right)^\alpha$$

where $\mathcal{L}_i^{\text{baseline}}$ is the per-term loss at end of warmup, $\tilde{\mathcal{L}}_i$ is an exponential moving average of the current loss, and α=0.7 controls adaptation strength. Terms improving faster than baseline get up-weighted; terms stagnating (physics conflicting with hidden nonlinearity) get down-weighted automatically.

### 2.3 Post-hoc InformingNet

**Problem with joint training.** When the InformingNet is trained simultaneously with the InformedNet, the solution network learns to lean on the correction. The residual signal is split between two networks, degrading both the PINN's solution quality and the distillation target.

**Post-hoc design.** After training the InformedNet to convergence:
1. Freeze all InformedNet parameters.
2. Compute the frozen PINN's physics residual at collocation points: $r(t) = x'' + cx' + kx - F(t) \approx -f_\text{hidden}(x, x')$
3. Train InformingNet $g_\phi(x, x')$ to minimise: $\|g_\phi(x, x') + r(t)\|^2 + \mu\|g_\phi\|^2$
4. Distill $g_\phi$ symbolically: its output is a clean neural approximation of the hidden term.

The L2 penalty μ enforces compactness (Occam's razor): the InformingNet is penalised for proposing large corrections, encouraging it toward sparse, physically interpretable outputs.

### 2.4 Symbolic distillation (SINDy-style)

Given samples of the supraphysical signal $s(x,v)$ (either raw residual or InformingNet output), we build a feature library $\Theta \in \mathbb{R}^{N \times K}$ of candidate nonlinear functions (x³, sign(v)v², x²v, ...) and solve:

$$\min_\xi \|\Theta\xi - s\|_2^2 + \text{sparsity via STLSQ}$$

Sequential Thresholded Least Squares (STLSQ) iteratively zeros coefficients below threshold τ and refits, producing a sparse symbolic expression.

---

## 3. Experiments

All experiments use PyTorch, Adam optimiser, 4-layer tanh MLP (64 hidden units), 2000-epoch data-only warmup before physics loss activation. Multi-trajectory residual pooling (4 trajectories) is used for ODE distillation to break phase-space collinearity.

### 3.1 Lambda sweep: multidimensional Pareto confirmation

We train 25 configurations on a 5×5 grid of (λ_stiffness, λ_damping) ∈ {0.005, 0.02, 0.05, 0.15, 0.4}², holding λ_inertia = λ_forcing = 0.05.

**Result:** Best SINDy R²=0.935 at (λ_s=0.40, λ_d=0.05) — off-diagonal. Best diagonal (equal weights) R²=0.913. The improvement (Δ=0.022) is consistent across runs.

**Interpretation:** The stiffness term most directly conflicts with the hidden Duffing nonlinearity (ε·x³ is a stiffness modification). Increasing λ_stiffness while holding λ_damping low forces the network to express the stiffness violation in the residual more cleanly — exactly the signal we want for distillation.

**Key claim confirmed:** The multidimensional Pareto space contains solutions unreachable by scalar λ weighting.

### 3.2 Adaptive weights

We compare three λ strategies:
- **A: Fixed equal** — λᵢ = 0.05 ∀i (R²=0.872)
- **B: Fixed hand-tuned** — asymmetric prior based on physics intuition (R²=0.842)
- **C: Adaptive** — relative-progress-based (R²=0.888)

Adaptive (C) outperforms both fixed strategies. The hand-tuned prior (B) underperforms because the human-specified weights don't match the optimal found by the sweep, while the adaptive scheme discovers a better balance from data.

In the PDE experiment, adaptive weights show clear divergence: temporal (↑155%), nonlinear reaction (↑98%), diffusion (↑57%), linear reaction (↑36%). The ordering is physically interpretable: temporal and nonlinear terms improve fastest (less conflict with hidden physics), while linear reaction stagnates (most entangled with the hidden source term).

### 3.3 Post-hoc informing network

Three modes compared:
- **A: No informing** — distill raw PINN residual (R²=0.900)
- **B: Post-hoc informing** — distill InformingNet output (R²=0.931)
- **C: Adaptive λ + post-hoc informing** — combined (R²=0.947)

The InformingNet loss decays smoothly from 3×10⁻² to 2×10⁻³ over 2000 post-hoc epochs, indicating clean fitting of a stable target. Modes B and C achieve higher correlation with ground truth (corr=0.902, 0.872) and tighter scatter in residual vs. truth plots.

**Joint training (not used)** was empirically found to split the residual signal: corr(correction, truth)=0.86, but corr(PINN residual, truth) drops when correction is used during training, because the InformedNet compensates.

### 3.4 PDE extension

The Fisher-KPP reaction-diffusion system presents new challenges: the hidden term ε·sin(πx)·u² is spatially modulated, small early (u small), and grows with u. Key findings:

**Library collinearity.** On a single symmetric Gaussian IC, sin(πx)·u² and x(1-x)·u² correlate at r=0.9996, making STLSQ unable to isolate the true term. Multi-IC training (7 asymmetric initial conditions) is required to break this degeneracy. Oracle R²=1.0 confirms the library and STLSQ are correct once the collinearity is resolved.

**Adaptive weights in PDE space.** All four PDE operators are up-weighted relative to baseline (all λᵢ increase), but at different rates, revealing their relative identifiability from data.

**Solution fit.** The multi-IC PINN fits the spatial profile well at early times (t<1) and improves significantly over single-IC training at late times.

---

## 4. Discussion

### What works

The three-component system (permissive physics loss + per-term decomposition + post-hoc symbolic distillation) consistently outperforms standard single-λ PINNs on the task of hidden physics discovery. The R² progression across methods (0.87 → 0.90 → 0.93 → 0.95) reflects each component's additive contribution.

### What doesn't work (yet)

**Distilled expressions are not sparse.** The recovered expressions typically contain 4-6 terms (e.g., `0.78*x - 0.80*x³ + 0.18*v - 0.76*x*v²`) rather than the clean 2-term ground truth. The R² is high because the extra terms are correlated with the true ones and absorb similar variance. This is a fundamental SINDy limitation under collinearity — the true terms are identifiable in coefficient space but not separable from correlated alternatives.

**Coefficients are scaled, not recovered exactly.** The x³ coefficient is ~2× the true ε=0.4 in most runs. This scaling arises because the PINN's learned trajectory covers a different region of (x,v) space than the true system — the network has learned a slightly different x(t) that satisfies the data + incomplete physics, not the true x(t).

**PDE late-time drift.** The PINN underestimates u at t=2 (0.59 vs truth 0.79). This is a training data coverage problem: the warmup phase is optimised for early dynamics where u is small, and the hidden term ε·sin(πx)·u² is negligible. Curriculum training (progressive time extension) would address this.

### Connections to prior work

- **PINNs** [Raissi 2019]: we invert the constraint direction
- **SINDy** [Brunton 2016]: we use STLSQ as distillation, not discovery from clean data
- **Multi-task learning / GradNorm** [Chen 2018, Yu 2020]: our adaptive λ scheme is a simpler relative-progress version
- **Symbolic regression** [Cranmer 2020]: PySR would replace our STLSQ once residual quality improves
- **Bayesian PINNs** [Yang 2021]: per-term λᵢ can be interpreted as precision parameters in a hierarchical prior

---

## 5. Conclusion

Physics-informing neural networks offer a principled framework for hidden physics discovery: train permissively, decompose the violation, distill the gap. The multidimensional Pareto decomposition is the key theoretical contribution — it provably enlarges the solution space, and the empirical off-diagonal optimum confirms that different physics terms require different epistemic confidence weights.

The most impactful practical finding is the post-hoc informing network design: training a separate network on the frozen PINN's residual avoids signal absorption and produces a cleaner, more distillable target.

**Immediate next steps:**
1. Replace STLSQ with PySR for larger symbolic search space
2. Curriculum training for PDE late-time coverage
3. Test on real experimental data (pendulum, fluid flow) where ground truth is unknown

---

## References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*.
- Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016). Discovering governing equations from data. *PNAS*.
- Chen, Z., et al. (2018). GradNorm: Gradient normalization for adaptive loss balancing. *ICML*.
- Cranmer, M., et al. (2020). Discovering symbolic models from deep learning. *NeurIPS*.
- Yang, L., et al. (2021). B-PINNs: Bayesian physics-informed neural networks. *Journal of Computational Physics*.

---

## Appendix: Experimental Details

| Parameter | ODE experiments | PDE experiments |
|-----------|----------------|-----------------|
| Architecture | 4-layer MLP, 64 hidden, tanh | 5-layer MLP, 64 hidden, tanh |
| Optimiser | Adam, lr=1e-3 | Adam, lr=3e-4 |
| Epochs | 5000 (2000 warmup) | 10000 (4000 warmup) |
| Collocation points | 800 (1D) | 6400 (80×80 grid) |
| Training trajectories | 1 (+ 3 for residual pooling) | 7 (multi-IC joint training) |
| SINDy threshold | τ=2.0 | τ=1.5 |
| InformingNet epochs | 2000 | 3000 |
| InformingNet reg μ | 0.02 | 0.02 |
| Adaptive α | 0.7 | 0.7 |
| Adaptive EMA β | 0.7 | 0.7 |

**ODE candidate library:** 1, x, x², x³, v, v², v³, sign(v)·v², x·v, x²·v, x·v²

**PDE candidate library:** u, u², u³, sin(πx), cos(πx), sin(πx)·u, sin(πx)·u², cos(πx)·u², u·(1-u)

**Compute:** All experiments run on CPU. ODE suite: ~25 min. PDE suite: ~60 min.
