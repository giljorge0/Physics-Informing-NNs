# Physics-Informing Neural Networks: Complete File Guide

## 📁 Repository Structure

```
physics-informing-nn/
├── README.md                     # Conceptual overview + research directions
├── QUICKSTART.md                 # How to run, tune, and iterate
├── .gitignore                    # Git exclusions
│
├── src/                          # Core libraries
│   ├── oscillator.py             # Ground-truth system, data generation
│   ├── pinn.py                   # PINN with decomposed physics loss
│   └── sindy_distill.py          # Symbolic regression on residuals
│
├── experiments/
│   └── run_experiment.py         # End-to-end pipeline (main entry point)
│
└── results/                      # Generated outputs (gitignored)
    ├── experiment_summary.png    # 3 plots: solution fit, lambda history, residual
    ├── loss_curves.png           # Training loss evolution
    └── summary.txt               # Numerical results
```

---

## 📄 File-by-file guide

### Top-level documentation

| File | Purpose | When to read |
|------|---------|--------------|
| `README.md` | Framework concept, theory, known limitations, future directions | First — to understand the ideas and what's implemented |
| `QUICKSTART.md` | How to install, run, and iterate with concrete examples | Second — before running or modifying code |
| `.gitignore` | Excludes Python cache, pycache, results | If version-controlling the repo |

### Source code (`src/`)

#### `oscillator.py` — Ground truth & data generation
**Lines:** ~90 | **Editable?** Yes — for different physical systems

**What it does:**
- Defines the true (hidden-term) ODE: `x'' + c x' + k x + eps x^3 + delta sign(v) v^2 = F0 cos(w_f t)`
- Defines the "known" (incomplete) physics: `x'' + c x' + k x = F0 cos(w_f t)` (gap is the discovery target)
- `generate_data(...)`: Integrates the true ODE, adds noise, returns `(t, x_noisy, v_noisy, x_clean, v_clean)`

**Key parameters to tune:**
- `TRUE_PARAMS` dict: `eps`, `delta` (hidden term strength), `c`, `k` (linear coefficients), `F0`, `w_f` (forcing)
- `generate_data(..., t_max=20.0, n_points=400, noise_std=0.01, y0=(1.0, 0.0), seed=0)`

**Example customization:**
```python
# Make the hidden term stronger
TRUE_PARAMS["eps"] = 0.8
TRUE_PARAMS["delta"] = 0.3

# Test a different forcing frequency
TRUE_PARAMS["w_f"] = 0.5
```

---

#### `pinn.py` — Physics-Informed Neural Network with decomposed loss
**Lines:** ~400 | **Editable?** Carefully — core training logic

**What it does:**
- `MLP`: Basic feedforward network (tanh activation, configurable depth/width)
- `InformedNet`: Solution network `x(t)` — the main PINN
- `InformingNet`: Auxiliary network predicting missing-physics correction from `(x, v)` — the adversarial counterpart
- `PhysicsTerms`: Decomposes the physics residual into named components (inertia, damping, stiffness, forcing), each with its own loss term `L_i`
- `AdaptiveWeights`: (Optional) GradNorm-style per-term weight manager `lambda_i`
- `train(...)`: Main training loop with warmup phase, adaptive/fixed weighting, logging

**Key functions to call:**
- `train(t_data, x_data, physics_terms, t_collocation, config)` → returns dict with trained networks, loss history, lambda history
- `evaluate_supraphysical_residual(informed, physics_terms, t_array)` → evaluates the network's implied "missing physics" at arbitrary time points

**Key parameters (via `TrainConfig`):**
- `epochs`, `lr`: Training budget and step size
- `warmup_epochs`: Data-only pretraining (protects against physics-induced collapse)
- `adaptive`: Boolean; if `True`, use gradient-norm balancing; if `False`, use fixed `lambda_fixed`
- `lambda_fixed`: Dict of per-term weights, e.g. `{"inertia": 0.05, "damping": 0.05, ...}`
- `use_informing_net`: Boolean; include the informing network (currently off by default in experiments)

**Example modification (change network size):**
```python
# Larger networks for harder problems
informed = InformedNet(hidden=128, layers=6)
informing = InformingNet(hidden=64, layers=4)
```

**Example modification (different physics equation):**
```python
@dataclass
class MyPhysicsTerms:
    # For a different system, override residual_components()
    def residual_components(self, t, x, x_t, x_tt):
        # Return named residual components
        return {...}
```

---

#### `sindy_distill.py` — Symbolic regression via sparse least squares
**Lines:** ~130 | **Editable?** Yes — add/remove candidate terms

**What it does:**
- `CANDIDATE_LIBRARY`: Dict of `{term_name: callable(x, v) -> array}` e.g. `"x^3"`, `"sign(v)*v^2"`
- `build_library(x, v, term_names=None)`: Constructs a design matrix `Theta` where each column is one candidate term, deduplicating identical columns
- `stlsq(Theta, y, threshold, max_iter, alpha)`: Sequential Thresholded Least Squares — iteratively zeros small coefficients, re-fits, repeats
- `distill(x, v, residual, threshold=2.0)`: Main entry point; returns `{terms, expr, r2, coef_full}`

**Key parameters:**
- `threshold`: Sparsity threshold (in normalized coefficient space). Higher → sparser (fewer terms). Default 2.0 works well for normalized data; try 1.0–5.0 depending on signal quality.
- `CANDIDATE_LIBRARY`: List of basis functions. Current set is mechanically-relevant (`x^3`, `sign(v) v^2`, etc., no trig). Extend as needed.

**Example: add a new term to the library:**
```python
CANDIDATE_LIBRARY["x^4*v"] = lambda x, v: x**4 * v
```

**Example: dial sparsity:**
```python
# Sparser (fewer terms)
result = distill(x, v, residual, threshold=3.0)

# Less sparse (more terms)
result = distill(x, v, residual, threshold=1.0)
```

---

### Experiments (`experiments/`)

#### `run_experiment.py` — Full pipeline
**Lines:** ~250 | **Editable?** Yes — this is the main knob-turning file

**What it does:**
1. **STEP 1:** Generate noisy data from `oscillator.py`
2. **STEP 2:** Train PINN on incomplete physics using `pinn.py`
3. **STEP 3:** Evaluate residual across single + extra trajectories (multi-trajectory pooling for phase-space coverage)
4. **STEP 4:** Distill residual to symbolic expression using `sindy_distill.py`; also run oracle check (distill true term as sanity test)
5. **STEP 5:** Save plots and summary to `results/`

**Key edit points:**
```python
# Change data generation
t, x_noisy, v_noisy, x_clean, v_clean = generate_data(
    t_max=30.0,         # Longer horizon
    n_points=600,       # More samples
    noise_std=0.005,    # Less noise
    seed=42,
    y0=(3.0, -1.0),     # Different initial condition
)

# Tune PINN training
config = TrainConfig(
    epochs=8000,
    warmup_epochs=3000,
    lr=5e-4,
    adaptive=True,  # Try adaptive (but see README caveat)
    lambda_fixed={"inertia": 0.02, "damping": 0.1, "stiffness": 0.05, "forcing": 0.01},
)

# Tune SINDy
result_distill = distill(x_pred, v_pred, residual, threshold=1.5)
```

**Output:**
- Prints progress to stdout (loss values, recovered expression, oracle check)
- Saves to `results/`:
  - `experiment_summary.png`: 3-panel figure (solution, lambda history, residual scatter)
  - `loss_curves.png`: Training loss curves
  - `summary.txt`: Numbers (MSE, R², oracle result, final losses)

---

### Results (`results/`)

Generated after running `experiments/run_experiment.py`. Gitignored by default.

| File | Content |
|------|---------|
| `experiment_summary.png` | Left: noisy data, true solution, PINN prediction; Middle: lambda history (currently flat since fixed weights); Right: residual (orange) vs true missing term (blue) scatter |
| `loss_curves.png` | Log-scale plot of data loss, physics loss, total loss vs epoch. Shows warmup phase (first 2000 epochs) and transition to physics loss |
| `summary.txt` | Text summary: ground truth params, recovered expression, R², MSE, oracle check, final loss values |

---

## 🔄 Typical iteration workflow

1. **Read** `README.md` → understand the idea
2. **Read** `QUICKSTART.md` → see concrete examples
3. **Run** `python experiments/run_experiment.py` → baseline results in `results/`
4. **Inspect** `results/summary.txt` and plots → identify bottleneck
5. **Edit** the relevant source file:
   - Stronger/weaker hidden term? → `src/oscillator.py` `TRUE_PARAMS`
   - PINN not converging? → `experiments/run_experiment.py` `TrainConfig`
   - SINDy missing terms? → `src/sindy_distill.py` `CANDIDATE_LIBRARY` or `threshold`
6. **Re-run** and compare
7. **Iterate**

## 🚀 Launching custom experiments

Example: compare fixed vs adaptive lambda on a sweep of hidden-term strengths:

```bash
# Create experiments/sweep_hidden_term.py
# Inside: loop over eps ∈ [0.1, 0.2, 0.4, 0.8]
# For each: run PINN with adaptive=True, log R² of distillation
# Plot results vs eps
```

Then: `python experiments/sweep_hidden_term.py`

---

## 📋 Checklist: what to change for a new physics system

If you want to study a different ODE (e.g., Lorenz, Van der Pol, a reaction-diffusion PDE):

1. **`src/oscillator.py`** (or new file): 
   - Define `true_rhs(t, y, p)` and `incomplete_physics_rhs(t, y, p)` 
   - Define `hidden_term(x, v, p)` (evaluation only)
   - Adjust `generate_data(...)` signature if needed (e.g., if state is 3D not 2D)

2. **`src/pinn.py`**: 
   - Adapt `PhysicsTerms.residual_components(...)` to your physics
   - Possibly change `InformedNet` input/output dims (if state ≠ 1D)

3. **`src/sindy_distill.py`**: 
   - Update `CANDIDATE_LIBRARY` with terms relevant to your system

4. **`experiments/run_experiment.py`**: 
   - Import your new system, call its `generate_data(...)` 
   - Update `PhysicsTerms` initialization 
   - Adjust `t_collocation` range if time horizon differs

---

## 🔗 Connections to proposal ideas

| Proposal concept | Implemented as | File(s) |
|---|---|---|
| Let NN violate incomplete physics, then recover gap as symbolic expression | PINN trained against partial loss + SINDy distillation | `src/pinn.py` + `src/sindy_distill.py` |
| Minimax: informing ↔ informed | `InformingNet` + `InformedNet` jointly trained | `src/pinn.py` classes + training loop |
| Multidimensional Pareto: per-term `lambda_i` instead of scalar `lambda` | `PhysicsTerms` + `AdaptiveWeights` | `src/pinn.py` |
| Different uncertainties about different physics → dynamic reweighting | `AdaptiveWeights` (gradient-norm based, though unstable) | `src/pinn.py` class + `train()` |

---

## 💡 Tips & tricks

- **Phase-space collinearity problem**: If SINDy R² stays low despite oracle check passing (R²=1.0), you likely need multi-trajectory data. See `experiments/run_experiment.py` `extra_trajectories` loop — expand it or generate data with more diverse initial conditions.
- **Adaptive lambda instability**: The current GradNorm scheme saturates too fast. See README "Future directions" #2 for a proposed fix (relative-loss scaling instead of absolute gradient norms).
- **PINN collapse to x ≈ 0**: Symptom of `lambda_fixed` values too large relative to data variance. Halve them and re-run.
- **SINDy picks wrong terms**: Either threshold too low (keeping noise), phase-space not diverse enough (single trajectory is 1D curve), or candidate library missing the true term. Try threshold sweep, add more trajectories, or inspect `CANDIDATE_LIBRARY`.

---

Happy iterating! 🔬
