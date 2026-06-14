# Quick Start Guide

## Installation

```bash
# Clone or extract the repo
cd physics-informing-nn

# Install dependencies
pip install torch numpy scipy matplotlib

# (Optional, for enhanced symbolic regression later)
# pip install pysr
```

## Run the default experiment

```bash
python experiments/run_experiment.py
```

This will:
1. Generate noisy synthetic data from a hidden-term oscillator
2. Train a PINN against incomplete physics
3. Evaluate the residual and symbolically distill it
4. Save plots and a summary to `results/`

Runtime: ~90 seconds on CPU.

---

## Iterate: Key knobs to tune

### System / data generation
**File:** `src/oscillator.py`

- `TRUE_PARAMS` dict: Change `eps`, `delta` (hidden term coefficients), damping `c`, stiffness `k`, forcing amplitude `F0` and frequency `w_f`.
- `generate_data()`: Adjust `t_max` (simulation horizon), `n_points` (data resolution), `noise_std` (measurement noise), `y0` (initial condition).

### PINN training
**File:** `src/pinn.py` and `experiments/run_experiment.py`

In `run_experiment.py`, the `TrainConfig`:
- `epochs`: Total training steps.
- `warmup_epochs`: How long to train on data alone before turning on physics loss (currently 2000, which is necessary because the incomplete physics is *hostile* to the true solution).
- `lambda_fixed`: Per-term physics loss weights. Currently `0.05` for all. Try `0.01` (softer physics constraint) or `0.1` (harder). Try asymmetric, e.g. `{"inertia": 0.02, "damping": 0.1, "stiffness": 0.05, "forcing": 0.01}` to assign different confidence to each term.
- `adaptive`: Set to `True` to enable GradNorm-style adaptive weighting (currently unstable -- see README "Interesting directions" #2 for how to fix).
- `lr`: Learning rate (try `1e-4` for slower, more stable convergence).

In `src/pinn.py`:
- `InformedNet` architecture: Change `hidden` or `layers` in the `TrainConfig` constructor call.
- `PhysicsTerms`: Modify the physics equation if testing a different system.
- `AdaptiveWeights`: Tune `alpha` (adaptation rate), `min_lambda`, `max_lambda` if you enable adaptive mode.

### Symbolic distillation
**File:** `src/sindy_distill.py`

- `CANDIDATE_LIBRARY`: Add or remove candidate terms (e.g., `"x^4"`, `"x^2*v^2"`, `"exp(-x)"`). Current library is trimmed to mechanical oscillator terms.
- `distill()` threshold: Currently `2.0` (in normalized coefficient space). Lower = more terms kept, higher = sparser. Try `1.0`, `3.0`, `5.0`.

### Multi-trajectory pooling
**File:** `experiments/run_experiment.py`, function `main()`

- Uncomment / modify the `extra_trajectories` loop. Currently pools residuals from 3 extra initial conditions (`y0` = `(2, 0), (0.5, 1.5), (-1.5, -1)`) to improve phase-space coverage.
- Try different `y0` values or add more trajectories (beware: more eval points = slower SINDy regression).

---

## Quick experiments to try

### 1. **Test adaptive lambda** (intended mini-fix)
Edit `experiments/run_experiment.py`:
```python
config = TrainConfig(
    ...,
    adaptive=True,
    lambda_init={"inertia": 1.0, "damping": 1.0, "stiffness": 1.0, "forcing": 1.0},
)
```
Then patch `src/pinn.py` `AdaptiveWeights.__init__` to use relative-loss scaling instead of gradient norms (see README section #2).

### 2. **Vary the hidden term strength**
Edit `src/oscillator.py`:
```python
TRUE_PARAMS = dict(
    ...
    eps=0.2,       # weaker Duffing (easier to fit with linear physics)
    delta=0.08,    # weaker drag
)
```
Re-run. Expect: weaker residual signal, harder for SINDy to recover. At `eps=0.0, delta=0.0` (no hidden term), the PINN should fit perfectly with the "known" incomplete physics.

### 3. **Test SINDy threshold sensitivity**
Edit `experiments/run_experiment.py`, function `main()`:
```python
for threshold in [0.5, 1.0, 2.0, 3.0, 5.0]:
    distilled = distill(x_pred, v_pred, residual, threshold=threshold)
    print(f"threshold={threshold}: {distilled['expr']}, R^2={distilled['r2']:.3f}")
```

### 4. **Longer training + richer library**
In `experiments/run_experiment.py`:
```python
config = TrainConfig(epochs=10000, warmup_epochs=3000, ...)
```
In `src/sindy_distill.py`, add terms:
```python
"x^4": lambda x, v: x ** 4,
"x^2*v^2": lambda x, v: x ** 2 * v ** 2,
```

### 5. **Add your own physics system**
Create `src/my_system.py` with your own ODE (e.g., Lorenz, Van der Pol, reaction-diffusion), then modify `experiments/run_experiment.py` to call your data generator and update `PhysicsTerms` accordingly.

---

## Debugging / monitoring

- **Training diverges or fits poorly**: Increase `warmup_epochs`, decrease `lambda_fixed` values (make physics loss weaker), or lower `lr`.
- **SINDy doesn't isolate the true terms**: Try lower threshold, add more trajectories, or extend `t_max` so the system explores a wider range.
- **"true missing term range" is [0, 0]**: This means the PINN is predicting x ≈ 0 everywhere (overfitting to physics at expense of data). Decrease `lambda_fixed`, check initial conditions, or verify data quality.

---

## Next steps (from README)

1. **Fix adaptive `lambda_i`**: Implement relative-loss (not gradient-norm) balancing.
2. **True multi-trajectory training**: Train a single PINN on all trajectories simultaneously, using initial condition as an input.
3. **Adversarial training**: Make `InformingNet` actually play a minimax game (alternating updates, like a GAN).
4. **Upgrade to PySR**: Once residual quality improves, swap `sindy_distill.py` for PySR for a much larger symbolic search space.
5. **Real systems**: Test on a 1D PDE (reaction-diffusion), or real experimental data where ground truth is unknown.

---

## Questions / issues?

Check the README's "Why the distillation isn't perfect" section — most issues trace back to phase-space collinearity (fixed by multi-trajectory data) or adaptive lambda instability (needs the relative-loss fix mentioned above).
