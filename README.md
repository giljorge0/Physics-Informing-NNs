# Physics-Informing Neural Networks (PI-NN, reversed)

A small research scaffold for the idea: instead of forcing a neural
network to obey known physics exactly (standard PINNs), let it deviate
from known physics where the data demands it, **decompose** that
deviation into a vector of per-term residuals with individually-tunable
confidence weights, and **distill** the leftover ("supraphysical")
signal back into a human-readable symbolic expression.

This maps directly onto the three ideas in the original proposal:

| Proposal idea | Implementation here |
|---|---|
| "Physics-informing" instead of "physics-informed": let the model make supraphysical predictions, then express the gap as known math | `src/pinn.py` (incomplete known physics + free residual) + `src/sindy_distill.py` (SINDy-style sparse symbolic regression on the residual) |
| Minimax pair: one network informing, one being informed | `InformedNet` (fits data + known physics) + `InformingNet` (proposes the missing-physics correction), trained jointly |
| Multidimensional Pareto fronts: decompose the physics loss term-by-term, each with its own confidence/weight | `PhysicsTerms` + `AdaptiveWeights` in `src/pinn.py` -- per-term `lambda_i` for inertia, damping, stiffness, forcing, instead of one scalar `lambda` |

## The test system

A driven, damped oscillator where the "true" system has two extra terms
hidden from the model's known physics:

```
TRUE:    x'' + c x' + k x + eps x^3 + delta sign(x') x'^2 = F0 cos(w_f t)
KNOWN:   x'' + c x' + k x                                  = F0 cos(w_f t)
```

`eps x^3` is a Duffing-type cubic stiffness term; `delta sign(v) v^2` is
quadratic (velocity-squared) drag. These have different functional forms
and would plausibly carry different epistemic uncertainty in a real
experiment -- exactly the motivation for decomposing the physics loss
rather than treating "physics" as one monolithic term.

## Pipeline

1. **`src/oscillator.py`** -- generates noisy synthetic data from the
   true (hidden-term) system.
2. **`src/pinn.py`** -- trains a PINN against the *incomplete* known
   physics:
   - `InformedNet`: x(t), the solution network.
   - `InformingNet`: an auxiliary network that proposes a correction
     term as a function of (x, v) -- the "informing" side of the
     minimax pair.
   - `PhysicsTerms`: splits the residual `x'' + c x' + k x - F(t)` into
     **named components** (inertia, damping, stiffness, forcing), each
     with its own loss and weight `lambda_i`.
   - `AdaptiveWeights`: optional GradNorm-style adaptive weighting
     (off by default in the included experiment -- see Results/Caveats).
3. **`src/sindy_distill.py`** -- once trained, evaluate the network's
   *implied* residual `x'' + c x' + k x - F(t)` along the learned
   trajectory. This residual is the "supraphysical gap" -- what the
   network had to do that the known physics didn't allow. Run
   Sequential-Thresholded-Least-Squares (the core of SINDy) over a
   library of candidate nonlinear terms to express that gap
   symbolically.
4. **`experiments/run_experiment.py`** -- ties it together end to end
   and saves plots + a summary to `results/`.

## Running it

```bash
pip install torch numpy scipy matplotlib
python experiments/run_experiment.py
```

Outputs go to `results/`: `experiment_summary.png`, `loss_curves.png`,
`summary.txt`.

## Results (current run)

- The PINN, trained against the *incomplete* (linear) known physics,
  fits the noisy data well (final data MSE ~7e-3) while the
  incomplete-physics-only solution would be visibly wrong (compare
  the red curve to the green dashed "true" curve in
  `experiment_summary.png` -- the PINN tracks the true nonlinear
  envelope, not the linear one).
- The network's implied residual (panel 3 of the figure) correlates
  with the true hidden term `eps x^3 + delta sign(v) v^2` in shape and
  sign, but is noisier -- MSE ~1.2e-2 against a signal whose own range
  is roughly [-0.24, 0.22].
- **Oracle check**: running the *same* symbolic distillation on the
  exact (noise-free) hidden term recovers it perfectly --
  `0.4000*x^3 + 0.1500*sign(v)*v^2`, R²=1.0. So the distillation method
  itself is correct; the bottleneck is residual quality from a single
  noisy trajectory.
- Distilling the PINN's actual residual gives R²≈0.87 with a
  qualitatively-plausible but not exactly-sparse expression (it spreads
  weight across several correlated polynomial terms instead of cleanly
  isolating `x^3` and `sign(v) v^2`).

## Why the distillation isn't perfect (and why that's a real, useful finding)

A single trajectory traces a roughly 1-D spiral in (x, v) phase space.
Along that spiral, `x`, `x^3`, `sin(x)`, `v`, `v^3`, etc. are all highly
correlated -- the regression library is nearly singular. This is a
**known SINDy limitation (structural non-identifiability from limited
excitation)**, and it's directly relevant to the original proposal's
point about "different uncertainties about different physics": you
can't separate two competing physical explanations using data that
doesn't excite the system in ways that distinguish them.

This becomes a concrete next experiment (see below) rather than a dead
end.

## Interesting directions to pursue next

1. **Multi-trajectory / multi-experiment training.** Train (or
   evaluate) across several initial conditions / forcing amplitudes
   that excite different regions of phase space, then pool residuals
   before distillation. This directly tests the "multidimensional
   Pareto front" idea: each trajectory is like a different experimental
   condition with its own informativeness about which physics term is
   active.

2. **Make the adaptive `lambda_i` scheme actually work.** The current
   experiment uses fixed, hand-tuned per-term weights because the
   GradNorm-style adaptive scheme (in `AdaptiveWeights`) is unstable --
   it saturates to its max bound almost immediately and collapses the
   solution back toward zero. A more careful relative-loss-based scheme
   (e.g. normalize each `L_i` by its value at the end of the data-only
   warmup, rather than by instantaneous gradient norms) is a good
   follow-up. This is the actual heart of the "multidimensional Pareto
   front" proposal and deserves its own focused experiment with proper
   ablations (fixed-equal vs fixed-handtuned vs adaptive, swept over
   several `eps`/`delta` ground-truth values).

3. **The minimax / adversarial framing.** `InformingNet` currently just
   adds a free correction term with a small L2 penalty -- it isn't yet
   playing an actual minimax game. A real version would have the
   informing network try to *maximize* the residual it can explain
   while the informed network tries to minimize total loss, with some
   adversarial training schedule (alternating updates, like a GAN).
   Worth testing whether this produces sparser / more interpretable
   corrections than the current joint-training setup.

4. **Replace the simple STLSQ with PySR** for a much larger search
   space (including rational functions, products of arbitrary terms,
   etc.) once you're ready to deal with the Julia dependency -- useful
   once the residual quality from (1) and (2) improves enough that the
   bottleneck shifts from "is the signal clean" to "is the function
   library expressive enough".

5. **Different test systems.** The oscillator is a good first sandbox
   because ground truth is known exactly. Good next candidates: a 1D
   reaction-diffusion / heat equation with a hidden nonlinear source
   term (PDE case, tests whether the same per-term decomposition scales
   to spatial operators), or a real experimental dataset (e.g. a damped
   pendulum with measured drag) where the "ground truth" correction is
   genuinely unknown and the symbolic output would be a real discovery
   rather than a recovery exercise.

## Repo structure

```
physics-informing-nn/
├── src/
│   ├── oscillator.py      # ground-truth system + data generation
│   ├── pinn.py             # MP-PINN: decomposed physics loss, adaptive weights,
│   │                        # informed/informing network pair
│   └── sindy_distill.py    # SINDy-style symbolic distillation of residuals
├── experiments/
│   └── run_experiment.py   # end-to-end pipeline
└── results/                 # generated plots + summary (gitignored if you like)
```
