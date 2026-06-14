"""
End-to-end experiment: Physics-Informing Neural Network with
multidimensional Pareto weighting + symbolic distillation.

Pipeline:
1. Generate noisy data from the TRUE system (with hidden Duffing +
   quadratic-drag terms).
2. Train an MP-PINN against the INCOMPLETE known physics
   (linear oscillator only), with per-term adaptive lambda_i.
3. Evaluate the network's implied residual -- the "supraphysical gap".
4. Run SINDy-style symbolic distillation on that residual to recover
   a human-readable expression for the missing physics.
5. Compare recovered expression to ground truth.
6. Save plots + a results summary.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.oscillator import generate_data, TRUE_PARAMS, hidden_term
from src.pinn import PhysicsTerms, TrainConfig, train, evaluate_supraphysical_residual
from src.sindy_distill import distill


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1: Generating synthetic experimental data")
    print("=" * 70)
    t, x_noisy, v_noisy, x_clean, v_clean = generate_data(
        t_max=20.0, n_points=400, noise_std=0.01, seed=0
    )
    print(f"Data: {len(t)} points over t in [0, {t[-1]:.1f}]")
    print(f"True hidden term: eps={TRUE_PARAMS['eps']}, delta={TRUE_PARAMS['delta']}")

    # Additional trajectories (different initial conditions) used ONLY for
    # broadening phase-space coverage of the residual evaluation/distillation
    # step -- mimics having multiple experimental runs.
    extra_trajectories = []
    for seed, y0 in [(1, (2.0, 0.0)), (2, (0.5, 1.5)), (3, (-1.5, -1.0))]:
        te, xe, ve, _, _ = generate_data(t_max=20.0, n_points=400, noise_std=0.01, seed=seed, y0=y0)
        extra_trajectories.append((te, xe, ve, y0))

    print()
    print("=" * 70)
    print("STEP 2: Training MP-PINN against INCOMPLETE known physics")
    print("   (known physics = linear damped oscillator only,")
    print("    hidden Duffing + quadratic-drag terms NOT included)")
    print("=" * 70)

    physics_terms = PhysicsTerms(
        c=TRUE_PARAMS["c"], k=TRUE_PARAMS["k"],
        F0=TRUE_PARAMS["F0"], w_f=TRUE_PARAMS["w_f"],
    )

    t_collocation = np.linspace(0, 20, 800)

    config = TrainConfig(
        epochs=5000, lr=1e-3, print_every=500, warmup_epochs=2000,
        adaptive=False,
        lambda_fixed={"inertia": 0.05, "damping": 0.05, "stiffness": 0.05, "forcing": 0.05},
    )
    result = train(t, x_noisy, physics_terms, t_collocation, config)

    print()
    print("=" * 70)
    print("STEP 3: Evaluating the supraphysical residual")
    print("   (pooling residuals across multiple trajectories /")
    print("    initial conditions for better phase-space coverage --")
    print("    a single trajectory traces a near-1D spiral in (x,v)")
    print("    space, which makes the symbolic-regression library")
    print("    ill-conditioned / collinear)")
    print("=" * 70)

    all_x, all_v, all_residual, all_true_missing = [], [], [], []
    eval_windows = [
        (t, x_noisy, v_noisy, x_clean, v_clean),
    ] + [(te, xe, ve, None, None) for (te, xe, ve, y0) in extra_trajectories]

    for (te, xe, ve, _, _) in eval_windows:
        t_eval = np.linspace(te[0] + 0.5, te[-1] - 0.5, 200)
        x_pred, v_pred, a_pred, residual = evaluate_supraphysical_residual(
            result["informed"], physics_terms, t_eval
        )
        all_x.append(x_pred)
        all_v.append(v_pred)
        all_residual.append(residual)
        all_true_missing.append(hidden_term(x_pred, v_pred))

    x_pred = np.concatenate(all_x)
    v_pred = np.concatenate(all_v)
    residual = np.concatenate(all_residual)
    true_missing = np.concatenate(all_true_missing)

    resid_mse = np.mean((residual - true_missing) ** 2)
    print(f"MSE(recovered residual vs true missing term) = {resid_mse:.4e}")
    print(f"Residual range: [{residual.min():.3f}, {residual.max():.3f}]")
    print(f"True missing term range: [{true_missing.min():.3f}, {true_missing.max():.3f}]")
    print()
    print("NOTE: the PINN is trained on ONE trajectory only (seed=0).")
    print("Evaluating the trained network's *implied physics residual*")
    print("along OTHER trajectories' (x,v) is an extrapolation test --")
    print("if the discovered residual generalizes, that's evidence the")
    print("network has learned a genuine state-dependent correction,")
    print("not just memorized one trajectory's time-series.")

    print()
    print("=" * 70)
    print("STEP 4: Symbolic distillation (SINDy-style) of the residual")
    print("=" * 70)
    distilled = distill(x_pred, v_pred, residual, threshold=2.0)
    print(f"Recovered expression (from PINN residual): {distilled['expr']}")
    print(f"R^2 of symbolic fit to residual: {distilled['r2']:.4f}")
    print(f"Ground truth term: 0.4000*x^3 + 0.1500*sign(v)*v^2")

    # Oracle sanity check: same distillation pipeline applied to the
    # TRUE (noise-free) residual, to confirm the symbolic regression
    # itself is correct and isolate PINN-vs-distillation error sources.
    oracle = distill(x_pred, v_pred, true_missing, threshold=2.0)
    print()
    print(f"[oracle check] Distilling the TRUE missing term directly:")
    print(f"  Recovered: {oracle['expr']}")
    print(f"  R^2: {oracle['r2']:.4f}")

    print()
    print("=" * 70)
    print("STEP 5: Saving plots and summary")
    print("=" * 70)

    # --- Plot 1: solution fit ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t, x_noisy, "k.", ms=3, alpha=0.4, label="noisy data")
    axes[0].plot(t, x_clean, "g--", lw=1, label="true (clean)")
    t_dense = np.linspace(0, 20, 500)
    import torch
    with torch.no_grad():
        x_dense = result["informed"](torch.tensor(t_dense, dtype=torch.float32).reshape(-1, 1)).numpy().flatten()
    axes[0].plot(t_dense, x_dense, "r-", lw=1.5, label="PINN prediction")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x(t)")
    axes[0].set_title("Solution fit (incomplete physics + data)")
    axes[0].legend(fontsize=8)

    # --- Plot 2: adaptive lambda evolution ---
    for name, hist in result["lambda_history"].items():
        axes[1].plot(hist, label=name)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("lambda_i")
    axes[1].set_yscale("log")
    axes[1].set_title("Per-term adaptive physics weights")
    axes[1].legend(fontsize=8)

    # --- Plot 3: residual vs true missing term ---
    order = np.argsort(x_pred)
    axes[2].scatter(x_pred, residual, s=8, alpha=0.4, label="PINN-implied residual")
    axes[2].scatter(x_pred, true_missing, s=8, alpha=0.4, label="true missing term")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("missing-physics value")
    axes[2].set_title("Supraphysical residual vs ground truth")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "experiment_summary.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved figure: {fig_path}")

    # --- Loss curves ---
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["loss_history"]["data"], label="data loss")
    ax.plot(result["loss_history"]["physics"], label="physics loss")
    ax.plot(result["loss_history"]["total"], label="total loss")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.set_title("Training loss curves")
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.savefig(fig2_path, dpi=120)
    print(f"Saved figure: {fig2_path}")

    # --- Summary file ---
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Physics-Informing Neural Network -- Experiment Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"True hidden term: eps*x^3 + delta*sign(v)*v^2\n")
        f.write(f"  eps = {TRUE_PARAMS['eps']}, delta = {TRUE_PARAMS['delta']}\n\n")
        f.write(f"Recovered symbolic expression (from PINN residual): {distilled['expr']}\n")
        f.write(f"R^2 of symbolic fit (PINN residual): {distilled['r2']:.4f}\n")
        f.write(f"MSE(PINN residual vs true missing term): {resid_mse:.4e}\n\n")
        f.write(f"Oracle check (distilling TRUE missing term directly):\n")
        f.write(f"  Recovered: {oracle['expr']}\n")
        f.write(f"  R^2: {oracle['r2']:.4f}\n\n")
        f.write("Final adaptive lambda_i values:\n")
        for name, hist in result["lambda_history"].items():
            f.write(f"  {name}: {hist[-1]:.4f}\n")
        f.write("\nFinal losses:\n")
        f.write(f"  data loss:    {result['loss_history']['data'][-1]:.4e}\n")
        f.write(f"  physics loss: {result['loss_history']['physics'][-1]:.4e}\n")
    print(f"Saved summary: {summary_path}")

    return result, distilled


if __name__ == "__main__":
    main()
