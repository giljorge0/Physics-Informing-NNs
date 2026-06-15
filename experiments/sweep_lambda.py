"""
Lambda sweep ablation — the core empirical test of the multidimensional Pareto idea.

We train the MP-PINN across a grid of (lambda_stiffness, lambda_damping) values,
holding lambda_inertia and lambda_forcing fixed. For each configuration we:
  1. Train the PINN
  2. Evaluate the supraphysical residual
  3. Run SINDy distillation
  4. Record R² of the symbolic fit

The hypothesis: the best (R², solution quality) lives off the diagonal
of the grid — i.e. asymmetric per-term weights outperform any scalar lambda.

This is the experiment that proves (or falsifies) the multidimensional Pareto claim.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from src.oscillator import generate_data, TRUE_PARAMS, hidden_term
from src.pinn import PhysicsTerms, TrainConfig, train, evaluate_supraphysical_residual
from src.sindy_distill import distill

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── sweep grid ────────────────────────────────────────────────────────
LAMBDA_VALS = [0.005, 0.02, 0.05, 0.15, 0.4]   # 5×5 = 25 runs
LAMBDA_INERTIA = 0.05
LAMBDA_FORCING  = 0.05
EPOCHS          = 4000
WARMUP          = 1500


def run_one(lam_stiff, lam_damp, t, x_noisy, physics_terms, extra_traj,
            t_collocation, run_id):
    config = TrainConfig(
        epochs=EPOCHS,
        warmup_epochs=WARMUP,
        lr=1e-3,
        adaptive=False,
        lambda_fixed={
            "inertia":   LAMBDA_INERTIA,
            "damping":   lam_damp,
            "stiffness": lam_stiff,
            "forcing":   LAMBDA_FORCING,
        },
        use_informing_net=False,
        print_every=EPOCHS + 1,  # silent
    )
    result = train(t, x_noisy, physics_terms, t_collocation, config)

    # Collect residuals across multiple trajectories
    all_x, all_v, all_res, all_true = [], [], [], []
    eval_windows = [(t,)] + [(te,) for (te, _, _) in extra_traj]
    for (te,) in eval_windows:
        t_eval = np.linspace(te[0] + 0.5, te[-1] - 0.5, 150)
        xp, vp, _, res = evaluate_supraphysical_residual(
            result["informed"], physics_terms, t_eval)
        all_x.append(xp); all_v.append(vp)
        all_res.append(res)
        all_true.append(hidden_term(xp, vp))

    x_pool = np.concatenate(all_x)
    v_pool = np.concatenate(all_v)
    res_pool = np.concatenate(all_res)
    true_pool = np.concatenate(all_true)

    dist = distill(x_pool, v_pool, res_pool, threshold=2.0)
    data_mse = result["loss_history"]["data"][-1]
    res_mse   = float(np.mean((res_pool - true_pool) ** 2))

    print(f"  [{run_id:3d}] λ_s={lam_stiff:.3f} λ_d={lam_damp:.3f} "
          f"| data_mse={data_mse:.3e} res_mse={res_mse:.3e} "
          f"sindy_R²={dist['r2']:.3f} | {dist['expr']}")

    return {
        "lam_stiff": lam_stiff,
        "lam_damp":  lam_damp,
        "data_mse":  data_mse,
        "res_mse":   res_mse,
        "sindy_r2":  dist["r2"],
        "expr":      dist["expr"],
    }


def main():
    print("=" * 65)
    print("LAMBDA SWEEP: multidimensional Pareto ablation")
    print(f"Grid: {len(LAMBDA_VALS)}×{len(LAMBDA_VALS)} = {len(LAMBDA_VALS)**2} runs")
    print("=" * 65)

    t, x_noisy, v_noisy, x_clean, v_clean = generate_data(
        t_max=20.0, n_points=400, noise_std=0.01, seed=0)
    extra_traj = []
    for seed, y0 in [(1, (2.0, 0.0)), (2, (0.5, 1.5)), (3, (-1.5, -1.0))]:
        te, xe, ve, _, _ = generate_data(t_max=20.0, n_points=400,
                                          noise_std=0.01, seed=seed, y0=y0)
        extra_traj.append((te, xe, ve))

    physics_terms = PhysicsTerms(c=TRUE_PARAMS["c"], k=TRUE_PARAMS["k"],
                                  F0=TRUE_PARAMS["F0"], w_f=TRUE_PARAMS["w_f"])
    t_col = np.linspace(0, 20, 800)

    records = []
    run_id = 0
    for lam_s in LAMBDA_VALS:
        for lam_d in LAMBDA_VALS:
            run_id += 1
            rec = run_one(lam_s, lam_d, t, x_noisy, physics_terms,
                          extra_traj, t_col, run_id)
            records.append(rec)

    # ── build result matrices ─────────────────────────────────────────
    N = len(LAMBDA_VALS)
    R2_grid   = np.zeros((N, N))
    MSE_grid  = np.zeros((N, N))
    DATA_grid = np.zeros((N, N))

    for i, ls in enumerate(LAMBDA_VALS):
        for j, ld in enumerate(LAMBDA_VALS):
            rec = next(r for r in records
                       if abs(r["lam_stiff"]-ls)<1e-9 and abs(r["lam_damp"]-ld)<1e-9)
            R2_grid[i, j]   = rec["sindy_r2"]
            MSE_grid[i, j]  = rec["res_mse"]
            DATA_grid[i, j] = rec["data_mse"]

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tick_labels = [f"{v:.3f}" for v in LAMBDA_VALS]

    for ax, grid, title, cmap in [
        (axes[0], R2_grid,   "SINDy R² (↑ better)",          "viridis"),
        (axes[1], MSE_grid,  "Residual MSE vs truth (↓ better)", "viridis_r"),
        (axes[2], DATA_grid, "Data MSE (↓ better)",           "viridis_r"),
    ]:
        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       vmin=np.nanpercentile(grid, 5),
                       vmax=np.nanpercentile(grid, 95))
        ax.set_xticks(range(N)); ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticks(range(N)); ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("λ_damping"); ax.set_ylabel("λ_stiffness")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)
        # Mark the best cell
        best = np.unravel_index(
            np.nanargmax(grid) if "↑" in title else np.nanargmin(grid),
            grid.shape)
        ax.add_patch(plt.Rectangle(
            (best[1]-0.5, best[0]-0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2))

    plt.suptitle(
        f"Multidimensional Pareto sweep\n"
        f"λ_inertia={LAMBDA_INERTIA}, λ_forcing={LAMBDA_FORCING} (fixed)\n"
        f"Red box = best configuration", fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "lambda_sweep.png")
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    # ── summary ───────────────────────────────────────────────────────
    best_r2_idx = int(np.nanargmax(R2_grid))
    bi, bj = np.unravel_index(best_r2_idx, R2_grid.shape)
    diag_r2 = np.array([R2_grid[i, i] for i in range(N)])

    print("\n" + "=" * 65)
    print("RESULT SUMMARY")
    print("=" * 65)
    print(f"Best SINDy R²: {R2_grid[bi,bj]:.4f} "
          f"at λ_s={LAMBDA_VALS[bi]:.3f}, λ_d={LAMBDA_VALS[bj]:.3f}")
    print(f"Best diagonal R² (equal λ_s=λ_d): {diag_r2.max():.4f} "
          f"at λ={LAMBDA_VALS[int(diag_r2.argmax())]:.3f}")

    if R2_grid[bi,bj] > diag_r2.max() + 0.01:
        print("✓ PARETO CONFIRMED: best result is OFF the diagonal")
        print("  → asymmetric per-term weights outperform any scalar lambda")
    else:
        print("~ Diagonal competitive with off-diagonal best.")
        print("  → try wider grid or more epochs for clearer separation")

    # Save json
    summary = {
        "grid_lambda_vals": LAMBDA_VALS,
        "lambda_inertia": LAMBDA_INERTIA,
        "lambda_forcing": LAMBDA_FORCING,
        "R2_grid": R2_grid.tolist(),
        "MSE_grid": MSE_grid.tolist(),
        "DATA_grid": DATA_grid.tolist(),
        "best_lam_stiff": LAMBDA_VALS[bi],
        "best_lam_damp": LAMBDA_VALS[bj],
        "best_r2": float(R2_grid[bi,bj]),
        "best_diagonal_r2": float(diag_r2.max()),
        "records": records,
    }
    jout = os.path.join(RESULTS_DIR, "lambda_sweep.json")
    with open(jout, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {jout}")

    return summary


if __name__ == "__main__":
    main()
