"""
PDE Physics-Informing experiment: 1D Reaction-Diffusion with hidden source.

TRUE PDE:    u_t = D*u_xx + r*u*(1-u) + eps*sin(pi*x)*u²
KNOWN:       u_t = D*u_xx + r*u*(1-u)
HIDDEN:      eps*sin(pi*x)*u²

Pipeline:
1. Generate noisy PDE data (integrate true system on t×x grid)
2. Train PDE-PINN against INCOMPLETE known physics (per-term adaptive λ_i)
3. Evaluate supraphysical residual at dense (t,x) collocation points
4. Symbolic distillation: find f(x,u) such that f ≈ residual
5. Compare recovered symbolic expression to ground truth
6. Lambda sweep: test multidimensional Pareto on PDE per-term weights

Key questions:
  - Does the PDE PINN correctly identify the residual as eps*sin(pi*x)*u²?
  - Does the per-term decomposition (temporal, diffusion, linear rxn, nonlinear rxn)
    show different convergence rates? (diffusion usually converges hardest)
  - Is the Pareto effect visible in PDE space too?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.react_diff import (
    generate_data_pde, TRUE_PARAMS_PDE, hidden_term_pde,
    build_pde_library, PDE_CANDIDATE_LIBRARY,
)
from src.pinn_pde import (
    PDEPhysicsTerms, PDETrainConfig, train_pde, evaluate_pde_residual,
)
from src.sindy_distill import stlsq

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── PDE SINDy distillation ────────────────────────────────────────────
def distill_pde(x_flat, u_flat, residual_flat, threshold=1.5):
    """Sparse regression on the PDE residual using spatial library."""
    Theta, names = build_pde_library(x_flat, u_flat)

    # Normalise columns
    scales = np.linalg.norm(Theta, axis=0)
    scales[scales == 0] = 1.0
    Theta_n = Theta / scales

    coef_n = stlsq(Theta_n, residual_flat, threshold=threshold)
    coef   = coef_n / scales

    terms = {n: c for n, c in zip(names, coef) if abs(c) > 1e-6}
    expr  = " + ".join(f"{c:.4f}*{n}" for n, c in terms.items()) or "0"

    y_pred = Theta @ coef
    ss_res = np.sum((residual_flat - y_pred) ** 2)
    ss_tot = np.sum((residual_flat - residual_flat.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"terms": terms, "expr": expr, "r2": r2}


def main():
    p = TRUE_PARAMS_PDE

    # ── Step 1: Generate data ─────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: Generate PDE data (Fisher-KPP + hidden source)")
    print("=" * 65)
    Nt, Nx = 60, 64
    t_eval, x_grid, u_noisy, u_clean = generate_data_pde(
        t_max=2.0, Nt=Nt, Nx=Nx, noise_std=0.005, seed=0,
    )
    print(f"Grid: {Nt} time points × {Nx} spatial points")
    print(f"u range: [{u_clean.min():.3f}, {u_clean.max():.3f}]")
    hidden_mag = np.abs(hidden_term_pde(x_grid[None,:], u_clean)).mean()
    print(f"Hidden term mean magnitude: {hidden_mag:.4f}")
    print(f"True hidden: {p['eps']:.3f}*sin(π·x)*u²")

    # Flatten (t,x,u) into 1D arrays for training
    T_mesh, X_mesh = np.meshgrid(t_eval, x_grid, indexing="ij")  # (Nt, Nx)
    t_flat   = T_mesh.flatten()
    x_flat   = X_mesh.flatten()
    u_flat   = u_noisy.flatten()

    # Collocation: denser than data
    t_col_1d = np.linspace(0, 2.0, 80)
    x_col_1d = np.linspace(0, 1.0, 80)
    T_col, X_col = np.meshgrid(t_col_1d, x_col_1d, indexing="ij")
    t_col_flat   = T_col.flatten()
    x_col_flat   = X_col.flatten()

    pde_terms = PDEPhysicsTerms(D=p["D"], r=p["r"])

    # ── Step 2: Train PDE-PINN ────────────────────────────────────────
    print()
    print("=" * 65)
    print("STEP 2: Train PDE-PINN (adaptive per-term λ_i)")
    print("  Known: u_t = D*u_xx + r*u*(1-u)")
    print("  Hidden: eps*sin(π*x)*u²  ← NOT in loss")
    print("=" * 65)

    config = PDETrainConfig(
        epochs=8000, warmup_epochs=3000, lr=5e-4,
        adaptive=True, adaptive_alpha=0.7, adaptive_ema=0.7,
        lambda_init={"temporal":0.1,"diffusion":0.1,"lin_rxn":0.1,"nonlin_rxn":0.1},
        t_max=2.0, x_max=1.0,
        print_every=1000,
    )
    result = train_pde(t_flat, x_flat, u_flat, pde_terms,
                       t_col_flat, x_col_flat, config)

    # ── Step 3: Evaluate supraphysical residual ───────────────────────
    print()
    print("=" * 65)
    print("STEP 3: Evaluate PDE supraphysical residual")
    print("=" * 65)

    # Evaluate on a dense grid
    t_eval_dense = np.linspace(0.2, 1.8, 60)
    x_eval_dense = np.linspace(0.05, 0.95, 64)
    T_ev, X_ev = np.meshgrid(t_eval_dense, x_eval_dense, indexing="ij")
    t_ev_flat  = T_ev.flatten()
    x_ev_flat  = X_ev.flatten()

    x_out, u_out, gap_out = evaluate_pde_residual(
        result["u_net"], pde_terms, t_ev_flat, x_ev_flat, config,
    )

    # True hidden term at same points
    true_hidden = hidden_term_pde(x_ev_flat, u_out)

    gap_mse  = float(np.mean((gap_out - true_hidden) ** 2))
    true_mag = float(np.abs(true_hidden).mean())
    print(f"Residual MSE vs true hidden:  {gap_mse:.4e}")
    print(f"True hidden term mean |·|:    {true_mag:.4f}")
    print(f"Relative error:               {gap_mse/true_mag**2:.3f}")

    # ── Oracle check ──────────────────────────────────────────────────
    oracle = distill_pde(x_ev_flat, u_out, true_hidden, threshold=1.5)
    print(f"\n[Oracle] Distilling true hidden directly:")
    print(f"  Recovered: {oracle['expr']}")
    print(f"  R²: {oracle['r2']:.4f}")

    # ── Step 4: Symbolic distillation ────────────────────────────────
    print()
    print("=" * 65)
    print("STEP 4: Symbolic distillation of PDE residual")
    print("=" * 65)
    dist = distill_pde(x_ev_flat, u_out, gap_out, threshold=1.5)
    print(f"Recovered expression: {dist['expr']}")
    print(f"R²: {dist['r2']:.4f}")
    print(f"Ground truth: {p['eps']:.4f}*sin(π*x)*u²")

    # ── Step 5: Lambda sweep (3×3 for PDE — faster) ──────────────────
    print()
    print("=" * 65)
    print("STEP 5: Lambda sweep — diffusion vs nonlin_rxn")
    print("  (These are the most physically uncertain terms)")
    print("=" * 65)

    SWEEP_VALS = [0.02, 0.1, 0.5]
    sweep_results = []
    run_id = 0
    for lam_diff in SWEEP_VALS:
        for lam_nlrxn in SWEEP_VALS:
            run_id += 1
            cfg_sweep = PDETrainConfig(
                epochs=6000, warmup_epochs=2500, lr=5e-4,
                adaptive=False,
                lambda_fixed={
                    "temporal":   0.1,
                    "diffusion":  lam_diff,
                    "lin_rxn":    0.1,
                    "nonlin_rxn": lam_nlrxn,
                },
                t_max=2.0, x_max=1.0,
                print_every=99999,  # silent
            )
            r = train_pde(t_flat, x_flat, u_flat, pde_terms,
                          t_col_flat, x_col_flat, cfg_sweep)
            x_s, u_s, gap_s = evaluate_pde_residual(
                r["u_net"], pde_terms, t_ev_flat, x_ev_flat, cfg_sweep)
            d = distill_pde(x_s, u_s, gap_s, threshold=1.5)
            data_mse = r["loss_history"]["data"][-1]
            print(f"  [{run_id:2d}] λ_diff={lam_diff:.2f} λ_nlrxn={lam_nlrxn:.2f} "
                  f"| data_mse={data_mse:.3e} sindy_R²={d['r2']:.3f} | {d['expr'][:50]}")
            sweep_results.append({
                "lam_diff": lam_diff, "lam_nlrxn": lam_nlrxn,
                "sindy_r2": d["r2"], "data_mse": data_mse,
                "expr": d["expr"],
            })

    # ── Plots ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("STEP 6: Saving plots")
    print("=" * 65)

    # ── Plot 1: PDE solution snapshots ───────────────────────────────
    t_snap_idx = [0, len(t_eval)//4, len(t_eval)//2, -1]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    # Get predictions on the data grid
    T_pred, X_pred = np.meshgrid(t_eval, x_grid, indexing="ij")
    t_pred_flat = T_pred.flatten()
    x_pred_flat = X_pred.flatten()
    x_p, u_p, gap_p = evaluate_pde_residual(
        result["u_net"], pde_terms, t_pred_flat, x_pred_flat, config)
    u_pred_grid = u_p.reshape(Nt, Nx)
    gap_grid    = gap_p.reshape(Nt, Nx)

    for col, ti in enumerate(t_snap_idx):
        ax = axes[0, col]
        ax.plot(x_grid, u_clean[ti], "g--", lw=1.5, label="true")
        ax.plot(x_grid, u_noisy[ti], "k.", ms=3, alpha=0.4, label="noisy")
        ax.plot(x_grid, u_pred_grid[ti], "r-", lw=1.5, label="PINN")
        ax.set_title(f"t={t_eval[ti]:.2f}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("u")
        if col == 0: ax.legend(fontsize=7)

        ax = axes[1, col]
        true_h = hidden_term_pde(x_grid, u_clean[ti])
        ax.plot(x_grid, true_h, "g--", lw=1.5, label="true hidden")
        ax.plot(x_grid, gap_grid[ti], "r-", lw=1.5, label="PINN residual")
        ax.set_title(f"Residual t={t_eval[ti]:.2f}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("hidden term")
        if col == 0: ax.legend(fontsize=7)

    plt.suptitle(
        f"PDE Physics-Informing: Fisher-KPP + hidden {p['eps']}·sin(π·x)·u²\n"
        f"Recovered: {dist['expr'][:60]}  R²={dist['r2']:.3f}", fontsize=10)
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "pde_solution.png")
    plt.savefig(out1, dpi=130); print(f"Saved: {out1}")

    # ── Plot 2: Lambda evolution + loss curves ────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    term_colors = {"temporal":"#2196F3","diffusion":"#FF9800",
                   "lin_rxn":"#4CAF50","nonlin_rxn":"#9C27B0"}
    ax = axes2[0]
    for name, hist in result["lambda_history"].items():
        if any(h > 0 for h in hist):
            ax.plot(hist, label=name, color=term_colors.get(name,"gray"), lw=1.5)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("λᵢ")
    ax.set_title("Per-term λ_i evolution (PDE)"); ax.legend(fontsize=8)

    ax = axes2[1]
    lh = result["loss_history"]
    ax.plot(lh["data"],    label="data",    lw=1)
    ax.plot(lh["physics"], label="physics", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training losses (PDE)"); ax.legend(fontsize=8)

    # Sweep heatmap
    ax = axes2[2]
    N  = len(SWEEP_VALS)
    R2_mat = np.zeros((N, N))
    for i, ld in enumerate(SWEEP_VALS):
        for j, ln in enumerate(SWEEP_VALS):
            rec = next(r for r in sweep_results
                       if abs(r["lam_diff"]-ld)<1e-9 and abs(r["lam_nlrxn"]-ln)<1e-9)
            R2_mat[i,j] = rec["sindy_r2"]
    im = ax.imshow(R2_mat, cmap="viridis",
                   vmin=np.nanmin(R2_mat), vmax=np.nanmax(R2_mat))
    tl = [f"{v}" for v in SWEEP_VALS]
    ax.set_xticks(range(N)); ax.set_xticklabels(tl)
    ax.set_yticks(range(N)); ax.set_yticklabels(tl)
    ax.set_xlabel("λ_nonlin_rxn"); ax.set_ylabel("λ_diffusion")
    ax.set_title("PDE Pareto sweep: SINDy R²")
    plt.colorbar(im, ax=ax)
    bi, bj = np.unravel_index(np.nanargmax(R2_mat), R2_mat.shape)
    ax.add_patch(plt.Rectangle((bj-.5,bi-.5),1,1,
                 fill=False, edgecolor="red", linewidth=2))

    plt.suptitle("PDE MP-PINN: adaptive weights & Pareto sweep", fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "pde_analysis.png")
    plt.savefig(out2, dpi=130); print(f"Saved: {out2}")

    # ── Text summary ──────────────────────────────────────────────────
    best_r2_rec = max(sweep_results, key=lambda r: r["sindy_r2"])
    diag_r2     = [r["sindy_r2"] for r in sweep_results
                   if abs(r["lam_diff"] - r["lam_nlrxn"]) < 1e-9]

    print()
    print("=" * 65)
    print("PDE EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"True hidden term: {p['eps']}·sin(π·x)·u²")
    print(f"Oracle R²:        {oracle['r2']:.4f}  ({oracle['expr']})")
    print(f"PINN residual R²: {dist['r2']:.4f}  ({dist['expr'][:60]})")
    print(f"Residual MSE:     {gap_mse:.4e}")
    print()
    print("PDE Pareto sweep:")
    print(f"  Best:     R²={best_r2_rec['sindy_r2']:.4f}  "
          f"λ_d={best_r2_rec['lam_diff']}, λ_nlrxn={best_r2_rec['lam_nlrxn']}")
    if diag_r2:
        print(f"  Diagonal: R²={max(diag_r2):.4f}")
        if best_r2_rec["sindy_r2"] > max(diag_r2) + 0.01:
            print("  ✓ PARETO CONFIRMED in PDE space too")
        else:
            print("  ~ Diagonal competitive (try wider grid)")

    # Save lambda history for final epoch
    print()
    print("Final adaptive λ_i (adaptive run):")
    for name, hist in result["lambda_history"].items():
        if hist:
            vals = [h for h in hist if h > 0]
            if vals:
                print(f"  {name:<15}: {vals[0]:.4f} → {vals[-1]:.4f}  "
                      f"({'↓' if vals[-1] < vals[0] else '↑'} "
                      f"{abs(vals[-1]-vals[0])/vals[0]*100:.1f}%)")

    return result, dist, sweep_results


if __name__ == "__main__":
    main()
