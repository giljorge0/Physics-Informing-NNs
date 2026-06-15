"""
Adversarial (minimax) informing experiment.

Compares three training modes side by side:
  A) No informing net   — baseline PINN with fixed lambdas
  B) Joint informing    — InformingNet trained jointly (original toy approach)
  C) Adversarial        — real minimax: InformingNet maximises residual
                          explained, InformedNet minimises total loss

For each mode: train, extract residual, distill, compare.

Key question from the proposal: does adversarial pressure produce
sparser / more physically interpretable corrections?
"""

import sys, os
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

LAMBDA_FIXED = {"inertia": 0.05, "damping": 0.05, "stiffness": 0.05, "forcing": 0.05}
EPOCHS  = 5000
WARMUP  = 2000


def run_mode(label, use_informing, adversarial, t, x_noisy, physics_terms,
             t_col, extra_traj):
    print(f"\n{'='*60}")
    print(f"MODE: {label}")
    print(f"{'='*60}")

    config = TrainConfig(
        epochs=EPOCHS,
        warmup_epochs=WARMUP,
        lr=1e-3,
        adaptive=False,
        lambda_fixed=LAMBDA_FIXED,
        use_informing_net=use_informing,
        adversarial=adversarial,
        informing_lr=5e-4,
        informing_steps=2,
        informing_reg=0.05,
        print_every=1000,
    )
    result = train(t, x_noisy, physics_terms, t_col, config)

    # Pool residuals
    all_x, all_v, all_res, all_true = [], [], [], []
    for te, xe, ve in [(t, x_noisy, None)] + [(te, xe, ve) for te, xe, ve in extra_traj]:
        t_eval = np.linspace(te[0] + 0.5, te[-1] - 0.5, 150)
        xp, vp, _, res = evaluate_supraphysical_residual(
            result["informed"], physics_terms, t_eval)
        all_x.append(xp); all_v.append(vp)
        all_res.append(res); all_true.append(hidden_term(xp, vp))

    x_pool = np.concatenate(all_x)
    v_pool = np.concatenate(all_v)
    res_pool = np.concatenate(all_res)
    true_pool = np.concatenate(all_true)

    dist = distill(x_pool, v_pool, res_pool, threshold=2.0)
    res_mse = float(np.mean((res_pool - true_pool)**2))

    print(f"\nRecovered: {dist['expr']}")
    print(f"R²={dist['r2']:.4f}   res_mse={res_mse:.4e}")
    print(f"Ground truth: 0.4000*x^3 + 0.1500*sign(v)*v^2")

    return {
        "label": label,
        "result": result,
        "x_pool": x_pool, "v_pool": v_pool,
        "res_pool": res_pool, "true_pool": true_pool,
        "dist": dist,
        "res_mse": res_mse,
    }


def main():
    t, x_noisy, _, x_clean, _ = generate_data(
        t_max=20.0, n_points=400, noise_std=0.01, seed=0)
    extra_traj = []
    for seed, y0 in [(1, (2.0, 0.0)), (2, (0.5, 1.5)), (3, (-1.5, -1.0))]:
        te, xe, ve, _, _ = generate_data(t_max=20.0, n_points=400,
                                          noise_std=0.01, seed=seed, y0=y0)
        extra_traj.append((te, xe, ve))

    physics_terms = PhysicsTerms(c=TRUE_PARAMS["c"], k=TRUE_PARAMS["k"],
                                  F0=TRUE_PARAMS["F0"], w_f=TRUE_PARAMS["w_f"])
    t_col = np.linspace(0, 20, 800)

    modes = [
        ("A: no informing net",    False, False),
        ("B: joint informing",     True,  False),
        ("C: adversarial minimax", True,  True),
    ]

    all_results = []
    for label, use_inf, adv in modes:
        r = run_mode(label, use_inf, adv, t, x_noisy, physics_terms,
                     t_col, extra_traj)
        all_results.append(r)

    # ── comparison plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, r in enumerate(all_results):
        # Row 0: residual vs truth scatter
        ax = axes[0, col]
        ax.scatter(r["x_pool"], r["true_pool"], s=6, alpha=0.3, label="true missing")
        ax.scatter(r["x_pool"], r["res_pool"],  s=6, alpha=0.3, label="PINN residual")
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("missing physics")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95, f"R²={r['dist']['r2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.text(0.05, 0.82, r["dist"]["expr"][:40],
                transform=ax.transAxes, va="top", fontsize=7, color="navy")

        # Row 1: loss curves
        ax = axes[1, col]
        lh = r["result"]["loss_history"]
        ax.plot(lh["data"],    label="data", lw=1)
        ax.plot(lh["physics"], label="physics", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(fontsize=7); ax.set_title("Training losses", fontsize=9)

    plt.suptitle(
        "Adversarial vs joint vs no-informing comparison\n"
        "Ground truth: 0.4·x³ + 0.15·sign(v)·v²", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "adversarial_comparison.png")
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    # ── text summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ADVERSARIAL EXPERIMENT SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"{r['label']:<30} R²={r['dist']['r2']:.4f}  "
              f"res_mse={r['res_mse']:.4e}  expr: {r['dist']['expr']}")

    return all_results


if __name__ == "__main__":
    main()
