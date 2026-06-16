"""
Adaptive weights experiment.

Compares three lambda strategies:
  A) Fixed equal        — lambda_i = 0.05 for all (original toy)
  B) Fixed hand-tuned   — asymmetric, physics-motivated prior
     (inertia trusted more than stiffness, damping uncertain)
  C) Adaptive           — relative-progress-based (the fixed version)

For each: show lambda evolution, distillation quality, and the
"residual landscape" — does the adaptive scheme automatically
down-weight the physics terms that conflict with the hidden nonlinearity?
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

EPOCHS = 5000
WARMUP = 2000


def run_strategy(label, adaptive, lambda_fixed, adaptive_alpha,
                 t, x_noisy, physics_terms, t_col, extra_traj):
    print(f"\n{'='*55}\nSTRATEGY: {label}\n{'='*55}")

    config = TrainConfig(
        epochs=EPOCHS,
        warmup_epochs=WARMUP,
        lr=1e-3,
        adaptive=adaptive,
        lambda_fixed=lambda_fixed,
        lambda_init={k: lambda_fixed[k] for k in lambda_fixed} if not adaptive else
                    {"inertia":0.05,"damping":0.05,"stiffness":0.05,"forcing":0.05},
        adaptive_alpha=adaptive_alpha,
        adaptive_ema=0.7,
        use_informing_net=False,
        print_every=1000,
    )
    result = train(t, x_noisy, physics_terms, t_col, config)

    # Pool residuals
    all_x, all_v, all_res, all_true = [], [], [], []
    for te, _, _ in [(t, x_noisy, None)] + extra_traj:
        t_eval = np.linspace(te[0]+0.5, te[-1]-0.5, 150)
        xp, vp, _, res = evaluate_supraphysical_residual(
            result["informed"], physics_terms, t_eval)
        all_x.append(xp); all_v.append(vp)
        all_res.append(res); all_true.append(hidden_term(xp, vp))

    x_p = np.concatenate(all_x); v_p = np.concatenate(all_v)
    r_p = np.concatenate(all_res); t_p = np.concatenate(all_true)

    dist = distill(x_p, v_p, r_p, threshold=2.0)
    res_mse = float(np.mean((r_p - t_p)**2))
    data_mse = result["loss_history"]["data"][-1]

    print(f"R²={dist['r2']:.4f}  data_mse={data_mse:.3e}  res_mse={res_mse:.3e}")
    print(f"Expr: {dist['expr']}")

    return {
        "label": label,
        "result": result,
        "x_pool": x_p, "v_pool": v_p,
        "res_pool": r_p, "true_pool": t_p,
        "dist": dist, "res_mse": res_mse, "data_mse": data_mse,
    }


def main():
    t, x_noisy, _, x_clean, _ = generate_data(
        t_max=20.0, n_points=400, noise_std=0.01, seed=0)
    extra_traj = []
    for seed, y0 in [(1,(2.,0.)),(2,(.5,1.5)),(3,(-1.5,-1.))]:
        te,xe,ve,_,_ = generate_data(t_max=20.,n_points=400,noise_std=.01,seed=seed,y0=y0)
        extra_traj.append((te,xe,ve))

    physics_terms = PhysicsTerms(c=TRUE_PARAMS["c"], k=TRUE_PARAMS["k"],
                                  F0=TRUE_PARAMS["F0"], w_f=TRUE_PARAMS["w_f"])
    t_col = np.linspace(0, 20, 800)

    strategies = [
        # (label, adaptive, lambda_fixed, adaptive_alpha)
        ("A: fixed equal",
         False,
         {"inertia":0.05,"damping":0.05,"stiffness":0.05,"forcing":0.05},
         0.0),

        ("B: fixed hand-tuned",
         False,
         # Physics-motivated: inertia very trusted (Newton's 2nd),
         # stiffness less trusted (nonlinear?), damping uncertain
         {"inertia":0.10,"damping":0.02,"stiffness":0.03,"forcing":0.08},
         0.0),

        ("C: adaptive (relative-progress)",
         True,
         {"inertia":0.05,"damping":0.05,"stiffness":0.05,"forcing":0.05},
         0.7),
    ]

    all_results = []
    for args in strategies:
        r = run_strategy(*args, t, x_noisy, physics_terms, t_col, extra_traj)
        all_results.append(r)

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    term_colors = {"inertia":"#2196F3","damping":"#FF9800",
                   "stiffness":"#4CAF50","forcing":"#9C27B0"}

    for col, r in enumerate(all_results):
        # Row 0: lambda history
        ax = axes[0, col]
        lh = r["result"]["lambda_history"]
        for name, hist in lh.items():
            if any(h > 0 for h in hist):
                ax.plot(hist, label=name, color=term_colors.get(name,"gray"), lw=1.5)
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("λᵢ")
        ax.set_yscale("log"); ax.legend(fontsize=7)
        ax.set_ylim(1e-5, 3)

        # Row 1: residual scatter + annotation
        ax = axes[1, col]
        ax.scatter(r["x_pool"], r["true_pool"], s=6, alpha=0.3, label="true missing")
        ax.scatter(r["x_pool"], r["res_pool"],  s=6, alpha=0.3, label="PINN residual")
        ax.set_xlabel("x"); ax.set_ylabel("missing physics")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95, f"SINDy R²={r['dist']['r2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round",fc="white",alpha=0.8))
        ax.text(0.05, 0.82, r["dist"]["expr"][:42],
                transform=ax.transAxes, va="top", fontsize=7, color="navy")

    plt.suptitle("Adaptive lambda strategies comparison\n"
                 "Ground truth: 0.4·x³ + 0.15·sign(v)·v²", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "adaptive_weights_comparison.png")
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    print("\n" + "="*55)
    print("ADAPTIVE WEIGHTS SUMMARY")
    print("="*55)
    for r in all_results:
        print(f"{r['label']:<35} R²={r['dist']['r2']:.4f}  "
              f"data_mse={r['data_mse']:.3e}")

    return all_results


if __name__ == "__main__":
    main()
