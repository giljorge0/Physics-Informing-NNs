"""
Informing net experiment — v3.

Three modes compared:
  A) No informing net      — baseline: distill raw PINN residual
  B) Post-hoc informing    — InformingNet trained AFTER InformedNet is frozen,
                             learns to fit the residual cleanly.
                             Distill the InformingNet output.
  C) Adaptive λ + post-hoc — combine best of both: adaptive weights find the
                             right physics balance, then post-hoc net cleans up.

Key insight from v2 diagnosis:
  - Joint training absorbs the residual signal into the correction term.
  - Post-hoc training keeps InformedNet's residual undisturbed, then fits it cleanly.
  - The InformingNet output (not the raw residual) is the better signal to distill.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.oscillator import generate_data, TRUE_PARAMS, hidden_term
from src.pinn import (PhysicsTerms, TrainConfig, train,
                      evaluate_supraphysical_residual, evaluate_informing_output)
from src.sindy_distill import distill

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS  = 5000
WARMUP  = 2000
LAMBDA_FIXED = {"inertia":0.05,"damping":0.05,"stiffness":0.05,"forcing":0.05}


def pool_residuals(informed, informing, pt, t_main, extra_trajs, use_informing=False):
    all_x, all_v, all_sig, all_true = [], [], [], []
    for te, _, _ in [(t_main, None, None)] + extra_trajs:
        t_eval = np.linspace(te[0]+0.5, te[-1]-0.5, 200)
        if use_informing and informing is not None:
            xp, vp, sig = evaluate_informing_output(informed, informing, pt, t_eval)
        else:
            xp, vp, _, sig = evaluate_supraphysical_residual(informed, pt, t_eval)
        all_x.append(xp); all_v.append(vp)
        all_sig.append(sig); all_true.append(hidden_term(xp, vp))
    return (np.concatenate(all_x), np.concatenate(all_v),
            np.concatenate(all_sig), np.concatenate(all_true))


def run_mode(label, config, t, x_noisy, pt, t_col, extra_trajs, use_informing_for_distill):
    print(f"\n{'='*60}\nMODE: {label}\n{'='*60}")
    result = train(t, x_noisy, pt, t_col, config)

    xp, vp, sig, true_h = pool_residuals(
        result["informed"], result["informing"], pt, t, extra_trajs,
        use_informing=use_informing_for_distill)

    dist = distill(xp, vp, sig, threshold=2.0)
    res_mse = float(np.mean((sig - true_h)**2))
    corr    = float(np.corrcoef(sig, true_h)[0,1])
    data_mse= result["loss_history"]["data"][-1]

    print(f"Signal corr with truth: {corr:.4f}")
    print(f"SINDy R²: {dist['r2']:.4f}   res_mse={res_mse:.4e}")
    print(f"Recovered: {dist['expr']}")
    print(f"Truth:     0.4000*x^3 + 0.1500*sign(v)*v^2")

    return {"label": label, "result": result,
            "xp": xp, "vp": vp, "sig": sig, "true_h": true_h,
            "dist": dist, "res_mse": res_mse, "corr": corr, "data_mse": data_mse}


def main():
    t, x_noisy, _, x_clean, _ = generate_data(
        t_max=20., n_points=400, noise_std=0.01, seed=0)
    extra_trajs = []
    for seed, y0 in [(1,(2.,0.)),(2,(.5,1.5)),(3,(-1.5,-1.))]:
        te,xe,ve,_,_ = generate_data(t_max=20.,n_points=400,noise_std=.01,seed=seed,y0=y0)
        extra_trajs.append((te,xe,ve))

    pt    = PhysicsTerms(c=TRUE_PARAMS["c"], k=TRUE_PARAMS["k"],
                         F0=TRUE_PARAMS["F0"], w_f=TRUE_PARAMS["w_f"])
    t_col = np.linspace(0, 20, 800)

    modes = [
        # (label, config, use_informing_for_distill)
        ("A: no informing — distill raw residual",
         TrainConfig(epochs=EPOCHS, warmup_epochs=WARMUP, lr=1e-3,
                     adaptive=False, lambda_fixed=LAMBDA_FIXED,
                     use_informing_net=False, print_every=1000),
         False),

        ("B: post-hoc informing — distill InformingNet output",
         TrainConfig(epochs=EPOCHS, warmup_epochs=WARMUP, lr=1e-3,
                     adaptive=False, lambda_fixed=LAMBDA_FIXED,
                     use_informing_net=True, adversarial=True,
                     informing_epochs=2000, informing_lr=5e-4, informing_reg=0.02,
                     print_every=1000),
         True),

        ("C: adaptive λ + post-hoc informing",
         TrainConfig(epochs=EPOCHS, warmup_epochs=WARMUP, lr=1e-3,
                     adaptive=True, adaptive_alpha=0.7, adaptive_ema=0.7,
                     lambda_init={"inertia":0.05,"damping":0.05,"stiffness":0.05,"forcing":0.05},
                     use_informing_net=True, adversarial=True,
                     informing_epochs=2000, informing_lr=5e-4, informing_reg=0.02,
                     print_every=1000),
         True),
    ]

    all_results = []
    for label, config, use_inf in modes:
        r = run_mode(label, config, t, x_noisy, pt, t_col, extra_trajs, use_inf)
        all_results.append(r)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, r in enumerate(all_results):
        ax = axes[0, col]
        ax.scatter(r["xp"], r["true_h"], s=6, alpha=0.3, label="true missing")
        ax.scatter(r["xp"], r["sig"],    s=6, alpha=0.3, label="recovered signal")
        ax.set_title(r["label"].split(" — ")[0], fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("missing physics")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95, f"R²={r['dist']['r2']:.3f}  corr={r['corr']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.text(0.05, 0.82, r["dist"]["expr"][:40],
                transform=ax.transAxes, va="top", fontsize=7, color="navy")

        ax = axes[1, col]
        lh = r["result"]["loss_history"]
        ax.plot(lh["data"],    label="data",    lw=1)
        ax.plot(lh["physics"], label="physics", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(fontsize=7)
        if r["result"]["informing_history"]["loss"]:
            ax2 = ax.twinx()
            ax2.plot(r["result"]["informing_history"]["loss"],
                     color="green", alpha=0.5, lw=1, label="inf loss")
            ax2.set_ylabel("inf loss", color="green", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="green")

    plt.suptitle("Informing net comparison (v3)\n"
                 "Ground truth: 0.4·x³ + 0.15·sign(v)·v²", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "adversarial_comparison.png")
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"{r['label'][:45]:<45} R²={r['dist']['r2']:.4f}  "
              f"corr={r['corr']:.3f}  data_mse={r['data_mse']:.3e}")

    return all_results


if __name__ == "__main__":
    main()
