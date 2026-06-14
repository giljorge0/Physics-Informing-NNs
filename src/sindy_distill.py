"""
Symbolic distillation of the "supraphysical" residual.
=========================================================

The PINN's residual eval gives us samples of:
    (x, v, residual) where residual ≈ eps*x^3 + delta*sign(v)*v^2

We build a dictionary of candidate nonlinear functions (SINDy-style)
and use sparse regression (sequential thresholded least squares, STLSQ)
to find a small set of terms that explain the residual -- yielding a
human-readable symbolic expression instead of a black-box correction.

This avoids a heavy PySR/Julia dependency while demonstrating the same
"black-box -> symbolic expression" pipeline described in the proposal.
"""

import numpy as np


# Library of candidate functions: name -> callable(x, v) -> array
CANDIDATE_LIBRARY = {
    "1":            lambda x, v: np.ones_like(x),
    "x":            lambda x, v: x,
    "x^2":          lambda x, v: x ** 2,
    "x^3":          lambda x, v: x ** 3,
    "v":            lambda x, v: v,
    "v^2":          lambda x, v: v ** 2,
    "v^3":          lambda x, v: v ** 3,
    "sign(v)*v^2":  lambda x, v: np.sign(v) * v ** 2,
    "x*v":          lambda x, v: x * v,
    "x^2*v":        lambda x, v: x ** 2 * v,
    "x*v^2":        lambda x, v: x * v ** 2,
}


def build_library(x, v, term_names=None):
    term_names = term_names or list(dict.fromkeys(CANDIDATE_LIBRARY.keys()))  # dedupe order-preserving
    cols = []
    names = []
    seen = set()
    for name in term_names:
        col = CANDIDATE_LIBRARY[name](x, v)
        key = tuple(np.round(col, 8))
        if key in seen:
            continue
        seen.add(key)
        cols.append(col)
        names.append(name)
    Theta = np.stack(cols, axis=1)
    return Theta, names


def stlsq(Theta, y, threshold=0.05, max_iter=50, alpha=1e-6):
    """Sequential Thresholded Least Squares (core of SINDy)."""
    n_features = Theta.shape[1]
    # ridge-regularized initial guess
    coef = np.linalg.lstsq(Theta.T @ Theta + alpha * np.eye(n_features),
                            Theta.T @ y, rcond=None)[0]

    for _ in range(max_iter):
        small = np.abs(coef) < threshold
        coef[small] = 0
        big = ~small
        if not np.any(big):
            break
        coef[big] = np.linalg.lstsq(
            Theta[:, big].T @ Theta[:, big] + alpha * np.eye(big.sum()),
            Theta[:, big].T @ y, rcond=None
        )[0]
    return coef


def distill(x, v, residual, threshold=2.0, normalize=True):
    """Run STLSQ and return a dict of {term_name: coefficient} plus
    a human-readable expression string."""
    Theta, names = build_library(x, v)

    if normalize:
        scales = np.linalg.norm(Theta, axis=0)
        scales[scales == 0] = 1.0
        Theta_n = Theta / scales
    else:
        scales = np.ones(Theta.shape[1])
        Theta_n = Theta

    coef_n = stlsq(Theta_n, residual, threshold=threshold)
    coef = coef_n / scales

    terms = {name: c for name, c in zip(names, coef) if abs(c) > 1e-8}

    expr = " + ".join(f"{c:.4f}*{name}" for name, c in terms.items()) or "0"

    # quality metric
    y_pred = Theta @ coef
    ss_res = np.sum((residual - y_pred) ** 2)
    ss_tot = np.sum((residual - residual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"terms": terms, "expr": expr, "r2": r2, "coef_full": dict(zip(names, coef))}


if __name__ == "__main__":
    # quick self-test: residual = 0.4*x^3 + 0.15*sign(v)*v^2
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 500)
    v = rng.uniform(-2, 2, 500)
    residual = 0.4 * x ** 3 + 0.15 * np.sign(v) * v ** 2 + rng.normal(0, 0.01, 500)
    result = distill(x, v, residual, threshold=2.0)
    print("Recovered expression:", result["expr"])
    print("R^2:", result["r2"])
