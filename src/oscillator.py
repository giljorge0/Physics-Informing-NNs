"""
Ground-truth system: a driven, damped oscillator with a HIDDEN nonlinear
stiffness term that the "known physics" model deliberately omits.

True ODE (ground truth, used only to generate synthetic experimental data):

    x'' + c * x' + k * x + eps * x**3 + delta * sign(x') * x'**2 = F0 * cos(w_f * t)

"Known physics" (incomplete model given to the PINN):

    x'' + c * x' + k * x = F0 * cos(w_f * t)

The PINN must reconcile the data with this *incomplete* physics. The gap
between what the incomplete physics predicts and what the data shows is the
"supraphysical" residual that we will later try to express symbolically
(via SINDy-style sparse regression) as:

    eps * x**3 + delta * sign(x') * x'**2

i.e. a cubic (Duffing-type) stiffness term + a quadratic (drag-type) damping
term -- two physically distinct "hidden physics" contributions with
different functional forms and different plausible "uncertainty" priors.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Ground truth parameters
# ----------------------------------------------------------------------
TRUE_PARAMS = dict(
    c=0.3,       # linear damping
    k=1.0,       # linear stiffness
    eps=0.4,     # Duffing (cubic stiffness) coefficient -- HIDDEN
    delta=0.15,  # quadratic drag coefficient -- HIDDEN
    F0=0.5,      # forcing amplitude
    w_f=1.2,     # forcing frequency
)


def true_rhs(t, y, p=TRUE_PARAMS):
    x, v = y
    dxdt = v
    dvdt = (
        p["F0"] * np.cos(p["w_f"] * t)
        - p["c"] * v
        - p["k"] * x
        - p["eps"] * x ** 3
        - p["delta"] * np.sign(v) * v ** 2
    )
    return [dxdt, dvdt]


def incomplete_physics_rhs(t, y, p=TRUE_PARAMS):
    """The 'known' physics, missing the eps and delta terms."""
    x, v = y
    dxdt = v
    dvdt = p["F0"] * np.cos(p["w_f"] * t) - p["c"] * v - p["k"] * x
    return [dxdt, dvdt]


def hidden_term(x, v, p=TRUE_PARAMS):
    """The exact 'missing physics' contribution, for evaluation only."""
    return p["eps"] * x ** 3 + p["delta"] * np.sign(v) * v ** 2


def generate_data(t_max=20.0, n_points=400, noise_std=0.01, seed=0,
                   y0=(1.0, 0.0), params=None):
    """Integrate the true system and return noisy (t, x, v) samples."""
    p = params or TRUE_PARAMS
    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(
        true_rhs, [0, t_max], y0, t_eval=t_eval, args=(p,),
        method="RK45", rtol=1e-9, atol=1e-9,
    )
    rng = np.random.default_rng(seed)
    x = sol.y[0] + rng.normal(0, noise_std, size=n_points)
    v = sol.y[1] + rng.normal(0, noise_std, size=n_points)
    return t_eval, x, v, sol.y[0], sol.y[1]  # noisy + clean


if __name__ == "__main__":
    t, x, v, x_clean, v_clean = generate_data()
    print(f"Generated {len(t)} points, t in [0, {t[-1]:.1f}]")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
