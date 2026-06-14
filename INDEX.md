# Physics-Informing Neural Networks — Complete File List

All files are individually downloadable below. Download them and organize as shown in the **Directory Structure** section.

---

## 📥 **DOCUMENTATION (Read First)**

Start with these to understand the project:

1. **README_physics_informing.md** — Main conceptual overview
   - What the project implements (the 3 proposal ideas)
   - How the test system works
   - Current results + why distillation isn't perfect
   - 5 future research directions

2. **QUICKSTART_physics_informing.md** — How to run & iterate
   - Installation (one command)
   - How to run the full experiment
   - Key parameters to tune (with examples)
   - 5 quick experiments to try

3. **FILES_physics_informing.md** — Detailed file-by-file guide
   - What each source file does
   - How to edit them for your own systems
   - Typical iteration workflow
   - Checklist for porting to a new physics problem

---

## 🐍 **PYTHON SOURCE CODE**

The 4 core modules. Download these and place in a folder structure (see below).

1. **oscillator.py** — Ground-truth system + data generation
   - Defines the true hidden-term ODE
   - Defines the incomplete (known) physics
   - `generate_data(...)` function for synthetic experiments

2. **pinn.py** — Physics-Informed Neural Network
   - `InformedNet`: solution network x(t)
   - `InformingNet`: auxiliary correction network
   - `PhysicsTerms`: decomposed per-term residuals
   - `AdaptiveWeights`: optional adaptive lambda_i
   - `train(...)`: main training loop
   - `evaluate_supraphysical_residual(...)`: extract residual for distillation

3. **sindy_distill.py** — Symbolic regression (SINDy)
   - `CANDIDATE_LIBRARY`: library of basis functions
   - `stlsq(...)`: sparse regression solver
   - `distill(...)`: main entry point for symbolic expression discovery

4. **run_experiment.py** — Complete end-to-end pipeline
   - Data generation (Step 1)
   - PINN training (Step 2)
   - Residual evaluation on multiple trajectories (Step 3)
   - Symbolic distillation (Step 4)
   - Plotting & summary (Step 5)

**How to use:** `python run_experiment.py`

---

## 📊 **RESULTS FROM A SUCCESSFUL RUN**

These are outputs from a completed experiment (included for reference):

1. **experiment_summary.png** — 3-panel figure
   - Left: PINN solution fit (red) vs. noisy data (black dots) vs. true clean solution (green dashed)
   - Middle: Lambda weight history across training epochs
   - Right: Recovered residual (orange) vs. ground truth missing term (blue)

2. **loss_curves.png** — Training dynamics
   - Data loss (blue): decreases from 0.4 to ~0.007
   - Physics loss (orange): active after warmup (~0.05)
   - Total loss (green): smooth decline
   - Vertical line at epoch 2000: warmup-to-physics transition

3. **results_summary.txt** — Numerical summary
   - True hidden term parameters (eps=0.4, delta=0.15)
   - Recovered symbolic expression from PINN residual
   - Oracle check: distilling the exact hidden term (R²=1.0, proving method works)
   - Final loss values

---

## 📁 **HOW TO ORGANIZE ON YOUR MACHINE**

After downloading, create this directory structure:

```
physics-informing-nn/
├── README.md                 ← README_physics_informing.md
├── QUICKSTART.md             ← QUICKSTART_physics_informing.md
├── FILES.md                  ← FILES_physics_informing.md
│
├── src/
│   ├── oscillator.py
│   ├── pinn.py
│   └── sindy_distill.py
│
├── experiments/
│   └── run_experiment.py
│
└── results/                  ← (will be created by run_experiment.py)
    ├── experiment_summary.png
    ├── loss_curves.png
    └── summary.txt
```

Or for quick testing, just put all `.py` files in one folder and run `python run_experiment.py`.

---

## 🚀 **QUICK START (3 STEPS)**

1. **Download all `.py` files** (oscillator.py, pinn.py, sindy_distill.py, run_experiment.py) into one folder

2. **Install dependencies:**
   ```bash
   pip install torch numpy scipy matplotlib
   ```

3. **Run the experiment:**
   ```bash
   python run_experiment.py
   ```

   ✅ Results appear in `results/` within ~90 seconds

---

## 📋 **WHAT EACH FILE CONTAINS**

| Filename | Lines | Purpose | Editable? |
|----------|-------|---------|-----------|
| README_physics_informing.md | 180 | Concepts, theory, results, future work | Read-only |
| QUICKSTART_physics_informing.md | 220 | Installation, running, tuning examples | Reference |
| FILES_physics_informing.md | 400 | Deep dive: file-by-file guide with code examples | Reference |
| oscillator.py | 95 | Ground truth system + data generation | ✅ Yes — change hidden term strength, system params |
| pinn.py | 410 | PINN architecture + training loop | ⚠️ Carefully — core logic |
| sindy_distill.py | 130 | Sparse regression on residual | ✅ Yes — add/remove candidate terms, tune threshold |
| run_experiment.py | 250 | End-to-end pipeline | ✅ Yes — main knob-turning file |
| experiment_summary.png | — | 3 plots from successful run | Reference |
| loss_curves.png | — | Training loss curves | Reference |
| results_summary.txt | — | Numbers from successful run | Reference |

---

## 💡 **KEY EDIT POINTS FOR ITERATION**

### Change the hidden physics term
**File:** `oscillator.py`
```python
TRUE_PARAMS["eps"] = 0.8      # Cubic stiffness strength
TRUE_PARAMS["delta"] = 0.3    # Quadratic drag strength
```

### Tune PINN training
**File:** `run_experiment.py`, line ~70
```python
config = TrainConfig(
    epochs=5000,
    warmup_epochs=2000,
    lr=1e-3,
    lambda_fixed={"inertia": 0.05, "damping": 0.05, ...}  # ← tune these
)
```

### Adjust symbolic regression sparsity
**File:** `run_experiment.py`, line ~120
```python
distilled = distill(x_pred, v_pred, residual, threshold=2.0)  # ← try 1.0, 3.0, 5.0
```

### Add candidate terms to the library
**File:** `sindy_distill.py`, line ~15
```python
CANDIDATE_LIBRARY["x^4"] = lambda x, v: x**4
CANDIDATE_LIBRARY["x^2*v^2"] = lambda x, v: x**2 * v**2
```

---

## ✅ **VERIFICATION**

After downloading and organizing:

```bash
cd physics-informing-nn
python -c "import torch; import numpy as np; from src import oscillator, pinn, sindy_distill; print('✅ All modules import successfully')"
```

Then run the full pipeline:
```bash
python experiments/run_experiment.py
```

Should complete in ~90 seconds and produce 3 files in `results/`.

---

## 🔗 **FILE DEPENDENCIES**

```
run_experiment.py
  ├── imports src/oscillator.py
  ├── imports src/pinn.py
  └── imports src/sindy_distill.py

src/pinn.py
  └── imports torch, numpy

src/sindy_distill.py
  └── imports numpy

src/oscillator.py
  └── imports scipy.integrate
```

All are pure Python with standard scientific libraries (torch, numpy, scipy, matplotlib). No external dependencies beyond these.

---

## 📞 **TROUBLESHOOTING**

**Q: "ModuleNotFoundError: No module named 'torch'"**
A: Run `pip install torch numpy scipy matplotlib`

**Q: "No such file or directory: 'src/oscillator.py'"**
A: Make sure you're running from the `physics-informing-nn/` root directory, or adjust the `sys.path` in `run_experiment.py`

**Q: Results look wrong / PINN didn't converge**
A: See QUICKSTART_physics_informing.md section "Debugging / monitoring" for common fixes (adjust `warmup_epochs`, `lambda_fixed`, etc.)

**Q: "Oracle check" R² is not 1.0**
A: This shouldn't happen — it indicates a bug in `sindy_distill.py`. Check that `threshold=2.0` is being used and the true hidden term is being passed correctly.

---

## 🎯 **Next Steps After Running**

1. **Inspect results:** Open the 3 PNG/TXT files generated in `results/`
2. **Read results:** Check `results/summary.txt` for numbers
3. **Modify & re-run:** Try one of the "Quick experiments" in QUICKSTART.md
4. **Iterate:** Follow the workflow in FILES_physics_informing.md

---

**All files are ready to download and use immediately!**
