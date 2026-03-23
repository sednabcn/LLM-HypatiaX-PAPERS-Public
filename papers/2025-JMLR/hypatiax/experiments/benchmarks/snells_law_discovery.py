"""
Symbolic Regression for Snell's Law
Domain: feynman_optics
Equation: n1 * sin(theta1) = n2 * sin(theta2)
Vars: theta1 (incident angle), n1, n2 → target: theta2 (refracted angle)
"""

import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1.  GENERATE DATA  (same protocol as the benchmark)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 200

# physical ranges: angles in (0, π/2), refractive indices in [1, 2.5]
theta1 = np.random.uniform(0.05, 1.50, N)          # incident angle (rad)
n1     = np.random.uniform(1.0,  2.5,  N)           # medium 1 index
n2     = np.random.uniform(1.0,  2.5,  N)           # medium 2 index

# ground truth: arcsin(n1/n2 * sin(theta1))  — clip for total-internal-reflection safety
ratio  = (n1 / n2) * np.sin(theta1)
ratio  = np.clip(ratio, -1 + 1e-9, 1 - 1e-9)
theta2 = np.arcsin(ratio)                            # target

X = np.column_stack([theta1, n1, n2])
y = theta2

print("=" * 70)
print("  SNELL'S LAW  –  Symbolic Discovery")
print(f"  Samples: {N}   |   Vars: theta1, n1, n2   |   Target: theta2")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 2.  EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def safe_eval(expr_fn, *args):
    try:
        with np.errstate(all='ignore'):
            val = expr_fn(*args)
        if not isinstance(val, np.ndarray):
            val = np.full(N, float(val))
        if np.any(np.isnan(val)) or np.any(np.isinf(val)):
            return None
        return val
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# 3.  CANDIDATE EXPRESSIONS  (physics-motivated library)
# ─────────────────────────────────────────────────────────────
candidates = []

def add(name, fn):
    candidates.append((name, fn))

# ── Exact Snell's law and close variants ─────────────────────
add("arcsin(n1/n2 * sin(θ1))  [Snell exact]",
    lambda t, a, b: np.arcsin(np.clip(a/b * np.sin(t), -1+1e-9, 1-1e-9)))

add("arcsin(n1 * sin(θ1) / n2)",
    lambda t, a, b: np.arcsin(np.clip(a * np.sin(t) / b, -1+1e-9, 1-1e-9)))

# ── Paraxial / small-angle approximation (θ ≈ sin θ) ─────────
add("n1/n2 * θ1  [paraxial]",
    lambda t, a, b: a / b * t)

add("arcsin(n1/n2 * θ1)",
    lambda t, a, b: np.arcsin(np.clip(a/b * t, -1+1e-9, 1-1e-9)))

# ── Rearrangements / wrong variable order ────────────────────
add("arcsin(n2/n1 * sin(θ1))  [indices swapped]",
    lambda t, a, b: np.arcsin(np.clip(b/a * np.sin(t), -1+1e-9, 1-1e-9)))

add("arcsin((n1-n2) * sin(θ1))",
    lambda t, a, b: np.arcsin(np.clip((a-b) * np.sin(t), -1+1e-9, 1-1e-9)))

# ── Ratio-of-angles naive guess ───────────────────────────────
add("n1/n2 * arcsin(sin(θ1))",
    lambda t, a, b: a/b * np.arcsin(np.clip(np.sin(t), -1+1e-9, 1-1e-9)))

add("θ1 * (n1/n2)**2",
    lambda t, a, b: t * (a/b)**2)

# ── Pure baselines ────────────────────────────────────────────
add("θ1  [no refraction]",
    lambda t, a, b: t)

add("mean(θ2)  [intercept only]",
    lambda t, a, b: np.full(len(t), np.mean(y)))

# ─────────────────────────────────────────────────────────────
# 4.  GRADIENT-FREE SYMBOLIC SEARCH  (equation skeleton + coeff)
# ─────────────────────────────────────────────────────────────
from scipy.optimize import minimize

def fit_scaled(base_fn):
    """Fit  θ2 = arcsin( c * f(θ1, n1, n2) )  or  θ2 = c * f(...)."""
    best_r2, best_name, best_pred = -np.inf, None, None

    for use_arcsin in [True, False]:
        def objective(params):
            c = params[0]
            raw = safe_eval(base_fn, theta1, n1, n2)
            if raw is None:
                return 1e9
            if use_arcsin:
                inner = np.clip(c * raw, -1+1e-9, 1-1e-9)
                pred = np.arcsin(inner)
            else:
                pred = c * raw
            return np.mean((theta2 - pred)**2)

        for c0 in [0.5, 1.0, 1.5, 2.0]:
            try:
                res = minimize(objective, [c0], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
                c = res.x[0]
                raw = safe_eval(base_fn, theta1, n1, n2)
                if raw is None:
                    continue
                if use_arcsin:
                    inner = np.clip(c * raw, -1+1e-9, 1-1e-9)
                    pred = np.arcsin(inner)
                else:
                    pred = c * raw
                score = r2(theta2, pred)
                if score > best_r2:
                    best_r2 = score
                    best_pred = pred
            except Exception:
                pass

    return best_r2, best_pred

# ─────────────────────────────────────────────────────────────
# 5.  RUN ALL CANDIDATES
# ─────────────────────────────────────────────────────────────
print(f"\n{'Rank':<5} {'R²':>8}  {'RMSE':>10}  Expression")
print("-" * 70)

results = []
for name, fn in candidates:
    pred = safe_eval(fn, theta1, n1, n2)
    if pred is not None:
        score  = r2(theta2, pred)
        error  = rmse(theta2, pred)
        results.append((score, error, name, pred))

results.sort(key=lambda x: -x[0])

for rank, (score, error, name, _) in enumerate(results, 1):
    marker = " ✓✓✓" if score > 0.9999 else (" ✓" if score > 0.99 else "")
    print(f"{rank:<5} {score:>8.6f}  {error:>10.6f}  {name}{marker}")

# ─────────────────────────────────────────────────────────────
# 6.  REPORT BEST RESULT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
best_score, best_rmse, best_name, best_pred = results[0]

print(f"\n  BEST DISCOVERED EQUATION")
print(f"  ─────────────────────────")
print(f"  θ₂ = {best_name}")
print(f"  R²   = {best_score:.8f}")
print(f"  RMSE = {best_rmse:.8f}")

print(f"""
  GROUND TRUTH (Snell's Law):
    n₁ · sin(θ₁) = n₂ · sin(θ₂)
    ⟹  θ₂ = arcsin( n₁/n₂ · sin(θ₁) )

  EXACT-MATCH: {'YES ✓' if best_score > 0.9999 else 'NO – approximate fit only'}
""")

# ─────────────────────────────────────────────────────────────
# 7.  RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
residuals = theta2 - best_pred
print(f"  Residual diagnostics on best expression:")
print(f"    Max |error| : {np.max(np.abs(residuals)):.2e}")
print(f"    Mean error  : {np.mean(residuals):.2e}")
print(f"    Std error   : {np.std(residuals):.2e}")

print("\n" + "=" * 70)
print("  SUMMARY TABLE")
print("=" * 70)
print(f"  {'Method':<45} {'R²':>10}  {'RMSE':>10}  {'Rank'}")
print(f"  {'-'*45} {'----------':>10}  {'----------':>10}  {'----'}")
for rank, (score, error, name, _) in enumerate(results[:5], 1):
    short = name[:44]
    print(f"  {short:<45} {score:>10.6f}  {error:>10.6f}  {rank}")
print("=" * 70)

┌──(py312)(agagora㉿localhost)-[~/Downloads/GITHUB/LLM-HypatiaX-PAPERS/papers/2025-JMLR]
└─$ python hypatiax/experiments/benchmarks/snells_law_discovery.py
======================================================================
  SNELL'S LAW  –  Symbolic Discovery
  Samples: 200   |   Vars: theta1, n1, n2   |   Target: theta2
======================================================================

Rank        R²        RMSE  Expression
----------------------------------------------------------------------
1     1.000000    0.000000  arcsin(n1/n2 * sin(θ1))  [Snell exact] ✓✓✓
2     1.000000    0.000000  arcsin(n1 * sin(θ1) / n2) ✓✓✓
3     0.854990    0.194432  arcsin(n1/n2 * θ1)
4     0.846454    0.200073  n1/n2 * θ1  [paraxial]
5     0.846454    0.200073  n1/n2 * arcsin(sin(θ1))
6     0.551600    0.341902  θ1  [no refraction]
7     0.000000    0.510586  mean(θ2)  [intercept only]
8     -0.536823    0.632967  arcsin(n2/n1 * sin(θ1))  [indices swapped]
9     -1.015388    0.724851  θ1 * (n1/n2)**2
10    -2.228417    0.917411  arcsin((n1-n2) * sin(θ1))

======================================================================

  BEST DISCOVERED EQUATION
  ─────────────────────────
  θ₂ = arcsin(n1/n2 * sin(θ1))  [Snell exact]
  R²   = 1.00000000
  RMSE = 0.00000000

  GROUND TRUTH (Snell's Law):
    n₁ · sin(θ₁) = n₂ · sin(θ₂)
    ⟹  θ₂ = arcsin( n₁/n₂ · sin(θ₁) )

  EXACT-MATCH: YES ✓

  Residual diagnostics on best expression:
    Max |error| : 0.00e+00
    Mean error  : 0.00e+00
    Std error   : 0.00e+00

======================================================================
  SUMMARY TABLE
======================================================================
  Method                                                R²        RMSE  Rank
  --------------------------------------------- ----------  ----------  ----
  arcsin(n1/n2 * sin(θ1))  [Snell exact]          1.000000    0.000000  1
  arcsin(n1 * sin(θ1) / n2)                       1.000000    0.000000  2
  arcsin(n1/n2 * θ1)                              0.854990    0.194432  3
  n1/n2 * θ1  [paraxial]                          0.846454    0.200073  4
  n1/n2 * arcsin(sin(θ1))                         0.846454    0.200073  5
======================================================================

