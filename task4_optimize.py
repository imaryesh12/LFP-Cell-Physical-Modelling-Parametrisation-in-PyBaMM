"""
Task 4 - Multi-Stage Optimization Pipeline (Manager)
Executes Differential Evolution (Global) & Nelder-Mead (Local) using the isolated worker architecture.
1:1 logic match with Jupyter Notebook (seed=0, custom initialization).
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import pybamm
from scipy.optimize import differential_evolution, minimize
import task4_objective as T

pybamm.set_logging_level("ERROR")

def evaluate_worker(theta_array):
    """Passes parameters to an isolated worker process to protect the main loop from C++ IDAKLU crashes."""
    theta_dict = {name: float(val) for name, val in zip(T.PARAM_NAMES, theta_array)}
    try:
        res = subprocess.run(
            [sys.executable, "task4_worker.py", json.dumps(theta_dict)],
            capture_output=True, text=True, check=True
        )
        return json.loads(res.stdout)["J"]
    except Exception:
        return T.PENALTY_V

class Tracker:
    def __init__(self, tag):
        self.tag = tag
        self.n = 0
        self.best = np.inf
        self.best_theta = None
        self.t0 = time.time()

    def __call__(self, theta_full):
        J = evaluate_worker(theta_full)
        self.n += 1
        if J < self.best:
            self.best = J
            self.best_theta = np.array(theta_full, float)
            print(f"[{self.tag}] eval {self.n:4d} | J={J:.5f} V | ({time.time()-self.t0:.0f} s)")
        return J

def seeded_population(theta_full, idx, bounds, popsize, n_params, seed=0):
    rng = np.random.default_rng(seed)
    n = max(5, popsize * n_params)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pop = rng.uniform(lo, hi, size=(n, n_params))
    pop[0] = np.clip(np.asarray(theta_full, float)[idx], lo, hi)
    return pop

def partial_objective(tracker, theta_full, idx):
    def f(sub):
        th = np.array(theta_full, float)
        th[idx] = sub
        return tracker(th)
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_task4")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    print("Optimization Manager Initiated (Strict Notebook Match). This will take approx. 40 minutes...")

    # Base parameters matching default_theta() exactly from your notebook
    base_params = pybamm.ParameterValues("Prada2013")
    theta = np.array([
        1.0, 1.0,
        base_params["Initial concentration in negative electrode [mol.m-3]"] / base_params["Maximum concentration in negative electrode [mol.m-3]"],
        np.log10(base_params["Negative particle diffusivity [m2.s-1]"]),
        0.0, 0.0, 0.0
    ])

    # --- STAGE 1: Capacity & Stoichiometry ---
    print("\n=== Stage 1: capacity & stoichiometry (L_neg_scale, x_init_neg) ===")
    tr1 = Tracker("stage1")
    f1 = partial_objective(tr1, theta, T.IDX_STAGE1)
    b1 = [T.BOUNDS[i] for i in T.IDX_STAGE1]
    
    differential_evolution(
        f1, b1, maxiter=12, popsize=6, tol=1e-4, seed=0, polish=False,
        init=seeded_population(theta, T.IDX_STAGE1, b1, 6, len(T.IDX_STAGE1), 0)
    )
    theta = tr1.best_theta.copy()
    print(f"[stage1] best {tr1.best:.5f} V")

    # --- STAGE 2: Dynamics ---
    print("\n=== Stage 2: dynamics (D_neg, k_neg, R_contact) ===")
    tr2 = Tracker("stage2")
    f2 = partial_objective(tr2, theta, T.IDX_STAGE2)
    b2 = [T.BOUNDS[i] for i in T.IDX_STAGE2]
    
    J_in = evaluate_worker(theta)
    differential_evolution(
        f2, b2, maxiter=12, popsize=6, tol=1e-4, seed=0, polish=False,
        init=seeded_population(theta, T.IDX_STAGE2, b2, 6, len(T.IDX_STAGE2), 0)
    )
    if tr2.best <= J_in:
        theta = tr2.best_theta.copy()
    print(f"[stage2] best {tr2.best:.5f} V")

    # --- STAGE 3: Joint Polish ---
    print("\n=== Stage 3: joint Nelder-Mead polish ===")
    tr3 = Tracker("stage3")
    lo = np.array([T.BOUNDS[i][0] for i in T.IDX_FREE])
    hi = np.array([T.BOUNDS[i][1] for i in T.IDX_FREE])

    def f3(sub):
        th = np.array(theta, float)
        th[T.IDX_FREE] = np.clip(sub, lo, hi)
        return tr3(th)

    minimize(f3, theta[T.IDX_FREE], method="Nelder-Mead", options={"maxfev": 120, "xatol": 1e-3, "fatol": 1e-5})
    if tr3.best < np.inf:
        theta = tr3.best_theta.copy()
    print(f"[stage3] best {tr3.best:.5f} V")

    # =====================================================================
    # Save Final Checkpoint
    # =====================================================================
    with open(os.path.join(args.out, "task4_optimization_results.json"), "w") as f:
        json.dump({"final": {"theta": theta.tolist()}}, f, indent=4)
        
    print(f"\nOptimization complete. Final parameters saved to {args.out}/task4_optimization_results.json")

if __name__ == "__main__":
    main()
































