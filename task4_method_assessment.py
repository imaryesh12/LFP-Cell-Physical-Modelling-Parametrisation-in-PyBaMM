"""
Task 4a - Method Assessment: 1-D Parameter Slices to verify landscape roughness.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pybamm
import task4_objective as T

pybamm.set_logging_level("ERROR")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--out", default="results_task4")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    ds = T.load_training_datasets(args.data)
    base_params = pybamm.ParameterValues("Prada2013")
    
    baseline_theta = {
        "L_neg_scale": 1.0, "L_pos_scale": 1.0,
        "x_init_neg": base_params["Initial concentration in negative electrode [mol.m-3]"] / base_params["Maximum concentration in negative electrode [mol.m-3]"],
        "log10_D_neg": np.log10(base_params["Negative particle diffusivity [m2.s-1]"]),
        "log10_k_neg": 0.0, "log10_k_pos": 0.0, "R_contact": 0.0
    }

    J_base = T.objective(baseline_theta, ds)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()

    for i, (name, _, bounds) in enumerate(T.PARAM_SPEC):
        vals = np.linspace(bounds[0], bounds[1], 6)
        objs = []
        for v in vals:
            th = baseline_theta.copy()
            th[name] = v
            objs.append(T.objective(th, ds))

        ax = axes_flat[i]
        ax.plot(vals, objs, "o-", lw=1.5, color="#1f77b4")
        ax.axhline(J_base, color="r", ls="--", lw=1.0)
        ax.set_title(f"{name} (range {(max(objs) - min(objs))*1000:.0f} mV)", fontsize=10)
        ax.set_ylabel("Objective [V]", fontsize=9)
        ax.grid(alpha=0.3)

    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task4_1D_slices.png"), dpi=150)
    print(f"1-D Slices saved to {args.out}/task4_1D_slices.png")

if __name__ == "__main__":
    main()