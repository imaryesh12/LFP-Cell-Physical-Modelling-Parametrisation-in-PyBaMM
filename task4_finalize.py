"""
Task 4 Finalizer: Loads the optimized checkpoint and renders reports/plots.
"""
import os
import json
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

    ds = T.load_training_datasets(args.data)
    base_params = pybamm.ParameterValues("Prada2013")
    
    # Load converged parameters
    with open(os.path.join(args.out, "task4_optimization_results.json"), "r") as f:
        theta_array = json.load(f)["final"]["theta"]
        
    opt_dict = {name: float(val) for name, val in zip(T.PARAM_NAMES, theta_array)}
    base_dict = {
        "L_neg_scale": 1.0, "L_pos_scale": 1.0,
        "x_init_neg": base_params["Initial concentration in negative electrode [mol.m-3]"] / base_params["Maximum concentration in negative electrode [mol.m-3]"],
        "log10_D_neg": np.log10(base_params["Negative particle diffusivity [m2.s-1]"]),
        "log10_k_neg": 0.0, "log10_k_pos": 0.0, "R_contact": 0.0
    }

    _, det0 = T.objective(base_dict, ds, return_detail=True)
    _, detf = T.objective(opt_dict, ds, return_detail=True)

    print("\\nTASK 4 FINAL NUMERIC RESULTS")
    print(f"{'Record':15s} | {'Base RMSE':>10s} | {'Opt RMSE':>10s}")
    for k in det0["per_dataset_RMSE_V"]:
        print(f"{k:15s} | {1e3*det0['per_dataset_RMSE_V'][k]:7.1f} mV | {1e3*detf['per_dataset_RMSE_V'][k]:7.1f} mV")

    # Fit plots (Figure 8)
    p_opt = T.apply_theta_dict(opt_dict, base_params)
    p_base = T.apply_theta_dict(base_dict, base_params)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex="col")

    for j, d in enumerate(ds):
        ax, axr = axes[0, j], axes[1, j]
        ax.plot(d["t"] / 60, d["V"], "k", lw=1.6, label="Measured")
        
        for p, lab, sty in ((p_base, "Prada2013", "r--"), (p_opt, "optimised", "g-")):
            q = p.copy()
            q.update({"Current function [A]": d["I_app"], "Lower voltage cut-off [V]": 2.0 if d["charge"] else T.V_LOW, "Upper voltage cut-off [V]": T.V_HIGH if d["charge"] else 3.7})
            opts = {"thermal": "isothermal"}
            if q.get("Contact resistance [Ohm]", 0) > 0: opts["contact resistance"] = "true"
            
            try:
                sim = pybamm.Simulation(pybamm.lithium_ion.DFN(options=opts), parameter_values=q, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6))
                sol = sim.solve([0, d["duration_s"] * 1.2], initial_soc=f"{d['V0']:.4f} V")
                ts, Vs = sol["Time [s]"].entries, sol["Voltage [V]"].entries
                ax.plot(ts / 60, Vs, sty, lw=1.1, label=lab)
                g_idx = d["t_grid"] <= ts[-1]
                axr.plot(d["t_grid"][g_idx] / 60, 1e3 * (np.interp(d["t_grid"][g_idx], ts, Vs) - d["V_grid"][g_idx]), sty[0], lw=.9)
            except Exception: pass

        ax.set_title(f"{d['name']}", fontsize=10)
        axr.axhline(0, color="k", lw=.6)
        if j == 0: ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task4_training_fits.png"), dpi=150)

if __name__ == "__main__":
    main()