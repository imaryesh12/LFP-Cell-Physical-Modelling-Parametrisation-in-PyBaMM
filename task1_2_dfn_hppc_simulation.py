"""
Task 1 & 2 - DFN model of the A123 ANR26650M1B (2.5 Ah LFP) in PyBaMM.
Builds the baseline DFN model, simulates the HPPC current profile, and generates plots.
"""

import argparse
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scipy.io as sio

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

CONFIG = {
    "model": "DFN",
    "parameter_set": "Prada2013",
    "thermal": "isothermal",
    "lower_voltage_cutoff_V": 2.0,
    "upper_voltage_cutoff_V": 4.0,
}

def load_hppc(path):
    meas = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    t = np.asarray(meas.Time, dtype=float)
    I = np.asarray(meas.Current, dtype=float)
    V = np.asarray(meas.Voltage, dtype=float)
    T = np.asarray(meas.Battery_Temp_degC, dtype=float)

    keep = np.r_[True, np.diff(t) > 0]
    t, I, V, T = t[keep], I[keep], V[keep], T[keep]
    return t, I, V, T

def build_model_and_parameters(t_data, I_data, T_mean_degC):
    model = pybamm.lithium_ion.DFN(options={"thermal": CONFIG["thermal"]})
    params = pybamm.ParameterValues(CONFIG["parameter_set"])

    I_pybamm = -I_data
    current_interp = pybamm.Interpolant(t_data, I_pybamm, pybamm.t, name="HPPC current", interpolator="linear")
    T_K = T_mean_degC + 273.15
    params.update({
        "Current function [A]": current_interp,
        "Ambient temperature [K]": T_K,
        "Initial temperature [K]": T_K,
        "Lower voltage cut-off [V]": CONFIG["lower_voltage_cutoff_V"],
        "Upper voltage cut-off [V]": CONFIG["upper_voltage_cutoff_V"],
    })
    return model, params

def simulate(model, params, t_data, V0_rest):
    output_variables = [
        "Voltage [V]", "Current [A]", "Discharge capacity [A.h]",
        "X-averaged negative particle surface stoichiometry",
        "X-averaged positive particle surface stoichiometry",
    ]
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, output_variables=output_variables)
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    
    initial_soc = f"{V0_rest:.4f} V"
    print(f"\n[sim] initial state set from measured rest voltage: {initial_soc}")
    
    t0 = time.time()
    sol = sim.solve(t_eval=t_data, initial_soc=initial_soc)
    print(f"[sim] solved in {time.time()-t0:.1f} s; termination: {sol.termination}")
    
    return sol, sol["Time [s]"].entries, sol["Voltage [V]"].entries, sol["Current [A]"].entries

def error_metrics(t, V_meas, V_sim, I_meas):
    res = V_sim - V_meas
    active = np.abs(I_meas) > 0.1
    def rmse(x): return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")
    def mae(x): return float(np.mean(np.abs(x))) if x.size else float("nan")
    
    return {
        "n_samples": int(len(res)),
        "RMSE_V": rmse(res),
        "MAE_V": mae(res),
        "MaxAbsError_V": float(np.max(np.abs(res))),
        "MeanError_V (bias, sim - meas)": float(np.mean(res)),
        "RMSE_pulse_V": rmse(res[active]),
        "RMSE_rest_V": rmse(res[~active]),
        "TimeWeighted_RMSE_V": float(np.sqrt(np.trapezoid(res**2, t) / (t[-1] - t[0]))),
    }

def make_plots(t, I_meas, V_meas, V_sim, out_dir, zoom_window=(0, 4000)):
    res = V_sim - V_meas
    th = t / 3600.0

    # Plot 1: measured vs simulated voltage, full test + current
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(th, V_meas, "k", lw=0.8, label="Measured (HPPC)")
    ax[0].plot(th, V_sim, "r", lw=0.8, alpha=0.8, label=f"Simulated DFN ({CONFIG['parameter_set']})")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].legend(loc="lower left")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Task 2: measured vs simulated voltage - HPPC profile")
    ax[1].plot(th, I_meas, "b", lw=0.6)
    ax[1].set_ylabel("Current [A]\n(neg = discharge)")
    ax[1].set_xlabel("Time [h]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task2_voltage_measured_vs_simulated.png"), dpi=150)
    plt.close(fig)

    # Plot 2: residual over the full test
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(th, 1e3 * res, "k", lw=0.6)
    ax.axhline(0, color="r", lw=0.8)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual (sim - meas) [mV]")
    ax.grid(alpha=0.3)
    ax.set_title("Task 2: voltage estimation error - HPPC profile")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task2_voltage_residual.png"), dpi=150)
    plt.close(fig)

    # Plot 3: zoom into the first pulse block
    m = (t >= zoom_window[0]) & (t <= zoom_window[1])
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(t[m], V_meas[m], "k", lw=1, label="Measured")
    ax[0].plot(t[m], V_sim[m], "r", lw=1, alpha=0.8, label="Simulated")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Task 2 (zoom {zoom_window[0]}-{zoom_window[1]} s): first pulse block")
    ax[1].plot(t[m], 1e3 * res[m], "k", lw=0.8)
    ax[1].axhline(0, color="r", lw=0.8)
    ax[1].set_ylabel("Residual [mV]")
    ax[1].grid(alpha=0.3)
    ax[2].plot(t[m], I_meas[m], "b", lw=0.8)
    ax[2].set_ylabel("Current [A]")
    ax[2].set_xlabel("Time [s]")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task2_zoom_first_pulse_block.png"), dpi=150)
    plt.close(fig)
    
    # Plot 4: residual vs discharged capacity
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(th, 1e3 * res, s=1, c=np.abs(I_meas), cmap="viridis")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual [mV]")
    ax.grid(alpha=0.3)
    ax.set_title("Task 2: residual coloured by |current| (dark = rest, bright = pulse)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task2_residual_vs_current.png"), dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/HPPC.mat")
    ap.add_argument("--out", default="results_task2")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    pybamm.set_logging_level("WARNING")

    t, I_meas, V_meas, T = load_hppc(args.data)
    model, params = build_model_and_parameters(t, I_meas, T.mean())
    sol, t_sim, V_sim, I_sim = simulate(model, params, t, V_meas[0])

    V_sim_on_meas = np.interp(t, t_sim, V_sim, left=np.nan, right=np.nan)
    ok = ~np.isnan(V_sim_on_meas)
    
    metrics = error_metrics(t[ok], V_meas[ok], V_sim_on_meas[ok], I_meas[ok])
    
    np.savetxt(
        os.path.join(args.out, "task2_hppc_measured_vs_simulated.csv"),
        np.c_[t, I_meas, V_meas, V_sim_on_meas, V_sim_on_meas - V_meas],
        delimiter=",", header="time_s,current_meas_A,voltage_meas_V,voltage_sim_V,residual_V", comments=""
    )
    
    with open(os.path.join(args.out, "task2_metrics.json"), "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
        
    make_plots(t[ok], I_meas[ok], V_meas[ok], V_sim_on_meas[ok], args.out)
    
    print(f"\n[done] outputs written to {args.out}/")

if __name__ == "__main__":
    main()