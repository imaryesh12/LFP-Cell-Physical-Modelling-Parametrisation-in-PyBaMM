"""
Task 5 - Validation against the held-out HPPC profile.
Calculates Common-Window RMSE metrics, prints comparison tables, and generates both
voltage validation (5a) and lithium plating risk (5b) graphs.
"""
import os
import json
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
import pybamm
import task4_objective as T

if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz
pybamm.set_logging_level("ERROR")

def calc_metrics(t_arr, V_m, V_s, I_m):
    res = V_s - V_m
    active = np.abs(I_m) > 0.1
    rmse = lambda x: float(np.sqrt(np.mean(x**2))) * 1000 if x.size else float("nan")
    return {
        "RMSE": rmse(res),
        "MAE": float(np.mean(np.abs(res))) * 1000,
        "MaxAbsError": float(np.max(np.abs(res))) * 1000,
        "Bias": float(np.mean(res)) * 1000,
        "RMSE_pulse": rmse(res[active]),
        "RMSE_rest": rmse(res[~active]),
        "TimeWeighted_RMSE": float(np.sqrt(np.trapezoid(res**2, t_arr) / (t_arr[-1] - t_arr[0]))) * 1000
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--task4", default="results_task4/task4_optimization_results.json")
    ap.add_argument("--out", default="results_task5")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading HPPC validation data...")
    base_params = pybamm.ParameterValues("Prada2013")
    
    with open(args.task4, "r") as f:
        theta_array = json.load(f)["final"]["theta"]
        
    opt_dict = {name: float(val) for name, val in zip(T.PARAM_NAMES, theta_array)}
    p_opt = T.apply_theta_dict(opt_dict, base_params)

    m = sio.loadmat(os.path.join(args.data, "HPPC.mat"), squeeze_me=True, struct_as_record=False)["meas"]
    t_meas, I_meas, V_meas, Temp_meas = (np.asarray(getattr(m, k), float) for k in ("Time", "Current", "Voltage", "Battery_Temp_degC"))
    keep = np.r_[True, np.diff(t_meas) > 0]
    t_meas, I_meas, V_meas, Temp_meas = t_meas[keep], I_meas[keep], V_meas[keep], Temp_meas[keep]

    T_K = float(Temp_meas.mean()) + 273.15
    p_opt.update({
        "Current function [A]": pybamm.Interpolant(t_meas, -I_meas, pybamm.t, interpolator="linear"),
        "Ambient temperature [K]": T_K, "Initial temperature [K]": T_K,
        "Lower voltage cut-off [V]": 2.0, "Upper voltage cut-off [V]": 4.0
    })

    PHYS_VARS = [
        "Voltage [V]", "Current [A]", "Discharge capacity [A.h]",
        "X-averaged negative electrode surface potential difference [V]",
        "Negative electrode surface potential difference at separator interface [V]"
    ]

    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal", "contact resistance": "true"})
    sim = pybamm.Simulation(model, parameter_values=p_opt, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, output_variables=PHYS_VARS))
    
    print("Simulating HPPC profile with new optimized parameters (takes ~1-2 mins)...")
    t0 = time.time()
    sol = sim.solve(t_eval=t_meas, initial_soc=f"{V_meas[0]:.4f} V")
    elapsed = time.time() - t0
    print(f"Simulation completed in {int(elapsed)} seconds.\n")

    t_sim = sol["Time [s]"].entries
    V_sim_raw = sol["Voltage [V]"].entries
    V_sim = np.interp(t_meas, t_sim, V_sim_raw, left=np.nan, right=np.nan)
    
    common_mask = t_meas <= 58572
    metrics = calc_metrics(t_meas[common_mask], V_meas[common_mask], V_sim[common_mask], I_meas[common_mask])
    
    # ------------------------------------------------------------------------
    # Exact Text Outputs to match Jupyter Notebook
    # ------------------------------------------------------------------------
    print("================================================================================")
    print("TABLE 1: PARAMETER COMPARISON (LITERATURE VS OPTIMISED)")
    print("================================================================================")
    print(f"{'Parameter':<35} | {'Base (Prada2013)':<20} | {'Optimised (New)':<20}")
    print("-" * 80)
    
    L_neg_base = base_params["Negative electrode thickness [m]"] * 1e6
    L_neg_opt = L_neg_base * opt_dict["L_neg_scale"]
    print(f"{'Negative electrode thickness':<35} | {L_neg_base:<4.1f} \u03bcm{' ':>11} | {L_neg_opt:.2f} \u03bcm (x{opt_dict['L_neg_scale']:.4f})")
    
    c_max_n = base_params["Maximum concentration in negative electrode [mol.m-3]"]
    c0_base = base_params["Initial concentration in negative electrode [mol.m-3]"]
    c0_opt = c_max_n * opt_dict["x_init_neg"]
    print(f"{'Initial conc. negative electrode':<35} | {c0_base:<7.1f} mol/m\u00b3{' ':>5} | {c0_opt:.1f} mol/m\u00b3")
    
    D_base = base_params["Negative particle diffusivity [m2.s-1]"]
    D_opt = 10**opt_dict["log10_D_neg"]
    print(f"{'Negative particle diffusivity':<35} | {D_base:.2e} m\u00b2/s{' ':>5} | {D_opt:.2e} m\u00b2/s")
    
    k_neg_opt = 10**opt_dict["log10_k_neg"]
    print(f"{'Neg. exchange-current density scale':<35} | 1.0{' ':>17} | {k_neg_opt:.3f}x")
    
    R_opt = opt_dict["R_contact"] * 1000
    print(f"{'Contact resistance':<35} | 0 m\u03a9{' ':>16} | {R_opt:.3f} m\u03a9")
    print(f"{'Positive thickness, positive k':<35} | 1.0 (held){' ':>10} | unchanged")
    print("\n================================================================================")
    print("TABLE 2: HPPC METRICS COMPARISON (COMMON WINDOW: 0 - 16.27h)")
    print("================================================================================")
    print(f"{'Metric (mV)':<20} | {'Baseline (Task 2)':<17} | {'Manual (Task 3)':<15} | {'Optimised (Your Set)':<20}")
    print("-" * 80)
    print(f"{'RMSE':<20} | 42.7{' ':>13} | 36.0{' ':>11} | {metrics['RMSE']:.1f}")
    print(f"{'MAE':<20} | 31.6{' ':>13} | 28.4{' ':>11} | {metrics['MAE']:.1f}")
    print(f"{'MaxAbsError':<20} | 685.6{' ':>12} | 208.8{' ':>10} | {metrics['MaxAbsError']:.1f}")
    print(f"{'Bias':<20} | -21.0{' ':>12} | -8.9{' ':>11} | {metrics['Bias']:.1f}")
    print("...")
    
    # ------------------------------------------------------------------------
    # FIXED: Dedicated masks for measured array (55k) and simulated array (379k)
    # ------------------------------------------------------------------------
    q_meas = -np.trapezoid(I_meas[common_mask], t_meas[common_mask]) / 3600.0
    sim_common_mask = t_sim <= 58572
    q_sim = sol["Discharge capacity [A.h]"].entries[sim_common_mask][-1]
    print(f"[Capacity Check] Measured = {q_meas:.4f} Ah, Model = {q_sim:.4f} Ah")
    
    phi_s_sep = sol["Negative electrode surface potential difference at separator interface [V]"].entries
    plating_mask = phi_s_sep < 0
    t_diff = np.diff(t_sim)
    plating_time = np.sum(t_diff[plating_mask[:-1]]) 
    
    print(f"[Plating Risk] Anode potential dropped below 0V for {plating_time:.1f} seconds total.")
    print("================================================================================\n")

    # ------------------------------------------------------------------------
    # Plotting Task 5a: HPPC Voltage Validation
    # ------------------------------------------------------------------------
    th_meas = t_meas / 3600.0
    fig1, ax1 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax1[0].plot(th_meas, V_meas, "k", lw=1.2, label="Measured (HPPC)")
    ax1[0].plot(th_meas, V_sim, "g", lw=1.0, alpha=0.9, label="Simulated (Optimised)")
    ax1[0].set_ylabel("Voltage [V]")
    ax1[0].legend(loc="lower left")
    ax1[0].grid(alpha=0.3)
    ax1[1].plot(th_meas, (V_sim - V_meas) * 1000, "k", lw=0.6)
    ax1[1].axhline(0, color="r", lw=0.8)
    ax1[1].set_ylabel("Error [mV]")
    ax1[1].grid(alpha=0.3)
    ax1[2].plot(th_meas, I_meas, "b", lw=0.6)
    ax1[2].set_ylabel("Current [A]")
    ax1[2].set_xlabel("Time [h]")
    ax1[2].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.out, "task5a_hppc_validation.png"), dpi=150)
    plt.close(fig1)
    print(f"Voltage validation graph saved to {args.out}/task5a_hppc_validation.png")

    # ------------------------------------------------------------------------
    # Plotting Task 5b: Anode Potential vs Li+ over HPPC (Safety Check)
    # ------------------------------------------------------------------------
    phi_s_avg = sol["X-averaged negative electrode surface potential difference [V]"].entries
    
    th_sim = t_sim / 3600.0
    fig2, ax2 = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
    ax2[0].plot(th_sim, phi_s_avg, color="blue", lw=0.8, label="x-averaged")
    ax2[0].plot(th_sim, phi_s_sep, color="crimson", lw=0.8, label="at separator interface")
    ax2[0].axhline(0, color="black", ls="--", label="0 V vs Li/Li+ (plating threshold)")
    ax2[0].set_ylabel("Anode potential vs Li/Li+\n[V]")
    ax2[0].set_title("Task 5b: Anode Potential vs Li$^+$ over HPPC (Safety Check)")
    ax2[0].legend(loc="upper left")
    ax2[0].grid(alpha=0.3)

    I_sim = sol["Current [A]"].entries
    ax2[1].plot(th_sim, I_sim, color="blue", lw=0.8)
    ax2[1].set_ylabel("Current [A]")
    ax2[1].set_xlabel("Time [h]")
    ax2[1].grid(alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "task5b_anode_potential.png"), dpi=150)
    plt.close(fig2)
    print(f"Safety check graph saved to {args.out}/task5b_anode_potential.png")

if __name__ == "__main__":
    main()