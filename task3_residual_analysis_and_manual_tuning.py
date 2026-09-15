"""
Task 3 - Residual Analysis, Sensitivity Study, and Manual Tuning.
"""

import os
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
import pybamm
from scipy.optimize import brentq

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

pybamm.set_logging_level("WARNING")

# ============================================================================
# 1. Data Loading Functions
# ============================================================================
def load_mat_simple(path):
    m = sio.loadmat(path, squeeze_me=True)
    t, I, V = (np.asarray(m[k], dtype=float) for k in ("Time", "Current", "Voltage"))
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], I[keep], V[keep]

def load_mat_struct(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    t, I, V, Ah = (np.asarray(getattr(m, k), dtype=float) for k in ("Time", "Current", "Voltage", "Ah"))
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], I[keep], V[keep], Ah[keep]

def run_cc(params, I_app, V0_rest, t_end, v_low=2.5, v_high=3.7):
    p = params.copy()
    p.update({
        "Current function [A]": I_app,
        "Lower voltage cut-off [V]": v_low,
        "Upper voltage cut-off [V]": v_high
    })
    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
    sol = sim.solve([0, t_end], initial_soc=f"{V0_rest:.4f} V")
    return sol["Time [s]"].entries, sol["Voltage [V]"].entries, float(sol["Discharge capacity [A.h]"].entries[-1])

def calc_rmse(t_sim, V_sim, t_meas, V_meas):
    n = int(min(np.searchsorted(t_meas, t_sim[-1]), len(t_meas)))
    if n < 2: return float("nan")
    r = np.interp(t_meas[:n], t_sim, V_sim) - V_meas[:n]
    return 1e3 * float(np.sqrt(np.mean(r**2)))

def scaled_function(f, factor):
    def g(*args):
        return factor * f(*args)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--task2", default="results_task2/task2_hppc_measured_vs_simulated.csv")
    ap.add_argument("--out", default="results_task3")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    BASE_SET = "Prada2013"
    base_params = pybamm.ParameterValues(BASE_SET)

    print("--- Loading 1C Discharge Data ---")
    td, Id, Vd, Ahd = load_mat_struct(os.path.join(args.data, "Discharge_1C.mat"))
    on_idx = np.argmax(np.abs(Id) > 0.05)
    V0d_rest = Vd[on_idx - 1]
    td_cc, Id_cc, Vd_cc = td[on_idx:] - td[on_idx], Id[on_idx:], Vd[on_idx:]
    stop_idx = np.argmax(np.abs(Id_cc) < 0.05) if (np.abs(Id_cc) < 0.05).any() else len(Id_cc)
    td_cc, Id_cc, Vd_cc = td_cc[:stop_idx], Id_cc[:stop_idx], Vd_cc[:stop_idx]
    Q_MEAS_1C = float(-Ahd.min()) 

    # ========================================================================
    # 3a. OCV Residuals
    # ========================================================================
    print("--- 3a: Analyzing Task 2 Residuals ---")
    d = np.genfromtxt(args.task2, delimiter=",", names=True)
    t_csv, I_csv, Vm_csv, Vs_csv = d["time_s"], d["current_meas_A"], d["voltage_meas_V"], d["voltage_sim_V"]
    
    ok = ~np.isnan(Vs_csv)
    t_csv, I_csv, Vm_csv, Vs_csv = t_csv[ok], I_csv[ok], Vm_csv[ok], Vs_csv[ok]
    res = Vs_csv - Vm_csv
    Q_csv = np.concatenate([[0.0], np.cumsum(-0.5 * (I_csv[1:] + I_csv[:-1]) * np.diff(t_csv))]) / 3600.0

    ocv_pts = []
    i = 0
    while i < len(t_csv):
        if abs(I_csv[i]) < 0.05:
            j = i
            while j + 1 < len(t_csv) and abs(I_csv[j + 1]) < 0.05:
                j += 1
            if t_csv[j] - t_csv[i] >= 1500:
                ocv_pts.append((Q_csv[j], Vm_csv[j], Vs_csv[j]))
            i = j + 1
        else:
            i += 1
    ocv_pts = np.array(ocv_pts)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(ocv_pts[:, 0], ocv_pts[:, 1], "ko-", label="Measured (end of >=25 min rest)")
    ax[0].plot(ocv_pts[:, 0], ocv_pts[:, 2], "rs--", label=f"Model ({BASE_SET})")
    ax[0].set_ylabel("Rest voltage (pseudo-OCV) [V]")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[0].set_title("3a: pseudo-OCV from HPPC rests vs model")
    ax[1].plot(ocv_pts[:, 0], 1e3 * (ocv_pts[:, 2] - ocv_pts[:, 1]), "k.-")
    ax[1].axhline(0, color="r", lw=.8)
    ax[1].set_xlabel("Discharged capacity [Ah]")
    ax[1].set_ylabel("OCV error (model-meas) [mV]")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task3a_ocv_error.png"), dpi=150)
    plt.close(fig)

    # ========================================================================
    # 3a Part 2: Pulse Resistance
    # ========================================================================
    print("--- 3a: Extracting Pulse Resistance ---")
    dI = np.diff(I_csv)
    pulse_starts = np.where(np.abs(dI) > 1.0)[0]
    discharge_pulses, charge_pulses = [], []

    for idx in pulse_starts:
        I_before, I_during = I_csv[idx], I_csv[idx + 1]
        if abs(I_before) < 0.1 and abs(I_during) > 1.0:
            end_idx = idx + 1
            while end_idx < len(I_csv) and abs(I_csv[end_idx] - I_during) < 0.5:
                end_idx += 1
            pulse_end = end_idx - 1
            delta_I = I_during - I_before
            R_meas_mOhm = abs((Vm_csv[pulse_end] - Vm_csv[idx]) / delta_I) * 1000
            R_sim_mOhm = abs((Vs_csv[pulse_end] - Vs_csv[idx]) / delta_I) * 1000
            C_rate = round(abs(I_during) / 2.5)
            if I_during > 0: discharge_pulses.append((Q_csv[idx], C_rate, R_meas_mOhm, R_sim_mOhm))
            else: charge_pulses.append((Q_csv[idx], C_rate, R_meas_mOhm, R_sim_mOhm))

    dis_data, chg_data = np.array(discharge_pulses), np.array(charge_pulses)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_rates = [1, 2, 4, 5, 6, 7, 10, 15]
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_rates)))

    for ax, data, title in zip(axes, [dis_data, chg_data], ["discharge pulses", "charge pulses"]):
        if len(data) == 0: continue
        for c_rate, color in zip(plot_rates, colors):
            mask = data[:, 1] == c_rate
            if not mask.any(): continue
            subset = data[mask]
            ax.plot(subset[:, 0], subset[:, 2], "o-", color=color, label=f"meas {c_rate}C")
            ax.plot(subset[:, 0], subset[:, 3], "s--", color=color, label=f"model {c_rate}C")
        ax.set_title(f"3a: end-of-pulse resistance, {title}")
        ax.set_xlabel("Discharged capacity [Ah]")
        if ax == axes[0]: ax.set_ylabel("dV/I at pulse end [mOhm]")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task3a_pulse_resistance.png"), dpi=150)
    plt.close(fig)

    # ========================================================================
    # 3b Sensitivity
    # ========================================================================
    print("--- 3b: Running Sensitivity Analysis ---")
    tc, Ic, Vc = load_mat_simple(os.path.join(args.data, "Charge_4C.mat"))
    on_idx_c = np.argmax(np.abs(Ic) > 0.05)
    V0c_rest = Vc[on_idx_c - 1]
    tc_cc, Ic_cc, Vc_cc = tc[on_idx_c:] - tc[on_idx_c], Ic[on_idx_c:], Vc[on_idx_c:]
    cc_end = int(np.argmax(Vc_cc >= 3.6 - 1e-3))
    tc_cc, Ic_cc, Vc_cc = tc_cc[:cc_end], Ic_cc[:cc_end], Vc_cc[:cc_end]

    t_d0, V_d0, _ = run_cc(base_params, 2.5, V0d_rest, 5000)
    t_c0, V_c0, _ = run_cc(base_params, -10.0, V0c_rest, 2000, v_low=2.0, v_high=3.6)
    rm_d0 = calc_rmse(t_d0, V_d0, td_cc, Vd_cc)
    rm_c0 = calc_rmse(t_c0, V_c0, tc_cc, Vc_cc)

    parameters_to_test = [
        "Negative electrode thickness [m]", "Positive electrode thickness [m]",
        "Negative electrode active material volume fraction", "Positive electrode active material volume fraction",
        "Electrode height [m]", "Negative particle diffusivity [m2.s-1]",
        "Positive particle diffusivity [m2.s-1]", "Negative particle radius [m]",
        "Positive particle radius [m]", "Negative electrode porosity",
        "Positive electrode porosity", "Separator porosity",
        "Negative electrode Bruggeman coefficient (electrolyte)", "Positive electrode Bruggeman coefficient (electrolyte)",
        "Initial concentration in electrolyte [mol.m-3]", "Contact resistance [Ohm]",
        "Negative electrode conductivity [S.m-1]", "Positive electrode conductivity [S.m-1]",
        "Maximum concentration in negative electrode [mol.m-3]", "Maximum concentration in positive electrode [mol.m-3]",
        "Negative electrode exchange-current density [A.m-2]", "Positive electrode exchange-current density [A.m-2]",
        "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]"
    ]

    rows = []
    factors = (0.8, 1.25)

    for name in parameters_to_test:
        if name not in base_params.keys(): continue
        for fac in factors:
            p = base_params.copy()
            if callable(base_params[name]):
                p.update({name: scaled_function(base_params[name], fac)})
            else:
                p.update({name: base_params[name] * fac})
            if name == "Contact resistance [Ohm]":
                p.update({name: 0.005 if fac > 1 else 0.002})
            try:
                t1, V1, _ = run_cc(p, 2.5, V0d_rest, 5000)
                rm_d = calc_rmse(t1, V1, td_cc, Vd_cc)
                t2, V2, _ = run_cc(p, -10.0, V0c_rest, 2000, v_low=2.0, v_high=3.6)
                rm_c = calc_rmse(t2, V2, tc_cc, Vc_cc)
                rows.append({"parameter": name, "factor": fac, "dRMSE_1C_dis_mV": rm_d - rm_d0, "dRMSE_4C_chg_mV": rm_c - rm_c0})
            except Exception:
                pass

    rank = {}
    for r in rows:
        eff = np.nanmax([abs(r["dRMSE_1C_dis_mV"]), abs(r["dRMSE_4C_chg_mV"])])
        rank[r["parameter"]] = max(rank.get(r["parameter"], 0), eff)
    order = sorted(rank, key=rank.get, reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(order))
    for fac, col in zip(factors, ("tab:blue", "tab:red")):
        v = [next((r["dRMSE_1C_dis_mV"] for r in rows if r["parameter"] == n and r["factor"] == fac), np.nan) for n in order]
        ax.barh(y + (0.2 if fac > 1 else -0.2), v, height=0.4, color=col, label=f"x{fac} (1C dis)")

    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(" [", "\n[") for n in order], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("change in 1C-discharge RMSE [mV]")
    ax.grid(alpha=.3, axis="x")
    ax.legend()
    ax.set_title("3b: one-at-a-time sensitivity (Prada2013 baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task3b_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ========================================================================
    # 3c Manual Tuning
    # ========================================================================
    print("--- 3c: Running Manual Tuning ---")
    Ln = base_params["Negative electrode thickness [m]"]
    Lp = base_params["Positive electrode thickness [m]"]

    def q_lowrate(k):
        p = base_params.copy()
        p.update({"Negative electrode thickness [m]": Ln * k, "Positive electrode thickness [m]": Lp * k})
        _, _, Q = run_cc(p, 0.1, V0d_rest, 3600 * 30)
        return Q

    k_thk = brentq(lambda k: q_lowrate(k) - Q_MEAS_1C, 1.0, 1.4, xtol=1e-3)
    p_step1 = base_params.copy()
    p_step1.update({
        "Negative electrode thickness [m]": Ln * k_thk, 
        "Positive electrode thickness [m]": Lp * k_thk,
        "Nominal cell capacity [A.h]": 2.5
    })

    D0 = p_step1["Negative particle diffusivity [m2.s-1]"]
    def q_1c(logf):
        p = p_step1.copy()
        p.update({"Negative particle diffusivity [m2.s-1]": D0 * 10**logf})
        _, _, Q = run_cc(p, 2.5, V0d_rest, 5000)
        return Q

    target = min(Q_MEAS_1C, q_1c(2) * 0.999)
    logf = brentq(lambda g: q_1c(g) - target, 0, 2, xtol=0.02)
    D_multiplier = 10**logf
    tuned_params = p_step1.copy()
    tuned_params.update({"Negative particle diffusivity [m2.s-1]": D0 * D_multiplier})

    t0_sim, V0_sim, _ = run_cc(base_params, 2.5, V0d_rest, 5000)
    t1_sim, V1_sim, _ = run_cc(p_step1, 2.5, V0d_rest, 5000)
    t2_sim, V2_sim, _ = run_cc(tuned_params, 2.5, V0d_rest, 5000)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(td_cc / 3600 * 2.5, Vd_cc, "k", lw=2, label="Measured 1C discharge")
    ax.plot(t0_sim / 3600 * 2.5, V0_sim, "r--", label=f"{BASE_SET} as-is")
    ax.plot(t1_sim / 3600 * 2.5, V1_sim, "b--", label=f"+ thickness x{k_thk:.3f}")
    ax.plot(t2_sim / 3600 * 2.5, V2_sim, "g", label=f"+ D_neg x{D_multiplier:.1f}")
    ax.set_xlabel("Discharged capacity [Ah]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_title("3c: manual tuning steps on the 1C discharge")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task3c_manual_tuning.png"), dpi=150)
    plt.close(fig)

    # ========================================================================
    # 4. Final Validation HPPC
    # ========================================================================
    print("--- 4: Final Validation HPPC Re-Evaluation ---")
    meas = sio.loadmat(os.path.join(args.data, "HPPC.mat"), squeeze_me=True, struct_as_record=False)["meas"]
    t_h, I_h, V_h, T_h = (np.asarray(getattr(meas, k), float) for k in ("Time", "Current", "Voltage", "Battery_Temp_degC"))
    keep = np.r_[True, np.diff(t_h) > 0]
    t_h, I_h, V_h, T_h = t_h[keep], I_h[keep], V_h[keep], T_h[keep]

    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    I_interp = pybamm.Interpolant(t_h, -I_h, pybamm.t, interpolator="linear")
    T_K = T_h.mean() + 273.15
    tuned_params.update({
        "Current function [A]": I_interp,
        "Ambient temperature [K]": T_K, "Initial temperature [K]": T_K,
        "Lower voltage cut-off [V]": 2.0, "Upper voltage cut-off [V]": 4.0,
    })

    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    sim = pybamm.Simulation(model, parameter_values=tuned_params, solver=solver)
    sol = sim.solve(t_eval=t_h, initial_soc=f"{V_h[0]:.4f} V")
    V_sim_aligned = np.interp(t_h, sol["Time [s]"].entries, sol["Voltage [V]"].entries, left=np.nan, right=np.nan)

    th_h = t_h / 3600.0
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(th_h, V_h, "k", lw=0.8, label="Measured (HPPC)")
    ax[0].plot(th_h, V_sim_aligned, "r", lw=0.8, alpha=0.8, label="Simulated DFN (Tuned)")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].legend(loc="lower left")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Task 3 (manually tuned): measured vs simulated voltage - HPPC profile")
    ax[1].plot(th_h, I_h, "b", lw=0.6)
    ax[1].set_ylabel("Current [A]\n(neg = discharge)")
    ax[1].set_xlabel("Time [h]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "task3_hppc_validation.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()