"""
Shared objective function and constraints for Task 4 Optimization.
"""
import os
import numpy as np
import scipy.io as sio
import pybamm

V_LOW, V_HIGH, GRID_DT, PENALTY_V = 2.5, 3.6, 5.0, 0.5

PARAM_SPEC = [
    ("L_neg_scale",    "linear", (0.80, 1.80)),   
    ("L_pos_scale",    "linear", (0.80, 1.80)),   
    ("x_init_neg",     "linear", (0.60, 0.92)),   
    ("log10_D_neg",    "log10",  (-15.0, -12.0)), 
    ("log10_k_neg",    "log10",  (-1.0, 2.0)),    
    ("log10_k_pos",    "log10",  (-1.0, 2.0)),    
    ("R_contact",      "linear", (0.0, 0.015))    
]
PARAM_NAMES = [s[0] for s in PARAM_SPEC]
BOUNDS = [s[2] for s in PARAM_SPEC]

IDX_FIXED = [1, 5]
IDX_STAGE1 = [0, 2]
IDX_STAGE2 = [3, 4, 6]
IDX_FREE = IDX_STAGE1 + IDX_STAGE2

def load_training_datasets(data_dir):
    def _dedup(t, *arrays):
        keep = np.r_[True, np.diff(t) > 0]
        return (t[keep],) + tuple(a[keep] for a in arrays)

    ds = []
    for name in ["Charge_1C", "Charge_2C", "Charge_3C", "Charge_4C", "Discharge_1C"]:
        path = os.path.join(data_dir, name + ".mat")
        if name.startswith("Charge"):
            m = sio.loadmat(path, squeeze_me=True)
            t, I, V = (np.asarray(m[k], float) for k in ("Time", "Current", "Voltage"))
            charge = True
        else:
            m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
            t, I, V = (np.asarray(getattr(m, k), float) for k in ("Time", "Current", "Voltage"))
            charge = False

        t, I, V = _dedup(t, I, V)
        on = np.abs(I) > 0.05
        i0 = int(np.argmax(on))
        V0 = float(V[i0 - 1])
        t, I, V = t[i0:] - t[i0], I[i0:], V[i0:]

        end = int(np.argmax(V >= V_HIGH - 1e-3)) if charge else int(np.argmax(V <= V_LOW + 1e-3))
        if end <= 1: end = len(V)
        t, I, V = t[:end], I[:end], V[:end]
        grid = np.arange(0.0, t[-1], GRID_DT)
        
        ds.append({
            "name": name, "charge": charge, "V0": V0, "I_app": float(-np.median(I)),
            "t": t, "V": V, "t_grid": grid, "V_grid": np.interp(grid, t, V), "duration_s": float(t[-1])
        })
    return ds

def apply_theta_dict(theta_dict, base_params):
    p = base_params.copy()
    cn = base_params["Maximum concentration in negative electrode [mol.m-3]"]
    
    def _scaled(f, fac): return lambda *a: fac * f(*a)
    
    p.update({
        "Negative electrode thickness [m]": float(theta_dict["L_neg_scale"]) * base_params["Negative electrode thickness [m]"],
        "Positive electrode thickness [m]": float(theta_dict["L_pos_scale"]) * base_params["Positive electrode thickness [m]"],
        "Initial concentration in negative electrode [mol.m-3]": float(theta_dict["x_init_neg"]) * cn,
        "Negative particle diffusivity [m2.s-1]": float(10.0 ** theta_dict["log10_D_neg"]),
        "Negative electrode exchange-current density [A.m-2]": _scaled(base_params["Negative electrode exchange-current density [A.m-2]"], 10.0 ** theta_dict["log10_k_neg"]),
        "Positive electrode exchange-current density [A.m-2]": _scaled(base_params["Positive electrode exchange-current density [A.m-2]"], 10.0 ** theta_dict["log10_k_pos"]),
        "Contact resistance [Ohm]": float(theta_dict["R_contact"]),
        "Nominal cell capacity [A.h]": 2.5,
    })
    return p

def objective(theta_dict, datasets, return_detail=False):
    base_params = pybamm.ParameterValues("Prada2013")
    p = apply_theta_dict(theta_dict, base_params)
    rms, cov = [], []
    
    for d in datasets:
        q = p.copy()
        q.update({
            "Current function [A]": d["I_app"],
            "Lower voltage cut-off [V]": 2.0 if d["charge"] else V_LOW,
            "Upper voltage cut-off [V]": V_HIGH if d["charge"] else 3.7
        })
        opts = {"thermal": "isothermal"}
        if q["Contact resistance [Ohm]"] > 0: opts["contact resistance"] = "true"
            
        try:
            sim = pybamm.Simulation(pybamm.lithium_ion.DFN(options=opts), parameter_values=q, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6))
            sol = sim.solve([0, d["duration_s"] * 1.2], initial_soc=f"{d['V0']:.4f} V")
            t_sim, V_sim = sol["Time [s]"].entries, sol["Voltage [V]"].entries
        except Exception:
            rms.append(PENALTY_V); cov.append(0.0)
            continue
            
        g = d["t_grid"]
        covered = g <= t_sim[-1]
        if covered.sum() < 2:
            rms.append(PENALTY_V); cov.append(0.0)
            continue
            
        r = np.interp(g[covered], t_sim, V_sim) - d["V_grid"][covered]
        sq = np.concatenate([r ** 2, np.full(int((~covered).sum()), PENALTY_V ** 2)])
        rms.append(float(np.sqrt(np.mean(sq))))
        cov.append(float(covered.mean()))
        
    J = float(np.sqrt(np.mean(np.array(rms) ** 2)))
    if return_detail:
        return J, {"per_dataset_RMSE_V": dict(zip([d["name"] for d in datasets], rms)), "coverage": dict(zip([d["name"] for d in datasets], cov))}
    return J