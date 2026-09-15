# LFP Cell Physical Modelling & Parametrisation in PyBaMM

Doyle–Fuller–Newman (DFN) model of the **A123 ANR26650M1B** (2.5 A·h, graphite / LiFePO₄) cell,
parametrised from measured charge/discharge data and validated on a held-out HPPC profile.

| | |
|---|---|
| **Model** | DFN, isothermal, contact resistance enabled (from Task 4 onward) |
| **Base parameter set** | `Prada2013` (PyBaMM built-in) |
| **Training data** | Charge 1C, 2C, 3C, 4C + Discharge 1C (constant-current phases) |
| **Validation data** | HPPC profile, 25.2 h — never used in the optimisation objective |
| **Result** | Training objective 0.2493 → 0.0347 V; HPPC RMSE 42.7 → 31.7 mV |

Full methodology, justification of every decision, and discussion are in
**`LFP_DFN_Case_Study_Report.pdf`**.

---

## 1. Requirements

```text
Python 3.12
pybamm  >= 26.0      # 26.8.0 used for all results in the report
numpy   >= 2.0
scipy
matplotlib
```

```bash
pip install "pybamm>=26" numpy scipy matplotlib
```

> **Version warning.** PyBaMM releases earlier than 26.x call `numpy.trapz`, which was
> **removed in NumPy 2.3**. The combination *PyBaMM 25.x + NumPy ≥ 2.3* fails with
> `AttributeError: module 'numpy' has no attribute 'trapz'` from inside PyBaMM.
> Either upgrade PyBaMM (preferred) or pin `numpy<2.3`. See §6.

The scripts use the **IDAKLU** (SUNDIALS) solver, which ships with PyBaMM wheels.
Verify with:

```bash
python -c "import pybamm, numpy; print(pybamm.__version__, numpy.__version__); pybamm.IDAKLUSolver()"
```

Everything runs on a **single CPU core**; peak memory is ~250 MB.

---

## 2. Repository layout

```text
.
├── Data/                                         # supplied measurement files (not modified)
│   ├── Charge_1C.mat  Charge_2C.mat
│   ├── Charge_3C.mat  Charge_4C.mat
│   ├── Discharge_1C.mat
│   └── HPPC.mat
│
├── task1_2_dfn_hppc_simulation.py                # Tasks 1 & 2
├── task3_residual_analysis_and_manual_tuning.py  # Task 3
├── task4_objective.py                            # Task 4 — objective (module, not run directly)
├── task4_worker.py                               # Task 4 — evaluation subprocess (not run directly)
├── task4_method_assessment.py                    # Task 4 — bullet 1
├── task4_optimize.py                             # Task 4 — bullets 2–4
├── task4_finalize.py                             # Task 4 — save parameters + training-fit plot
├── task5_validation.py                           # Task 5
│
├── results_task2/ … results_task5/               # generated outputs
└── LFP_DFN_Case_Study_Report.pdf
```

**All scripts must be run from the repository root**, because they resolve `Data/...` and
each other's output folders relative to the current working directory.

---

## 3. Running everything

The scripts are a pipeline: each reads the previous stage's output. Run them in this order.

```bash
# --- Tasks 1 & 2 : build the DFN, simulate the HPPC profile, error metrics  (~3 min)
python task1_2_dfn_hppc_simulation.py --data Data/HPPC.mat --out results_task2

# --- Task 3 : residual analysis, sensitivity study, manual tuning          (~15 min)
python task3_residual_analysis_and_manual_tuning.py --data Data --task2 results_task2 --out results_task3

# --- Task 4 : optimisation
python task4_method_assessment.py --data Data --out results_task4                      # (~6 min)
python task4_optimize.py --out results_task4                                           # (~40 min)
python task4_finalize.py --data Data --out results_task4                               # (~2 min)

# --- Task 5 : validation on the held-out HPPC profile                       (~5 min)
python task5_validation.py --data Data --task4 results_task4/task4_optimization_results.json --out results_task5
```

Total ≈ 70 minutes on one core.

> **Note on `--data`.** `task1_2_dfn_hppc_simulation.py` takes the **path to the HPPC file**
> (`Data/HPPC.mat`). Every other script takes the **data directory** (`Data`), because they
> read several files.

---

## 4. Script reference

### 4.1 `task1_2_dfn_hppc_simulation.py` — Tasks 1 & 2

Builds the DFN with the `Prada2013` set, applies the measured HPPC current as a
`pybamm.Interpolant`, and compares simulated with measured voltage.

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `Data/HPPC.mat` | path to the HPPC `.mat` file |
| `--out` | `results_task2` | output directory |

**Outputs** → `results_task2/`

- `task2_voltage_measured_vs_simulated.png` — measured vs simulated voltage + current
- `task2_voltage_residual.png` — voltage estimation error
- `task2_zoom_first_pulse_block.png` — first pulse block, 0–4000 s
- `task2_residual_vs_current.png` — residual coloured by |current|
- `task2_metrics.json` — RMSE, MAE, max error, bias, pulse/rest split, time-weighted RMSE
- `task2_hppc_measured_vs_simulated.csv` — time, current, measured V, simulated V, residual

---

### 4.2 `task3_residual_analysis_and_manual_tuning.py` — Task 3

Classifies the Task 2 residual by operating condition, runs a one-at-a-time sensitivity study,
performs manual tuning of the capacity and rate parameters, and re-simulates the HPPC profile.

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `Data` | data directory |
| `--task2` | `results_task2` | Task 2 output directory (reads the CSV and metrics JSON) |
| `--out` | `results_task3` | output directory |

**Requires** `task1_2_dfn_hppc_simulation.py` in the same folder — it imports `CONFIG`,
`load_hppc`, `error_metrics`, `build_model_and_parameters`, `simulate` and `make_plots` from it
so that Task 2 and Task 3 numbers are computed identically.

**Outputs** → `results_task3/`

- `task3a_residual_share.png`, `task3a_ocv_vs_capacity.png`, `task3a_pulse_resistance.png`
- `task3a_residual_analysis.json` — segment shares, 8 OCV points, 71 pulse resistances
- `task3b_sensitivity.png` / `.json` — OAT ranking of 24 candidate parameters
- `task3c_1C_discharge_tuning.png`, `task3c_ocv_diagnosis.png`, `task3c_ocp_diagnosis.json`
- `parameters_task3_manual_tuned.json` — manually tuned set
- `task3_hppc_tuned_measured_vs_simulated.csv` and `hppc_tuned_plots/`

---

### 4.3 Task 4 — optimisation

Four files. `task4_objective.py` and `task4_worker.py` are **modules/helpers, not run directly**.

#### `task4_objective.py` (module)
Loads the five training records, maps a 7-element parameter vector onto `ParameterValues`, and
evaluates the objective: the equal-weighted RMS of the per-record voltage RMSE on a uniform 5 s
grid.

#### `task4_worker.py` (helper)
A persistent evaluation subprocess to protect the main loop from IDAKLU crashes during aggressive parameter exploration. 

#### `task4_method_assessment.py` — bullet 1
Produces 1-D objective slices, finite-difference gradients at several step sizes, and a
repeatability check.

#### `task4_optimize.py` — bullets 2–4
Three stages, all using all five training records:
1. capacity & stoichiometry (`L_neg_scale`, `x_init_neg`) — global differential evolution
2. dynamics (`log10 D_neg`, `log10 k_neg`, `R_contact`) — global differential evolution
3. joint Nelder–Mead polish over the five free parameters

| Argument | Default | Meaning |
|---|---|---|
| `--out` | `results_task4` | output directory |

*Note: Hyperparameters and seeds (e.g., `seed=0`, `maxiter=12`) are hardcoded directly into `task4_optimize.py` to ensure 1:1 reproducibility with the exploratory Jupyter Notebook and to strictly replicate the validated PDF report findings.*

#### `task4_finalize.py`
Collects the best checkpoint, evaluates it, writes the parameter file and the training-fit plot.

**Outputs** → `results_task4/`

- **`parameters_task4_optimized.json`** — the optimised parameter set (the Task 4 deliverable)
- `task4_optimization_results.json` — baseline vs final, per-record RMSE and coverage
- `task4_training_fit.png` — fit on all five records, before and after

---

### 4.4 `task5_validation.py` — Task 5

Validates the optimised set on the HPPC profile, plots the physical variables, and compares all
three parameter sets over the time window they all cover.

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `Data` | data directory |
| `--task4` | `results_task4/task4_optimization_results.json` | output file from Task 4 |
| `--out` | `results_task5` | output directory |

**Outputs** → `results_task5/`

- `task5a_hppc_validation.png` — measured vs simulated voltage, error, and current
- `task5b_anode_potential.png` — anode potential vs Li/Li⁺ with the 0 V plating threshold
- Output terminal metric tables matching Jupyter Notebook deliverables.

---

## 5. Expected results

| Stage | Training objective | HPPC RMSE (common 0–16.27 h window) | HPPC coverage |
|---|---|---|---|
| Baseline `Prada2013` | 0.2493 V | 42.7 mV | 63.3 % |
| Manual tuning (Task 3) | — | 36.0 mV | 100 % |
| Optimised (Task 4) | 0.0347 V | 31.7 mV | 100 % |

Per-record RMSE with the optimised set: Charge 1C 31.1, 2C 34.0, 3C 35.6, 4C 35.2,
Discharge 1C 37.3 mV — all at 100 % coverage.

---

## 6. Troubleshooting

**`AttributeError: module 'numpy' has no attribute 'trapz'`**
Old PyBaMM with NumPy ≥ 2.3. Check the traceback: if the failing line is inside
`site-packages/pybamm/...`, upgrade PyBaMM. Use the same interpreter that runs the script:

```bash
python -m pip install --upgrade pybamm scipy
python -c "import pybamm, numpy; print(pybamm.__version__, numpy.__version__)"
```

If PyBaMM cannot be upgraded, pin NumPy instead: `python -m pip install "numpy<2.3"`.

**`AttributeError: module 'numpy' has no attribute 'trapezoid'`**
The opposite case — NumPy < 2.0. Upgrade NumPy, or add after `import numpy as np`:

```python
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
```

**`FileNotFoundError: 'Data/HPPC.mat'`**
The working directory is not the repository root. Check with:
```python
import os, glob; print(os.getcwd()); print(glob.glob("**/HPPC.mat", recursive=True))
```

---

## 7. Known limitations

Stated in full in report §9. In brief:

- Both electrode OCP curves are borrowed (graphite from Chen 2020 / LG M50, LFP from Afshar
  2017). This is the largest remaining error source.
- The optimised electrode thickness (×1.2068) is a **compensating parameter** absorbing the OCP
  shape error, not a physical measurement of the electrode.
- Capacity levers are degenerate: thickness and `c_max` + initial concentration produce the same
  cell to within 0.87 mV RMS. Optimise only one per electrode.
- No temperature dependence is identified — the data spans 22.8–24.1 °C only.
- CV phases were excluded from parametrisation and remain unvalidated.
