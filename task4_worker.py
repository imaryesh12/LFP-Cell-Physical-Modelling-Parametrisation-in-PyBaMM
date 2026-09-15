"""
Task 4 Worker: Executes a single PyBaMM evaluation.
Safely isolates IDAKLU C++ solver crashes from the main optimization loop.
"""
import sys
import json
import pybamm
import task4_objective as T

pybamm.set_logging_level("ERROR")

if __name__ == "__main__":
    try:
        theta_dict = json.loads(sys.argv[1])
        datasets = T.load_training_datasets("Data")
        J = T.objective(theta_dict, datasets)
        print(json.dumps({"J": J}))
    except Exception as e:
        print(json.dumps({"J": T.PENALTY_V, "error": str(e)}))