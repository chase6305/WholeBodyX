"""Compare the four Python/C++ WBC and MPC combinations on the same experiment."""

import argparse
import json

import numpy as np

from _common import run


def compare(steps=400, scenario="mpc_retargeting"):
    if scenario not in ("mpc_retargeting", "disturbance_recovery"):
        raise ValueError("mixed comparison requires an MPC scenario")
    results = [run(scenario, wbc, mpc, steps)
               for wbc, mpc in (("python", "python"), ("python", "native"),
                                ("native", "python"), ("native", "native"))]
    errors = [r["final_position_error_m"] for r in results]
    # Compare end-to-end behavior, not timing: machines and OSQP versions differ.
    if not np.allclose(errors, errors[0], atol=2e-6, rtol=0):
        raise RuntimeError("backend combinations disagree on final tracking error")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--scenario", choices=("mpc_retargeting", "disturbance_recovery"), default="mpc_retargeting")
    args = parser.parse_args()
    print(json.dumps(compare(args.steps, args.scenario), indent=2))
