"""Register independently configured WBC/MPC factories for headless or GUI runs.

These factories use the built-in algorithms; replace them with adapters from your
own packages that return ControlResult / PlanResult to connect external libraries.
"""

import argparse
from functools import partial
import json
from pathlib import Path

from wholebodyx import BackendRegistry, JointMPC, KinematicWBC
from wholebodyx.runner import run as run_experiment


def make_registry():
    registry = BackendRegistry()
    registry.register("smooth_wbc", controller=partial(KinematicWBC, regularization=1e-5, smoothing=.02))
    registry.register("long_mpc", planner=partial(JointMPC, horizon=20, control_weight=.2))
    return registry


def run(steps=400, csv_path=None):
    return run_experiment("disturbance_recovery", backend="smooth_wbc", mpc_backend="long_mpc",
                          steps=steps, csv_path=csv_path, registry=make_registry())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.viewer:
        if args.steps is not None or args.csv is not None:
            parser.error("--steps and --csv apply to headless runs")
        from wholebodyx.visualization.viewer import ControlViewer
        ControlViewer(port=args.port, scenario="disturbance_recovery", backend="smooth_wbc",
                      mpc_backend="long_mpc", registry=make_registry()).run()
    else:
        print(json.dumps(run(400 if args.steps is None else args.steps, args.csv), indent=2))
