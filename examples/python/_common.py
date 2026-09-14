"""Compatibility entry for source-tree examples; execution lives in the package."""

import argparse
import json
from pathlib import Path

from wholebodyx.experiments import SCENARIOS
from wholebodyx.runner import run
from wholebodyx.session import TrackingSession

def main(scenario):
    parser = argparse.ArgumentParser(description=SCENARIOS[scenario].description)
    parser.add_argument("--backend", choices=("python", "native"), default="python", help="WBC backend")
    parser.add_argument("--mpc-backend", choices=("python", "native"), default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--csv", type=Path, help="Write every headless step, including q/v and errors")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.viewer:
        if args.csv is not None or args.steps is not None:
            parser.error("--csv and --steps apply to headless runs")
        from wholebodyx.visualization.viewer import ControlViewer
        ControlViewer(port=args.port, backend=args.backend, scenario=scenario, mpc_backend=args.mpc_backend).run()
    else:
        print(json.dumps(run(scenario, args.backend, args.mpc_backend, args.steps, args.csv), indent=2))
