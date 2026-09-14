"""Installed entry points for experiments, diagnostics and the optional viewer."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .experiments import SCENARIOS


def _control_arguments(parser):
    parser.add_argument("--backend", choices=("python", "native"), default="python", help="WBC backend")
    parser.add_argument("--mpc-backend", choices=("python", "native"))
    parser.add_argument("--mode", choices=("WBC", "MPC + WBC"), help="tracking defaults to MPC + WBC")


def main(argv=None):
    parser = argparse.ArgumentParser(description="WholeBodyX control experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the packaged simulation scenarios as JSON")
    run_parser = commands.add_parser("run", help="Run a headless experiment and print JSON metrics")
    run_parser.add_argument("scenario", choices=tuple(SCENARIOS))
    _control_arguments(run_parser)
    run_parser.add_argument("--steps", type=int)
    run_parser.add_argument("--csv", type=Path)
    viewer_parser = commands.add_parser("view", help="Open the Viser experiment server")
    viewer_parser.add_argument("scenario", choices=tuple(SCENARIOS), nargs="?", default="tracking")
    _control_arguments(viewer_parser)
    viewer_parser.add_argument("--host", default="127.0.0.1")
    viewer_parser.add_argument("--port", type=int, default=8080)
    doctor_parser = commands.add_parser("doctor", help="Check optional imports and both control implementations")
    doctor_parser.add_argument("--require", nargs="+", choices=("native", "urdf", "viz"), default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = {name: asdict(scenario) for name, scenario in SCENARIOS.items()}
        elif args.command == "doctor":
            from .diagnostics import diagnose
            result = diagnose()
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0 if all(result["checks"][key]["available"] for key in ["python", *args.require]) else 1
        elif args.command == "run":
            from .runner import run
            result = run(args.scenario, args.backend, args.mpc_backend, args.steps, args.csv, mode=args.mode)
        else:
            from .visualization.viewer import ControlViewer
            required = SCENARIOS[args.scenario].mode
            if args.mode is not None and required is not None and args.mode != required:
                raise ValueError(f"{args.scenario} requires {required}")
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            ControlViewer(host=args.host, port=args.port, backend=args.backend,
                          mode=args.mode or required or "MPC + WBC", scenario=args.scenario,
                          mpc_backend=args.mpc_backend).run()
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, RuntimeError, ImportError, OSError) as exc:
        print(f"wholebodyx: {exc}", file=sys.stderr)
        return 1
