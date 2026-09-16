"""Compare H1 inverse-dynamics implementations on the same closed-loop trajectory.

Optional --reference is a saved wholebodyx/dynamics.py implementation. Timings
include Python overhead and are observations, not real-time guarantees.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def benchmark(steps=1600, repeats=3, reference=None, *, controller="wbc", amplitude=.035, push_force=0.):
    if steps < 1 or repeats < 1:
        raise ValueError("steps and repeats must be positive")
    import wholebodyx.dynamics as current
    demo = module_at("h1_benchmark_demo", ROOT / "examples/python/h1_dynamics_wbc.py")
    implementations = {"current": current}
    if reference is not None:
        implementations["reference"] = module_at("wholebodyx._benchmark_reference", reference)
    results = {name: [] for name in implementations}
    traces = {}
    torque_differences, configuration_differences = [], []
    for repeat in range(repeats):
        order = list(implementations) if repeat % 2 == 0 else list(reversed(implementations))
        for name in order:
            implementation = implementations[name]
            robot_class = implementation.FloatingBaseDynamics
            if "external_force" not in inspect.signature(robot_class.forward).parameters:
                class CompatibleReference(robot_class):
                    def forward(self, state, torque, contact_acceleration, terms, *, external_force=None):
                        if external_force is not None and np.any(external_force):
                            raise ValueError("reference benchmark does not support external forces")
                        return super().forward(state, torque, contact_acceleration, terms)
                robot_class = CompatibleReference
            demo.FloatingBaseDynamics = robot_class
            demo.ContactWBC = implementation.ContactWBC
            session = demo.DynamicsDemo(controller=controller, amplitude=amplitude, push_force=push_force)
            latency, torque, configurations = [], [], []
            started = time.perf_counter()
            for _ in range(steps):
                begin = time.perf_counter()
                session.step()
                latency.append((time.perf_counter() - begin) * 1000)
                torque.append(session.controller.previous[session.robot.nv:session.robot.nv + session.robot.na].copy())
                configurations.append(session.state.q.copy())
            elapsed = time.perf_counter() - started
            results[name].append(dict(wall_s=elapsed, mean_step_ms=float(np.mean(latency)),
                                      p95_step_ms=float(np.percentile(latency, 95)),
                                      p99_step_ms=float(np.percentile(latency, 99)),
                                      max_step_ms=float(np.max(latency)),
                                      solver_recoveries=getattr(session.controller, "solver_recoveries", 0),
                                      solver_setups=getattr(session.controller, "solver_setups", steps),
                                      max_contact_error_m=session.max_contact_error,
                                      max_dynamics_residual=session.max_dynamics_residual))
            traces[name] = (np.asarray(torque), np.asarray(configurations))
        if reference is not None:
            torque_differences.append(float(np.max(np.abs(traces["current"][0] - traces["reference"][0]))))
            configuration_differences.append(float(np.max(np.abs(traces["current"][1] - traces["reference"][1]))))
    report = dict(steps=steps, repeats=repeats, controller=controller, amplitude_m=amplitude,
                  push_force_n=push_force, results=results,
                  dynamics_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for name, module in implementations.items()})
    if reference is not None:
        report["max_torque_difference_nm"] = max(torque_differences)
        report["max_configuration_difference"] = max(configuration_differences)
        report["median_speedup"] = float(np.median([r["wall_s"] for r in results["reference"]]) /
                                         np.median([r["wall_s"] for r in results["current"]]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--controller", choices=("wbc", "mpc"), default="wbc")
    parser.add_argument("--amplitude", type=float, default=.035)
    parser.add_argument("--push-force", type=float, default=0.)
    args = parser.parse_args()
    report = benchmark(args.steps, args.repeats, args.reference, controller=args.controller,
                       amplitude=args.amplitude, push_force=args.push_force)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
