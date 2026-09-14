"""Bounded-memory endurance check of the H1 fixed-contact dynamics demo."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(seconds, amplitude, output, controller="wbc", push_force=0.):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive")
    spec = importlib.util.spec_from_file_location("h1_soak_demo", ROOT / "examples/python/h1_dynamics_wbc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.DynamicsDemo(amplitude=amplitude, controller=controller, push_force=push_force)
    steps = int(np.ceil(seconds / demo.dt))
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
               (ROOT / "python/wholebodyx/dynamics.py", ROOT / "examples/python/h1_dynamics_wbc.py",
                ROOT / "python/wholebodyx/centroidal.py", ROOT / "python/wholebodyx/qp.py",
                ROOT / "python/wholebodyx/contact.py")}
    started = heartbeat = time.monotonic()
    report = dict(status="running", controller=controller, push_force_n=push_force, requested_seconds=seconds, amplitude=amplitude,
                  source_sha256=sources, steps=0, min_force_n=float("inf"), max_com_error_m=0.,
                  max_contact_position_error_m=0., max_forward_contact_violation=0.,
                  max_constraint_violation=0., max_forward_dynamics_residual=0., max_quaternion_norm_error=0., max_wbc_ms=0.,
                  initial_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        for index in range(steps):
            row = demo.step()
            report["steps"] = index + 1
            report["min_force_n"] = min(report["min_force_n"], row["left_fz_n"], row["right_fz_n"])
            for key, source in (("max_com_error_m", "com_error_m"),
                                ("max_contact_position_error_m", "contact_position_error_m"),
                                ("max_forward_contact_violation", "forward_contact_violation"),
                                ("max_constraint_violation", "constraint_violation"),
                                ("max_forward_dynamics_residual", "forward_dynamics_residual"),
                                ("max_wbc_ms", "wbc_ms")):
                report[key] = max(report[key], row[source])
            report["max_quaternion_norm_error"] = max(report["max_quaternion_norm_error"],
                                                       abs(float(np.linalg.norm(demo.state.q[3:7])) - 1))
            if index % 1000 == 0:
                if demo.max_contact_error > 1e-4:
                    raise AssertionError("contact position drift exceeds 0.1 mm")
                if np.any(demo.state.q[7:] < demo.robot.model.lowerPositionLimit[7:] - 1e-6) or np.any(
                        demo.state.q[7:] > demo.robot.model.upperPositionLimit[7:] + 1e-6):
                    raise AssertionError("joint position limit exceeded")
                if time.monotonic() - heartbeat >= 30:
                    heartbeat = time.monotonic()
                    print(json.dumps(dict(simulation_s=demo.state.stamp, wall_s=heartbeat - started,
                                          contact_error_m=demo.max_contact_error,
                                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update(simulation_s=demo.state.stamp, wall_s=time.monotonic() - started,
                      mpc_continuity_fallbacks=demo.mpc_continuity_fallbacks,
                      solver_setups=demo.controller.solver_setups, solver_recoveries=demo.controller.solver_recoveries, mpc_replans=0 if demo.planner is None else demo.planner.replans,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-force", type=float, default=0.)
    parser.add_argument("--controller", choices=("wbc", "mpc"), default="wbc")
    parser.add_argument("--seconds", type=float, default=3600.)
    parser.add_argument("--amplitude", type=float, default=.05)
    parser.add_argument("--output", type=Path, default=Path("/tmp/h1-dynamics-soak.json"))
    args = parser.parse_args()
    run(args.seconds, args.amplitude, args.output, args.controller, args.push_force)
