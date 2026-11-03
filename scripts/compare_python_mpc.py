"""Compare Python MPC revisions in fresh, alternating worker processes.

RSS is sampled before dense diagnostic formulation. Both revisions use the
same interpreter, dependencies and workload. Set BLAS threads when launching.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

from _comparison import case, invoke_worker, rss_kib

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    from time import perf_counter
    import numpy as np
    import scipy
    import osqp
    import wholebodyx
    from wholebodyx import JointLimits, JointMPC, RobotState

    package = Path(wholebodyx.__file__).resolve().parent
    if package.parent != args.python_dir.resolve():
        raise RuntimeError(f"loaded {package}, expected a package in {args.python_dir}")
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in package.rglob("*.py")}
    joints, horizon = args.case
    limits = JointLimits([-2.] * joints, [2.] * joints, [1.] * joints, [4.] * joints)
    state, goal = RobotState(np.zeros(joints), np.zeros(joints)), np.full(joints, .3)
    report = dict(joints=joints, horizon=horizon, workload=args.workload,
                  package=str(package), source_sha256=hashes,
                  dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__),
                  initial_rss_kib=rss_kib(), started_utc=datetime.now(timezone.utc).isoformat())
    before = perf_counter()
    planner = JointMPC(limits, horizon=horizon)
    report.update(initialization_ms=(perf_counter() - before) * 1000,
                  after_initialization_rss_kib=rss_kib(),
                  internal_fixed_matrix_bytes=getattr(planner, "fixed_matrix_bytes", None))

    def solve(measurement, target):
        start = perf_counter()
        outcome = planner.solve(measurement, target)
        elapsed = (perf_counter() - start) * 1000
        if outcome.plan is None:
            raise RuntimeError(outcome.solver)
        plan = outcome.plan
        violation = max(float(np.max(limits.lower - plan.positions[1:])),
                        float(np.max(plan.positions[1:] - limits.upper)),
                        float(np.max(np.abs(plan.velocities) - limits.velocity)),
                        float(np.max(np.abs(np.diff(np.vstack((measurement.v, plan.velocities)), axis=0))
                                     - .1 * limits.acceleration)), 0.)
        if violation > 1e-6:
            raise RuntimeError(f"invalid physical plan: {violation}")
        return elapsed, outcome.solver, violation

    report["first_solve_ms"], _, _ = solve(state, goal)
    report["after_first_solve_rss_kib"] = rss_kib()
    for _ in range(args.warmup - 1):
        solve(state, goal)
    rng = np.random.default_rng(args.seed)
    times, iterations, maximum, physical = [], [], 0., 0.
    for k in range(args.iterations):
        if args.workload == "random":
            state = RobotState(rng.uniform(-.25, .25, joints), rng.uniform(-.1, .1, joints), k * .07)
            target = rng.uniform(-3., 3., joints)
        else:
            target = goal + .01 * np.sin(k)
        elapsed, result, violation = solve(state, target)
        times.append(elapsed)
        iterations.append(result.iterations)
        maximum = max(maximum, result.violation)
        physical = max(physical, violation)
    report.update(p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                  max_ms=max(times), mean_solve_ms=float(np.mean(times)),
                  mean_solver_iterations=float(np.mean(iterations)),
                  max_constraint_violation=maximum, max_physical_violation=physical,
                  after_updates_rss_kib=rss_kib(),
                  source_unchanged=all(hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
                                       for name, digest in hashes.items()))
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path, default=ROOT / "python")
    parser.add_argument("--cases", type=case, nargs="+", default=[(6, 15), (19, 30), (30, 60)])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--workload", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-python-comparison.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--python-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", type=case, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.rounds) < 1:
        parser.error("iterations, warmup and rounds must be positive")
    if args.worker:
        worker(args)
        return
    if args.baseline_python is None:
        parser.error("--baseline-python is required")
    builds = {"baseline": args.baseline_python.resolve(strict=True),
              "candidate": args.candidate_python.resolve(strict=True)}
    report = dict(schema_version=1, status="running", python=platform.python_version(),
                  platform=platform.platform(), rounds=args.rounds, iterations=args.iterations,
                  warmup=args.warmup, workload=args.workload, seed=args.seed, results=[],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).with_name("_comparison.py").read_bytes()).hexdigest(),
                  measurement_sha256=hashlib.sha256(Path(__file__).with_name("_measurement.py").read_bytes()).hexdigest(),
                  blas_threads={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    try:
        for index in range(args.rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for joints, horizon in args.cases:
                for name in order:
                    row = invoke_worker(__file__, builds[name], [
                        "--python-dir", str(builds[name]), "--case", f"{joints}x{horizon}",
                        "--iterations", str(args.iterations), "--warmup", str(args.warmup),
                        "--workload", args.workload, "--seed", str(args.seed)])
                    row.update(build=name, round=index + 1)
                    report["results"].append(row)
                    print(json.dumps({key: row[key] for key in
                                      ("build", "round", "joints", "horizon", "p50_ms", "p95_ms",
                                       "initialization_ms", "after_first_solve_rss_kib")}), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
