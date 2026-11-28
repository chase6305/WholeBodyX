"""Compare two native builds in fresh, alternating worker processes.

Both builds use the current Python adapter and identical MPC workloads. RSS is
sampled before any dense diagnostic formulation is requested. Set BLAS thread
counts in the launching environment. This is not a control-rate guarantee.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import platform

from _comparison import case, invoke_worker, joint_mpc_workload, joint_plan_violation, rss_kib

from _measurement import source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    from time import perf_counter, thread_time_ns
    import numpy as np
    import wholebodyx_native
    from wholebodyx import JointLimits, RobotState
    from wholebodyx.native import NativeMPC, NativeSparseMPC

    binary = Path(wholebodyx_native.__file__).resolve()
    if binary.parent != args.native_dir.resolve():
        raise RuntimeError(f"loaded {binary}, expected a build in {args.native_dir}")
    joints, horizon = args.case
    if args.workload == "retarget":
        limits = JointLimits([-1.] * joints, [1.] * joints, [.8] * joints, [2.] * joints)
    else:
        limits = JointLimits([-2.] * joints, [2.] * joints, [1.] * joints, [4.] * joints)
    state = RobotState(np.zeros(joints), np.zeros(joints))
    goal = np.full(joints, .02 if args.workload == "near" else .3)
    if args.workload == "retarget":
        goal = .55 * np.sin(np.arange(joints) * .37)
    report = dict(
        joints=joints,
        horizon=horizon,
        dt=args.dt,
        backend=args.backend,
        workload=args.workload,
        native_binary=str(binary),
        native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        initial_rss_kib=rss_kib(),
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    before = perf_counter()
    planner_type = NativeMPC if args.backend == "native" else NativeSparseMPC
    planner = planner_type(limits, horizon=horizon, dt=args.dt)
    report.update(
        initialization_ms=(perf_counter() - before) * 1000,
        after_initialization_rss_kib=rss_kib(),
        internal_fixed_matrix_bytes=getattr(planner, "fixed_matrix_bytes", None),
    )

    def solve(measurement, target):
        cpu_start = thread_time_ns()
        start = perf_counter()
        outcome = planner.solve(measurement, target)
        elapsed = (perf_counter() - start) * 1000
        cpu_ms = (thread_time_ns() - cpu_start) / 1e6
        if outcome.plan is None:
            raise RuntimeError(outcome.solver)
        violation = joint_plan_violation(limits, outcome.plan, measurement.v)
        if violation > 1e-6:
            raise RuntimeError(f"invalid physical plan: {violation}")
        return elapsed, cpu_ms, outcome.solver, violation

    report["first_solve_ms"], report["first_thread_cpu_ms"], _, _ = solve(state, goal)
    report["after_first_solve_rss_kib"] = rss_kib()
    for _ in range(args.warmup - 1):
        solve(state, goal)
    times, cpu_times, iterations = [], [], []
    maximum, physical, shortened = 0.0, 0.0, 0
    work = joint_mpc_workload(args.workload, state, goal, args.iterations, args.seed)
    for measurement, target in work:
        elapsed, cpu_ms, result, violation = solve(measurement, target)
        times.append(elapsed)
        cpu_times.append(cpu_ms)
        iterations.append(result.iterations)
        shortened += int(result.detail == "unconstrained suffix")
        maximum = max(maximum, result.violation)
        physical = max(physical, violation)
    report.update(
        p50_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        max_ms=max(times),
        thread_cpu_p50_ms=float(np.median(cpu_times)),
        thread_cpu_p95_ms=float(np.percentile(cpu_times, 95)),
        thread_cpu_p99_ms=float(np.percentile(cpu_times, 99)),
        mean_solver_iterations=float(np.mean(iterations)),
        zero_iteration_solves=iterations.count(0),
        unconstrained_suffix_solves=shortened,
        max_constraint_violation=maximum,
        max_physical_violation=physical,
        after_updates_rss_kib=rss_kib(),
    )
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-native", type=Path)
    parser.add_argument("--candidate-native", type=Path, default=ROOT / "build/native")
    parser.add_argument("--backend", choices=("native", "native-sparse"), default="native")
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument(
        "--workload", choices=("fixed", "random", "near", "alternating", "retarget"), default="fixed"
    )
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--baseline-source-receipt", type=Path)
    parser.add_argument("--cases", type=case, nargs="+", default=[(6, 15), (19, 30), (30, 60)])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/wholebodyx-native-comparison.json")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", type=case, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.rounds) < 1:
        parser.error("iterations, warmup and rounds must be positive")
    if not isfinite(args.dt) or args.dt <= 0.:
        parser.error("dt must be positive and finite")
    if args.worker:
        worker(args)
        return
    if args.baseline_native is None:
        parser.error("--baseline-native is required")
    builds = {
        "baseline": args.baseline_native.resolve(strict=True),
        "candidate": args.candidate_native.resolve(strict=True),
    }
    sources = list((ROOT / "cpp").glob("*.*")) + list((ROOT / "include/wholebodyx").glob("*.hpp"))
    sources += list((ROOT / "python/wholebodyx").glob("*.py"))
    sources += [
        Path(__file__).with_name("_comparison.py").resolve(),
        Path(__file__).resolve(),
        ROOT / "bindings/module.cpp",
        ROOT / "CMakeLists.txt",
        ROOT / "conanfile.py",
        ROOT / "conan.lock",
        ROOT / "profiles/linux-gcc12-release",
    ]
    hashes = source_hashes(ROOT, sources)
    report = dict(
        schema_version=1,
        status="running",
        python=platform.python_version(),
        platform=platform.platform(),
        rounds=args.rounds,
        iterations=args.iterations,
        warmup=args.warmup,
        backend=args.backend,
        dt=args.dt,
        workload=args.workload,
        seed=args.seed,
        candidate_source_sha256=hashes,
        results=[],
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        blas_threads={
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
    )
    if args.baseline_source_receipt:
        report["baseline_source_receipt"] = json.loads(args.baseline_source_receipt.read_text())
    try:
        for index in range(args.rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for joints, horizon in args.cases:
                for name in order:
                    row = invoke_worker(
                        __file__,
                        os.pathsep.join([str(ROOT / "python"), str(builds[name])]),
                        [
                            "--native-dir",
                            str(builds[name]),
                            "--case",
                            f"{joints}x{horizon}",
                            "--iterations",
                            str(args.iterations),
                            "--warmup",
                            str(args.warmup),
                            "--backend",
                            args.backend,
                            "--dt",
                            str(args.dt),
                            "--workload",
                            args.workload,
                            "--seed",
                            str(args.seed),
                        ],
                    )
                    row.update(build=name, round=index + 1)
                    report["results"].append(row)
                    print(json.dumps(row), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["candidate_source_unchanged"] = sources_unchanged(ROOT, hashes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
