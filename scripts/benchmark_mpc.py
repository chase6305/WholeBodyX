"""Measure MPC initialization and repeated parameter updates on a fixed state.

For reproducible BLAS scheduling, launch with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
These are observed wall times, not a hard real-time or closed-loop benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import osqp
import scipy

from wholebodyx import JointLimits, JointMPC, RobotState


def measure(backend, joints, horizon, iterations, warmup):
    if backend == "native":
        from wholebodyx.native import NativeMPC as Planner
    elif backend == "sparse":
        from wholebodyx.sparse_mpc import SparseJointMPC as Planner
    elif backend == "native-sparse":
        from wholebodyx.native import NativeSparseMPC as Planner
    else:
        Planner = JointMPC
    limits = JointLimits(np.full(joints, -2.), np.full(joints, 2.), np.ones(joints), np.full(joints, 4.))
    state, goal = RobotState(np.zeros(joints), np.zeros(joints)), np.full(joints, .3)
    start = perf_counter()
    planner = Planner(limits, horizon=horizon)
    initialization_ms = (perf_counter() - start) * 1000

    def solve(target):
        start = perf_counter()
        result = planner.solve(state, target)
        elapsed = (perf_counter() - start) * 1000
        if result.plan is None:
            raise RuntimeError(f"benchmark solve failed: {result.solver}")
        return elapsed, result.solver

    first_solve_ms, _ = solve(goal)
    for _ in range(warmup - 1):
        solve(goal)
    times, solver_iterations, max_violation = [], [], 0.0
    for k in range(iterations):
        elapsed, result = solve(goal + .01 * np.sin(k))
        times.append(elapsed)
        solver_iterations.append(result.iterations)
        max_violation = max(max_violation, result.violation)
    owned = getattr(planner, "fixed_matrix_bytes", None)
    problem = planner.formulate(state, goal)
    from scipy import sparse
    def storage(matrix):
        if sparse.issparse(matrix):
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        return matrix.nbytes
    report = {"backend": backend, "joints": joints, "horizon": horizon, "iterations": iterations,
            "problem_matrix_bytes": storage(problem.hessian) + storage(problem.constraint),
            "initialization_ms": initialization_ms, "first_solve_ms": first_solve_ms,
            "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "max_ms": max(times), "mean_solver_iterations": float(np.mean(solver_iterations)),
            "max_constraint_violation": max_violation}
    if owned is not None:
        report["internal_fixed_matrix_bytes"] = int(owned)
    return report


def case(value):
    try:
        joints, horizon = map(int, value.split("x"))
        if joints < 1 or horizon < 1:
            raise ValueError
        return joints, horizon
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be positive JOINTSxHORIZON, e.g. 14x30") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native", "sparse", "native-sparse", "all"), default="all")
    parser.add_argument("--cases", nargs="+", type=case, default=[(6, 15), (14, 30)])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1,
                        help="repeat comparisons, reversing backend order on alternate rounds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1 or args.rounds < 1:
        parser.error("iterations, warmup and rounds must be positive")
    backends = ("python", "native", "sparse", "native-sparse") if args.backend == "all" else (args.backend,)
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "python/wholebodyx").glob("*.py")) + [Path(__file__).resolve()]
    sources += list((root / "cpp").glob("*.*")) + list((root / "include/wholebodyx").glob("*.hpp"))
    sources += [root / "bindings/module.cpp", root / "CMakeLists.txt"]
    if any(name.startswith("native") for name in backends):
        import wholebodyx_native
        sources.append(Path(wholebodyx_native.__file__).resolve())
    digests = {str(path.relative_to(root) if path.is_relative_to(root) else path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sources}
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "python_osqp": osqp.__version__,
                              "platform": platform.platform(),
                              "blas_threads": {name: os.environ.get(name) for name in
                                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}},
              "warmup": args.warmup, "rounds": args.rounds, "source_sha256": digests,
              "results": []}
    for round_index in range(args.rounds):
        order = backends if round_index % 2 == 0 else tuple(reversed(backends))
        for n, h in args.cases:
            for backend in order:
                result = measure(backend, n, h, args.iterations, args.warmup)
                if args.rounds > 1:
                    result["round"] = round_index + 1
                report["results"].append(result)
    report["source_unchanged"] = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                                      for name, digest in digests.items())
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
