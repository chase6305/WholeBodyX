"""Measure native WBC with changing dense Jacobians, including Python adapters.

Select a saved baseline or current extension through PYTHONPATH. Timing excludes
task construction and the first five solves. Optional traces support numerical
comparison between builds; observed percentiles are not real-time guarantees.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from wholebodyx import JointLimits, LinearTask, RobotState
from wholebodyx.native import NativeWBC, _extension


def benchmark(joints, iterations):
    results, traces = [], {}
    for n in joints:
        rng = np.random.default_rng(n)
        jacobian = rng.normal(size=(2 * n, n))
        limits = JointLimits([-10.0] * n, [10.0] * n, [1.0] * n)
        controller = NativeWBC(limits, regularization=0.1)
        samples, velocities, iteration_counts = [], [], []
        maximum_violation = 0.0
        for k in range(iterations + 5):
            state = RobotState(np.zeros(n), np.zeros(n), k * 0.01)
            task = LinearTask(
                jacobian + 0.01 * np.sin(k) * np.eye(2 * n, n), np.full(2 * n, 0.1 * np.cos(k))
            )
            started = perf_counter()
            result = controller.step(state, [task], 0.01)
            elapsed = (perf_counter() - started) * 1e6
            if not result.solver.success or result.command is None:
                raise RuntimeError(result.solver)
            lower, upper = limits.velocity_bounds(state, 0.01)
            velocity = result.command.velocity
            violation = max(float(np.max(lower - velocity)), float(np.max(velocity - upper)), 0.0)
            if violation > 1e-6:
                raise RuntimeError(f"invalid physical command: {violation}")
            maximum_violation = max(maximum_violation, violation)
            if k >= 5:
                samples.append(elapsed)
                velocities.append(velocity)
                iteration_counts.append(result.solver.iterations)
        results.append(
            dict(
                joints=n,
                p50_us=float(np.median(samples)),
                p95_us=float(np.percentile(samples, 95)),
                max_us=max(samples),
                mean_iterations=float(np.mean(iteration_counts)),
                max_physical_violation=maximum_violation,
            )
        )
        traces[str(n)] = np.asarray(velocities)
    extension = Path(_extension().__file__)
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        warmup=5,
        iterations=iterations,
        threads={
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        native_binary=str(extension.resolve()),
        native_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        results=results,
    ), traces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", nargs="+", type=int, default=[6, 19, 29])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or any(n < 1 for n in args.joints):
        parser.error("iterations and joint counts must be positive")
    report, traces = benchmark(args.joints, args.iterations)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.trace, **traces)
    print(rendered)
