"""Interleave opt-in Hessian reuse with uncached WBC on identical inputs."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np

from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState
from wholebodyx.wbc import ControlResult


def run(joints=(6, 30, 60, 120), steps=300, repeats=3, baseline=None):
    if steps < 25 or repeats < 1 or not joints or min(joints) < 1:
        raise ValueError("use positive dimensions/repeats and at least 25 steps")
    source = Path(inspect.getfile(KinematicWBC)).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256(Path(baseline).read_bytes()).hexdigest() if baseline else None
    modes = {"default": lambda limits: KinematicWBC(limits),
             "cached": lambda limits: KinematicWBC(limits, cache_task_hessian=True)}
    if baseline is not None:
        spec = importlib.util.spec_from_file_location("wholebodyx._wbc_benchmark_baseline", baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.ControlResult = ControlResult
        modes["baseline"] = lambda limits: module.KinematicWBC(limits)
    results = []
    for n in joints:
        rng = np.random.default_rng(20261027 + n)
        jacobian = rng.normal(size=(n, n)) / np.sqrt(n)
        limits = JointLimits([-2.] * n, [2.] * n, [1.] * n, [3.] * n)
        state = RobotState(np.zeros(n), np.zeros(n))
        for dynamic in (False, True):
            tasks = [[LinearTask(jacobian * (1 + .001 * k if dynamic else 1),
                                 .1 * np.sin(np.arange(n) + k * .01))] for k in range(steps)]
            for operation in ("formulate", "step"):
                for repeat in range(repeats):
                    order = list(modes) if repeat % 2 == 0 else list(reversed(modes))
                    expected = None
                    for mode in order:
                        controller = modes[mode](limits)
                        times, signatures = [], []
                        for k, batch in enumerate(tasks):
                            start = perf_counter_ns()
                            outcome = getattr(controller, operation)(state, batch, .01)
                            elapsed = (perf_counter_ns() - start) / 1000
                            if k >= 20:
                                times.append(elapsed)
                            if operation == "step":
                                if not outcome.solver.success:
                                    raise RuntimeError(outcome.solver.detail)
                                signatures.append(np.r_[outcome.command.velocity, outcome.solver.iterations,
                                                        outcome.solver.violation])
                            else:
                                signatures.append(np.r_[outcome.hessian.ravel(), outcome.gradient,
                                                        outcome.lower, outcome.upper])
                        if expected is None:
                            expected = signatures
                        else:
                            for a, b in zip(signatures, expected):
                                np.testing.assert_array_equal(a, b)
                        results.append(dict(joints=n, dynamic_jacobian=dynamic, operation=operation,
                                            repeat=repeat, mode=mode, p50_us=float(np.median(times)),
                                            p95_us=float(np.percentile(times, 95))))
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("loaded WBC source changed during the benchmark")
    if baseline and hashlib.sha256(Path(baseline).read_bytes()).hexdigest() != baseline_hash:
        raise RuntimeError("baseline source changed during the benchmark")
    return dict(status="passed", python=platform.python_version(), numpy=np.__version__, steps=steps,
                warmup_steps_excluded=20, repeats=repeats, results=results,
                source_file=str(source), source_sha256=source_hash, source_unchanged=True,
                baseline_sha256=baseline_hash,
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                scope="Repeated fixed measured state with varying targets; exact QP/solution parity; no scheduling guarantee")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=int, nargs="+", default=[6, 30, 60, 120])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline", type=Path, help="optional saved wbc.py from before the change")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.joints, args.steps, args.repeats, args.baseline)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
