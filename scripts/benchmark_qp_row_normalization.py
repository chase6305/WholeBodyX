"""Compare caller and adapter row normalization on full QP construction/solve.

All variants use identical physical problems and independently constructed KKT
optima. Caller-cached reuses normalized A when the workload says A is fixed;
caller recomputes it each call. Preparation of physical inputs is outside timing.
Run serially with one BLAS thread; shared-host timings are not real-time bounds.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import osqp
from scipy import sparse
import scipy
import wholebodyx

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram

from _measurement import source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


def measure(n, compact, mode, variant, iterations, warmup, unit_rows):
    rng = np.random.default_rng(20261112 + n)
    b = rng.normal(size=(n, n)) / np.sqrt(n)
    base_h = b.T @ b + np.eye(n)
    base_a = rng.normal(size=(2 * n, n))
    base_a *= np.geomspace(.1, 3., 2 * n)[:, None]
    if unit_rows:
        base_a /= np.max(np.abs(base_a), axis=1)[:, None]
    expected0 = rng.uniform(-.3, .3, n)
    fixed_norm = np.max(np.abs(base_a), axis=1)
    fixed_normalized_a = base_a / fixed_norm[:, None]
    solver = OSQPSolver(variable_scaling=np.geomspace(.2, 3., n),
                         normalize_constraints=variant == 'solver')
    constructor = SparseQuadraticProgram if compact else QuadraticProgram
    wall, cpu = [], []
    max_difference = max_violation = max_objective_gap = 0.
    total_iterations = 0
    for k in range(iterations + warmup):
        h_scale = 1. + .1 * np.sin(k) if mode in ('hessian', 'both') else 1.
        a_scale = 1. + .1 * np.cos(k) if mode in ('constraint', 'both') else 1.
        h, a = h_scale * base_h, a_scale * base_a
        expected = expected0 + .01 * np.sin(k)
        ax = a @ expected
        lower, upper = ax - 1., ax + 1.
        # One active upper constraint, with a complementary positive dual.
        upper[0] = ax[0]
        gradient = -h @ expected - a[0]
        started_wall, started_cpu = time.perf_counter_ns(), time.thread_time_ns()
        solver_a, solver_lower, solver_upper = a, lower, upper
        if variant != 'solver':
            if variant == 'caller-cached' and mode in ('fixed', 'hessian'):
                row_norm, solver_a = fixed_norm, fixed_normalized_a
            else:
                row_norm = np.max(np.abs(a), axis=1)
                row_norm[row_norm == 0] = 1.
                solver_a = a / row_norm[:, None]
            solver_lower, solver_upper = lower / row_norm, upper / row_norm
        p = constructor(sparse.csc_matrix(h) if compact else h, gradient,
                        sparse.csc_matrix(solver_a) if compact else solver_a,
                        solver_lower, solver_upper)
        result = solver.solve(p)
        elapsed_cpu = (time.thread_time_ns() - started_cpu) / 1000.
        elapsed_wall = (time.perf_counter_ns() - started_wall) / 1000.
        if not result.success:
            raise RuntimeError((n, compact, mode, variant, k, result))
        x = result.x
        difference = float(np.max(np.abs(x - expected)))
        actual_ax = a @ x
        violation = float(max(np.max(lower - actual_ax), np.max(actual_ax - upper), 0.))
        objective = lambda value: .5 * value @ h @ value + gradient @ value
        gap = float(abs(objective(x) - objective(expected)) / (1 + abs(objective(expected))))
        if difference > 3e-5 or violation > 1e-6 or gap > 1e-6:
            raise RuntimeError((n, compact, mode, variant, k, difference, violation, gap))
        if k >= warmup:
            wall.append(elapsed_wall)
            cpu.append(elapsed_cpu)
            total_iterations += result.iterations
            max_difference, max_violation = max(max_difference, difference), max(max_violation, violation)
            max_objective_gap = max(max_objective_gap, gap)
    return dict(joints=n, storage='sparse' if compact else 'dense', mode=mode, variant=variant,
                wall_p50_us=float(np.median(wall)), wall_p95_us=float(np.percentile(wall, 95)),
                wall_max_us=max(wall), cpu_p50_us=float(np.median(cpu)),
                cpu_p95_us=float(np.percentile(cpu, 95)), cpu_mean_us=float(np.mean(cpu)),
                mean_iterations=total_iterations / iterations,
                max_solution_difference=max_difference, max_physical_violation=max_violation,
                max_objective_gap=max_objective_gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joints', nargs='+', type=int, default=[6, 19, 30])
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--storage', choices=['dense', 'sparse', 'both'], default='both')
    parser.add_argument('--unit-rows', action='store_true',
                        help='use physical A with unit row maxima before dynamic A changes')
    parser.add_argument('--variants', nargs='+', choices=['caller', 'caller-cached', 'solver'],
                        default=['caller', 'caller-cached', 'solver'])
    parser.add_argument('--output', type=Path,
                        default=Path('/tmp/wholebodyx-qp-row-normalization.json'))
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.warmup, *args.joints) < 1:
        parser.error('rounds, iterations, warmup and joints must be positive')
    package = Path(wholebodyx.__file__).resolve().parent
    paths = list(package.rglob('*.py')) + [Path(__file__).resolve()]
    hashes = source_hashes(ROOT, paths)
    report = dict(status='running', python=platform.python_version(), rounds=args.rounds,
                  iterations=args.iterations, warmup=args.warmup, joints=args.joints,
                  unit_rows=args.unit_rows,
                  variants=args.variants,
                  package=str(package),
                  source_sha256=hashes,
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  threads={key: os.environ.get(key) for key in
                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS')},
                  dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__),
                  results=[])
    started = time.monotonic()
    variants = tuple(args.variants)
    storage = (False, True) if args.storage == 'both' else (args.storage == 'sparse',)
    try:
        for index in range(args.rounds):
            # Rotate variant order, preserving identical inputs and fresh workspaces.
            shift = index % len(variants)
            order = variants[shift:] + variants[:shift]
            for n in args.joints:
                for compact in storage:
                    for mode in ('fixed', 'hessian', 'constraint', 'both'):
                        for variant in order:
                            row = measure(n, compact, mode, variant, args.iterations, args.warmup,
                                          args.unit_rows)
                            row['round'] = index + 1
                            report['results'].append(row)
                            print(json.dumps(row, allow_nan=False), flush=True)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(wall_s=time.monotonic() - started,
                      source_unchanged=sources_unchanged(ROOT, hashes))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
