"""Compare full scaled QP construction and solve in isolated revisions.

Dense A and identical row bounds make the baseline's reversed row order an
identical feasible set in this benchmark. Every case has a constructed unique
optimum; correctness with unequal bounds is tested separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

from _comparison import comparison_report, revision_rows

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    from time import perf_counter
    import numpy as np
    import osqp
    import scipy
    import wholebodyx
    from wholebodyx import OSQPSolver, QuadraticProgram

    package = Path(wholebodyx.__file__).resolve().parent
    if package.parent != args.python_dir.resolve():
        raise RuntimeError(f'loaded {package}, expected package in {args.python_dir}')
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob('*.py')}
    results = []
    for n in args.joints:
        for mode in ('fixed', 'hessian', 'constraint', 'both'):
            rng = np.random.default_rng(20261024 + n)
            b = rng.normal(size=(n, n))
            base_h = b.T @ b / n + np.eye(n)
            expected0 = rng.uniform(.1, .3, n)
            base_a = rng.normal(size=(2 * n, n))
            base_a *= (.4 / np.maximum(1., np.abs(base_a @ expected0)))[:, None]
            base_a[0] = expected0 / (expected0 @ expected0)
            solver = OSQPSolver(variable_scaling=np.geomspace(.2, 3., n))
            times, max_difference, max_violation = [], 0., 0.
            for k in range(args.iterations + args.warmup):
                h_scale = 1 + .1 * np.sin(k) if mode in ('hessian', 'both') else 1.
                a_scale = 1 + .1 * np.cos(k) if mode in ('constraint', 'both') else 1.
                h, a, expected = h_scale * base_h, a_scale * base_a, expected0 / a_scale
                # One active upper constraint, all remaining constraints slack.
                gradient = -h @ expected - a[0]
                lower, upper = -np.ones(2 * n), np.ones(2 * n)
                started = perf_counter()
                problem = QuadraticProgram(h, gradient, a, lower, upper)
                result = solver.solve(problem)
                elapsed = (perf_counter() - started) * 1e6
                if not result.success:
                    raise RuntimeError(result)
                difference = float(np.max(np.abs(result.x - expected)))
                ax = a @ result.x
                violation = float(max(np.max(lower - ax), np.max(ax - upper), 0.))
                if difference > 3e-5 or violation > 1e-6:
                    raise RuntimeError((mode, difference, violation))
                max_difference = max(max_difference, difference)
                max_violation = max(max_violation, violation)
                if k >= args.warmup:
                    times.append(elapsed)
            results.append(dict(joints=n, mode=mode, p50_us=float(np.median(times)),
                                p95_us=float(np.percentile(times, 95)), mean_us=float(np.mean(times)),
                                max_us=max(times), max_solution_difference=max_difference,
                                max_physical_violation=max_violation))
    print(json.dumps(dict(results=results, package=str(package), source_sha256=hashes,
                         dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__),
                         source_unchanged=all(hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
                                              for name, digest in hashes.items())), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-python', type=Path)
    parser.add_argument('--candidate-python', type=Path, default=ROOT / 'python')
    parser.add_argument('--joints', nargs='+', type=int, default=[6, 19, 30])
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-scaled-qp-comparison.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--python-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.warmup, *args.joints) < 1:
        parser.error('rounds, iterations, warmup and joints must be positive')
    if args.worker:
        worker(args)
        return
    if args.baseline_python is None:
        parser.error('--baseline-python is required')
    builds = dict(baseline=args.baseline_python.resolve(strict=True),
                  candidate=args.candidate_python.resolve(strict=True))
    report = comparison_report(args, __file__)
    try:
        report['results'].extend(revision_rows(args, __file__, builds,
                                  ('build', 'round', 'joints', 'mode', 'p50_us', 'p95_us')))
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
