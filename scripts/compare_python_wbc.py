"""Compare complete WBC steps using alternating, isolated Python revisions.

Task construction and independent bound checks are outside each timed step.
Both fixed and changing coupled Jacobians use deterministic targets/states.
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
    from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState

    package = Path(wholebodyx.__file__).resolve().parent
    if package.parent != args.python_dir.resolve():
        raise RuntimeError(f'loaded {package}, expected package in {args.python_dir}')
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob('*.py')}
    results = []
    for n in args.joints:
        for changing in (False, True):
            bounds = JointLimits([-2.] * n, [2.] * n, [1.] * n)
            controller = KinematicWBC(bounds)
            base = np.eye(n) + .1 * np.ones((n, n)) / n
            times, velocities, max_violation = [], [], 0.
            for k in range(args.iterations + args.warmup):
                state = RobotState(np.full(n, .01 * np.sin(k)), np.zeros(n), k * .02)
                scale = 1 + .1 * np.sin(k) if changing else 1.
                tasks = [LinearTask(scale * base, np.full(n, .2 + .01 * np.cos(k)))]
                start = perf_counter()
                result = controller.step(state, tasks, .02)
                elapsed = (perf_counter() - start) * 1e6
                if result.command is None or not result.solver.success:
                    raise RuntimeError(result.solver)
                lo, hi = bounds.velocity_bounds(state, .02)
                v = result.command.velocity
                violation = max(float(np.max(lo - v)), float(np.max(v - hi)), 0.)
                if violation > 1e-6:
                    raise RuntimeError(f'invalid command: {violation}')
                max_violation = max(max_violation, violation, result.solver.violation)
                if k >= args.warmup:
                    times.append(elapsed)
                    velocities.append(v)
            results.append(dict(joints=n, changing_jacobian=changing,
                                p50_us=float(np.median(times)), p95_us=float(np.percentile(times, 95)),
                                mean_us=float(np.mean(times)), max_us=max(times),
                                velocity_sha256=hashlib.sha256(np.asarray(velocities).tobytes()).hexdigest(),
                                max_constraint_violation=max_violation))
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
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-wbc-comparison.json'))
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
                                  ('build', 'round', 'joints', 'changing_jacobian', 'p50_us', 'p95_us')))
        for n in args.joints:
            for changing in (False, True):
                digests = {r['velocity_sha256'] for r in report['results']
                           if r['joints'] == n and r['changing_jacobian'] == changing}
                if len(digests) != 1:
                    raise RuntimeError(f'command sequences differ for {n}, changing={changing}')
        report.update(status='passed', command_sequences_bitwise_equal=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
