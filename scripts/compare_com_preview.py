"""Compare complete CoM planning calls across revisions on identical previews.

Run serially with one numerical-library thread. Accepted plans are independently
checked after timing; finite-budget rejections are retained separately. Shared
host latency observations are not execution-time guarantees.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

from _comparison import invoke_worker
from _measurement import rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


def _stationarity(plan, targets, velocities):
    """Differentiate the physical rollout backwards, without the QP matrices."""
    import numpy as np

    dt = plan.dt
    transition = np.array([[1., dt, .5*dt**2], [0., 1., dt], [0., 0., 1.]])
    injection = np.array([dt**3/6, .5*dt**2, dt])
    adjoint = np.zeros((3, 2))
    largest = 0.
    for k in reversed(range(len(plan.jerks))):
        adjoint[0] += plan.states[k+1, 0] - targets[k]
        adjoint[1] += .01*(plan.states[k+1, 1] - velocities[k])
        derivative = 1e-5*plan.jerks[k] + injection @ adjoint
        largest = max(largest, float(np.max(np.abs(derivative))))
        adjoint = transition.T @ adjoint
    return largest


def worker(args):
    import inspect
    import numpy as np
    import scipy
    import osqp
    import wholebodyx
    from wholebodyx.centroidal import CoMPreviewMPC

    package = Path(wholebodyx.__file__).resolve().parent
    if package.parent != args.python_dir.resolve():
        raise RuntimeError(f'loaded {package}, expected package in {args.python_dir}')
    paths = list(package.rglob('*.py')) + [Path(__file__).resolve(),
             ROOT/'scripts/_comparison.py', ROOT/'scripts/_measurement.py']
    hashes = source_hashes(ROOT, paths)
    report = dict(status='running', package=str(package), source_sha256=hashes,
                  python=platform.python_version(), results=[],
                  dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__),
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  threads={name: os.environ.get(name) for name in
                           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    rotation = np.array([[np.cos(.25), -np.sin(.25)], [np.sin(.25), np.cos(.25)]])
    support = np.array([[-.12, -.09], [.12, -.09], [.12, .09], [-.12, .09]]) @ rotation.T
    initial = ([0., 0.], [.01, -.02], [.005, .01])
    try:
        for n in args.horizons:
            for mode in args.modes:
                options = {}
                if args.disable_fast_path and 'unconstrained_fast_path' in inspect.signature(CoMPreviewMPC).parameters:
                    options['unconstrained_fast_path'] = False
                began = time.perf_counter()
                planner = CoMPreviewMPC(support, .8, horizon=n, **options)
                initialization_ms = (time.perf_counter() - began)*1000
                wall, cpu, accepted_wall = [], [], []
                rejections, accepted, warmup_rejected = [], 0, 0
                direct = 0 if hasattr(planner, '_direct_solves') else None
                max_violation = max_stationarity = max_objective = 0.
                for k in range(args.warmup + args.iterations):
                    phase = k*.03 + np.arange(n)*.1
                    active = mode == 'active' or (mode == 'mixed' and (k//5) % 2 == 1)
                    if active:
                        center = [.1, .07] if n >= 6 else [.6, .4]
                        targets = np.tile(center, (n, 1)) + .001*np.column_stack((np.sin(phase), np.cos(phase)))
                        velocities = np.zeros((n, 2))
                    else:
                        targets = np.column_stack((.015*np.sin(phase), .01*np.cos(phase)))
                        velocities = np.column_stack((.015*np.cos(phase), -.01*np.sin(phase)))
                    before_direct = getattr(planner, '_direct_solves', 0)
                    start_w, start_c = time.perf_counter_ns(), time.thread_time_ns()
                    try:
                        plan = planner.plan(*initial, targets, stamp=k*.01, target_velocities=velocities)
                    except RuntimeError as exc:
                        elapsed_w = (time.perf_counter_ns()-start_w)/1000.
                        elapsed_c = (time.thread_time_ns()-start_c)/1000.
                        if not str(exc).startswith('CoM MPC failed:'):
                            raise
                        if k >= args.warmup:
                            rejections.append(dict(attempt=k, detail=str(exc), wall_us=elapsed_w))
                        else:
                            warmup_rejected += 1
                    else:
                        elapsed_c = (time.thread_time_ns()-start_c)/1000.
                        elapsed_w = (time.perf_counter_ns()-start_w)/1000.
                        violation = planner.validate_plan(plan)
                        if not np.isfinite(violation) or not 0 <= violation <= 1e-6:
                            raise AssertionError((n, mode, k, violation))
                        used_direct = getattr(planner, '_direct_solves', 0) > before_direct
                        derivative = _stationarity(plan, targets, velocities) if used_direct or not active else 0.
                        if used_direct and derivative > 1e-8:
                            raise AssertionError(('direct stationarity', n, mode, k, derivative))
                        objective = (.5*np.sum((plan.states[1:, 0]-targets)**2)
                                     + .005*np.sum((plan.states[1:, 1]-velocities)**2)
                                     + .5e-5*np.sum(plan.jerks**2))
                        max_violation, max_stationarity = max(max_violation, violation), max(max_stationarity, derivative)
                        max_objective = max(max_objective, float(objective))
                        if k >= args.warmup:
                            accepted += 1
                            accepted_wall.append(elapsed_w)
                            if direct is not None:
                                direct += int(used_direct)
                    if k == 0:
                        first_call_us = elapsed_w
                    if k >= args.warmup:
                        wall.append(elapsed_w)
                        cpu.append(elapsed_c)
                factor = getattr(planner, '_unconstrained_factor', None)
                report['results'].append(dict(horizon=n, mode=mode, initialization_ms=initialization_ms,
                    first_call_us=first_call_us, accepted=accepted, rejected=len(rejections),
                    warmup_rejected=warmup_rejected, rejection_samples=rejections[:10], direct_accepts=direct,
                    wall_p50_us=float(np.median(wall)), wall_p95_us=float(np.percentile(wall, 95)), wall_max_us=max(wall),
                    cpu_p50_us=float(np.median(cpu)), cpu_p95_us=float(np.percentile(cpu, 95)),
                    accepted_wall_p50_us=float(np.median(accepted_wall)) if accepted_wall else None,
                    max_physical_violation=max_violation, max_free_stationarity=max_stationarity,
                    max_objective=max_objective, fast_path=getattr(planner, 'unconstrained_fast_path', False),
                    fixed_qp_bytes=planner.problem._hessian.nbytes + planner.problem._constraint.nbytes,
                    factor_bytes=0 if factor is None else factor.nbytes))
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(source_unchanged=sources_unchanged(ROOT, hashes), final_rss_kib=rss_kib())
        print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-python', type=Path)
    parser.add_argument('--candidate-python', type=Path, default=ROOT/'python')
    parser.add_argument('--horizons', nargs='+', type=int, default=[20, 60, 100])
    parser.add_argument('--modes', nargs='+', choices=['free', 'active', 'mixed'],
                        default=['free', 'active', 'mixed'])
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--disable-fast-path', action='store_true',
                        help='disable the shortcut in revisions that expose the option')
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-com-preview.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--python-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.warmup) < 1 or not all(2 <= n <= 100 for n in args.horizons):
        parser.error('rounds, iterations and warmup must be positive; horizons must be in [2, 100]')
    if args.worker:
        if args.python_dir is None:
            parser.error('--python-dir is required for a worker')
        worker(args)
        return
    if args.baseline_python is None:
        parser.error('--baseline-python is required')
    builds = dict(baseline=args.baseline_python.resolve(strict=True),
                  candidate=args.candidate_python.resolve(strict=True))
    paths = [Path(__file__).resolve(), ROOT/'scripts/_comparison.py', ROOT/'scripts/_measurement.py']
    hashes = source_hashes(ROOT, paths)
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  rounds=args.rounds, iterations=args.iterations, warmup=args.warmup,
                  horizons=args.horizons, modes=args.modes, source_sha256=hashes,
                  disabled_fast_path=args.disable_fast_path, results=[], workers=[])
    started = time.monotonic()
    try:
        for index in range(args.rounds):
            order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
            for name in order:
                arguments = ['--python-dir', str(builds[name]), '--horizons', *map(str, args.horizons),
                             '--modes', *args.modes, '--iterations', str(args.iterations),
                             '--warmup', str(args.warmup)]
                if args.disable_fast_path:
                    arguments.append('--disable-fast-path')
                data = invoke_worker(__file__, builds[name], arguments)
                if data['status'] != 'completed' or not data['source_unchanged']:
                    raise RuntimeError('worker failed or its sources changed')
                for row in data.pop('results'):
                    row.update(build=name, round=index+1, worker=len(report['workers']))
                    report['results'].append(row)
                    print(json.dumps({key: row[key] for key in
                          ('build', 'round', 'horizon', 'mode', 'wall_p50_us', 'cpu_p50_us',
                           'accepted', 'rejected', 'direct_accepts')}), flush=True)
                data.update(build=name, round=index+1)
                report['workers'].append(data)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(wall_s=time.monotonic()-started, source_unchanged=sources_unchanged(ROOT, hashes))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
