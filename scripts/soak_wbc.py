"""Bounded-memory Python/native WBC parity under changing coupled tasks.

Each case integrates Python commands through independent physical box checks.
Wall duration measures endurance, not a control-rate or hard-real-time bound.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState
from wholebodyx.native import NativeWBC

from _measurement import rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


class Case:
    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed + n)
        self.n, self.dt = n, .01
        self.limits = JointLimits([-1.] * n, [1.] * n, [.7] * n, [2.] * n)
        self.python = KinematicWBC(self.limits, regularization=.01, smoothing=.02)
        self.native = NativeWBC(self.limits, regularization=.01, smoothing=.02)
        self.state = RobotState(np.full(n, .8), np.zeros(n))
        self.report = dict(steps=0, resets=0, rejected_infeasible=0, metadata_mutations=0,
                           max_velocity_difference=0., max_normalized_objective_difference=0.,
                           max_physical_violation=0., max_solver_violation=0., max_abs_position=0.,
                           python_mean_ms=0., native_mean_ms=0., python_max_ms=0., native_max_ms=0.)

    def step(self):
        r, n, state = self.report, self.n, self.state
        k, dt = r['steps'], self.dt
        jac = self.rng.normal(size=(max(1, n // 2), n))
        task = LinearTask(jac, self.rng.normal(size=jac.shape[0]), self.rng.uniform(.1, 2., jac.shape[0]))
        if k and k % 197 == 0:
            self.python.reset()
            self.native.reset()
            r['resets'] += 1
        if k and k % 101 == 0:
            impossible = RobotState(self.limits.upper + 2 * dt * self.limits.velocity, np.zeros(n), state.stamp)
            for controller in (self.python, self.native):
                rejected = controller.step(impossible, [task], dt)
                assert rejected.command is None and not rejected.solver.success
            r['rejected_infeasible'] += 1
        if k and k % 307 == 0:
            p = self.python.formulate(state, [task], dt)
            p.hessian.shape = (n * n,)
            p.constraint.dtype = np.int64
            p.lower.setflags(write=True)
            p.lower[:] = np.nan
            r['metadata_mutations'] += 1
        results = {}
        order = ('python', 'native') if k % 2 == 0 else ('native', 'python')
        for name in order:
            before = time.perf_counter()
            result = getattr(self, name).step(state, [task], dt)
            elapsed = (time.perf_counter() - before) * 1000
            assert result.command is not None and result.solver.success, (name, result.solver)
            r[name + '_mean_ms'] += (elapsed - r[name + '_mean_ms']) / (k + 1)
            r[name + '_max_ms'] = max(r[name + '_max_ms'], elapsed)
            r['max_solver_violation'] = max(r['max_solver_violation'], result.solver.violation)
            results[name] = result.command.velocity
        lo, hi = self.limits.velocity_bounds(state, dt)
        costs = {}
        for name, velocity in results.items():
            q = state.q + dt * velocity
            violation = max(float(np.max(lo - velocity)), float(np.max(velocity - hi)),
                            float(np.max(self.limits.lower - q)), float(np.max(q - self.limits.upper)),
                            float(np.max(np.abs(velocity) - self.limits.velocity)),
                            float(np.max(np.abs(velocity - state.v) - dt * self.limits.acceleration)), 0.)
            assert violation <= 1e-6, (name, violation)
            r['max_physical_violation'] = max(r['max_physical_violation'], violation)
            residual = jac @ velocity - task.target_velocity
            costs[name] = .5 * (np.dot(task.weight * residual, residual)
                                + .01 * np.dot(velocity, velocity)
                                + .02 * np.sum((velocity - state.v)**2))
        difference = float(np.max(np.abs(results['python'] - results['native'])))
        cost_difference = abs(costs['python'] - costs['native']) / (1 + abs(costs['python']))
        assert difference <= 3e-5 and cost_difference <= 1e-6, (difference, cost_difference)
        r['max_velocity_difference'] = max(r['max_velocity_difference'], difference)
        r['max_normalized_objective_difference'] = max(r['max_normalized_objective_difference'], cost_difference)
        self.state = RobotState(state.q + dt * results['python'], results['python'], state.stamp + dt)
        r['max_abs_position'] = max(r['max_abs_position'], float(np.max(np.abs(self.state.q))))
        r['steps'] += 1


def run(seconds, output, joints, seed):
    if not np.isfinite(seconds) or seconds <= 0 or not joints or min(joints) < 1:
        raise ValueError('duration and joint counts must be positive')
    import wholebodyx_native
    paths = list((ROOT / 'python/wholebodyx').rglob('*.py')) + [Path(__file__).resolve()]
    paths += list((ROOT / 'cpp').glob('*.*')) + list((ROOT / 'include/wholebodyx').glob('*.hpp'))
    paths += [ROOT / 'bindings/module.cpp', ROOT / 'CMakeLists.txt', Path(wholebodyx_native.__file__).resolve()]
    hashes = source_hashes(ROOT, paths)
    cases = [Case(n, seed) for n in joints]
    report = dict(status='running', requested_wall_s=seconds, seed=seed, joints=joints,
                  dt=.01, steps=0, initial_rss_kib=rss_kib(), max_sampled_rss_kib=rss_kib(),
                  rss_sample_period_s=30., source_sha256=hashes, cases={})
    started = heartbeat = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            cases[report['steps'] % len(cases)].step()
            report['steps'] += 1
            now = time.monotonic()
            if now - heartbeat >= 30:
                heartbeat = now
                rss = rss_kib()
                report['max_sampled_rss_kib'] = max(report['max_sampled_rss_kib'] or 0, rss or 0)
                print(json.dumps(dict(wall_s=now - started, steps=report['steps'], current_rss_kib=rss,
                                      max_velocity_difference=max(c.report['max_velocity_difference'] for c in cases))), flush=True)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(wall_s=time.monotonic() - started, final_rss_kib=rss_kib(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_unchanged=sources_unchanged(ROOT, hashes),
                      cases={str(case.n): dict(case.report, simulation_s=case.state.stamp) for case in cases})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--joints', nargs='+', type=int, default=[1, 6, 19, 30])
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-wbc-soak.json'))
    args = parser.parse_args()
    run(args.seconds, args.output, args.joints, args.seed)
