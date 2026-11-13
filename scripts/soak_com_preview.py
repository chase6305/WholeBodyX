"""Bounded-memory changing CoM previews with independent physical witnesses."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import wholebodyx
from wholebodyx.centroidal import CoMPlan, CoMPreviewMPC

from compare_com_preview import _stationarity
from _measurement import rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


class Case:
    def __init__(self, horizon, mode, seed=20261112, planner_factory=CoMPreviewMPC):
        if mode not in ('free', 'active', 'mixed'):
            raise ValueError('unknown preview workload')
        self.mode = mode
        self.rng = np.random.default_rng(seed + 100*horizon + ('free', 'active', 'mixed').index(mode))
        rotation = np.array([[np.cos(.25), -np.sin(.25)], [np.sin(.25), np.cos(.25)]])
        support = np.array([[-.12, -.09], [.12, -.09], [.12, .09], [-.12, .09]]) @ rotation.T
        self.planner = planner_factory(support, .8, horizon=horizon)
        self.report = dict(horizon=horizon, mode=mode, seed=seed, planner=planner_factory.__name__,
            attempts=0, accepted=0, direct_accepts=0, backend_calls=0, resets=0,
            injected_infeasible=0, rejected_initial=0, cold_recoveries=0, rejected_cold=0,
            witnesses=0, max_physical_violation=0., max_direct_stationarity=0.,
            max_objective=0., max_plan_attempt_ms=0., rejection_samples=[])

    def _witness(self, initial):
        """Two jerk pulses stop the small supplied state, then hold equilibrium."""
        planner, n = self.planner, self.planner.horizon
        dt = planner.dt
        jerk = np.zeros((n, 2))
        jerk[0] = -initial[1]/dt**2 - 1.5*initial[2]/dt
        jerk[1] = initial[1]/dt**2 + .5*initial[2]/dt
        states = np.empty((n+1, 3, 2))
        states[0] = initial
        for k in range(2):
            p, v, a = states[k]
            states[k+1] = [p+dt*v+.5*dt**2*a+dt**3/6*jerk[k],
                            v+dt*a+.5*dt**2*jerk[k], a+dt*jerk[k]]
        states[2:, 0] = states[2, 0]
        states[2:, 1:] = 0.
        witness = CoMPlan(states, jerk, 0., dt, planner.height, planner.gravity)
        assert planner.validate_plan(witness) <= 1e-12
        self.report['witnesses'] += 1

    def _call(self, initial, targets, velocities, stamp):
        planner, r = self.planner, self.report
        before_direct = getattr(planner, '_direct_solves', 0)
        before_backend = getattr(planner, '_qp_solves', 0)
        try:
            plan = planner.plan(*initial, targets, stamp=stamp, target_velocities=velocities)
        finally:
            r['direct_accepts'] += max(0, getattr(planner, '_direct_solves', 0)-before_direct)
            r['backend_calls'] += max(0, getattr(planner, '_qp_solves', 0)-before_backend)
        used_direct = getattr(planner, '_direct_solves', 0) > before_direct
        violation = planner.validate_plan(plan)
        assert np.isfinite(violation) and 0 <= violation <= 1e-6, violation
        if used_direct:
            gradient = _stationarity(plan, targets, velocities)
            assert gradient <= 1e-8, gradient
            r['max_direct_stationarity'] = max(r['max_direct_stationarity'], gradient)
        objective = (.5*np.sum((plan.states[1:, 0]-targets)**2)
                     + .005*np.sum((plan.states[1:, 1]-velocities)**2)
                     + .5e-5*np.sum(plan.jerks**2))
        r['max_objective'] = max(r['max_objective'], float(objective))
        r['max_physical_violation'] = max(r['max_physical_violation'], violation)
        return plan

    def step(self):
        planner, rng, r = self.planner, self.rng, self.report
        k, n = r['attempts'], planner.horizon
        initial = rng.uniform(-.01, .01, (3, 2))
        phase = k*.03 + np.arange(n)*.1
        active = self.mode == 'active' or (self.mode == 'mixed' and (k//8) % 2 == 1)
        if active:
            signs = rng.choice([-1., 1.], 2)
            targets = np.tile(signs*[.1, .07], (n, 1)) + .001*np.column_stack((np.sin(phase), np.cos(phase)))
            velocities = np.zeros((n, 2))
        else:
            targets = initial[0] + np.column_stack((.01*np.sin(phase), .01*np.cos(phase)))
            velocities = np.column_stack((.01*np.cos(phase), -.01*np.sin(phase)))
        self.current_input = (initial, targets, velocities)
        self._witness(initial)
        if k and k % 197 == 0:
            try:
                planner.plan([2., 2.], [0., 0.], [0., 0.], targets)
            except RuntimeError:
                assert planner.last_plan is None
            else:
                raise AssertionError('infeasible measured state was accepted')
            r['injected_infeasible'] += 1
            planner.reset()
            r['resets'] += 1
        elif k and k % 151 == 0:
            planner.reset()
            r['resets'] += 1
        began = time.perf_counter()
        r['attempts'] += 1
        sample = None
        for attempt in range(2):
            try:
                self._call(initial, targets, velocities, k*.01)
            except RuntimeError as exc:
                if not str(exc).startswith('CoM MPC failed:'):
                    raise
                assert planner.last_plan is None
                if attempt == 0:
                    r['rejected_initial'] += 1
                    sample = dict(attempt=k, detail=str(exc))
                else:
                    r['rejected_cold'] += 1
                    sample.update(cold_detail=str(exc), initial=initial.tolist(),
                                  targets=targets.tolist(), target_velocities=velocities.tolist())
                planner.reset()
                r['resets'] += 1
            else:
                r['accepted'] += 1
                if attempt:
                    r['cold_recoveries'] += 1
                    sample['cold_recovered'] = True
                break
        if sample is not None and len(r['rejection_samples']) < 5:
            r['rejection_samples'].append(sample)
        r['max_plan_attempt_ms'] = max(r['max_plan_attempt_ms'], (time.perf_counter()-began)*1000)


def run(seconds, output, horizons=(20, 60, 100), modes=('free', 'mixed'), seed=20261112):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('duration must be finite and positive')
    cases = {f'{n}-{mode}': Case(n, mode, seed) for n in horizons for mode in modes}
    sequence = list(cases.values())
    if not sequence:
        raise ValueError('at least one workload is required')
    package = Path(wholebodyx.__file__).resolve().parent
    paths = list(package.rglob('*.py')) + [Path(__file__).resolve(), ROOT/'scripts/compare_com_preview.py',
             ROOT/'scripts/_measurement.py', ROOT/'scripts/_comparison.py']
    hashes = source_hashes(ROOT, paths)
    report = dict(status='running', requested_wall_s=seconds, seed=seed,
                  source_sha256=hashes, package=str(package), initial_rss_kib=rss_kib(),
                  max_sampled_rss_kib=rss_kib(), attempts=0)
    began = heartbeat = time.monotonic()
    began_cpu = time.thread_time()
    current = None
    try:
        while time.monotonic()-began < seconds:
            current = sequence[report['attempts'] % len(sequence)]
            current.step()
            report['attempts'] += 1
            if time.monotonic()-heartbeat >= 30:
                heartbeat = time.monotonic()
                report['max_sampled_rss_kib'] = max(report['max_sampled_rss_kib'] or 0, rss_kib() or 0)
                print(json.dumps(dict(wall_s=heartbeat-began, attempts=report['attempts'],
                                      current_rss_kib=rss_kib())), flush=True)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if current is not None and hasattr(current, 'current_input'):
            initial, targets, velocities = current.current_input
            report['failure_input'] = dict(horizon=current.planner.horizon, mode=current.mode,
                initial=initial.tolist(), targets=targets.tolist(), target_velocities=velocities.tolist())
        raise
    finally:
        report.update(wall_s=time.monotonic()-began, thread_s=time.thread_time()-began_cpu,
                      cases={name: value.report for name, value in cases.items()},
                      final_rss_kib=rss_kib(), source_unchanged=sources_unchanged(ROOT, hashes))
        report['accepted'] = sum(case.report['accepted'] for case in sequence)
        report['rejected_cold'] = sum(case.report['rejected_cold'] for case in sequence)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'], attempts=report['attempts'], accepted=report['accepted'],
                              rejected_cold=report['rejected_cold'], wall_s=report['wall_s'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--horizons', nargs='+', type=int, default=[20, 60, 100])
    parser.add_argument('--modes', nargs='+', choices=['free', 'active', 'mixed'], default=['free', 'mixed'])
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-com-preview-soak.json'))
    args = parser.parse_args()
    run(args.seconds, args.output, args.horizons, args.modes, args.seed)
