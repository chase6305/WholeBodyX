"""Exercise changing scaled QPs with independently constructed KKT optima.

Positive definite H and feasible x with complementary signed multipliers give
g = -H*x - A.T*y. Thus x is the unique optimum, without a second QP solver.
Reports retain bounded aggregate state and sample RSS every 30 seconds.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram

from _measurement import rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


class Case:
    def __init__(self, n, compact, seed, tolerance=1e-8, normalize_rows=False):
        self.n, self.compact = n, compact
        self.rng = np.random.default_rng(seed + n + 100 * compact)
        self.normalize_rows = normalize_rows
        self.solver = OSQPSolver(variable_scaling=np.geomspace(.2, 3., n),
                                 eps_abs=tolerance, eps_rel=tolerance,
                                 eps_prim_inf=tolerance, eps_dual_inf=tolerance)
        self.h = self.a = None
        self.report = dict(attempts=0, solves=0, resets=0, rejected_infeasible=0,
                           rejected_initial=0, cold_recoveries=0, rejected_cold=0,
                           total_iterations=0, cold_iterations=0, max_attempt_ms=0.,
                           rejection_samples=[],
                           max_solution_difference=0., max_objective_gap=0.,
                           max_physical_violation=0., max_kkt_residual=0.,
                           mean_solve_ms=0., max_solve_ms=0.)

    def step(self):
        r, rng, n = self.report, self.rng, self.n
        k, m = r['attempts'], 2 * n
        # Rotate fixed matrices, H-only, A-only and simultaneous updates.
        mode = (k // 25) % 4
        if self.h is None or mode in (1, 3):
            b = rng.normal(size=(n, n)) / np.sqrt(n)
            self.h = b.T @ b + np.eye(n)
        if self.a is None or mode in (2, 3):
            self.a = rng.normal(size=(m, n))
            if k % 13 == 0:
                self.a[rng.random((m, n)) < .35] = 0.
        h, a = self.h, self.a
        expected = rng.uniform(-.5, .5, n)
        ax = a @ expected
        lower, upper = ax - rng.uniform(.1, 2., m), ax + rng.uniform(.1, 2., m)
        dual = np.zeros(m)
        active = rng.choice(m, size=max(1, n // 2), replace=False)
        signs = rng.choice([-1., 1.], active.size)
        dual[active] = signs * rng.uniform(.2, 1., active.size)
        lower[active[signs < 0]] = ax[active[signs < 0]]
        upper[active[signs > 0]] = ax[active[signs > 0]]
        gradient = -h @ expected - a.T @ dual
        kkt = float(np.max(np.abs(h @ expected + gradient + a.T @ dual)))
        assert kkt <= 1e-10
        constructor = SparseQuadraticProgram if self.compact else QuadraticProgram
        solver_a, solver_lower, solver_upper = a, lower, upper
        if self.normalize_rows:
            # Positive row scaling preserves the feasible set. A residual in
            # a tiny row need not imply a small distance to the optimum; retain
            # original A/bounds below for an independent physical check.
            row_norm = np.max(np.abs(a), axis=1)
            row_norm[row_norm == 0] = 1.
            solver_a = a / row_norm[:, None]
            solver_lower, solver_upper = lower / row_norm, upper / row_norm
        p = constructor(sparse.csc_matrix(h) if self.compact else h, gradient,
                        sparse.csc_matrix(solver_a) if self.compact else solver_a,
                        solver_lower, solver_upper)
        if k and k % 197 == 0:
            self.solver.reset()
            r['resets'] += 1
        if k and k % 101 == 0:
            bad = constructor(p.hessian, gradient, p.constraint, p.upper + 1., p.lower)
            result = self.solver.solve(bad)
            assert not result.success and result.x is None
            r['rejected_infeasible'] += 1
        before = time.perf_counter()
        result = self.solver.solve(p)
        r['total_iterations'] += result.iterations
        r['attempts'] += 1
        if not result.success:
            # Finite-budget ADMM may reject a feasible problem. Exercise an
            # explicit caller reset, while recording every rejected attempt.
            # Never treat an inaccurate or physically invalid x as a solution.
            assert result.x is None
            assert (result.status == 'iteration_limit' or
                    result.detail.lower() == 'solved inaccurate'), result
            r['rejected_initial'] += 1
            sample = dict(attempt=k, status=result.status, detail=result.detail,
                          iterations=result.iterations)
            self.solver.reset()
            result = self.solver.solve(p)
            r['total_iterations'] += result.iterations
            r['cold_iterations'] += result.iterations
            sample.update(cold_status=result.status, cold_iterations=result.iterations)
            if len(r['rejection_samples']) < 5:
                r['rejection_samples'].append(sample)
            if result.success:
                r['cold_recoveries'] += 1
            else:
                assert result.x is None
                assert (result.status == 'iteration_limit' or
                        result.detail.lower() == 'solved inaccurate'), result
                r['rejected_cold'] += 1
                self.solver.reset()
                r['max_attempt_ms'] = max(r['max_attempt_ms'], (time.perf_counter() - before) * 1000)
                return
        elapsed = (time.perf_counter() - before) * 1000
        r['max_attempt_ms'] = max(r['max_attempt_ms'], elapsed)
        assert result.success, result
        x = result.x
        difference = float(np.max(np.abs(x - expected)))
        objective = lambda v: .5 * v @ h @ v + gradient @ v
        gap = abs(objective(x) - objective(expected)) / (1 + abs(objective(expected)))
        actual_ax = a @ x
        violation = float(max(np.max(lower - actual_ax), np.max(actual_ax - upper), 0.))
        assert difference <= 3e-5 and gap <= 1e-6 and violation <= 1e-6, (difference, gap, violation)
        r['solves'] += 1
        r['max_solution_difference'] = max(r['max_solution_difference'], difference)
        r['max_objective_gap'] = max(r['max_objective_gap'], float(gap))
        r['max_physical_violation'] = max(r['max_physical_violation'], violation)
        r['max_kkt_residual'] = max(r['max_kkt_residual'], kkt)
        r['mean_solve_ms'] += (elapsed - r['mean_solve_ms']) / r['solves']
        r['max_solve_ms'] = max(r['max_solve_ms'], elapsed)


def run(seconds, output, seed, tolerance=1e-8, normalize_rows=False):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('duration must be finite and positive')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    paths = list((ROOT / 'python/wholebodyx').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = source_hashes(ROOT, paths)
    cases = {f'{n}-{"sparse" if compact else "dense"}': Case(n, compact, seed, tolerance, normalize_rows)
             for n in (3, 6, 19, 30) for compact in (False, True)}
    sequence = list(cases.values())
    report = dict(status='running', requested_wall_s=seconds, seed=seed, attempts=0,
                  tolerance=tolerance, normalized_rows=normalize_rows,
                  initial_rss_kib=rss_kib(), max_sampled_rss_kib=rss_kib(),
                  rss_sample_period_s=30., source_sha256=hashes)
    started = heartbeat = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            sequence[report['attempts'] % len(sequence)].step()
            report['attempts'] += 1
            now = time.monotonic()
            if now - heartbeat >= 30:
                heartbeat = now
                rss = rss_kib()
                report['max_sampled_rss_kib'] = max(report['max_sampled_rss_kib'] or 0, rss or 0)
                print(json.dumps(dict(wall_s=now - started, attempts=report['attempts'],
                                      solves=sum(c.report['solves'] for c in sequence), rss_kib=rss,
                                      rejected_initial=sum(c.report['rejected_initial'] for c in sequence),
                                      rejected_cold=sum(c.report['rejected_cold'] for c in sequence),
                                      max_solution_difference=max(c.report['max_solution_difference']
                                                                  for c in sequence))), flush=True)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(wall_s=time.monotonic() - started, final_rss_kib=rss_kib(),
                      solves=sum(c.report['solves'] for c in sequence),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_unchanged=sources_unchanged(ROOT, hashes),
                      cases={name: case.report for name, case in cases.items()})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600.)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--normalize-rows', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/wholebodyx-scaled-qp-soak.json'))
    args = parser.parse_args()
    run(args.seconds, args.output, args.seed, args.tolerance, args.normalize_rows)
