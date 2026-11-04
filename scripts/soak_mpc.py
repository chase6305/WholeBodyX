"""Bounded-memory MPC stress with randomized bounds, states and soft targets.

This is a solver/plan endurance test, not a robot control-rate measurement.
Each round uses a fresh feasible measured state and compares full trajectories.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from wholebodyx import JointLimits, JointMPC, RobotState

from _measurement import rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


def planner_type(name):
    if name == "python":
        return JointMPC
    if name == "sparse":
        from wholebodyx.sparse_mpc import SparseJointMPC
        return SparseJointMPC
    from wholebodyx.native import NativeMPC, NativeSparseMPC
    return NativeMPC if name == "native" else NativeSparseMPC


def check_plan(plan, state, limits, horizon, dt, goal):
    assert plan.stamp == state.stamp and plan.dt == dt
    assert plan.positions.shape == (horizon + 1, state.q.size)
    assert plan.velocities.shape == (horizon, state.q.size)
    assert np.isfinite(plan.positions).all() and np.isfinite(plan.velocities).all()
    np.testing.assert_array_equal(plan.positions[0], state.q)
    np.testing.assert_allclose(np.diff(plan.positions, axis=0), dt * plan.velocities, atol=1e-8, rtol=1e-7)
    violations = [np.max(limits.lower - plan.positions[1:]),
                  np.max(plan.positions[1:] - limits.upper),
                  np.max(np.abs(plan.velocities) - limits.velocity), 0.]
    change = np.diff(np.vstack((state.v, plan.velocities)), axis=0)
    violations.append(np.max(np.abs(change) - dt * limits.acceleration))
    violation = float(max(violations))
    assert violation <= 1e-6, violation
    cost = .5 * (10 * np.sum((plan.positions[1:-1] - goal)**2)
                  + 30 * np.sum((plan.positions[-1] - goal)**2)
                  + .1 * np.sum(plan.velocities**2))
    assert np.isfinite(cost)
    return violation, float(cost)


def run(seconds, output, *, backends, joints=30, horizon=60, seed=20261021, start_round=0):
    if not np.isfinite(seconds) or seconds <= 0 or joints < 1 or horizon < 1:
        raise ValueError("duration, joints and horizon must be positive")
    if isinstance(start_round, bool) or not isinstance(start_round, (int, np.integer)) or start_round < 0:
        raise ValueError("start round must be a nonnegative integer")
    dt = .1
    rng = np.random.default_rng(seed)
    limits = JointLimits(-rng.uniform(.7, 2, joints), rng.uniform(.7, 2, joints),
                          rng.uniform(.3, 1, joints), rng.uniform(2, 4, joints))
    for _ in range(start_round):
        rng.uniform(-.25, .25, joints)
        rng.uniform(-.1, .1, joints)
        rng.uniform(-3., 3., joints)
    planners = {name: planner_type(name)(limits, horizon=horizon, dt=dt) for name in backends}
    sources = list((ROOT / "python/wholebodyx").glob("*.py")) + [Path(__file__).resolve()]
    sources += list((ROOT / "cpp").glob("*.*")) + list((ROOT / "include/wholebodyx").glob("*.hpp"))
    sources += [ROOT / "CMakeLists.txt", ROOT / "bindings/module.cpp"]
    if any(name.startswith("native") for name in backends):
        import wholebodyx_native
        sources.append(Path(wholebodyx_native.__file__).resolve())
    hashes = source_hashes(ROOT, sources)
    report = dict(status="running", requested_wall_s=seconds, joints=joints, horizon=horizon,
                  seed=seed, start_round=int(start_round), dt=dt, rounds=0, source_sha256=hashes,
                  initial_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  initial_rss_kib=rss_kib(), max_sampled_rss_kib=rss_kib(), rss_sample_period_s=30.,
                  parity_tolerances=dict(position=1e-5, velocity=3e-4, normalized_objective=1e-6),
                  max_position_difference=0., max_velocity_difference=0., max_normalized_cost_difference=0.,
                  backends={name: dict(solves=0, resets=0, rejected_infeasible=0,
                                       cold_retries=0, retry_samples=[], max_physical_violation=0., max_solver_violation=0.,
                                       mean_solve_ms=0., max_solve_ms=0.) for name in backends})
    started = heartbeat = time.monotonic()
    current = {}
    try:
        while time.monotonic() - started < seconds:
            index = start_round + report["rounds"]
            stamp = 1e6 if index % 499 == 498 else index * .07
            state = RobotState(rng.uniform(-.25, .25, joints), rng.uniform(-.1, .1, joints), stamp)
            goal = rng.uniform(-3., 3., joints)
            current = dict(round=index, q=state.q.tolist(), v=state.v.tolist(), stamp=stamp, goal=goal.tolist())
            results = {}
            order = backends if index % 2 == 0 else tuple(reversed(backends))
            for name in order:
                planner, stats = planners[name], report["backends"][name]
                if index and index % 197 == 0:
                    planner.reset()
                    stats["resets"] += 1
                if index and index % 101 == 0:
                    # The next position cannot reach its box even at max speed.
                    impossible = RobotState(limits.upper + 2 * dt * limits.velocity, np.zeros(joints), stamp)
                    rejected = planner.solve(impossible, np.zeros(joints))
                    assert rejected.plan is None and not rejected.solver.success
                    stats["rejected_infeasible"] += 1
                before = time.perf_counter()
                result = planner.solve(state, goal)
                elapsed = (time.perf_counter() - before) * 1000
                assert result.plan is not None, (name, result.solver.status, result.solver.detail)
                violation, cost = check_plan(result.plan, state, limits, horizon, dt, goal)
                if result.solver.detail.startswith("cold retry after "):
                    stats["cold_retries"] += 1
                    if len(stats["retry_samples"]) < 5:
                        stats["retry_samples"].append(dict(round=index, iterations=result.solver.iterations,
                                                           detail=result.solver.detail))
                stats["solves"] += 1
                stats["mean_solve_ms"] += (elapsed - stats["mean_solve_ms"]) / stats["solves"]
                stats["max_solve_ms"] = max(stats["max_solve_ms"], elapsed)
                stats["max_physical_violation"] = max(stats["max_physical_violation"], violation)
                stats["max_solver_violation"] = max(stats["max_solver_violation"], result.solver.violation)
                results[name] = result.plan, cost
            reference, reference_cost = results[backends[0]]
            for plan, cost in results.values():
                qdiff = float(np.max(np.abs(plan.positions - reference.positions)))
                vdiff = float(np.max(np.abs(plan.velocities - reference.velocities)))
                cdiff = abs(cost - reference_cost) / max(1., abs(reference_cost))
                report["max_position_difference"] = max(report["max_position_difference"], qdiff)
                report["max_velocity_difference"] = max(report["max_velocity_difference"], vdiff)
                report["max_normalized_cost_difference"] = max(report["max_normalized_cost_difference"], cdiff)
                assert qdiff <= 1e-5 and vdiff <= 3e-4 and cdiff <= 1e-6, (qdiff, vdiff, cdiff)
            report["rounds"] += 1
            if time.monotonic() - heartbeat >= 30:
                heartbeat = time.monotonic()
                rss = rss_kib()
                report["max_sampled_rss_kib"] = max(report["max_sampled_rss_kib"] or 0, rss or 0) or None
                print(json.dumps(dict(wall_s=heartbeat - started, rounds=report["rounds"], rss_kib=rss,
                                      max_position_difference=report["max_position_difference"],
                                      max_velocity_difference=report["max_velocity_difference"])), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_input=current)
        raise
    finally:
        report.update(wall_s=time.monotonic() - started, final_rss_kib=rss_kib(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_unchanged=sources_unchanged(ROOT, hashes))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600.)
    parser.add_argument("--backend", choices=("python", "native", "sparse", "native-sparse", "all"), default="all")
    parser.add_argument("--joints", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-mpc-soak.json"))
    args = parser.parse_args()
    selected = ("python", "native", "sparse", "native-sparse") if args.backend == "all" else (args.backend,)
    run(args.seconds, args.output, backends=selected, joints=args.joints, horizon=args.horizon, seed=args.seed, start_round=args.start_round)
