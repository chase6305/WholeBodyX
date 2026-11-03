"""Measure solver isolation, separate from closed-loop tracking endurance.

Both paths solve identical fixed-base MPC inputs and execute a WBC hold task.
The synchronous path solves inside its control cycle; the process path polls
and submits inside cycles while a child solves. Result application/goal latency
is measured by soak_async_mpc instead. Startup and shutdown are excluded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from wholebodyx import JointLimits, KinematicWBC, ProcessMPCWorker, RobotState, VelocitySession
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import PostureTask
from soak_mpc import planner_type, check_plan

ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(samples=len(values), mean_ms=float(values.mean()),
                p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()),
                over_5ms=int(np.count_nonzero(values >= 5.)))


def inputs(joints, count, seed, start_round):
    rng = np.random.default_rng(seed)
    bounds = JointLimits(-rng.uniform(.7, 2, joints), rng.uniform(.7, 2, joints),
                         rng.uniform(.3, 1, joints), rng.uniform(2, 4, joints))
    work = []
    for index in range(start_round + count):
        q, v, goal = rng.uniform(-.25, .25, joints), rng.uniform(-.1, .1, joints), rng.uniform(-3., 3., joints)
        if index >= start_round:
            work.append((RobotState(q, v, index * .07), goal))
    return bounds, work


def make_session(bounds, backend, callback):
    if backend.startswith("native"):
        from wholebodyx.native import NativeWBC
        controller = NativeWBC(bounds)
    else:
        controller = KinematicWBC(bounds)
    n = bounds.lower.size
    model = SimpleNamespace(joint_names=tuple(f"joint_{i}" for i in range(n)), limits=bounds)
    driver = JointIntegrator(bounds, RobotState(np.zeros(n), np.zeros(n)), control_dt=.01)

    def tasks(model, state):
        callback()
        return [PostureTask(np.zeros(n)).linearize(model, state)]

    session = VelocitySession(model, driver, controller, .01, task_factory=tasks)
    session.reset()
    return session


def measure(name, bounds, work, horizon, warmup, interval):
    direct = planner_type(name)(bounds, horizon=horizon, dt=.1)
    sync_plans, sync_times = [], []
    index = 0

    def solve():
        nonlocal index
        result = direct.solve(*work[index])
        assert result.solver.success and result.plan is not None, (name, index, result.solver.status, result.solver.detail)
        sync_plans.append(result.plan)
        index += 1

    session = make_session(bounds, name, solve)
    for _ in work:
        step = session.step()
        np.testing.assert_array_equal(step.command.velocity, np.zeros(bounds.lower.size))
        sync_times.append(step.cycle.cycle_ms)
    index = 0
    replies, cycle_times, child_times = [], [], []
    max_qdiff = max_vdiff = max_violation = 0.
    with ProcessMPCWorker(bounds, backend=name, horizon=horizon, dt=.1) as worker:
        submitted = False

        def exchange():
            nonlocal index, submitted
            if worker.error:
                raise RuntimeError(worker.error)
            reply = worker.poll()
            if reply is not None:
                replies.append(reply)
                child_times.append(reply.solve_ms)
                index += 1
                submitted = False
            if not submitted and index < len(work):
                submitted = worker.submit(*work[index]) is not None

        session = make_session(bounds, name, exchange)
        deadline = time.monotonic() + max(60., 10 * len(work))
        due = time.monotonic()
        measured = False
        while index < len(work):
            if time.monotonic() > deadline:
                raise TimeoutError("benchmark process path timed out")
            measured = index >= warmup
            step = session.step()
            np.testing.assert_array_equal(step.command.velocity, np.zeros(bounds.lower.size))
            if measured:
                cycle_times.append(step.cycle.cycle_ms)
            due += interval
            now = time.monotonic()
            if due < now - interval:
                due = now
            if due > now:
                time.sleep(due - now)
        info, stats = worker.runtime_info, worker.statistics
    assert len(replies) == len(sync_plans) == len(work)
    assert worker._shutdown_complete
    for i, reply in enumerate(replies):
        assert reply.outcome.solver.success and reply.outcome.plan is not None
        plan = reply.outcome.plan
        check_plan(sync_plans[i], work[i][0], bounds, horizon, .1, work[i][1])
        violation, _ = check_plan(plan, work[i][0], bounds, horizon, .1, work[i][1])
        max_violation = max(max_violation, violation)
        max_qdiff = max(max_qdiff, float(np.max(np.abs(plan.positions - sync_plans[i].positions))))
        max_vdiff = max(max_vdiff, float(np.max(np.abs(plan.velocities - sync_plans[i].velocities))))
    assert max_qdiff <= 1e-5 and max_vdiff <= 3e-4
    return dict(synchronous_cycles=describe(sync_times[warmup:]), process_cycles=describe(cycle_times),
                child_solve=describe(child_times[warmup:]), max_plan_position_difference=max_qdiff,
                max_plan_velocity_difference=max_vdiff, max_physical_violation=max_violation,
                worker_runtime=info, worker_statistics=stats, all_workers_reaped=True)


def run(output, *, backends, joints=30, horizon=60, samples=24, warmup=2,
        seed=20261026, start_round=9850, interval=.005):
    if samples < 1 or warmup < 0 or joints < 1 or horizon < 1 or not np.isfinite(interval) or interval <= 0:
        raise ValueError("samples, dimensions and interval must be positive; warmup must be nonnegative")
    bounds, work = inputs(joints, samples + warmup, seed, start_round)
    digest = hashlib.sha256()
    for state, goal in work:
        for value in (state.q, state.v, np.array([state.stamp]), goal):
            digest.update(value.tobytes())
    sources = list((ROOT / "python/wholebodyx").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/soak_mpc.py"]
    if any(b.startswith("native") for b in backends):
        import wholebodyx_native
        sources.append(Path(wholebodyx_native.__file__).resolve())
    hashes = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(status="running", joints=joints, horizon=horizon, samples=samples, warmup=warmup,
                  seed=seed, start_round=start_round, process_control_period_s=interval,
                  input_sha256=digest.hexdigest(), source_sha256=hashes, backends={})
    started = time.monotonic()
    try:
        for name in backends:
            report["backends"][name] = measure(name, bounds, work, horizon, warmup, interval)
            print(json.dumps(dict(backend=name, synchronous=report["backends"][name]["synchronous_cycles"],
                                  process=report["backends"][name]["process_cycles"])), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(wall_s=time.monotonic() - started,
                      source_unchanged=all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--joints", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--backend", choices=("python", "native", "sparse", "native-sparse", "all"), default="all")
    parser.add_argument("--reverse", action="store_true", help="Reverse backend measurement order")
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-process-mpc-benchmark.json"))
    args = parser.parse_args()
    selected = ("python", "native", "sparse", "native-sparse") if args.backend == "all" else (args.backend,)
    if args.reverse:
        selected = tuple(reversed(selected))
    run(args.output, backends=selected, joints=args.joints, horizon=args.horizon,
        samples=args.samples, warmup=args.warmup)
