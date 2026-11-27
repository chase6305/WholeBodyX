"""Measure control-thread receipt and adoption of real process MPC replies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep, thread_time_ns

import numpy as np

from wholebodyx import AsyncMPCReferenceManager, JointLimits, JointPlan, ProcessMPCWorker, RobotState
from soak_mpc import check_plan


def _quantiles(samples):
    return dict(zip(("p50", "p95", "p99", "max"),
                    (np.percentile(samples, (50, 95, 99, 100)) / 1_000).tolist()))


def measure(backend, joints, horizon, samples, warmup):
    if joints < 1 or horizon < 10 or samples < 1 or warmup < 0:
        raise ValueError("require positive joints and samples, horizon >= 10 and nonnegative warmup")
    dt, control_dt = .02, .01
    zero = np.zeros(joints)
    goal = .02 * np.cos(np.arange(joints) * .31)
    limits = JointLimits(zero - 1., zero + 1., zero + 1., zero + 3.)
    state = RobotState(zero, zero)
    seed = JointPlan(np.zeros((horizon + 1, joints)), np.zeros((horizon, joints)), 0., dt)
    adoption_wall, adoption_cpu, reuse_cpu = [], [], []
    violations = []
    trace = hashlib.sha256()
    deferred_polls = 0
    with ProcessMPCWorker(limits, backend=backend, horizon=horizon, dt=dt) as worker:
        manager = AsyncMPCReferenceManager(worker, lookahead=.04, max_adoption_delay=0.)
        manager.seed(seed, state, goal, control_dt)
        for index in range(samples + warmup):
            deadline = monotonic() + 10.
            while manager._pending is None:
                update = manager.update(state, goal, control_dt)
                if not update.success or monotonic() >= deadline:
                    raise RuntimeError(f"could not submit plan request: {update.reason}")
                if manager._pending is None:
                    sleep(.0005)
            request = manager._pending
            # Wait outside the timed call and adopt at the first eligible
            # tick, using the reference that produced the forecast.
            while worker._reply is None:
                if worker.error is not None or monotonic() >= deadline:
                    raise RuntimeError(worker.error or "planning reply timed out")
                sleep(.0005)
            stamp = request.nominal_start
            position, _ = manager.plan.sample(stamp)
            state = RobotState(position, request.state.v, stamp)
            while True:
                wall_start, cpu_start = perf_counter_ns(), thread_time_ns()
                update = manager.update(state, goal, control_dt)
                cpu_elapsed = thread_time_ns() - cpu_start
                wall_elapsed = perf_counter_ns() - wall_start
                if update.replanned:
                    break
                if not update.success or monotonic() >= deadline:
                    raise RuntimeError(f"could not adopt plan: {update.reason}")
                deferred_polls += 1
                sleep(.0005)
            plan = manager.plan
            violation, _ = check_plan(plan, request.state, limits, horizon, dt, goal)
            position, velocity = plan.sample(stamp + .024)
            reuse_state = RobotState(position, velocity, stamp + .024)
            started = thread_time_ns()
            reused = manager.update(reuse_state, goal, control_dt)
            reuse_elapsed = thread_time_ns() - started
            if reused.reason != "reused" or not reused.success:
                raise RuntimeError(f"could not reuse plan: {reused.reason}")
            if index >= warmup:
                adoption_wall.append(wall_elapsed)
                adoption_cpu.append(cpu_elapsed)
                reuse_cpu.append(reuse_elapsed)
                violations.append(violation)
                for values in (plan.positions, plan.velocities, update.solver.x,
                               update.position, update.velocity):
                    trace.update(values.tobytes())
            stamp += .105
            position, velocity = plan.sample(stamp)
            state = RobotState(position, velocity, stamp)
        info = worker.runtime_info
        statistics = worker.statistics
    return dict(backend=backend, joints=joints, horizon=horizon, samples=samples, warmup=warmup,
                adoption_wall_us=_quantiles(adoption_wall), adoption_cpu_us=_quantiles(adoption_cpu),
                reuse_cpu_us=_quantiles(reuse_cpu), deferred_polls=deferred_polls,
                max_physical_violation=max(violations), trace_sha256=trace.hexdigest(),
                worker_runtime=info, worker_statistics=statistics,
                all_workers_reaped=worker._shutdown_complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "sparse", "native", "native-sparse"),
                        default="sparse")
    parser.add_argument("--joints", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = measure(args.backend, args.joints, args.horizon, args.samples, args.warmup)
    if hasattr(os, "sched_getaffinity"):
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
