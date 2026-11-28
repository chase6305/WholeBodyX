"""Shared process isolation and provenance for revision comparisons."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from _measurement import SOURCE as MEASUREMENT_SOURCE, rss_kib as rss_kib


def joint_plan_violation(limits, plan, state, horizon, dt):
    """Check the exported plan against its measurement and physical problem."""
    import numpy as np

    if (
        plan.stamp != state.stamp or plan.dt != dt
        or plan.positions.shape != (horizon + 1, state.q.size)
        or plan.velocities.shape != (horizon, state.q.size)
        or not np.array_equal(plan.positions[0], state.q)
        or not np.isfinite(plan.positions).all() or not np.isfinite(plan.velocities).all()
    ):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        expected = dt * plan.velocities
        if not np.isfinite(expected).all() or not np.allclose(
            np.diff(plan.positions, axis=0), expected, atol=1e-8, rtol=1e-7,
        ):
            return float("inf")
    violation = max(
        float(np.max(limits.lower - plan.positions[1:])),
        float(np.max(plan.positions[1:] - limits.upper)),
        float(np.max(np.abs(plan.velocities) - limits.velocity)),
        0.0,
    )
    if limits.acceleration is not None:
        changes = np.diff(np.vstack((state.v, plan.velocities)), axis=0)
        violation = max(violation, float(np.max(np.abs(changes) - plan.dt * limits.acceleration)))
    return violation


def joint_mpc_initial_sample(joints, workload):
    import numpy as np
    from wholebodyx import JointLimits, RobotState

    state = RobotState(np.zeros(joints), np.zeros(joints))
    if workload == "retarget":
        limits = JointLimits([-1.] * joints, [1.] * joints, [.8] * joints, [2.] * joints)
        goal = .55 * np.sin(np.arange(joints) * .37)
    else:
        limits = JointLimits([-2.] * joints, [2.] * joints, [1.] * joints, [4.] * joints)
        goal = np.full(joints, .02 if workload == "near" else .3)
    return limits, state, goal


def joint_mpc_workload(name, state, goal, iterations, seed, dt):
    import numpy as np
    from wholebodyx import RobotState

    joints = state.q.size
    phases = np.linspace(0, np.pi, joints)
    rng = np.random.default_rng(seed)
    for k in range(iterations):
        if name == "random":
            measurement = RobotState(
                rng.uniform(-.25, .25, joints), rng.uniform(-.1, .1, joints), k * .07
            )
            target = rng.uniform(-3., 3., joints)
        elif name in ("near", "alternating"):
            measurement = RobotState(
                .03 * np.cos(phases + .17 * k), .05 * np.sin(phases + .1 * k), k * .02
            )
            distance = .32 if name == "alternating" and k % 6 >= 3 else .02
            target = measurement.q + distance * np.cos(phases + .003 * k)
        elif name == "retarget":
            phase, tick = divmod(k, 6)
            angles = np.arange(joints) * .37 + phase * 1.9
            target = .55 * np.sin(angles)
            previous = .55 * np.sin(angles - 1.9)
            position = previous + min(1., tick / 4) * (target - previous)
            velocity = .02 * np.cos(angles + tick * .3)
            measurement = RobotState(position, velocity, k * dt)
        else:
            measurement = state
            target = goal + .01 * np.sin(k)
        yield measurement, target


def invoke_worker(script, pythonpath, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(pythonpath)
    command = [sys.executable, str(Path(script).resolve()), "--worker", *arguments]
    process = subprocess.run(command, env=environment, cwd="/tmp", capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"worker exited {process.returncode}: {process.stderr}\n{process.stdout}"
        )
    return json.loads(process.stdout)


def comparison_report(args, script):
    return dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        rounds=args.rounds,
        iterations=args.iterations,
        warmup=args.warmup,
        joints=args.joints,
        results=[],
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurement_sha256=hashlib.sha256(MEASUREMENT_SOURCE.read_bytes()).hexdigest(),
        blas_threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
    )


def revision_rows(args, script, builds, columns):
    for index in range(args.rounds):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for name in order:
            data = invoke_worker(
                script,
                builds[name],
                [
                    "--python-dir",
                    str(builds[name]),
                    "--joints",
                    *map(str, args.joints),
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                ],
            )
            if not data["source_unchanged"]:
                raise RuntimeError("worker sources changed during measurement")
            for row in data.pop("results"):
                row.update(data, build=name, round=index + 1)
                # Retain completed rows even if output or a later worker fails.
                yield row
                print(json.dumps({k: row[k] for k in columns}), flush=True)


def case(value):
    try:
        result = tuple(map(int, value.split("x")))
        if len(result) != 2 or min(result) <= 0:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be positive JOINTSxHORIZON") from exc
