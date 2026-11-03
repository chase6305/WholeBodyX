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


def joint_plan_violation(limits, plan, initial_velocity):
    """Independently check exported positions, velocities and first-step slew."""
    import numpy as np

    if not np.isfinite(plan.positions).all() or not np.isfinite(plan.velocities).all():
        return float("inf")
    violation = max(
        float(np.max(limits.lower - plan.positions[1:])),
        float(np.max(plan.positions[1:] - limits.upper)),
        float(np.max(np.abs(plan.velocities) - limits.velocity)),
        0.0,
    )
    if limits.acceleration is not None:
        changes = np.diff(np.vstack((initial_velocity, plan.velocities)), axis=0)
        violation = max(violation, float(np.max(np.abs(changes) - plan.dt * limits.acceleration)))
    return violation


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
