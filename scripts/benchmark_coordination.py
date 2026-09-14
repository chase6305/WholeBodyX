"""Interleaved full coordination-session steps, before/after geometry reuse."""

import json
import os
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np

from wholebodyx import session as session_module
from wholebodyx.experiments import FRAMES, wbc_tasks
from wholebodyx.session import TrackingSession
from wholebodyx.tasks import PostureTask, RelativePositionTask
from wholebodyx.types import LinearTask


def repeated_geometry(model, state, scenario, target):
    """Original coordination task recipe, retained for comparison."""
    positions = [model.pose(state.q, frame).translation for frame in FRAMES]
    center = .5 * (positions[0] + positions[1])
    center_goal = .5 * (target.poses[0].translation + target.poses[1].translation)
    center_jacobian = .5 * sum(model.jacobian(state.q, frame)[:3] for frame in FRAMES)
    return [
        LinearTask(center_jacobian, target.twists[0][:3] + 4 * (center_goal - center)),
        RelativePositionTask(*FRAMES, target.poses[1].translation - target.poses[0].translation,
                             weight=20).linearize(model, state),
        PostureTask(target.q, weight=1e-5).linearize(model, state),
    ]


def main():
    results = []
    for run in range(3):
        modes = (("repeated", repeated_geometry), ("shared", wbc_tasks))
        if run % 2:
            modes = modes[::-1]
        baseline = None
        for mode, recipe in modes:
            session = TrackingSession(scenario="coordination")
            times, states, max_violation = [], [], 0.
            with patch.object(session_module, "wbc_tasks", recipe):
                for k in range(400):
                    started = perf_counter()
                    sample = session.step()
                    elapsed = (perf_counter() - started) * 1e6
                    if k >= 5:
                        times.append(elapsed)
                    state = session.plant.read()
                    states.append(np.r_[state.q, state.v])
                    max_violation = max(max_violation, sample["violation"])
            if baseline is None:
                baseline = states
            else:
                np.testing.assert_allclose(states, baseline, rtol=0, atol=1e-10)
            results.append({"run": run, "mode": mode,
                            "p50_us": float(np.median(times)),
                            "p95_us": float(np.percentile(times, 95)),
                            "max_violation": max_violation,
                            "final_error_m": max(sample["left_error"], sample["right_error"])})
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(),
                      "threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                      "steps": 400, "warmup_steps_excluded": 5,
                      "results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
