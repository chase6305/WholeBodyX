"""Compare full-pose and position-only diagnostics in complete session steps.

The full-pose baseline includes packing positions into the batch API shape.
Browser rendering is excluded.
"""

import json
import os
import platform
from time import perf_counter

import numpy as np

from wholebodyx.session import TrackingSession


def main():
    results = []
    for run in range(3):
        modes = ("full_pose", "position_only") if run % 2 == 0 else ("position_only", "full_pose")
        for scenario in ("tracking", "coordination", "cartesian_circle"):
            baseline = None
            for mode in modes:
                session = TrackingSession(scenario=scenario)
                if mode == "full_pose":
                    model = session.model

                    def full_pose(configurations, frame):
                        return np.array([model.pose(q, frame).translation for q in configurations])

                    model.translations = full_pose
                times, trajectory = [], []
                for k in range(400):
                    start = perf_counter()
                    sample = session.step()
                    elapsed = (perf_counter() - start) * 1e6
                    if k >= 5:
                        times.append(elapsed)
                    state = session.plant.read()
                    trajectory.append(np.r_[state.q, state.v, sample["left_error"],
                                             sample["right_error"], sample["relative_error"]])
                if baseline is None:
                    baseline = trajectory
                else:
                    np.testing.assert_allclose(trajectory, baseline, rtol=0, atol=1e-12)
                results.append({"run": run, "scenario": scenario, "mode": mode,
                                "p50_us": float(np.median(times)),
                                "p95_us": float(np.percentile(times, 95))})
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(),
                      "threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                      "steps": 400, "warmup_steps_excluded": 5,
                      "results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
