"""Interleave complete scenarios with general and fixed-size pose checks."""

import json
import os
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np

from wholebodyx.model import Pose
from wholebodyx.session import TrackingSession
from wholebodyx.types import array


def original_validation(self):
    p = array(self.translation, 1, "translation")
    r = array(self.rotation, 2, "rotation")
    if p.shape != (3,) or r.shape != (3, 3):
        raise ValueError("pose requires 3-D translation and 3x3 rotation")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=1e-8):
        raise ValueError("rotation must belong to SO(3)")
    object.__setattr__(self, "translation", p)
    object.__setattr__(self, "rotation", r)


def main():
    optimized = Pose.__post_init__
    results = []
    for run in range(3):
        modes = (("before", original_validation), ("after", optimized))
        if run % 2:
            modes = modes[::-1]
        for scenario in ("tracking", "coordination", "cartesian_circle"):
            baseline = None
            for mode, validate in modes:
                states, times = [], []
                with patch.object(Pose, "__post_init__", validate):
                    session = TrackingSession(scenario=scenario)
                    for k in range(400):
                        start = perf_counter()
                        sample = session.step()
                        elapsed = (perf_counter() - start) * 1e6
                        if k >= 5:
                            times.append(elapsed)
                        state = session.plant.read()
                        states.append(np.r_[state.q, state.v, sample["left_error"], sample["right_error"]])
                if baseline is None:
                    baseline = states
                else:
                    np.testing.assert_array_equal(states, baseline)
                results.append(dict(run=run, scenario=scenario, mode=mode,
                                    p50_us=float(np.median(times)), p95_us=float(np.percentile(times, 95))))
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                         threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                         steps=400, warmup_steps_excluded=5, results=results), indent=2))


if __name__ == "__main__":
    main()
