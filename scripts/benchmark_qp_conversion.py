"""Interleave complete scenes with sparse versus dense upper-triangle extraction."""

import json
import os
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy import sparse

from wholebodyx import qp
from wholebodyx.session import TrackingSession


def original(hessian):
    return sparse.triu(sparse.csc_matrix(hessian), format="csc")


def main():
    optimized = qp._upper_csc
    results = []
    for run in range(3):
        modes = (("before", original), ("after", optimized))
        if run % 2:
            modes = modes[::-1]
        for scenario in ("tracking", "coordination", "cartesian_circle"):
            baseline = None
            for mode, conversion in modes:
                times, trajectory = [], []
                with patch.object(qp, "_upper_csc", conversion):
                    session = TrackingSession(scenario=scenario)
                    for k in range(400):
                        start = perf_counter()
                        sample = session.step()
                        elapsed = (perf_counter() - start) * 1e6
                        if k >= 5:
                            times.append(elapsed)
                        state = session.plant.read()
                        trajectory.append(np.r_[state.q, state.v, sample["violation"], sample["iterations"]])
                if baseline is None:
                    baseline = trajectory
                else:
                    np.testing.assert_array_equal(trajectory, baseline)
                results.append(dict(run=run, scenario=scenario, mode=mode,
                                    p50_us=float(np.median(times)), p95_us=float(np.percentile(times, 95))))
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                         threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                         steps=400, warmup_steps_excluded=5, results=results), indent=2))


if __name__ == "__main__":
    main()
