"""Interleave complete MPC steps and prediction reads with joint-only targets."""

import json
import os
from time import perf_counter

import numpy as np

from wholebodyx.session import TrackingSession


class FullReferenceSession(TrackingSession):
    def _reference_goal(self, stamp):
        return self.reference(stamp).q


def main():
    results = []
    for run in range(3):
        classes = (FullReferenceSession, TrackingSession)
        if run % 2:
            classes = classes[::-1]
        for scenario in ("tracking", "mpc_retargeting", "disturbance_recovery"):
            baseline = None
            for cls in classes:
                session = cls(scenario=scenario)
                times, reads, states, reasons = [], [], [], []
                for k in range(400):
                    start = perf_counter()
                    sample = session.step()
                    elapsed = (perf_counter() - start) * 1e6
                    start = perf_counter()
                    paths = [session.prediction(frame) for frame in session.frames]
                    read_elapsed = (perf_counter() - start) * 1e6
                    if k >= 5:
                        times.append(elapsed)
                        reads.append(read_elapsed)
                    state = session.plant.read()
                    states.append(np.r_[state.q, state.v, sample["left_error"], sample["right_error"],
                                         *(len(path) for path in paths)])
                    reasons.append(sample["replan_reason"])
                if baseline is None:
                    baseline = (states, reasons)
                else:
                    np.testing.assert_allclose(states, baseline[0], rtol=0, atol=1e-12)
                    assert reasons == baseline[1]
                results.append({"run": run, "scenario": scenario, "mode": cls.__name__,
                                "step_p50_us": float(np.median(times)),
                                "step_p95_us": float(np.percentile(times, 95)),
                                "prediction_p50_us": float(np.median(reads)),
                                "prediction_p95_us": float(np.percentile(reads, 95))})
    print(json.dumps({"threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                      "steps": 400, "warmup_steps_excluded": 5, "results": results}, indent=2))


if __name__ == "__main__":
    main()
