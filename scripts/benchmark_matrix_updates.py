"""Compare current solver against a saved qp.py using --baseline PATH."""

import argparse
import importlib.util
import json
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

from wholebodyx import qp
from wholebodyx.session import TrackingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("wholebodyx._benchmark_baseline", args.baseline)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.QPResult = qp.QPResult
    optimized, results = qp.OSQPSolver.solve, []
    for run in range(3):
        for scenario in ("tracking", "coordination", "cartesian_circle"):
            modes = (("before", module.OSQPSolver.solve), ("after", optimized))
            if run % 2:
                modes = modes[::-1]
            baseline = None
            for mode, solve in modes:
                times, states, iterations = [], [], []
                with patch.object(qp.OSQPSolver, "solve", solve):
                    session = TrackingSession(scenario=scenario)
                    for k in range(400):
                        start = perf_counter()
                        sample = session.step()
                        elapsed = (perf_counter() - start) * 1e6
                        if k >= 5:
                            times.append(elapsed)
                        state = session.plant.read()
                        states.append(np.r_[state.q, state.v, sample["violation"]])
                        iterations.append(sample["iterations"])
                difference = None
                if baseline is None:
                    baseline = np.array(states)
                else:
                    np.testing.assert_allclose(states, baseline, atol=1e-8, rtol=0)
                    difference = float(np.max(np.abs(np.array(states) - baseline)))
                results.append(dict(run=run, scenario=scenario, mode=mode,
                                    p50_us=float(np.median(times)), p95_us=float(np.percentile(times, 95)),
                                    max_difference=difference, mean_iterations=float(np.mean(iterations))))
    print(json.dumps(dict(steps=400, warmup_steps_excluded=5, results=results), indent=2))


if __name__ == "__main__":
    main()
