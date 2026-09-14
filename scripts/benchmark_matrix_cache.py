"""Interleave MPC runs with value scans and immutable-storage reuse enabled.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The scan baseline retains
key construction overhead, isolating the benefit of skipping matrix scans.
"""

import json
import os
import platform
from unittest.mock import patch

import numpy as np

from benchmark_mpc import measure
from wholebodyx import qp


def main():
    optimized = qp._same_matrix

    def scanned(snapshot, value, previous_key, key):
        return np.array_equal(snapshot, value)

    results = []
    for run in range(3):
        modes = (("scan", scanned), ("immutable", optimized))
        if run % 2:
            modes = modes[::-1]
        for mode, implementation in modes:
            with patch.object(qp, "_same_matrix", implementation):
                for joints, horizon in ((6, 15), (14, 30)):
                    result = measure("python", joints, horizon, 200, 5)
                    results.append(dict(result, run=run, mode=mode))
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(),
                      "threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                      "results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
