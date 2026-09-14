"""Compare scalar and batch planar geometry; excludes rendering and control."""

import json
import os
import platform
from timeit import repeat

import numpy as np

from wholebodyx.model import PlanarDualArm


def measure():
    model = PlanarDualArm()
    configurations = np.random.default_rng(18).uniform(-2.8, 2.8, (16, 6))
    frames = ("left_ee", "right_ee")

    def scalar_prediction():
        return {f: np.array([model.pose(q, f).translation for q in configurations]) for f in frames}

    def batch_prediction():
        return {f: model.translations(configurations, f) for f in frames}

    def scalar_jacobian():
        theta, offset = model._chain(configurations[0], frames[0])
        j = np.zeros((6, 6))
        for k in range(3):
            j[0, offset + k] = -model.lengths[k:] @ np.sin(theta[k:])
            j[1, offset + k] = model.lengths[k:] @ np.cos(theta[k:])
            j[5, offset + k] = 1
        return j

    def batch_jacobian():
        return model.jacobian(configurations[0], frames[0])

    expected, actual = scalar_prediction(), batch_prediction()
    for frame in frames:
        np.testing.assert_allclose(actual[frame], expected[frame], atol=1e-14, rtol=0)
    np.testing.assert_allclose(batch_jacobian(), scalar_jacobian(), atol=1e-14, rtol=0)
    results = {}
    for name, fn in (("scalar_prediction", scalar_prediction), ("batch_prediction", batch_prediction),
                     ("scalar_jacobian", scalar_jacobian), ("batch_jacobian", batch_jacobian)):
        fn()
        timings = np.array(repeat(fn, number=200, repeat=5)) / 200 * 1e6
        results[name] = {"min_mean_us": float(timings.min()), "median_mean_us": float(np.median(timings))}
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "trajectory_points": 16, "arms": 2, "repeats": 5, "calls_per_repeat": 200,
            "results": results}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
