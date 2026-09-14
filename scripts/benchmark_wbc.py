"""Interleaved full WBC step benchmark, including changing Jacobians."""

import json
import os
import platform
from time import perf_counter

import numpy as np

from wholebodyx import JointLimits, KinematicWBC, LinearTask, QuadraticProgram, RobotState


class UncachedWBC(KinematicWBC):
    """Pre-cache formulation retained only as a measurement baseline."""

    def formulate(self, state, tasks, dt):
        lo, hi = self.limits.velocity_bounds(state, dt)
        n = state.q.size
        h = (self.regularization + self.smoothing) * np.eye(n)
        g = -self.smoothing * state.v
        for task in tasks:
            if task.jacobian.shape[1] != n:
                raise ValueError("task and state dimensions differ")
            h += task.jacobian.T @ (task.weight[:, None] * task.jacobian)
            g -= task.jacobian.T @ (task.weight * task.target_velocity)
        return QuadraticProgram(h, g, np.eye(n), lo, hi)


def main():
    results = []
    for run in range(3):
        classes = (UncachedWBC, KinematicWBC) if run % 2 == 0 else (KinematicWBC, UncachedWBC)
        for changing in (False, True):
            for cls in classes:
                controller = cls(JointLimits([-2] * 6, [2] * 6, [1] * 6))
                samples, velocities = [], []
                for k in range(205):
                    state = RobotState(np.full(6, .01 * np.sin(k)), np.zeros(6), k * .02)
                    scale = 1 + .1 * np.sin(k) if changing else 1
                    tasks = [LinearTask(scale * np.eye(6), np.full(6, .2 + .01 * np.cos(k)))]
                    start = perf_counter()
                    result = controller.step(state, tasks, .02)
                    elapsed = (perf_counter() - start) * 1e6
                    if not result.solver.success:
                        raise RuntimeError(result.solver)
                    if k >= 5:
                        samples.append(elapsed)
                        velocities.append(result.command.velocity)
                results.append({"run": run, "mode": cls.__name__, "changing_jacobian": changing,
                                "p50_us": float(np.median(samples)),
                                "p95_us": float(np.percentile(samples, 95))})
                if cls is classes[0]:
                    baseline = velocities
                else:
                    np.testing.assert_allclose(velocities, baseline, rtol=0, atol=1e-8)
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                      "warmup": 5, "iterations": 200, "results": results}, indent=2))


if __name__ == "__main__":
    main()
