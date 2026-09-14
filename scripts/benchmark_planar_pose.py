"""Compare general axis-angle and analytic planar poses in complete scenes."""

import json
import os
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx.model import PlanarDualArm, Pose
from wholebodyx.session import TrackingSession


def original_pose(self, q, frame):
    theta, offset = self._chain(q, frame)
    translation = np.array([self.lengths @ np.cos(theta),
                            self.lengths @ np.sin(theta) + (.4 if offset == 0 else -.4), 0.])
    return Pose(translation, Rotation.from_rotvec([0, 0, theta[-1]]).as_matrix())


def main():
    optimized, results = PlanarDualArm.pose, []
    for run in range(3):
        modes = (("before", original_pose), ("after", optimized))
        if run % 2:
            modes = modes[::-1]
        for scenario in ("tracking", "coordination", "cartesian_circle"):
            baseline = None
            for mode, pose in modes:
                times, trajectory = [], []
                with patch.object(PlanarDualArm, "pose", pose):
                    session = TrackingSession(scenario=scenario)
                    for k in range(400):
                        start = perf_counter()
                        sample = session.step()
                        elapsed = (perf_counter() - start) * 1e6
                        if k >= 5:
                            times.append(elapsed)
                        state = session.plant.read()
                        trajectory.append(np.r_[state.q, state.v, sample["left_error"], sample["right_error"]])
                difference = None
                if baseline is None:
                    baseline = np.array(trajectory)
                else:
                    np.testing.assert_allclose(trajectory, baseline, rtol=0, atol=1e-8)
                    difference = float(np.max(np.abs(np.array(trajectory) - baseline)))
                results.append(dict(run=run, scenario=scenario, mode=mode, max_difference=difference,
                                    p50_us=float(np.median(times)), p95_us=float(np.percentile(times, 95))))
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                         threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                         steps=400, warmup_steps_excluded=5, results=results), indent=2))


if __name__ == "__main__":
    main()
