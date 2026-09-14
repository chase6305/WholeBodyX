"""Compare frame updates or name lookup using synthetic fixed URDF frames."""

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import repeat

import numpy as np
import pinocchio

from wholebodyx.model import PinocchioModel, Pose
from wholebodyx.types import array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("frames", "lookup"), default="frames")
    args = parser.parse_args()
    results = []
    with TemporaryDirectory() as directory:
        for count in (10, 1000):
            parts = ['<robot name="frames"><link name="base"/><link name="arm"/>',
                     '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>',
                     '<axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>']
            for k in range(count):
                parts.append(f'<link name="frame_{k}"/><joint name="fixed_{k}" type="fixed">'
                             f'<parent link="arm"/><child link="frame_{k}"/>'
                             f'<origin xyz="{k * .001} 0 0"/></joint>')
            parts.append('</robot>')
            path = Path(directory) / "model.urdf"
            path.write_text(''.join(parts))
            model = PinocchioModel(path)
            q, frame = np.array([.4]), f"frame_{count - 1}"

            def before():
                if args.operation == "lookup":
                    configuration = array(q, 1, "q")
                    if configuration.size != model._model.nq:
                        raise ValueError("configuration dimensions differ")
                    if not model._model.existFrame(frame):
                        raise ValueError(f"unknown frame: {frame}")
                    frame_id = model._model.getFrameId(frame)
                else:
                    configuration, frame_id = model._prepare(q, frame)
                model._pin.forwardKinematics(model._model, model._data, configuration)
                if args.operation == "lookup":
                    placement = model._pin.updateFramePlacement(model._model, model._data, frame_id)
                else:
                    model._pin.updateFramePlacements(model._model, model._data)
                    placement = model._data.oMf[frame_id]
                return Pose(placement.translation, placement.rotation)

            def after():
                return model.pose(q, frame)

            expected, actual = before(), after()
            np.testing.assert_array_equal(actual.translation, expected.translation)
            np.testing.assert_array_equal(actual.rotation, expected.rotation)
            timings = {"before": [], "after": []}
            for run in range(5):
                modes = (("before", before), ("after", after))
                if run % 2:
                    modes = modes[::-1]
                for name, fn in modes:
                    timings[name].append(repeat(fn, number=500, repeat=1)[0] / 500 * 1e6)
            results.append(dict(auxiliary_frames=count, model_frames=model._model.nframes,
                                **{name + "_median_us": float(np.median(values)) for name, values in timings.items()}))
    print(json.dumps(dict(operation=args.operation, python=platform.python_version(), pinocchio=pinocchio.__version__,
                         repetitions=5, calls_per_repeat=500, results=results), indent=2))


if __name__ == "__main__":
    main()
