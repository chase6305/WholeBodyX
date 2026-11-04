"""Injected torque loop and independent fixed-contact plant, with no H1 assets."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wholebodyx import (CoMReference, ContactSpec, DynamicsState,
                        FixedContactSimulator, TorqueControlLoop)
from wholebodyx.dynamics import ContactWBC, FloatingBaseDynamics


DEFAULT_URDF = Path(__file__).resolve().parents[1] / "models/standing_robot.urdf"


def run(steps=1000, csv_path=None, disturbance=.2):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(disturbance) or not 0 <= disturbance <= 1:
        raise ValueError("disturbance must be between zero and one Nm")
    robot = FloatingBaseDynamics(DEFAULT_URDF, contact_specs=[ContactSpec("base")])
    q = robot.pin.neutral(robot.model)
    q[2], q[-1] = .07, .2
    initial = DynamicsState(q, np.zeros(robot.nv))
    target = robot.terms(initial).com.copy()
    dt = .005

    def force(state):
        value = np.zeros(robot.nv)
        if .5 <= state.stamp < .7:
            value[-1] = disturbance
        return value

    plant = FixedContactSimulator(robot, initial, dt, external_force=force)
    controller = ContactWBC(robot, initial)
    loop = TorqueControlLoop(plant, robot, controller, dt)
    loop.reset()
    stream = None
    max_error = max_violation = max_cycle_ms = max_prediction_error = 0.
    try:
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", newline="", encoding="utf-8")
        writer = None
        for _ in range(steps):
            result = loop.step(lambda state: CoMReference(target, state.stamp, state.stamp + dt))
            state = plant.read()
            error = float(np.max(np.abs(state.q[7:] - initial.q[7:])))
            max_error = max(max_error, error)
            max_violation = max(max_violation, result.control.violation)
            max_cycle_ms = max(max_cycle_ms, result.cycle_ms)
            max_prediction_error = max(max_prediction_error,
                                        float(np.max(np.abs(plant.acceleration - result.control.acceleration))))
            row = dict(time=state.stamp, q_rad=float(state.q[-1]), torque_nm=float(result.command.torque[-1]),
                       posture_error_rad=error, normal_force_n=float(plant.wrench[2]),
                       cycle_ms=result.cycle_ms, control_ms=result.control_ms,
                       violation=result.control.violation)
            if stream is not None:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
    finally:
        plant.stop()
        if stream is not None:
            stream.close()
    return dict(steps=steps, duration_s=plant.read().stamp, actuated_joints=robot.na,
                final_posture_error_rad=error, max_posture_error_rad=max_error,
                max_constraint_violation=max_violation, max_cycle_ms=max_cycle_ms,
                max_predicted_acceleration_difference=max_prediction_error,
                disturbance_nm=disturbance, contact_mode="prescribed_bilateral")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--csv", dest="csv_path", type=Path)
    parser.add_argument("--disturbance", type=float, default=.2)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, allow_nan=False))
