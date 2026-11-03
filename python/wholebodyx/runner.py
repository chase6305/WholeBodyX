"""Packaged headless experiments with streaming CSV and online error metrics."""

import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .experiments import SCENARIOS
from .session import TrackingSession


def run(scenario, backend="python", mpc_backend=None, steps=None, csv_path=None, *,
        mode=None, registry=None):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    required_mode = SCENARIOS[scenario].mode
    if mode is not None and required_mode is not None and mode != required_mode:
        raise ValueError(f"{scenario} requires {required_mode}")
    step_count = SCENARIOS[scenario].steps if steps is None else steps
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 1:
        raise ValueError("steps must be a positive integer")
    session = TrackingSession(backend=backend, scenario=scenario, mpc_backend=mpc_backend,
                              mode=mode or required_mode or "MPC + WBC", registry=registry)
    squared_error = max_position = max_orientation = max_relative = max_violation = 0.0
    with ExitStack() as stack:
        stream, writer = None, None
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
        for _ in range(step_count):
            sample = session.step()
            # Only export needs an expanded row. Keep the session's history
            # untouched and avoid creating dozens of unused scalar fields.
            row = dict(sample) if stream is not None else sample
            state, target = session.plant.read(), session.reference()
            for k, frame in enumerate(session.frames):
                pose = session.model.pose(state.q, frame)
                orientation_error = float(np.linalg.norm(
                    Rotation.from_matrix(target.poses[k].rotation @ pose.rotation.T).as_rotvec()))
                max_orientation = max(max_orientation, orientation_error)
                if stream is not None:
                    row[f"orientation_error_{k}_rad"] = orientation_error
                    for j, axis in enumerate("xyz"):
                        row[f"actual_{k}_{axis}_m"] = float(pose.translation[j])
                        row[f"target_{k}_{axis}_m"] = float(target.poses[k].translation[j])
            if stream is not None:
                for k in range(len(state.q)):
                    row[f"q_{k}"] = float(state.q[k])
                    row[f"v_{k}"] = float(state.v[k])
                    row[f"goal_{k}"] = float(target.q[k])
            left, right = row["left_error"], row["right_error"]
            squared_error += left * left + right * right
            final_position = max(left, right)
            max_position = max(max_position, final_position)
            max_relative = max(max_relative, row["relative_error"])
            max_violation = max(max_violation, row["violation"])
            if stream is not None:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
    return {
        "scenario": scenario, "mode": session.mode, "control_dt_s": session.dt,
        "wbc_backend": session.backend, "mpc_backend": session.mpc_backend,
        "steps": step_count, "duration_s": row["time"],
        "final_position_error_m": final_position,
        "rms_position_error_m": float(np.sqrt(squared_error / (2 * step_count))),
        "max_position_error_m": max_position,
        "max_orientation_error_rad": max_orientation,
        "orientation_controlled": scenario != "coordination",
        "max_relative_error_m": max_relative,
        "max_constraint_violation": max_violation,
        "replans": session.replans, "scheduled_goal_changes": session.goal_changes,
        "tracking_replans": session.tracking_replans, "disturbances": session.disturbances,
        "replan_stamps_s": list(session.replan_stamps),
        "replan_history_truncated": session.replans > len(session.replan_stamps),
    }
