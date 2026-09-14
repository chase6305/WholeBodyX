"""Pinocchio URDF FK/Jacobian -> backend-neutral pose task -> Python/C++ WBC."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx import BackendRegistry, JointOrder, ReorderedController, RobotState
from wholebodyx.model import PinocchioModel
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import FramePoseTask, PostureTask

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "models/planar_arm.urdf"


def run(backend="python", urdf=DEFAULT_URDF, frame="tool", q0=None, goal=None, steps=250, backend_joints=None):
    model = PinocchioModel(urdf)
    order = JointOrder(model.joint_names, model.joint_names if backend_joints is None else backend_joints)
    if q0 is None:
        q0 = [0.2, 0.6, -0.4] if Path(urdf).resolve() == DEFAULT_URDF else np.clip(
            np.zeros(len(model.joint_names)), model.limits.lower, model.limits.upper)
    if goal is None:
        goal = [0.5, 0.4, -0.3] if Path(urdf).resolve() == DEFAULT_URDF else np.clip(
            np.array(q0) + 0.1, model.limits.lower, model.limits.upper)
    goal = np.asarray(goal)
    target = model.pose(goal, frame)
    plant = JointIntegrator(model.limits, RobotState(q0, np.zeros(len(model.joint_names))))
    if steps < 1:
        raise ValueError("steps must be positive")
    controller = ReorderedController(BackendRegistry().create_controller(backend, order.limits_to_backend(model.limits)), order)
    violation = 0.0
    for _ in range(steps):
        state = plant.read()
        tasks = [FramePoseTask(frame, target).linearize(model, state),
                 PostureTask(goal, weight=1e-3).linearize(model, state)]
        result = controller.step(state, tasks, 0.02)
        if result.command is None:
            raise RuntimeError(result.solver.detail)
        plant.write(result.command, 0.02)
        violation = max(violation, result.solver.violation)
    actual = model.pose(plant.read().q, frame)
    return {"backend": backend, "joints": model.joint_names, "frame": frame,
            "backend_joint_order": order.backend_names,
            "final_position_error_m": float(np.linalg.norm(actual.translation - target.translation)),
            "final_orientation_error_rad": float(np.linalg.norm(
                Rotation.from_matrix(target.rotation @ actual.rotation.T).as_rotvec())),
            "max_constraint_violation": violation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--frame", default="tool")
    parser.add_argument("--q0", nargs="+", type=float)
    parser.add_argument("--goal", nargs="+", type=float)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--backend-joints", nargs="+", help="Backend joint names in its own order; q0/goal stay in model order")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
