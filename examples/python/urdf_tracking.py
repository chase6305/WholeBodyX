"""Pinocchio URDF FK/Jacobian -> backend-neutral pose task -> Python/C++ WBC."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx import (BackendRegistry, JointOrder, ReorderedController,
                        ReorderedPlanner, RobotState, VelocitySession)
from wholebodyx.model import PinocchioModel
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import FramePoseTask, PostureTask

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "models/planar_arm.urdf"


def run(backend="python", urdf=DEFAULT_URDF, frame="tool", q0=None, goal=None, steps=250,
        backend_joints=None, *, mpc=False, mpc_backend=None):
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
    plant = JointIntegrator(model.limits, RobotState(q0, np.zeros(len(model.joint_names))), control_dt=.02)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    registry = BackendRegistry()
    limits = order.limits_to_backend(model.limits)
    controller = ReorderedController(registry.create_controller(backend, limits), order)
    planner = ReorderedPlanner(registry.create_planner(mpc_backend or backend, limits), order) if mpc else None

    def tasks(model, state):
        result = [FramePoseTask(frame, target).linearize(model, state)]
        if not mpc:
            result.append(PostureTask(goal, weight=1e-3).linearize(model, state))
        return result

    session = VelocitySession(model, plant, controller, .02, planner=planner,
                              task_factory=tasks, posture_weight=1e-3)
    session.reset()
    violation = 0.0
    replans = 0
    for _ in range(steps):
        result = session.step(goal if mpc else None)
        violation = max(violation, result.cycle.control.solver.violation)
        replans += int(result.reference is not None and result.reference.replanned)
    actual = model.pose(plant.read().q, frame)
    return {"backend": backend, "joints": model.joint_names, "frame": frame,
            "mode": "MPC + WBC" if mpc else "WBC", "replans": replans,
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
    parser.add_argument("--mpc", action="store_true", help="Combine MPC posture references and Cartesian WBC tasks")
    parser.add_argument("--mpc-backend", choices=("python", "native", "sparse", "native-sparse"))
    parser.add_argument("--backend-joints", nargs="+", help="Backend joint names in its own order; q0/goal stay in model order")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
