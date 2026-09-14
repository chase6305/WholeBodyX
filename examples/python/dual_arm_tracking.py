"""Run a deterministic planar dual-arm WBC or MPC -> WBC tracking experiment."""

import argparse
import json

import numpy as np

from wholebodyx import BackendRegistry, MPCReferenceManager, RobotState
from wholebodyx.model import PlanarDualArm
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import FramePoseTask, PostureTask


def run(backend="python", with_mpc=False, steps=250):
    model = PlanarDualArm()
    q0 = np.array([0.2, 0.6, -0.4, -0.2, -0.6, 0.4])
    goal = np.array([0.5, 0.4, -0.3, -0.5, -0.4, 0.3])
    plant = JointIntegrator(model.limits, RobotState(q0, np.zeros(6)))
    registry = BackendRegistry()
    controller = registry.create_controller(backend, model.limits)
    manager = MPCReferenceManager(registry.create_planner(backend, model.limits), model.limits) if with_mpc else None
    dt = 0.02
    frames = ("left_ee", "right_ee")
    initial_error = sum(np.linalg.norm(model.pose(q0, f).translation - model.pose(goal, f).translation) for f in frames)
    max_violation = 0.0
    for _ in range(steps):
        state = plant.read()
        if with_mpc:
            update = manager.update(state, goal, dt)
            if not update.success:
                raise RuntimeError(f"MPC failed: {update.solver}")
            tasks = [PostureTask(update.position, feedforward=update.velocity).linearize(model, state)]
        else:
            tasks = [FramePoseTask(f, model.pose(goal, f)).linearize(model, state) for f in frames]
            tasks.append(PostureTask(goal, weight=1e-3).linearize(model, state))
        result = controller.step(state, tasks, dt)
        if result.command is None:
            raise RuntimeError(f"WBC failed: {result.solver}")
        max_violation = max(max_violation, result.solver.violation)
        plant.write(result.command, dt)
    state = plant.read()
    final_error = sum(np.linalg.norm(model.pose(state.q, f).translation - model.pose(goal, f).translation) for f in frames)
    return {"backend": backend, "mpc": with_mpc, "steps": steps,
            "initial_position_error_m": float(initial_error), "final_position_error_m": float(final_error),
            "joint_error_rad": float(np.linalg.norm(state.q - goal)), "max_constraint_violation": max_violation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--mpc", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.backend, args.mpc), indent=2))
