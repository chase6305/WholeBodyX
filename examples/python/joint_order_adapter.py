"""Run WBC and MPC with different joint orders while the robot stays in model order."""

import argparse
import json

import numpy as np

from wholebodyx import BackendRegistry, JointOrder, ReorderedController, ReorderedPlanner
from wholebodyx.model import PlanarDualArm
from wholebodyx.session import TrackingSession


def make_registry(backend="python", mpc_backend=None):
    mpc_backend = backend if mpc_backend is None else mpc_backend
    source = BackendRegistry()
    names = PlanarDualArm.joint_names
    wbc_order = JointOrder(names, tuple(names[i] for i in (4, 2, 1, 5, 0, 3)))
    mpc_order = JointOrder(names, tuple(names[i] for i in (2, 0, 5, 1, 4, 3)))
    registry = BackendRegistry()
    registry.register("reordered_wbc", controller=lambda limits: ReorderedController(
        source.create_controller(backend, wbc_order.limits_to_backend(limits)), wbc_order))
    registry.register("reordered_mpc", planner=lambda limits: ReorderedPlanner(
        source.create_planner(mpc_backend, mpc_order.limits_to_backend(limits)), mpc_order))
    return registry, wbc_order, mpc_order


def run(backend="python", mpc_backend=None, steps=400, scenario="disturbance_recovery"):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    registry, wbc_order, mpc_order = make_registry(backend, mpc_backend)
    baseline = TrackingSession(backend=backend, mpc_backend=mpc_backend, scenario=scenario)
    mapped = TrackingSession(backend="reordered_wbc", mpc_backend="reordered_mpc", scenario=scenario, registry=registry)
    position_difference = velocity_difference = cartesian_difference = constraint_violation = 0.0
    for _ in range(steps):
        baseline.step()
        sample = mapped.step()
        a, b = baseline.plant.read(), mapped.plant.read()
        position_difference = max(position_difference, float(np.max(np.abs(a.q - b.q))))
        velocity_difference = max(velocity_difference, float(np.max(np.abs(a.v - b.v))))
        cartesian_difference = max(cartesian_difference, *(float(np.linalg.norm(x[-1] - y[-1]))
                                    for x, y in zip(baseline.trails, mapped.trails)))
        constraint_violation = max(constraint_violation, sample["violation"])
    # Redundant tasks with 1e-6 regularization permit small joint-velocity
    # differences under OSQP ordering/scaling. Check task-space motion as well.
    if position_difference > 2e-6 or velocity_difference > 2e-5 or cartesian_difference > 1e-6:
        raise RuntimeError("joint permutation changed the model-order closed-loop trajectory")
    if mapped.tracking_replans != baseline.tracking_replans or mapped.disturbances != baseline.disturbances:
        raise RuntimeError("joint permutation changed recovery events")
    return {
        "scenario": scenario, "wbc_backend": backend, "mpc_backend": baseline.mpc_backend, "steps": steps,
        "model_order": wbc_order.model_names, "wbc_order": wbc_order.backend_names, "mpc_order": mpc_order.backend_names,
        "max_position_difference_rad": position_difference, "max_velocity_difference_rad_s": velocity_difference,
        "max_cartesian_difference_m": cartesian_difference,
        "final_position_error_m": max(sample["left_error"], sample["right_error"]),
        "max_constraint_violation": constraint_violation, "tracking_replans": mapped.tracking_replans,
        "disturbances": mapped.disturbances,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--mpc-backend", choices=("python", "native"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scenario", choices=("disturbance_recovery", "coordination"), default="disturbance_recovery")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.viewer:
        if args.steps is not None:
            parser.error("--steps applies to headless comparison")
        from wholebodyx.visualization.viewer import ControlViewer
        registry, _, _ = make_registry(args.backend, args.mpc_backend)
        ControlViewer(backend="reordered_wbc", mpc_backend="reordered_mpc", registry=registry,
                      scenario=args.scenario, port=args.port).run()
    else:
        print(json.dumps(run(args.backend, args.mpc_backend, 400 if args.steps is None else args.steps, args.scenario), indent=2))
