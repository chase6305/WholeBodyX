"""An unreachable soft target saturates; contradictory hard bounds reject commands."""

import argparse
import json

import numpy as np

from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState
from wholebodyx.simulation import JointIntegrator


def run(backend="python"):
    if backend == "native":
        from wholebodyx.native import NativeWBC as Controller
    elif backend == "python":
        Controller = KinematicWBC
    else:
        raise ValueError("unknown backend")
    limits = JointLimits([-1], [1], [0.5])
    controller = Controller(limits)
    plant = JointIntegrator(limits, RobotState([0.9], [0]))
    max_violation = 0.0
    for _ in range(50):
        state = plant.read()
        # Target 2 rad exceeds q_max=1, but only the objective is unreachable.
        result = controller.step(state, [LinearTask([[1]], 4 * (2 - state.q))], 0.02)
        if result.command is None:
            raise RuntimeError("reachable hard constraints unexpectedly failed")
        plant.write(result.command, 0.02)
        max_violation = max(max_violation, result.solver.violation)
    # Fault injection, separate from the plant: q=2 cannot enter [-1,1] in 20 ms.
    fault = controller.step(RobotState([2], [0], plant.read().stamp), [], 0.02)
    if fault.command is not None or fault.solver.status != "infeasible":
        raise RuntimeError("infeasible input must not generate a command")
    controller.reset()
    # Resume only with a valid measured state. Never replace bad measurements by clipping.
    state = plant.read()
    recovered = controller.step(state, [LinearTask([[1]], -state.q)], 0.02)
    if recovered.command is None:
        raise RuntimeError("solver did not recover from infeasible input")
    return {"backend": backend, "bounded_position_rad": float(state.q[0]),
            "unreachable_target_error_rad": float(2 - state.q[0]),
            "max_constraint_violation": max_violation, "fault_status": fault.solver.status,
            "fault_has_command": fault.command is not None,
            "recovered_velocity_rad_s": float(recovered.command.velocity[0])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    print(json.dumps(run(parser.parse_args().backend), indent=2))
