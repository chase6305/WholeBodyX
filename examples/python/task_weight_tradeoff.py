"""Conflicting soft tasks: compare the QP answer against its analytic optimum."""

import argparse
import json

from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState


def run(backend="python"):
    if backend == "native":
        from wholebodyx.native import NativeWBC as Controller
    elif backend == "python":
        Controller = KinematicWBC
    else:
        raise ValueError("unknown backend")
    regularization = 0.1
    controller = Controller(JointLimits([-2], [2], [2]), regularization=regularization)
    state, rows = RobotState([0], [0]), []
    for weight in (0.1, 1.0, 10.0, 100.0):
        result = controller.step(state, [LinearTask([[1]], [1], weight), LinearTask([[1]], [-1])], 0.02)
        if result.command is None:
            raise RuntimeError(result.solver.detail)
        expected = (weight - 1) / (weight + 1 + regularization)
        actual = float(result.command.velocity[0])
        if abs(actual - expected) > 1e-6:
            raise RuntimeError("weighted QP differs from analytic optimum")
        rows.append({"positive_task_weight": weight, "velocity_rad_s": actual,
                     "analytic_velocity_rad_s": expected, "positive_task_residual": abs(1 - actual)})
    return {"backend": backend, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    print(json.dumps(run(parser.parse_args().backend), indent=2))
