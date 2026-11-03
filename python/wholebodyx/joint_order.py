"""Named joint permutations at a fixed-base controller/planner boundary.

Only joint order changes: units, signs, task rows and coordinate frames do not.
QPResult.x remains in the wrapped solver's decision-variable order.
"""

from dataclasses import dataclass, field

import numpy as np

from .mpc import JointPlan, PlanResult
from .qp import QPResult
from .types import JointLimits, LinearTask, RobotState, VelocityCommand, array
from .wbc import ControlResult


def _names(values, label):
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence of joint names")
    values = tuple(values)
    if not values or any(not isinstance(name, str) or not name or name.strip() != name for name in values):
        raise ValueError(f"{label} must contain nonempty joint names without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate joint names")
    return values


@dataclass(frozen=True)
class JointOrder:
    model_names: tuple[str, ...]
    backend_names: tuple[str, ...]
    _forward: tuple[int, ...] = field(init=False, repr=False)
    _inverse: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        model = _names(self.model_names, "model_names")
        backend = _names(self.backend_names, "backend_names")
        if set(model) != set(backend):
            missing, extra = sorted(set(model) - set(backend)), sorted(set(backend) - set(model))
            raise ValueError(f"joint name sets differ: missing={missing}, extra={extra}")
        indices = {name: i for i, name in enumerate(model)}
        forward = tuple(indices[name] for name in backend)
        inverse = tuple(int(i) for i in np.argsort(forward))
        for name, value in (("model_names", model), ("backend_names", backend),
                            ("_forward", forward), ("_inverse", inverse)):
            object.__setattr__(self, name, value)

    def _map(self, values, indices):
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != len(indices):
            raise ValueError("expected a joint vector or matrix with joints along its last axis")
        # np.take already gives the mapped values an independent buffer.
        mapped = np.take(values, indices, axis=-1)
        if not np.isfinite(mapped).all():
            raise ValueError(f"joint values must be a finite {values.ndim}-D array")
        mapped.setflags(write=False)
        return mapped

    def to_backend(self, values):
        """Map a model-order vector or joint columns to backend order; returns a copy."""
        return self._map(values, self._forward)

    def to_model(self, values):
        return self._map(values, self._inverse)

    def state_to_backend(self, state):
        return RobotState(self.to_backend(state.q), self.to_backend(state.v), state.stamp)

    def state_to_model(self, state):
        return RobotState(self.to_model(state.q), self.to_model(state.v), state.stamp)

    def limits_to_backend(self, limits):
        acceleration = self.to_backend(limits.acceleration) if limits.acceleration is not None else None
        return JointLimits(self.to_backend(limits.lower), self.to_backend(limits.upper),
                           self.to_backend(limits.velocity), acceleration)

    def task_to_backend(self, task):
        return LinearTask(self.to_backend(task.jacobian), task.target_velocity, task.weight)

    def command_to_model(self, command):
        return VelocityCommand(self.to_model(command.velocity), command.stamp, command.valid_until)

    def plan_to_model(self, plan):
        return JointPlan(self.to_model(plan.positions), self.to_model(plan.velocities), plan.stamp, plan.dt)


def _solver_snapshot(result):
    if not isinstance(result, QPResult):
        raise ValueError("joint-order adapter requires a QPResult")
    x = array(result.x, 1, "solver decision vector") if result.x is not None else None
    if result.status == "solved" and (x is None or not x.size):
        raise ValueError("solved result must contain a decision vector")
    return QPResult(result.status, x, result.iterations, result.violation, result.detail)


class ReorderedController:
    """Wrap a Controller already constructed with backend-order limits/settings."""

    def __init__(self, controller, order):
        if not isinstance(order, JointOrder):
            raise TypeError("order must be a JointOrder")
        if not all(callable(getattr(controller, name, None)) for name in ("reset", "step")):
            raise TypeError("controller must implement reset and step")
        self.controller, self.order = controller, order

    def reset(self):
        self.controller.reset()

    def step(self, state, tasks, dt):
        result = self.controller.step(self.order.state_to_backend(state),
                                      [self.order.task_to_backend(task) for task in tasks], dt)
        if not isinstance(result, ControlResult):
            raise ValueError("wrapped controller must return ControlResult")
        solver = _solver_snapshot(result.solver)
        if not solver.success:
            if result.command is not None:
                raise ValueError("failed controller returned a command")
            return ControlResult(solver)
        if not isinstance(result.command, VelocityCommand):
            raise ValueError("solved controller must return VelocityCommand")
        return ControlResult(solver, self.order.command_to_model(result.command))


class ReorderedPlanner:
    """Wrap a Planner already constructed with backend-order limits/settings."""

    def __init__(self, planner, order):
        if not isinstance(order, JointOrder):
            raise TypeError("order must be a JointOrder")
        if not all(callable(getattr(planner, name, None)) for name in ("reset", "solve")):
            raise TypeError("planner must implement reset and solve")
        self.planner, self.order = planner, order

    def reset(self):
        self.planner.reset()

    def solve(self, state, goal):
        goal = array(goal, 1, "goal")
        result = self.planner.solve(self.order.state_to_backend(state), self.order.to_backend(goal))
        if not isinstance(result, PlanResult):
            raise ValueError("wrapped planner must return PlanResult")
        solver = _solver_snapshot(result.solver)
        if not solver.success:
            if result.plan is not None:
                raise ValueError("failed planner returned a plan")
            return PlanResult(solver)
        if not isinstance(result.plan, JointPlan):
            raise ValueError("solved planner must return JointPlan")
        return PlanResult(solver, self.order.plan_to_model(result.plan))
