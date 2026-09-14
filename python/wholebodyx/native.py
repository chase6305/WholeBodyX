"""Adapters from the optional extension to the pure Python data contracts."""

import numpy as np

from .mpc import JointPlan, PlanResult, _MPCSettings
from .qp import QPResult, QuadraticProgram
from .types import VelocityCommand, array
from .wbc import ControlResult, KinematicWBC


def _extension():
    try:
        import wholebodyx_native
    except (ImportError, OSError) as exc:
        raise ImportError(f"Cannot load wholebodyx-native: {exc}. Build/install the matching native wheel; see README.md") from exc
    required = ("JointLimits", "RobotState", "LinearTask", "JointPlan", "KinematicWBC", "JointMPC")
    missing = [name for name in required if not hasattr(wholebodyx_native, name)]
    if missing:
        raise ImportError(f"Incompatible wholebodyx-native: missing {', '.join(missing)}; rebuild the native wheel")
    return wholebodyx_native


def _limits(module, limits):
    a = limits.acceleration if limits.acceleration is not None else np.empty(0)
    return module.JointLimits(limits.lower, limits.upper, limits.velocity, a)


def _state(module, state):
    return module.RobotState(state.q, state.v, state.stamp)


def _result(result):
    return QPResult(result.status, array(result.x, 1, "solution") if result.success else None,
                    result.iterations, result.violation, result.detail)


def _problem(p):
    return QuadraticProgram(p.hessian, p.gradient, p.constraint, p.lower, p.upper)


class NativeWBC:
    def __init__(self, limits, regularization=1e-6, smoothing=0.0):
        # Keep configuration validation identical across language frontends.
        KinematicWBC(limits, regularization, smoothing)
        self._module = _extension()
        self._impl = self._module.KinematicWBC(_limits(self._module, limits), regularization, smoothing)

    def reset(self):
        self._impl.reset()

    def _tasks(self, tasks):
        return [self._module.LinearTask(t.jacobian, t.target_velocity, t.weight) for t in tasks]

    def formulate(self, state, tasks, dt):
        return _problem(self._impl.formulate(_state(self._module, state), self._tasks(tasks), dt))

    def step(self, state, tasks, dt):
        r = self._impl.step(_state(self._module, state), self._tasks(tasks), dt)
        c = r.command
        command = VelocityCommand(c.velocity, c.stamp, c.valid_until) if c is not None else None
        return ControlResult(_result(r.solver), command)


class NativeMPC:
    def __init__(self, limits, horizon=15, dt=0.1, tracking_weight=10.0,
                 control_weight=0.1, terminal_weight=30.0):
        _MPCSettings(limits, horizon, dt, tracking_weight, control_weight, terminal_weight)
        self._module = _extension()
        self._impl = self._module.JointMPC(_limits(self._module, limits), horizon, dt,
                                          tracking_weight, control_weight, terminal_weight)

    def reset(self):
        self._impl.reset()

    def formulate(self, state, goal):
        return _problem(self._impl.formulate(_state(self._module, state), goal))

    def solve(self, state, goal):
        r = self._impl.solve(_state(self._module, state), goal)
        p = r.plan
        plan = JointPlan(p.positions, p.velocities, p.stamp, p.dt) if p is not None else None
        return PlanResult(_result(r.solver), plan)
