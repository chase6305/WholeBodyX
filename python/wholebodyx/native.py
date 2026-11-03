"""Adapters from the optional extension to the pure Python data contracts."""

import numpy as np

from .mpc import JointPlan, PlanResult, _MPCSettings
from .qp import QPResult, QuadraticProgram
from .types import VelocityCommand, array
from .wbc import ControlResult, KinematicWBC
from ._solver_execution import OSQP_EXECUTION_LOCK


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


def _sparse_extension():
    module = _extension()
    if not hasattr(module, "SparseJointMPC"):
        raise ImportError("Incompatible wholebodyx-native: missing SparseJointMPC; rebuild the native wheel")
    return module


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
        native_state, native_tasks = _state(self._module, state), self._tasks(tasks)
        with OSQP_EXECUTION_LOCK:
            r = self._impl.step(native_state, native_tasks, dt)
        c = r.command
        command = VelocityCommand(c.velocity, c.stamp, c.valid_until) if c is not None else None
        return ControlResult(_result(r.solver), command)


class _NativeMPC:
    """Shared reset and result conversion; formulations keep separate types."""
    def reset(self):
        self._impl.reset()

    def solve(self, state, goal):
        native_state = _state(self._module, state)
        with OSQP_EXECUTION_LOCK:
            r = self._impl.solve(native_state, goal)
        p = r.plan
        plan = JointPlan(p.positions, p.velocities, p.stamp, p.dt) if p is not None else None
        return PlanResult(_result(r.solver), plan)


class NativeMPC(_NativeMPC):
    def __init__(self, limits, horizon=15, dt=0.1, tracking_weight=10.0,
                 control_weight=0.1, terminal_weight=30.0):
        _MPCSettings(limits, horizon, dt, tracking_weight, control_weight, terminal_weight)
        self._module = _extension()
        self._impl = self._module.JointMPC(_limits(self._module, limits), horizon, dt,
                                          tracking_weight, control_weight, terminal_weight)

    @property
    def fixed_matrix_bytes(self):
        """C++ model H/A storage, excluding solver workspaces; old wheels: None."""
        return getattr(self._impl, "fixed_matrix_bytes", None)

    def formulate(self, state, goal):
        return _problem(self._impl.formulate(_state(self._module, state), goal))


class NativeSparseMPC(_NativeMPC):
    """C++ uncondensed MPC; matrix storage and parameter updates stay sparse.

    This optional feature requires a native wheel exposing SparseJointMPC.
    Older wheels can still provide NativeMPC and NativeWBC independently.
    """

    def __init__(self, limits, horizon=15, dt=.1, tracking_weight=10.,
                 control_weight=.1, terminal_weight=30.):
        self._settings = _MPCSettings(limits, horizon, dt, tracking_weight,
                                      control_weight, terminal_weight)
        self._module = _sparse_extension()
        self._impl = self._module.SparseJointMPC(_limits(self._module, self.limits), self.horizon,
                                               self.dt, self.tracking_weight, self.control_weight,
                                               self.terminal_weight)

    limits = property(lambda self: self._settings.limits)
    horizon = property(lambda self: self._settings.horizon)
    dt = property(lambda self: self._settings.dt)
    tracking_weight = property(lambda self: self._settings.tracking_weight)
    control_weight = property(lambda self: self._settings.control_weight)
    terminal_weight = property(lambda self: self._settings.terminal_weight)

    def formulate(self, state, goal):
        from .sparse_qp import SparseQuadraticProgram
        p = self._impl.formulate(_state(self._module, state), goal)
        return SparseQuadraticProgram(p.hessian, p.gradient, p.constraint, p.lower, p.upper)
