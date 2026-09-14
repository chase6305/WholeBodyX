"""Synchronous, velocity-mode execution boundary for robot-driver adapters.

Drivers supply model-order RobotState values and execute VelocityCommand values.
This module validates the transfer; it does not open devices, schedule threads, or
claim a hard real-time deadline.
"""

from dataclasses import dataclass
from math import ulp
from time import perf_counter
from typing import Callable, Protocol

import numpy as np

from .joint_order import JointOrder
from .qp import QPResult
from .types import JointLimits, LinearTask, RobotState, VelocityCommand, array, positive
from .wbc import ControlResult


def _same_time(a, b):
    return abs(a - b) <= 4 * max(ulp(a), ulp(b))


class VelocityRobotBackend(Protocol):
    """One owner calls read/write/stop serially from the control-loop thread."""

    def read(self) -> RobotState: ...

    def write(self, command: VelocityCommand) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class CommandGate:
    """Checks a sampled state and candidate velocity before a driver write."""

    limits: JointLimits
    control_dt: float

    def __post_init__(self):
        positive(self.control_dt, "control dt")
        object.__setattr__(self, "_last_stamp", None)
        object.__setattr__(self, "_observed_stamp", None)
        object.__setattr__(self, "_fault", None)

    @property
    def fault(self):
        return self._fault

    def reset(self):
        object.__setattr__(self, "_last_stamp", None)
        object.__setattr__(self, "_observed_stamp", None)
        object.__setattr__(self, "_fault", None)

    def latch_fault(self, detail):
        if self._fault is None:
            object.__setattr__(self, "_fault", str(detail))

    def observe(self, state):
        if self._fault is not None:
            raise RuntimeError(f"command gate is faulted: {self._fault}; reset before continuing")
        if not isinstance(state, RobotState):
            raise ValueError("robot backend must return RobotState")
        self.limits.check_state(state)
        tolerance = 1e-6
        if np.any(state.q < self.limits.lower - tolerance) or np.any(state.q > self.limits.upper + tolerance):
            raise ValueError("measured configuration is outside configured joint limits")
        if np.any(np.abs(state.v) > self.limits.velocity + tolerance):
            raise ValueError("measured velocity exceeds configured limits")
        if self._last_stamp is not None and state.stamp <= self._last_stamp:
            raise ValueError("robot state clock did not advance")
        object.__setattr__(self, "_last_stamp", state.stamp)
        object.__setattr__(self, "_observed_stamp", state.stamp)
        return state

    def validate(self, state, command):
        if self._fault is not None:
            raise RuntimeError(f"command gate is faulted: {self._fault}; reset before continuing")
        if not isinstance(command, VelocityCommand):
            raise ValueError("controller must return VelocityCommand")
        if self._observed_stamp is None or not _same_time(state.stamp, self._observed_stamp):
            raise ValueError("command was not validated against the latest observed state")
        if not _same_time(command.stamp, state.stamp):
            raise ValueError("command timestamp does not match measured state")
        end = state.stamp + self.control_dt
        if not np.isfinite(end) or end <= state.stamp:
            raise ValueError("control interval cannot be represented")
        if command.valid_until < end - 4 * max(ulp(end), ulp(command.valid_until)):
            raise ValueError("command does not cover the complete control interval")
        if command.velocity.shape != state.q.shape:
            raise ValueError("command and measured state dimensions differ")
        lo, hi = self.limits.velocity_bounds(state, self.control_dt)
        if np.any(command.velocity < lo - 1e-6) or np.any(command.velocity > hi + 1e-6):
            raise ValueError("command violates configured position, velocity, or slew limits")
        return VelocityCommand(command.velocity, command.stamp, command.valid_until)


class ModelOrderRobotBackend:
    """Make a named external velocity backend appear in the model's joint order."""

    def __init__(self, backend, model_names, backend_names):
        if not all(callable(getattr(backend, name, None)) for name in ("read", "write", "stop")):
            raise TypeError("robot backend must implement read, write, and stop")
        self.backend = backend
        self.order = JointOrder(model_names, backend_names)
        self.joint_names = self.order.model_names

    def read(self):
        state = self.backend.read()
        if not isinstance(state, RobotState):
            raise ValueError("robot backend must return RobotState")
        return self.order.state_to_model(state)

    def write(self, command):
        if not isinstance(command, VelocityCommand):
            raise ValueError("expected VelocityCommand")
        self.backend.write(VelocityCommand(self.order.to_backend(command.velocity), command.stamp, command.valid_until))

    def stop(self):
        self.backend.stop()


@dataclass(frozen=True)
class CycleResult:
    state: RobotState
    control: ControlResult
    control_ms: float

    @property
    def command(self):
        return self.control.command


class VelocityControlLoop:
    """One synchronous control cycle: read → task factory → WBC → gate → write.

    Any read, task, solve, validation, or write error latches a fault and calls the
    backend's stop method once. Call reset() before attempting another cycle.
    """

    def __init__(self, backend, limits, controller, control_dt):
        if not all(callable(getattr(backend, name, None)) for name in ("read", "write", "stop")):
            raise TypeError("robot backend must implement read, write, and stop")
        if not all(callable(getattr(controller, name, None)) for name in ("reset", "step")):
            raise TypeError("controller must implement reset and step")
        self.backend, self.controller = backend, controller
        self.gate = CommandGate(limits, control_dt)
        self.control_dt = self.gate.control_dt

    @property
    def fault(self):
        return self.gate.fault

    def reset(self):
        self.controller.reset()
        self.gate.reset()

    def _fail(self, exc):
        self.gate.latch_fault(exc)
        detail = str(exc)
        try:
            self.backend.stop()
        except Exception as stop_exc:
            detail = f"{detail}; backend stop failed: {type(stop_exc).__name__}: {stop_exc}"
        raise RuntimeError(f"velocity control cycle fault: {detail}") from exc

    def step(self, task_factory: Callable[[RobotState], list[LinearTask]]):
        if not callable(task_factory):
            raise TypeError("task_factory must be callable")
        if self.fault is not None:
            raise RuntimeError(f"velocity control loop is faulted: {self.fault}; reset before continuing")
        try:
            state = self.gate.observe(self.backend.read())
            tasks = task_factory(state)
            if not isinstance(tasks, (list, tuple)) or not all(isinstance(task, LinearTask) for task in tasks):
                raise ValueError("task_factory must return a list or tuple of LinearTask values")
            started = perf_counter()
            control = self.controller.step(state, list(tasks), self.control_dt)
            elapsed = (perf_counter() - started) * 1000
            if not isinstance(control, ControlResult) or not isinstance(control.solver, QPResult):
                raise ValueError("controller must return ControlResult with QPResult")
            if not control.solver.success or control.command is None:
                raise RuntimeError(f"WBC rejected command: {control.solver.status} — {control.solver.detail}")
            if not np.isfinite(control.solver.violation) or control.solver.violation < 0 or control.solver.violation > 1e-6:
                raise ValueError("controller reported an invalid constraint violation")
            command = self.gate.validate(state, control.command)
            self.backend.write(command)
            return CycleResult(state, ControlResult(control.solver, command), elapsed)
        except Exception as exc:
            self._fail(exc)
