"""Synchronous torque execution for prescribed floating-base contacts.

This is a driver boundary, not a contact switcher or a real-time scheduler.
One owner serializes all model, controller, gate and driver calls.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .dynamics_types import (CoMReference, DynamicsControlResult, DynamicsState,
                             TorqueCommand)
from .hardware import ExecutionDeadlineError, _check_cycle_budget
from .types import JointLimits, array, positive


class TorqueRobotBackend(Protocol):
    def read(self) -> DynamicsState: ...

    def write(self, command: TorqueCommand) -> None: ...

    def stop(self) -> None: ...


class TorqueCommandGate:
    """Bind accepted torque/dynamics output to one fresh model-order state."""

    def __init__(self, dynamics, control_dt, *, limits=None, effort_limits=None, friction=.6):
        self.dynamics = dynamics
        self.control_dt = positive(control_dt, "control dt")
        if self.control_dt > .02:
            raise ValueError("torque control dt must not exceed 0.02 s")
        model = dynamics.model
        self.limits = (JointLimits(model.lowerPositionLimit[7:], model.upperPositionLimit[7:],
                                    model.velocityLimit[6:]) if limits is None else limits)
        effort = model.effortLimit[6:] if effort_limits is None else effort_limits
        self.effort_limits = array(effort, 1, "effort limits")
        if (self.limits.lower.size != dynamics.na or self.effort_limits.shape != (dynamics.na,)
                or np.any(self.effort_limits <= 0)):
            raise ValueError("positive effort and joint limits must match actuated dimensions")
        self.friction = positive(friction, "friction")
        self._cones = tuple(spec.wrench_cone(self.friction) for spec in dynamics.contact_specs)
        self.reset()

    def reset(self):
        self._last_stamp = self._observed = self._fault = None

    @property
    def fault(self):
        return self._fault

    def latch_fault(self, detail):
        if self._fault is None:
            self._fault = str(detail)

    def observe(self, state):
        if self.fault is not None:
            raise RuntimeError(f"torque gate is faulted: {self.fault}; reset before continuing")
        if not isinstance(state, DynamicsState):
            raise ValueError("torque backend must return DynamicsState")
        self.dynamics.validate_state(state)
        if self._last_stamp is not None and state.stamp <= self._last_stamp:
            raise ValueError("robot state clock did not advance")
        if (np.any(state.q[7:] < self.limits.lower - 1e-6)
                or np.any(state.q[7:] > self.limits.upper + 1e-6)
                or np.any(np.abs(state.v[6:]) > self.limits.velocity + 1e-6)):
            raise ValueError("measured actuated state violates joint limits")
        self._observed = DynamicsState(state.q, state.v, state.stamp)
        self._last_stamp = state.stamp
        return state

    def validate(self, state, control):
        if self.fault is not None:
            raise RuntimeError(f"torque gate is faulted: {self.fault}; reset before continuing")
        observed = self._observed
        if (observed is None or not isinstance(state, DynamicsState) or state.stamp != observed.stamp
                or not np.array_equal(state.q, observed.q) or not np.array_equal(state.v, observed.v)):
            raise ValueError("torque command state differs from the latest observed state")
        if not isinstance(control, DynamicsControlResult):
            raise ValueError("torque controller must return DynamicsControlResult")
        command = control.command
        if not command.covers(state.stamp, self.control_dt):
            raise ValueError("torque command must cover the current control interval")
        nv, na, nc = self.dynamics.nv, self.dynamics.na, 6 * len(self._cones)
        if (control.acceleration.shape != (nv,) or command.torque.shape != (na,)
                or control.wrench.shape != (nc,) or control.contact_rhs.shape != (nc,)):
            raise ValueError("torque result and model dimensions differ")
        if (not control.solver.success or not np.isfinite(control.violation)
                or not 0 <= control.violation <= 1e-5):
            raise ValueError("torque controller reported an invalid constraint violation")
        if np.any(np.abs(command.torque) > self.effort_limits + 1e-5):
            raise ValueError("torque command exceeds configured effort limits")
        for cone, wrench in zip(self._cones, control.wrench.reshape(-1, 6)):
            if np.max(cone @ wrench) > 1e-5:
                raise ValueError("torque result violates physical contact bounds")
        # Recompute terms from the observed state. Do not accept a reported zero
        # residual or a controller's stale dynamics evaluation as evidence.
        terms = self.dynamics.terms(observed)
        dynamics_residual = (terms.mass @ control.acceleration + terms.bias
                             - self.dynamics.selection @ command.torque
                             - terms.contact_jacobian.T @ control.wrench)
        contact_residual = terms.contact_jacobian @ control.acceleration - control.contact_rhs
        if (not np.isfinite(dynamics_residual).all() or not np.isfinite(contact_residual).all()
                or np.max(np.abs(dynamics_residual)) > 1e-5
                or np.max(np.abs(contact_residual)) > 1e-5):
            raise ValueError("torque result fails independently evaluated dynamics constraints")
        dt = self.control_dt
        velocity = state.v[6:] + dt * control.acceleration[6:]
        position = state.q[7:] + dt * state.v[6:] + .5 * dt**2 * control.acceleration[6:]
        if (self.limits.acceleration is not None
                and np.any(np.abs(control.acceleration[6:]) > self.limits.acceleration + 1e-6)):
            raise ValueError("torque result exceeds configured actuated acceleration limits")
        if (not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or np.any(position < self.limits.lower - 1e-6)
                or np.any(position > self.limits.upper + 1e-6)
                or np.any(np.abs(velocity) > self.limits.velocity + 1e-6)):
            raise ValueError("torque result predicts an actuated joint limit violation")
        return TorqueCommand(command.torque, command.stamp, command.valid_until)


@dataclass(frozen=True)
class TorqueCycleResult:
    state: DynamicsState
    control: DynamicsControlResult
    control_ms: float
    cycle_ms: float

    @property
    def command(self):
        return self.control.command


class TorqueControlLoop:
    """read → reference factory → dynamics controller → torque gate → write.

    Any cycle/reset failure stops the backend once and latches a fault. Only a
    successful reset permits more I/O. Optional wall budgets detect overruns at
    call boundaries; blocking calls cannot be preempted.
    """

    def __init__(self, backend, dynamics, controller, control_dt=.005, *, limits=None,
                 effort_limits=None, friction=.6, max_cycle_time=None):
        if not all(callable(getattr(backend, name, None)) for name in ("read", "write", "stop")):
            raise TypeError("torque backend must implement read, write, and stop")
        if not all(callable(getattr(controller, name, None)) for name in ("reset", "step")):
            raise TypeError("dynamics controller must implement reset and step")
        self.backend, self.controller = backend, controller
        self.gate = TorqueCommandGate(dynamics, control_dt, limits=limits,
                                      effort_limits=effort_limits, friction=friction)
        self.control_dt = self.gate.control_dt
        self._max_cycle_time = None if max_cycle_time is None else positive(max_cycle_time, "max cycle time")

    max_cycle_time = property(lambda self: self._max_cycle_time)
    fault = property(lambda self: self.gate.fault)

    def _fail(self, exc, operation):
        self.gate.latch_fault(str(exc) or type(exc).__name__)
        detail = str(exc)
        try:
            self.backend.stop()
        except BaseException as stop_exc:
            if not isinstance(exc, Exception):
                raise exc from stop_exc
            detail += f"; backend stop failed: {type(stop_exc).__name__}: {stop_exc}"
        if not isinstance(exc, Exception):
            raise exc
        raise RuntimeError(f"torque control {operation} fault: {detail}") from exc

    def reset(self):
        try:
            self.controller.reset()
        except BaseException as exc:
            self._fail(exc, "reset")
        self.gate.reset()

    def step(self, reference_factory):
        if self.fault is not None:
            raise RuntimeError(f"torque loop is faulted: {self.fault}; reset before continuing")
        started = perf_counter()

        def deadline(stage, command=None):
            return _check_cycle_budget(perf_counter() - started, self.max_cycle_time,
                                       "torque", stage, command)

        try:
            state = self.gate.observe(self.backend.read())
            deadline("after_read")
            reference = reference_factory(state) if callable(reference_factory) else reference_factory
            if not isinstance(reference, CoMReference) or not reference.covers(state.stamp, self.control_dt):
                raise ValueError("CoM reference must cover the current torque control interval")
            deadline("after_reference")
            before = perf_counter()
            control = self.controller.step(state, reference, self.control_dt)
            control_ms = (perf_counter() - before) * 1000
            command = self.gate.validate(state, control)
            deadline("before_write", command)
            self.backend.write(command)
            cycle_ms = deadline("after_write", command)
            return TorqueCycleResult(state, control, control_ms, cycle_ms)
        except BaseException as exc:
            self._fail(exc, "cycle")
