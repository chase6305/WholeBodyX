"""Deterministic velocity-integrator backend for algorithm tests."""

import numpy as np

from .types import RobotState, array, positive


class JointIntegrator:
    def __init__(self, limits, initial, *, control_dt=None):
        limits.check_state(initial)
        if np.any(initial.q < limits.lower) or np.any(initial.q > limits.upper):
            raise ValueError("initial configuration is outside joint limits")
        if np.any(np.abs(initial.v) > limits.velocity):
            raise ValueError("initial velocity exceeds limits")
        self.limits = limits
        self._control_dt = None if control_dt is None else positive(control_dt, "simulation control dt")
        self._state = initial

    control_dt = property(lambda self: self._control_dt)

    def read(self):
        return self._state

    def inject_configuration_offset(self, offset):
        """Simulation-only state jump; preserves clock/velocity, does not model force."""
        offset = array(offset, 1, "configuration offset")
        state = self._state
        if offset.shape != state.q.shape:
            raise ValueError("configuration offset dimensions differ")
        candidate = RobotState(state.q + offset, state.v, state.stamp)
        if np.any(candidate.q < self.limits.lower) or np.any(candidate.q > self.limits.upper):
            raise ValueError("configuration offset exceeds joint limits")
        self._state = candidate

    def write(self, command, dt=None):
        if dt is None:
            if self.control_dt is None:
                raise ValueError("provide simulation dt or configure control_dt at construction")
            dt = self.control_dt
        elif self.control_dt is not None and float(dt) != self.control_dt:
            raise ValueError("simulation dt differs from configured control_dt")
        dt = positive(dt, "simulation dt")
        state = self._state
        if command.mode != "velocity" or command.velocity.shape != state.q.shape:
            raise ValueError("command mode or dimensions do not match backend")
        if abs(command.stamp - state.stamp) > 1e-9 or state.stamp + dt > command.valid_until + 1e-9:
            raise ValueError("command does not cover the simulation interval")
        lo, hi = self.limits.velocity_bounds(state, dt)
        if np.any(command.velocity < lo - 1e-6) or np.any(command.velocity > hi + 1e-6):
            raise ValueError("command violates configured limits")
        self._state = RobotState(state.q + dt * command.velocity, command.velocity, state.stamp + dt)

    def stop(self):
        """Velocity-mode simulation stop; preserve configuration and clock."""
        state = self._state
        self._state = RobotState(state.q, np.zeros_like(state.v), state.stamp)
