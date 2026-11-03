"""Torque-driven simulation with prescribed persistent bilateral contacts.

The plant computes its own contact stabilization and forward dynamics. It does
not consume the controller's predicted acceleration, wrench, or contact RHS.
No contact release, impact, slip, or collision dynamics are modeled.
"""

import numpy as np

from .dynamics_types import DynamicsState, TorqueCommand
from .types import positive


class FixedContactSimulator:
    def __init__(self, dynamics, initial, control_dt=0.005, *, friction=0.6, external_force=None):
        dynamics.validate_state(initial)
        self.dynamics = dynamics
        self._control_dt = positive(control_dt, "simulation control dt")
        if self.control_dt > 0.02:
            raise ValueError("simulation control dt must not exceed 0.02 s")
        if external_force is not None and not callable(external_force):
            raise TypeError("external_force must be a callable of the measured state")
        self.external_force = external_force
        self.friction = positive(friction, "friction")
        self._cones = tuple(spec.wrench_cone(self.friction) for spec in dynamics.contact_specs)
        self._state = DynamicsState(initial.q, initial.v, initial.stamp)
        self.anchors = tuple(pose.copy() for pose in dynamics.terms(initial).contact_poses)
        self.acceleration = self.wrench = self.last_command = None
        self.steps = self.stops = 0

    control_dt = property(lambda self: self._control_dt)

    def read(self):
        return self._state

    def write(self, command):
        state, robot, dt = self._state, self.dynamics, self.control_dt
        if not isinstance(command, TorqueCommand) or not command.covers(state.stamp, dt):
            raise ValueError("torque command must cover the complete simulation interval")
        if command.torque.shape != (robot.na,):
            raise ValueError("torque command and simulation actuator dimensions differ")
        if np.any(np.abs(command.torque) > robot.model.effortLimit[6:] + 1e-5):
            raise ValueError("torque command exceeds simulation effort limits")
        terms = robot.terms(state)
        error = np.concatenate(
            [
                robot.pin.log6(pose.inverse() * anchor).vector
                for pose, anchor in zip(terms.contact_poses, self.anchors)
            ]
        )
        rhs = -terms.contact_drift + 100 * error - 20 * (terms.contact_jacobian @ state.v)
        external = None if self.external_force is None else self.external_force(state)
        acceleration, wrench = robot.forward(
            state, command.torque, rhs, terms, external_force=external
        )
        if (
            np.shape(acceleration) != (robot.nv,)
            or np.shape(wrench) != (6 * len(self._cones),)
            or not np.isfinite(acceleration).all()
            or not np.isfinite(wrench).all()
        ):
            raise RuntimeError("invalid forward dynamics result")
        for cone, force in zip(self._cones, wrench.reshape(-1, 6)):
            if np.max(cone @ force) > 1e-5:
                raise RuntimeError("forward contact wrench violates physical support bounds")
        candidate = robot.integrate(state, acceleration, dt)
        model = robot.model
        if (
            np.any(candidate.q[7:] < model.lowerPositionLimit[7:] - 1e-6)
            or np.any(candidate.q[7:] > model.upperPositionLimit[7:] + 1e-6)
            or np.any(np.abs(candidate.v[6:]) > model.velocityLimit[6:] + 1e-6)
        ):
            raise RuntimeError("forward dynamics violates actuated joint limits")
        # Commit only after forward dynamics, contact and state checks succeed.
        self._state = candidate
        self.acceleration, self.wrench = acceleration.copy(), wrench.copy()
        self.last_command = TorqueCommand(command.torque, command.stamp, command.valid_until)
        self.steps += 1

    def stop(self):
        """Stop advancing the simulation; no instantaneous physical brake is modeled."""
        self.stops += 1
        self.last_command = None
