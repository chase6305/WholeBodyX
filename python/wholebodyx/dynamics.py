"""Experimental floating-base inverse-dynamics WBC with prescribed rigid contacts.

Requires the optional Pinocchio dependency. Contact switches and impacts are not
implemented; the forward plant enforces the same persistent bilateral contacts.
"""

from pathlib import Path

import numpy as np
from ._solver_execution import OSQP_EXECUTION_LOCK

import osqp
from scipy import sparse

from .qp import QPResult, QuadraticProgram
from .contact import ContactSpec
from .dynamics_types import (
    CoMReference,
    ContactFormulation,
    DynamicsControlResult,
    DynamicsState,
    DynamicsTerms,
    TorqueCommand,
)
from .types import positive


class FloatingBaseDynamics:
    def __init__(self, urdf, feet=("left_ankle_link", "right_ankle_link"), *, contact_specs=None):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf)), pin.JointModelFreeFlyer())
        if self.model.nq != self.model.nv + 1:
            raise ValueError("requires scalar actuated joints after the free-flyer")
        if contact_specs is None:
            contact_specs = tuple(ContactSpec(name) for name in feet)
        else:
            contact_specs = tuple(contact_specs)
            if not all(isinstance(spec, ContactSpec) for spec in contact_specs):
                raise TypeError("contact_specs must contain ContactSpec values")
        names = tuple(spec.frame for spec in contact_specs)
        if not names or len(set(names)) != len(names):
            raise ValueError("at least one distinct contact frame is required")
        self.contact_specs = contact_specs
        self.contacts = []
        for spec in self.contact_specs:
            name = spec.frame
            parent = self.model.getFrameId(name)
            if parent >= self.model.nframes:
                raise ValueError(f"unknown contact frame: {name}")
            frame = self.model.frames[parent]
            placement = frame.placement * pin.SE3(spec.rotation, spec.translation)
            self.contacts.append(
                self.model.addFrame(
                    pin.Frame(name + "_sole", frame.parentJoint, parent, placement, pin.OP_FRAME)
                )
            )
        self.data = self.model.createData()
        self.nv, self.na = self.model.nv, self.model.nv - 6
        self.selection = np.eye(self.nv)[:, 6:]
        self._mass_lower_rows, self._mass_lower_columns = np.tril_indices(self.nv, -1)

    def validate_state(self, state):
        q, v = np.asarray(state.q, dtype=float), np.asarray(state.v, dtype=float)
        if (
            q.shape != (self.model.nq,)
            or v.shape != (self.nv,)
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or not np.isfinite(state.stamp)
            or state.stamp < 0
        ):
            raise ValueError("invalid floating-base state")
        if abs(np.linalg.norm(q[3:7]) - 1.0) > 1e-8:
            raise ValueError("floating-base quaternion must be normalized")
        return q, v

    def terms(self, state):
        pin, model, data = self.pin, self.model, self.data
        q, v = self.validate_state(state)
        mass = pin.crba(model, data, q).copy()
        # CRBA writes the upper triangle; reflect it in this owned snapshot.
        rows, columns = self._mass_lower_rows, self._mass_lower_columns
        mass[rows, columns] = mass[columns, rows]
        bias = pin.nonLinearEffects(model, data, q, v).copy()
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        pin.updateFramePlacements(model, data)
        poses = [data.oMf[f].copy() for f in self.contacts]
        contact_jacobian = np.vstack(
            [pin.getFrameJacobian(model, data, f, pin.LOCAL) for f in self.contacts]
        )
        contact_drift = np.concatenate(
            [
                pin.getFrameJacobianTimeVariation(model, data, f, pin.LOCAL) @ v
                for f in self.contacts
            ]
        )
        com = pin.centerOfMass(model, data, q, v, np.zeros(model.nv)).copy()
        com_velocity, com_drift = data.vcom[0].copy(), data.acom[0].copy()
        com_jacobian = pin.jacobianCenterOfMass(model, data, q).copy()
        return DynamicsTerms(
            mass,
            bias,
            contact_jacobian,
            contact_drift,
            poses,
            com,
            com_velocity,
            com_drift,
            com_jacobian,
        )

    def forward(self, state, torque, contact_acceleration, terms, *, external_force=None):
        """Solve constrained forward dynamics from actuator torque, not QP ddq."""
        self.validate_state(state)
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (self.na,) or not np.isfinite(torque).all():
            raise ValueError("invalid actuator torque")
        mass, bias, contact_jacobian = terms[:3]
        contact_acceleration = np.asarray(contact_acceleration, dtype=float)
        nc = len(contact_jacobian)
        # Block assignment must not broadcast malformed model terms.
        if np.shape(mass) != (self.nv, self.nv) or np.shape(contact_jacobian) != (nc, self.nv):
            raise ValueError("dynamics term dimensions do not match the model")
        if contact_acceleration.shape != (nc,) or not np.isfinite(contact_acceleration).all():
            raise ValueError("invalid contact acceleration")
        external_force = (
            np.zeros(self.nv) if external_force is None else np.asarray(external_force, dtype=float)
        )
        if external_force.shape != (self.nv,) or not np.isfinite(external_force).all():
            raise ValueError("invalid generalized external force")
        system = np.empty(
            (self.nv + nc, self.nv + nc), dtype=np.result_type(mass, contact_jacobian, float)
        )
        system[: self.nv, : self.nv] = mass
        system[: self.nv, self.nv :] = -contact_jacobian.T
        system[self.nv :, : self.nv] = contact_jacobian
        system[self.nv :, self.nv :] = 0.0
        rhs = np.concatenate(
            (self.selection @ torque + external_force - bias, contact_acceleration)
        )
        # H1 has five joints per leg; the two six-axis constraints contain
        # a redundant row at symmetric poses. Acceleration is unique, while
        # internal contact wrench can be indeterminate. Use a rank-aware solve.
        solution = np.linalg.lstsq(system, rhs, rcond=1e-10)[0]
        if not np.isfinite(solution).all():
            raise RuntimeError("nonfinite constrained forward dynamics")
        residual = system @ solution - rhs
        if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1e-5:
            raise RuntimeError("inconsistent constrained forward dynamics")
        return solution[: self.nv], solution[self.nv :]

    def integrate(self, state, acceleration, dt):
        dt = float(dt)
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError("dt must be in (0, 0.02]")
        q, v = self.validate_state(state)
        start = float(state.stamp)
        stamp = start + dt
        if not np.isfinite(stamp) or stamp <= start:
            raise ValueError("integration interval cannot be represented")
        acceleration = np.asarray(acceleration, dtype=float)
        if acceleration.shape != (self.nv,) or not np.isfinite(acceleration).all():
            raise ValueError("invalid generalized acceleration")
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = v + dt * acceleration
            displacement = dt * (v + 0.5 * dt * acceleration)
        if not np.isfinite(velocity).all() or not np.isfinite(displacement).all():
            raise ValueError("dynamics integration overflow")
        q = self.pin.integrate(self.model, q, displacement)
        if not np.isfinite(q).all():
            raise ValueError("dynamics integration produced a nonfinite configuration")
        return DynamicsState(q, velocity, stamp)


class ContactWBC:
    """QP for prescribed contacts in physical coordinates.

    The decision order is generalized acceleration, actuator torque, then local
    sole wrenches. Each contact contributes six [force; moment] coordinates.
    """

    def __init__(self, dynamics, initial, friction=0.6):
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("friction must be positive")
        self.robot, self._friction = dynamics, float(friction)
        self.nominal = initial.q.copy()
        self.anchors = dynamics.terms(initial)[4]
        self._contact_reserve = 0.0
        nv, na, nc = dynamics.nv, dynamics.na, 6 * len(self.anchors)
        decision_count = nv + na + nc
        self._hessian_template = np.diag(
            np.r_[np.full(nv, 1e-3), np.full(na, 1e-5), np.full(nc, 1e-6)]
        )
        self._posture_indices = np.arange(6, nv)
        self._orientation_indices = np.arange(3, 6)
        self._bounds_start = nv + nc
        self._cone_start = self._bounds_start + nv + na
        cones = [spec.wrench_cone(self._friction) for spec in dynamics.contact_specs]
        self._physical_wrench_constraint = np.zeros((sum(map(len, cones)), nc))
        row = 0
        for foot, cone in enumerate(cones):
            self._physical_wrench_constraint[row : row + len(cone), 6 * foot : 6 * foot + 6] = cone
            row += len(cone)
        self._planning_wrench_constraint = self._physical_wrench_constraint.copy()
        rows = self._cone_start + row
        self._constraint_template = np.zeros((rows, decision_count))
        self._constraint_template[self._bounds_start : self._cone_start, : nv + na] = np.eye(
            nv + na
        )
        self._lower_template, self._upper_template = np.zeros(rows), np.zeros(rows)
        self._lower_template[self._cone_start :] = -np.inf
        self._scale = np.r_[np.ones(nv), np.full(na, 50.0), np.full(nc, 300.0)]
        self._update_contact_cone()
        ContactWBC.reset(self)

    def reset(self):
        """Clear solver history and counters, retaining posture/contact anchors."""
        self.previous = None
        self._solver = None
        self.solver_setups = 0
        self.solver_recoveries = 0
        self._p_pattern = self._a_pattern = None
        self._p_columns = self._a_columns = None

    @property
    def friction(self):
        return self._friction

    @property
    def contact_reserve(self):
        return self._contact_reserve

    @contact_reserve.setter
    def contact_reserve(self, value):
        if not np.isfinite(value) or not 0 <= value <= 0.75:
            raise ValueError("contact reserve must be in [0, 0.75]")
        self._contact_reserve = float(value)
        self._planning_wrench_constraint = self._physical_wrench_constraint.copy()
        self._planning_wrench_constraint[:, 2::6] *= 1 - self._contact_reserve
        self._update_contact_cone()

    def _update_contact_cone(self):
        start = self.robot.nv + self.robot.na
        self._constraint_template[self._cone_start :, start:] = self._planning_wrench_constraint

    def contact_violation(self, wrench):
        """Check actual local sole wrenches using the controller's contact bounds."""
        force = np.asarray(wrench, dtype=float)
        if force.shape != (6 * len(self.anchors),) or not np.isfinite(force).all():
            raise ValueError("invalid contact wrench")
        return float(max(0.0, np.max(self._physical_wrench_constraint @ force)))

    def step(self, state, reference, dt=0.005):
        """Consume a typed, current CoM reference with full interval coverage."""
        dt = positive(dt, "control dt")
        if not isinstance(reference, CoMReference):
            raise ValueError("dynamics controller requires a CoMReference")
        if not reference.covers(state.stamp, dt):
            raise ValueError("CoM reference must cover the current control interval")
        return self.solve(
            state,
            reference.position,
            dt,
            target_velocity=reference.velocity,
            target_acceleration=reference.acceleration,
        )

    def formulate(
        self, state, target_com, dt=0.005, *, target_velocity=None, target_acceleration=None
    ):
        """Build the physical-units QP without advancing solver history."""
        robot = self.robot
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError("dt must be in (0, 0.02]")
        end = float(state.stamp) + float(dt)
        if not np.isfinite(end) or end <= state.stamp:
            raise ValueError("dynamics control interval cannot be represented")
        target_com = np.asarray(target_com, dtype=float)
        if target_com.shape != (3,) or not np.isfinite(target_com).all():
            raise ValueError("target_com must be a finite 3-vector")
        references = []
        for value in (target_velocity, target_acceleration):
            value = np.zeros(3) if value is None else np.asarray(value, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("CoM feedforward must be a finite 3-vector")
            references.append(value)
        target_velocity, target_acceleration = references
        terms = robot.terms(state)
        (
            mass,
            bias,
            contact_jacobian,
            contact_drift,
            poses,
            com,
            com_velocity,
            com_drift,
            com_jacobian,
        ) = terms
        nv, na, nc = robot.nv, robot.na, len(contact_jacobian)
        decision_count = nv + na + nc
        error = np.concatenate(
            [
                robot.pin.log6(pose.inverse() * anchor).vector
                for pose, anchor in zip(poses, self.anchors)
            ]
        )
        contact_rhs = -contact_drift + 100 * error - 20 * (contact_jacobian @ state.v)
        hessian = self._hessian_template.copy()
        gradient = np.zeros(decision_count)

        desired_com_acceleration = (
            target_acceleration
            + 60 * (target_com - com)
            + 16 * (target_velocity - com_velocity)
            - com_drift
        )
        hessian[:nv, :nv] += 1000.0 * com_jacobian.T @ com_jacobian
        gradient[:nv] -= 1000.0 * com_jacobian.T @ desired_com_acceleration
        posture = robot.pin.difference(robot.model, state.q, self.nominal)
        # These task Jacobians select coordinates; their normal matrices are
        # diagonal. Preserve task order without multiplying dense identities.
        hessian[self._posture_indices, self._posture_indices] += 1.0
        gradient[6:nv] -= 20 * posture[6:] - 8 * state.v[6:]
        hessian[self._orientation_indices, self._orientation_indices] += 10.0
        gradient[3:6] -= 10.0 * (30 * posture[3:6] - 10 * state.v[3:6])
        # Soft load-sharing target for this two-foot support controller.
        if len(poses) == 2:
            spacing = poses[0].translation[1] - poses[1].translation[1]
            if abs(spacing) > 1e-6:
                fraction = np.clip((target_com[1] - poses[1].translation[1]) / spacing, 0.1, 0.9)
                weight = robot.pin.computeTotalMass(robot.model) * np.linalg.norm(
                    robot.model.gravity.linear
                )
                for foot, share in enumerate((fraction, 1 - fraction)):
                    index = nv + na + 6 * foot + 2
                    hessian[index, index] += 0.001
                    gradient[index] -= 0.001 * weight * share
        constraint = self._constraint_template.copy()
        constraint[:nv, :nv], constraint[:nv, nv : nv + na], constraint[:nv, nv + na :] = (
            mass,
            -robot.selection,
            -contact_jacobian.T,
        )
        constraint[nv : nv + nc, :nv] = contact_jacobian
        lower, upper = self._lower_template.copy(), self._upper_template.copy()
        lower[:nv] = upper[:nv] = -bias
        lower[nv : nv + nc] = upper[nv : nv + nc] = contact_rhs
        acceleration_lower, acceleration_upper = np.full(nv, -40.0), np.full(nv, 40.0)
        velocity = np.minimum(robot.model.velocityLimit[6:], 2.0)
        acceleration_lower[6:] = np.maximum(acceleration_lower[6:], (-velocity - state.v[6:]) / dt)
        acceleration_upper[6:] = np.minimum(acceleration_upper[6:], (velocity - state.v[6:]) / dt)
        acceleration_lower[6:] = np.maximum(
            acceleration_lower[6:],
            2 * (robot.model.lowerPositionLimit[7:] - state.q[7:] - dt * state.v[6:]) / dt**2,
        )
        acceleration_upper[6:] = np.minimum(
            acceleration_upper[6:],
            2 * (robot.model.upperPositionLimit[7:] - state.q[7:] - dt * state.v[6:]) / dt**2,
        )
        # Reserve travel for braking with an assumed deceleration of 20.
        lower_speed = np.minimum(
            velocity,
            np.sqrt(2 * 20.0 * np.maximum(state.q[7:] - robot.model.lowerPositionLimit[7:], 0.0)),
        )
        upper_speed = np.minimum(
            velocity,
            np.sqrt(2 * 20.0 * np.maximum(robot.model.upperPositionLimit[7:] - state.q[7:], 0.0)),
        )
        acceleration_lower[6:] = np.maximum(
            acceleration_lower[6:], (-lower_speed - state.v[6:]) / dt
        )
        acceleration_upper[6:] = np.minimum(
            acceleration_upper[6:], (upper_speed - state.v[6:]) / dt
        )
        lower[self._bounds_start : self._bounds_start + nv] = acceleration_lower
        upper[self._bounds_start : self._bounds_start + nv] = acceleration_upper
        lower[self._bounds_start + nv : self._cone_start] = -robot.model.effortLimit[6:]
        upper[self._bounds_start + nv : self._cone_start] = robot.model.effortLimit[6:]
        problem = QuadraticProgram(hessian, gradient, constraint, lower, upper)
        if np.any(problem.lower > problem.upper):
            raise RuntimeError("dynamics WBC has inconsistent state/limit bounds")
        return ContactFormulation(problem, terms, contact_rhs)

    def solve(self, state, target_com, dt=0.005, *, target_velocity=None, target_acceleration=None):
        formulation = self.formulate(
            state,
            target_com,
            dt,
            target_velocity=target_velocity,
            target_acceleration=target_acceleration,
        )
        problem, terms, contact_rhs = formulation
        robot = self.robot
        mass, bias, contact_jacobian = terms[:3]
        nv, na, nc = robot.nv, robot.na, len(contact_jacobian)
        decision_count = nv + na + nc
        # Scale accelerations, torques and wrenches into comparable QP units.
        scale = self._scale
        matrix = problem.constraint * scale
        row_scale = 1 / np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
        scaled_hessian = problem.hessian * scale[:, None] * scale
        scaled_constraint = matrix * row_scale[:, None]
        for attempt in range(2):
            if self._solver is None:
                # Explicit zeros preserve the symbolic pattern as the robot moves.
                # Only the acceleration Hessian block is dense; all force/torque
                # regularizers are diagonal. Constraint couplings stay fixed.
                hessian_mask = np.eye(decision_count, dtype=bool)
                hessian_mask[:nv, :nv] = np.triu(np.ones((nv, nv), dtype=bool))
                constraint_mask = problem.constraint != 0
                constraint_mask[:nv, :nv] = True
                constraint_mask[:nv, nv + na :] = True
                constraint_mask[nv : nv + nc, :nv] = True
                self._p_pattern = sparse.csc_matrix(hessian_mask, dtype=float)
                self._a_pattern = sparse.csc_matrix(constraint_mask, dtype=float)
                self._p_columns = np.repeat(np.arange(decision_count), np.diff(self._p_pattern.indptr))
                self._a_columns = np.repeat(np.arange(decision_count), np.diff(self._a_pattern.indptr))
                self._p_pattern.data[:] = scaled_hessian[self._p_pattern.indices, self._p_columns]
                self._a_pattern.data[:] = scaled_constraint[
                    self._a_pattern.indices, self._a_columns
                ]
                self._solver = osqp.OSQP()
                tolerance = 1e-8 if attempt == 0 else 1e-9
                self._solver.setup(
                    P=self._p_pattern,
                    q=problem.gradient * scale,
                    A=self._a_pattern,
                    l=problem.lower * row_scale,
                    u=problem.upper * row_scale,
                    eps_abs=tolerance,
                    eps_rel=tolerance,
                    max_iter=30000,
                    polishing=True,
                    verbose=False,
                    adaptive_rho_interval=200,
                    alpha=1.4,
                )
                self.solver_setups += 1
            else:
                self._solver.update(
                    Px=scaled_hessian[self._p_pattern.indices, self._p_columns],
                    Ax=scaled_constraint[self._a_pattern.indices, self._a_columns],
                    q=problem.gradient * scale,
                    l=problem.lower * row_scale,
                    u=problem.upper * row_scale,
                )
            # Do not inherit an extreme adaptive rho from the preceding robot
            # configuration. Re-seed ADMM scaling while retaining the workspace
            # and primal/dual warm start; physical acceptance thresholds stay fixed.
            self._solver.update_settings(rho=0.1)
            if self.previous is not None:
                self._solver.warm_start(x=self.previous / scale)
            with OSQP_EXECUTION_LOCK:
                result = self._solver.solve(raise_error=False)
            if result.info.status_val == 1:
                if (
                    result.x is None
                    or np.shape(result.x) != (decision_count,)
                    or not np.isfinite(result.x).all()
                ):
                    self._solver, self.previous = None, None
                    raise RuntimeError("dynamics WBC returned an invalid solution")
                physical = result.x * scale
                evaluation = problem.constraint @ physical
                violation = float(
                    max(np.max(problem.lower - evaluation), np.max(evaluation - problem.upper), 0.0)
                )
                if violation <= 1e-5:
                    break
                # Scaled solver tolerances are not physical acceptance bounds.
                # Refine once in a fresh workspace, retaining the same absolute
                # limits; reject the step if the tighter solve still fails.
                self._solver, self.previous = None, None
                if attempt == 0:
                    self.solver_recoveries += 1
                    continue
                raise RuntimeError(f"dynamics WBC constraint violation: {violation}")
            self._solver, self.previous = None, None
            if attempt == 0 and result.info.status_val in (2, 7):
                self.solver_recoveries += 1
                continue
            raise RuntimeError(f"dynamics WBC failed: {result.info.status}")
        acceleration, torque, wrench = np.split(physical, [nv, nv + na])
        outcome = DynamicsControlResult(
            solver=QPResult("solved", physical, result.info.iter, violation, result.info.status),
            command=TorqueCommand(torque, state.stamp, float(state.stamp) + float(dt)),
            acceleration=acceleration,
            wrench=wrench,
            contact_rhs=contact_rhs,
            terms=terms,
            dynamics_residual=float(
                np.max(
                    np.abs(
                        mass @ acceleration
                        + bias
                        - robot.selection @ torque
                        - contact_jacobian.T @ wrench
                    )
                )
            ),
            contact_residual=float(np.max(np.abs(contact_jacobian @ acceleration - contact_rhs))),
        )
        self.previous = physical.copy()
        return outcome
