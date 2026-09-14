"""Experimental floating-base inverse-dynamics WBC with prescribed rigid contacts.

Requires the optional Pinocchio dependency. Contact switches and impacts are not
implemented; the forward plant enforces the same persistent bilateral contacts.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import osqp
from scipy import sparse

from .qp import QuadraticProgram
from .contact import rectangular_wrench_cone


@dataclass
class DynamicsState:
    q: np.ndarray
    v: np.ndarray
    stamp: float = 0.


class FloatingBaseDynamics:
    def __init__(self, urdf, feet=("left_ankle_link", "right_ankle_link")):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf)), pin.JointModelFreeFlyer())
        if self.model.nq != self.model.nv + 1:
            raise ValueError("requires scalar actuated joints after the free-flyer")
        if not feet or len(set(feet)) != len(feet):
            raise ValueError("at least one distinct contact frame is required")
        self.contacts = []
        for name in feet:
            parent = self.model.getFrameId(name)
            if parent >= self.model.nframes:
                raise ValueError(f"unknown contact frame: {name}")
            frame = self.model.frames[parent]
            # Assumed sole plane for this demo; patch dimensions are not vendor data.
            placement = frame.placement * pin.SE3(np.eye(3), np.array([0., 0., -.07]))
            self.contacts.append(self.model.addFrame(pin.Frame(name + "_sole", frame.parentJoint,
                                                              parent, placement, pin.OP_FRAME)))
        self.data = self.model.createData()
        self.nv, self.na = self.model.nv, self.model.nv - 6
        self.selection = np.eye(self.nv)[:, 6:]

    def validate_state(self, state):
        q, v = np.asarray(state.q, dtype=float), np.asarray(state.v, dtype=float)
        if (q.shape != (self.model.nq,) or v.shape != (self.nv,)
                or not np.isfinite(q).all() or not np.isfinite(v).all()
                or not np.isfinite(state.stamp)):
            raise ValueError("invalid floating-base state")
        if abs(np.linalg.norm(q[3:7]) - 1.) > 1e-8:
            raise ValueError("floating-base quaternion must be normalized")
        return q, v

    def terms(self, state):
        p, m, d = self.pin, self.model, self.data
        q, v = self.validate_state(state)
        mass = p.crba(m, d, q).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        bias = p.nonLinearEffects(m, d, q, v).copy()
        p.computeJointJacobiansTimeVariation(m, d, q, v)
        p.updateFramePlacements(m, d)
        poses = [d.oMf[f].copy() for f in self.contacts]
        jac = np.vstack([p.getFrameJacobian(m, d, f, p.LOCAL) for f in self.contacts])
        drift = np.concatenate([p.getFrameJacobianTimeVariation(m, d, f, p.LOCAL) @ v for f in self.contacts])
        com = p.centerOfMass(m, d, q, v, np.zeros(m.nv)).copy()
        com_v, com_drift = d.vcom[0].copy(), d.acom[0].copy()
        com_jac = p.jacobianCenterOfMass(m, d, q).copy()
        return mass, bias, jac, drift, poses, com, com_v, com_drift, com_jac

    def forward(self, state, torque, contact_acceleration, terms, *, external_force=None):
        """Solve constrained forward dynamics from actuator torque, not QP ddq."""
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (self.na,) or not np.isfinite(torque).all():
            raise ValueError("invalid actuator torque")
        mass, bias, jac = terms[:3]
        contact_acceleration = np.asarray(contact_acceleration, dtype=float)
        nc = len(jac)
        if contact_acceleration.shape != (nc,) or not np.isfinite(contact_acceleration).all():
            raise ValueError("invalid contact acceleration")
        system = np.block([[mass, -jac.T], [jac, np.zeros((nc, nc))]])
        external_force = np.zeros(self.nv) if external_force is None else np.asarray(external_force, dtype=float)
        if external_force.shape != (self.nv,) or not np.isfinite(external_force).all():
            raise ValueError("invalid generalized external force")
        rhs = np.r_[self.selection @ torque + external_force - bias, contact_acceleration]
        # H1 has five joints per leg; the two six-axis constraints contain
        # a redundant row at symmetric poses. Acceleration is unique, while
        # internal contact wrench can be indeterminate. Use a rank-aware solve.
        solution = np.linalg.lstsq(system, rhs, rcond=1e-10)[0]
        if np.max(np.abs(system @ solution - rhs)) > 1e-5:
            raise RuntimeError("inconsistent constrained forward dynamics")
        return solution[:self.nv], solution[self.nv:]

    def integrate(self, state, acceleration, dt):
        if not np.isfinite(dt) or not 0 < dt <= .02:
            raise ValueError("dt must be in (0, 0.02]")
        q, v = self.validate_state(state)
        acceleration = np.asarray(acceleration, dtype=float)
        if acceleration.shape != (self.nv,) or not np.isfinite(acceleration).all():
            raise ValueError("invalid generalized acceleration")
        velocity = v + dt * acceleration
        q = self.pin.integrate(self.model, q, dt * (v + .5 * dt * acceleration))
        return DynamicsState(q, velocity, state.stamp + dt)


class ContactWBC:
    """Acceleration/torque/local-sole-wrench QP; no dynamic MPC yet."""
    def __init__(self, dynamics, initial, friction=.6):
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("friction must be positive")
        self.robot, self._friction = dynamics, float(friction)
        self._physical_cone = np.vstack((rectangular_wrench_cone(self._friction),
                                        [0., 0., -.015, 0., 0., 1.], [0., 0., -.015, 0., 0., -1.]))
        self._planning_cone = self._physical_cone.copy()
        self.nominal = initial.q.copy()
        self.anchors = dynamics.terms(initial)[4]
        self._contact_reserve = 0.
        self.previous = None
        self._solver = None
        self.solver_setups = 0
        self.solver_recoveries = 0
        self._p_pattern = self._a_pattern = None

    @property
    def friction(self):
        return self._friction

    @property
    def contact_reserve(self):
        return self._contact_reserve

    @contact_reserve.setter
    def contact_reserve(self, value):
        if not np.isfinite(value) or not 0 <= value <= .75:
            raise ValueError("contact reserve must be in [0, 0.75]")
        self._contact_reserve = float(value)
        self._planning_cone = self._physical_cone.copy()
        self._planning_cone[:, 2] *= 1 - self._contact_reserve

    def contact_violation(self, wrench):
        """Check actual local sole wrenches using the controller's contact bounds."""
        force = np.asarray(wrench, dtype=float)
        if force.shape != (6 * len(self.anchors),) or not np.isfinite(force).all():
            raise ValueError("invalid contact wrench")
        force = force.reshape(-1, 6)
        return float(max(0., np.max(self._physical_cone @ force.T)))

    def solve(self, state, target_com, dt=.005, *, target_velocity=None, target_acceleration=None):
        r = self.robot
        if not np.isfinite(dt) or not 0 < dt <= .02:
            raise ValueError("dt must be in (0, 0.02]")
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
        terms = r.terms(state)
        mass, bias, jac, drift, poses, com, com_v, com_drift, com_jac = terms
        nv, na, nc = r.nv, r.na, len(jac)
        n = nv + na + nc
        error = np.concatenate([r.pin.log6(pose.inverse() * anchor).vector for pose, anchor in zip(poses, self.anchors)])
        contact_rhs = -drift + 100 * error - 20 * (jac @ state.v)
        hessian = np.diag(np.r_[np.full(nv, 1e-3), np.full(na, 1e-5), np.full(nc, 1e-6)])
        gradient = np.zeros(n)

        def task(matrix, desired, weight):
            nonlocal hessian, gradient
            hessian[:nv, :nv] += weight * matrix.T @ matrix
            gradient[:nv] -= weight * matrix.T @ desired

        task(com_jac, target_acceleration + 60 * (target_com - com) + 16 * (target_velocity - com_v) - com_drift, 1000.)
        posture = r.pin.difference(r.model, state.q, self.nominal)
        task(np.eye(nv)[6:], 20 * posture[6:] - 8 * state.v[6:], 1.)
        task(np.eye(nv)[3:6], 30 * posture[3:6] - 10 * state.v[3:6], 10.)
        # Soft load-sharing target for this two-foot support controller.
        if len(poses) == 2:
            spacing = poses[0].translation[1] - poses[1].translation[1]
            if abs(spacing) > 1e-6:
                fraction = np.clip((target_com[1] - poses[1].translation[1]) / spacing, .1, .9)
                weight = r.pin.computeTotalMass(r.model) * np.linalg.norm(r.model.gravity.linear)
                for foot, share in enumerate((fraction, 1 - fraction)):
                    index = nv + na + 6 * foot + 2
                    hessian[index, index] += .001
                    gradient[index] -= .001 * weight * share
        eq = np.zeros((nv + nc, n))
        eq[:nv, :nv], eq[:nv, nv:nv + na], eq[:nv, nv + na:] = mass, -r.selection, -jac.T
        eq[nv:, :nv] = jac
        constraints, lower, upper = [eq], [np.r_[-bias, contact_rhs]], [np.r_[-bias, contact_rhs]]
        bounds = np.zeros((nv + na, n))
        bounds[:, :nv + na] = np.eye(nv + na)
        alo, ahi = np.full(nv, -40.), np.full(nv, 40.)
        velocity = np.minimum(r.model.velocityLimit[6:], 2.)
        alo[6:] = np.maximum(alo[6:], (-velocity - state.v[6:]) / dt)
        ahi[6:] = np.minimum(ahi[6:], (velocity - state.v[6:]) / dt)
        alo[6:] = np.maximum(alo[6:], 2 * (r.model.lowerPositionLimit[7:] - state.q[7:] - dt * state.v[6:]) / dt**2)
        ahi[6:] = np.minimum(ahi[6:], 2 * (r.model.upperPositionLimit[7:] - state.q[7:] - dt * state.v[6:]) / dt**2)
        # Brake before position limits, rather than waiting until the last step.
        lower_speed = np.minimum(velocity, np.sqrt(2 * 20. * np.maximum(state.q[7:] - r.model.lowerPositionLimit[7:], 0.)))
        upper_speed = np.minimum(velocity, np.sqrt(2 * 20. * np.maximum(r.model.upperPositionLimit[7:] - state.q[7:], 0.)))
        alo[6:] = np.maximum(alo[6:], (-lower_speed - state.v[6:]) / dt)
        ahi[6:] = np.minimum(ahi[6:], (upper_speed - state.v[6:]) / dt)
        constraints.append(bounds)
        lower.append(np.r_[alo, -r.model.effortLimit[6:]])
        upper.append(np.r_[ahi, r.model.effortLimit[6:]])
        for foot in range(nc // 6):
            cone = self._planning_cone
            block = np.zeros((len(cone), n))
            block[:, nv + na + 6 * foot:nv + na + 6 * foot + 6] = cone
            constraints.append(block)
            lower.append(np.full(len(cone), -np.inf))
            upper.append(np.zeros(len(cone)))
        problem = QuadraticProgram(hessian, gradient, np.vstack(constraints), np.concatenate(lower), np.concatenate(upper))
        if np.any(problem.lower > problem.upper):
            raise RuntimeError("dynamics WBC has inconsistent state/limit bounds")
        # Scale accelerations, torques and wrenches into comparable QP units.
        scale = np.r_[np.ones(nv), np.full(na, 50.), np.full(nc, 300.)]
        matrix = problem.constraint * scale
        row_scale = 1 / np.maximum(np.max(np.abs(matrix), axis=1), 1.)
        scaled_hessian = problem.hessian * scale[:, None] * scale
        scaled_constraint = matrix * row_scale[:, None]
        for attempt in range(2):
            if self._solver is None:
                # Explicit zeros preserve the symbolic pattern as the robot moves.
                # Only the acceleration Hessian block is dense; all force/torque
                # regularizers are diagonal. Constraint couplings stay fixed.
                p_mask = np.eye(n, dtype=bool)
                p_mask[:nv, :nv] = np.triu(np.ones((nv, nv), dtype=bool))
                a_mask = problem.constraint != 0
                a_mask[:nv, :nv] = True
                a_mask[:nv, nv + na:] = True
                a_mask[nv:nv + nc, :nv] = True
                self._p_pattern = sparse.csc_matrix(p_mask, dtype=float)
                self._a_pattern = sparse.csc_matrix(a_mask, dtype=float)
                self._p_columns = np.repeat(np.arange(n), np.diff(self._p_pattern.indptr))
                self._a_columns = np.repeat(np.arange(n), np.diff(self._a_pattern.indptr))
                self._p_pattern.data[:] = scaled_hessian[self._p_pattern.indices, self._p_columns]
                self._a_pattern.data[:] = scaled_constraint[self._a_pattern.indices, self._a_columns]
                self._solver = osqp.OSQP()
                self._solver.setup(P=self._p_pattern, q=problem.gradient * scale,
                                   A=self._a_pattern, l=problem.lower * row_scale, u=problem.upper * row_scale,
                                   eps_abs=1e-9, eps_rel=1e-9, max_iter=30000, polishing=True, verbose=False, adaptive_rho_interval=50)
                self.solver_setups += 1
            else:
                self._solver.update(Px=scaled_hessian[self._p_pattern.indices, self._p_columns],
                                    Ax=scaled_constraint[self._a_pattern.indices, self._a_columns],
                                    q=problem.gradient * scale,
                                    l=problem.lower * row_scale, u=problem.upper * row_scale)
            # Do not inherit an extreme adaptive rho from the preceding robot
            # configuration. Re-seed ADMM scaling while retaining the workspace
            # and primal/dual warm start; physical acceptance thresholds stay fixed.
            self._solver.update_settings(rho=.1)
            if self.previous is not None:
                self._solver.warm_start(x=self.previous / scale)
            result = self._solver.solve(raise_error=False)
            if result.info.status_val == 1:
                if result.x is None or np.shape(result.x) != (n,) or not np.isfinite(result.x).all():
                    self._solver, self.previous = None, None
                    raise RuntimeError("dynamics WBC returned an invalid solution")
                break
            self._solver, self.previous = None, None
            if attempt == 0 and result.info.status_val in (2, 7):
                self.solver_recoveries += 1
                continue
            raise RuntimeError(f"dynamics WBC failed: {result.info.status}")
        result.x *= scale
        violation = float(max(np.max(problem.lower - problem.constraint @ result.x),
                              np.max(problem.constraint @ result.x - problem.upper), 0.))
        if violation > 1e-5:
            self._solver, self.previous = None, None
            raise RuntimeError(f"dynamics WBC constraint violation: {violation}")
        self.previous = result.x.copy()
        acceleration, torque, wrench = np.split(result.x, [nv, nv + na])
        return dict(acceleration=acceleration, torque=torque, wrench=wrench,
                    contact_rhs=contact_rhs, terms=terms, violation=violation, iterations=result.info.iter,
                    dynamics_residual=float(np.max(np.abs(mass @ acceleration + bias - r.selection @ torque - jac.T @ wrench))),
                    contact_residual=float(np.max(np.abs(jac @ acceleration - contact_rhs))))
