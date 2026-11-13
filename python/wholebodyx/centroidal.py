"""Fixed-height CoM/ZMP preview MPC for a fixed, planar support polygon.

The decision is piecewise-constant planar CoM jerk. ZMP = CoM - h/g *
CoM acceleration is the linear inverted-pendulum approximation, not a full
centroidal momentum or contact-switching model.
"""

from dataclasses import dataclass
from typing import Protocol
from weakref import ref

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.spatial import ConvexHull

from .qp import (OSQPSolver, _FixedQP, _immutable, _immutable_matrix_key,
                 _same_immutable_key, _OSQP_SOLVE)
from .mpc import _time_window
from .types import _snapshot, array


@dataclass(frozen=True)
class CoMPlan:
    states: np.ndarray  # (N+1, 3, 2): position, velocity, acceleration
    jerks: np.ndarray
    stamp: float
    dt: float
    height: float
    gravity: float

    def __post_init__(self):
        for name in ("stamp", "dt", "height", "gravity"):
            object.__setattr__(self, name, float(getattr(self, name)))
        states = _snapshot(self.states, 3, "CoM states")
        jerks = _snapshot(self.jerks, 2, "CoM jerks")
        if jerks.shape[1:] != (2,) or not len(jerks) or states.shape != (len(jerks) + 1, 3, 2):
            raise ValueError("CoM plan dimensions must be (N+1, 3, 2) and (N, 2)")
        for value in (self.dt, self.height, self.gravity):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("plan timing, height and gravity must be positive")
        _time_window(self.stamp, self.dt, len(jerks))
        p, v, a = states[:-1, 0], states[:-1, 1], states[:-1, 2]
        predicted = np.stack(
            (
                p + self.dt * v + 0.5 * self.dt**2 * a + self.dt**3 / 6 * jerks,
                v + self.dt * a + 0.5 * self.dt**2 * jerks,
                a + self.dt * jerks,
            ),
            axis=1,
        )
        if not np.allclose(predicted, states[1:], atol=1e-9, rtol=1e-8):
            raise ValueError("CoM plan must satisfy jerk integration")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "jerks", jerks)

    def sample(self, stamp):
        stamp = float(stamp)
        if (
            not np.isfinite(stamp)
            or stamp < self.stamp
            or stamp >= self.stamp + self.dt * len(self.jerks)
        ):
            raise ValueError("CoM plan is future-dated or expired")
        elapsed = stamp - self.stamp
        index = min(int(elapsed / self.dt), len(self.jerks) - 1)
        t = elapsed - index * self.dt
        p, v, a = self.states[index]
        jerk = self.jerks[index]
        return (
            p + t * v + 0.5 * t**2 * a + t**3 / 6 * jerk,
            v + t * a + 0.5 * t**2 * jerk,
            a + t * jerk,
        )

    @property
    def valid_until(self):
        return self.stamp + self.dt * len(self.jerks)

    @property
    def zmp(self):
        return self.states[:, 0] - self.height / self.gravity * self.states[:, 2]


class CoMPlanner(Protocol):
    horizon: int
    dt: float

    def reset(self) -> None: ...

    def plan(
        self, position, velocity, acceleration, targets, stamp=0.0, *, target_velocities=None
    ) -> CoMPlan: ...

    def validate_plan(self, plan: CoMPlan) -> float: ...


@dataclass(frozen=True)
class _PreviewSettings:
    horizon: int
    dt: float
    height: float
    gravity: float
    velocity_limit: float
    acceleration_limit: float
    jerk_limit: float
    unconstrained_fast_path: bool


class CoMPreviewMPC:
    horizon = property(lambda self: self._settings.horizon)
    dt = property(lambda self: self._settings.dt)
    height = property(lambda self: self._settings.height)
    gravity = property(lambda self: self._settings.gravity)
    velocity_limit = property(lambda self: self._settings.velocity_limit)
    acceleration_limit = property(lambda self: self._settings.acceleration_limit)
    jerk_limit = property(lambda self: self._settings.jerk_limit)
    unconstrained_fast_path = property(lambda self: self._settings.unconstrained_fast_path)

    def __init__(
        self,
        support_points,
        height,
        horizon=20,
        dt=0.1,
        margin=0.005,
        velocity_limit=0.2,
        acceleration_limit=0.6,
        jerk_limit=3.0,
        gravity=9.81,
        *,
        unconstrained_fast_path=True,
    ):
        points = array(support_points, 2, "support points")
        if (
            points.shape[1:] != (2,)
            or len(points) < 3
            or np.linalg.matrix_rank(points - points[0]) != 2
        ):
            raise ValueError("support points must form a planar polygon")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or not 2 <= horizon <= 100:
            raise ValueError("horizon must be an integer in [2, 100]")
        for value in (height, dt, velocity_limit, acceleration_limit, jerk_limit, gravity):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("MPC height, timing and bounds must be finite and positive")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("support margin must be nonnegative")
        if not isinstance(unconstrained_fast_path, bool):
            raise ValueError("unconstrained_fast_path must be boolean")
        hull = ConvexHull(points)
        self.support_vertices = _immutable(points[hull.vertices])
        self.normals = _immutable(hull.equations[:, :2])
        self.offsets = _immutable(-hull.equations[:, 2] - margin)
        self._settings = _PreviewSettings(
            horizon,
            float(dt),
            float(height),
            float(gravity),
            float(velocity_limit),
            float(acceleration_limit),
            float(jerk_limit),
            unconstrained_fast_path,
        )
        self.solver = OSQPSolver(eps_abs=1e-7, eps_rel=1e-7, max_iter=50000)
        # Identifying the original instance must not retain its workspace after
        # an application replaces the solver.
        self._default_solver_ref = ref(self.solver)
        self.last_plan = None
        self.replans = 0
        self._direct_solves = self._qp_solves = 0
        self._prepare()

    def _prepare(self):
        n = self.horizon
        start, control = np.eye(3), np.zeros((3, n))
        starts, controls, z_starts, z_controls = [], [], [], []
        v_starts, v_controls = [], []
        dt = self.dt
        height_ratio = self.height / self.gravity
        transition = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        injection = np.array([dt**3 / 6, 0.5 * dt**2, dt])
        # Bernstein control points enclose the complete ZMP and velocity
        # curves, so these bounds also constrain the intervals between knots.
        zmp_coefficients = (
            (np.array([1.0, 0.0, -height_ratio]), 0.0),
            (np.array([1.0, dt / 3, -height_ratio]), -height_ratio * dt / 3),
            (np.array([1.0, 2 * dt / 3, -height_ratio + dt**2 / 6]), -2 * height_ratio * dt / 3),
            (np.array([1.0, dt, -height_ratio + dt**2 / 2]), -height_ratio * dt + dt**3 / 6),
        )
        velocity_coefficients = (
            (np.array([0.0, 1.0, 0.0]), 0.0),
            (np.array([0.0, 1.0, dt / 2]), 0.0),
            (np.array([0.0, 1.0, dt]), 0.5 * dt**2),
        )
        for k in range(n):
            for coefficients, jerk_coefficient in zmp_coefficients:
                z_starts.append(coefficients @ start)
                z_map = coefficients @ control
                z_map[k] += jerk_coefficient
                z_controls.append(z_map)
            for coefficients, jerk_coefficient in velocity_coefficients:
                v_starts.append(coefficients @ start)
                v_map = coefficients @ control
                v_map[k] += jerk_coefficient
                v_controls.append(v_map)
            start, control = transition @ start, transition @ control
            control[:, k] += injection
            starts.append(start.copy())
            controls.append(control.copy())
        self.state_start, self.state_control = np.asarray(starts), np.asarray(controls)
        self.z_start, self.z_control = np.asarray(z_starts), np.asarray(z_controls)
        self.v_start, self.v_control = np.asarray(v_starts), np.asarray(v_controls)
        self.position_map = np.kron(self.state_control[:, 0], np.eye(2))
        self.velocity_map = np.kron(self.state_control[:, 1], np.eye(2))
        self.acceleration_map = np.kron(self.state_control[:, 2], np.eye(2))
        self.zmp_map = np.kron(self.z_control, self.normals)
        hessian = (
            1000 * self.position_map.T @ self.position_map
            + 10 * self.velocity_map.T @ self.velocity_map
            + 0.01 * np.eye(2 * n)
        )
        capture = np.array([1.0, np.sqrt(self.height / self.gravity), 0.0])
        self.capture_start = capture @ self.state_start[-1]
        self.capture_control = np.kron((capture @ self.state_control[-1])[None], self.normals)
        constraint = np.vstack(
            (
                np.eye(2 * n),
                np.kron(self.v_control, np.eye(2)),
                self.acceleration_map,
                self.zmp_map,
                self.capture_control,
            )
        )
        self.problem = _FixedQP(hessian / 1000.0, constraint)
        self._unconstrained_factor = None
        self._unconstrained_keys = (self.problem._hessian_key, self.problem._constraint_key)
        if self.unconstrained_fast_path:
            try:
                # OSQP reads H's upper triangle. Use that same symmetric
                # operator, including any accepted roundoff asymmetry.
                factor = cholesky(self.problem._hessian, lower=False, check_finite=False)
            except np.linalg.LinAlgError:
                # A numerically singular cost retains the existing QP path.
                pass
            else:
                # Own immutable bytes in LAPACK's column-major order.
                if np.isfinite(factor).all():
                    self._unconstrained_factor = _immutable(factor.T).T

    def reset(self):
        self.solver.reset()
        self.last_plan = None
        self.replans = 0
        self._direct_solves = self._qp_solves = 0

    def validate_plan(self, plan):
        """Independent physical validation, including complete interval curves.

        Return the largest absolute bound violation; zero means feasible. This
        does not use the QP matrices or the solver workspace.
        """
        if not isinstance(plan, CoMPlan):
            raise ValueError("CoM planner must return a CoMPlan")
        if plan.height != self.height or plan.gravity != self.gravity:
            raise ValueError("CoM plan uses a different height or gravity")
        p, v, a = np.moveaxis(plan.states[:-1], 1, 0)
        j, dt, h = plan.jerks, plan.dt, self.height / self.gravity
        z0 = p - h * a
        zmp_controls = np.stack(
            (
                z0,
                z0 + dt / 3 * (v - h * j),
                z0 + 2 * dt / 3 * (v - h * j) + dt**2 / 6 * a,
                plan.zmp[1:],
            ),
            axis=1,
        )
        velocity_controls = np.stack((v, v + dt / 2 * a, plan.states[1:, 1]), axis=1)
        capture = plan.states[-1, 0] + np.sqrt(h) * plan.states[-1, 1]
        values = (
            np.max(np.abs(j) - self.jerk_limit),
            np.max(np.abs(velocity_controls) - self.velocity_limit),
            np.max(np.abs(plan.states[:, 2]) - self.acceleration_limit),
            np.max(zmp_controls @ self.normals.T - self.offsets),
            np.max(self.normals @ capture - self.offsets),
        )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite CoM plan constraint evaluation")
        return float(max(0.0, *values))

    def _plan_from_decision(self, initial, nominal, decision, stamp):
        jerks = decision.reshape(self.horizon, 2)
        states = np.concatenate((initial[None], nominal + self.state_control @ jerks))
        return CoMPlan(states, jerks, float(stamp), self.dt, self.height, self.gravity)

    def _unconstrained_plan(self, problem, initial, nominal, stamp):
        """Accept the unique free optimum only for a certified, feasible model."""
        factor = self._unconstrained_factor
        if (
            factor is None
            or self.solver is not self._default_solver_ref()
            or type(self.solver) is not OSQPSolver
            or getattr(self.solver.solve, "__func__", None) is not _OSQP_SOLVE
        ):
            return None
        for expected, matrix in zip(
            self._unconstrained_keys, (problem.hessian, problem.constraint)
        ):
            if not _same_immutable_key(expected, _immutable_matrix_key(matrix)):
                return None
        # Retain the adapter's public-buffer checks even when a custom problem
        # factory changed vectors after construction. LAPACK receives finite RHS.
        problem._check_vectors()
        # The negated gradient is owned; LAPACK may overwrite this RHS only.
        with np.errstate(over="ignore", invalid="ignore"):
            decision = cho_solve((factor, False), -problem.gradient,
                                 overwrite_b=True, check_finite=False)
        # The certified model starts with the jerk identity block. Avoid a
        # full matrix-vector product when this box already rejects the point.
        if (not np.isfinite(decision).all()
                or not (decision >= problem.lower[:decision.size]).all()
                or not (decision <= problem.upper[:decision.size]).all()):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            values = problem.constraint @ decision
        if (
            not np.isfinite(values).all()
            or not (values >= problem.lower).all()
            or not (values <= problem.upper).all()
        ):
            return None
        # Independently evaluate the returned physical trajectory, including
        # Bernstein interval bounds and terminal capture, before bypassing QP.
        try:
            plan = self._plan_from_decision(initial, nominal, decision, stamp)
            violation = self.validate_plan(plan)
        except ValueError:
            return None
        if not np.isfinite(violation) or not 0 <= violation <= 1e-6:
            return None
        self._direct_solves += 1
        return plan

    def plan(self, position, velocity, acceleration, targets, stamp=0.0, *, target_velocities=None):
        initial = np.stack([array(x, 1, "CoM state") for x in (position, velocity, acceleration)])
        targets = array(targets, 2, "CoM targets")
        if initial.shape != (3, 2) or targets.shape != (self.horizon, 2):
            raise ValueError("CoM state must be 2-D and targets must match horizon")
        _time_window(stamp, self.dt, self.horizon)
        if target_velocities is None:
            target_velocities = np.zeros_like(targets)
        else:
            target_velocities = array(target_velocities, 2, "target velocities")
            if target_velocities.shape != targets.shape:
                raise ValueError("target velocities must match horizon")
        nominal = self.state_start @ initial
        # Apply scalar weights to vectors, avoiding scaled copies of both maps.
        gradient = 1000 * (self.position_map.T @ (nominal[:, 0] - targets).ravel())
        gradient += 10 * (self.velocity_map.T @ (nominal[:, 1] - target_velocities).ravel())
        n = self.horizon * 2
        support = (self.z_start @ initial) @ self.normals.T
        velocity_offset = (self.v_start @ initial).ravel()
        lower = np.r_[
            np.full(n, -self.jerk_limit),
            -self.velocity_limit - velocity_offset,
            -self.acceleration_limit - nominal[:, 2].ravel(),
            np.full(support.size + len(self.normals), -np.inf),
        ]
        upper = np.r_[
            np.full(n, self.jerk_limit),
            self.velocity_limit - velocity_offset,
            self.acceleration_limit - nominal[:, 2].ravel(),
            (self.offsets - support).ravel(),
            self.offsets - self.normals @ (self.capture_start @ initial),
        ]
        problem = self.problem.problem(gradient / 1000.0, lower, upper)
        plan = self._unconstrained_plan(problem, initial, nominal, stamp)
        if plan is None:
            self._qp_solves += 1
            result = self.solver.solve(problem)
            if not result.success:
                self.last_plan = None
                raise RuntimeError(f"CoM MPC failed: {result.status}: {result.detail}")
            plan = self._plan_from_decision(initial, nominal, result.x, stamp)
        self.last_plan = plan
        self.replans += 1
        return plan
