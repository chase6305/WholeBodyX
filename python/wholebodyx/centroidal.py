"""Fixed-height CoM/ZMP preview MPC for a fixed, planar support polygon.

The decision is piecewise-constant planar CoM jerk. ZMP = CoM - h/g *
CoM acceleration is the linear inverted-pendulum approximation, not a full
centroidal momentum or contact-switching model.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from .qp import OSQPSolver, _FixedQP, _immutable
from .mpc import _time_window
from .types import array


@dataclass(frozen=True)
class CoMPlan:
    states: np.ndarray  # (N+1, 3, 2): position, velocity, acceleration
    jerks: np.ndarray
    stamp: float
    dt: float
    height: float
    gravity: float

    def __post_init__(self):
        states = array(self.states, 3, "CoM states")
        jerks = array(self.jerks, 2, "CoM jerks")
        if jerks.shape[1:] != (2,) or not len(jerks) or states.shape != (len(jerks) + 1, 3, 2):
            raise ValueError("CoM plan dimensions must be (N+1, 3, 2) and (N, 2)")
        for value in (self.dt, self.height, self.gravity):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("plan timing, height and gravity must be positive")
        _time_window(self.stamp, self.dt, len(jerks))
        p, v, a = states[:-1, 0], states[:-1, 1], states[:-1, 2]
        predicted = np.stack((p + self.dt * v + .5 * self.dt**2 * a + self.dt**3 / 6 * jerks,
                              v + self.dt * a + .5 * self.dt**2 * jerks, a + self.dt * jerks), axis=1)
        if not np.allclose(predicted, states[1:], atol=1e-9, rtol=1e-8):
            raise ValueError("CoM plan must satisfy jerk integration")
        object.__setattr__(self, "states", _immutable(states))
        object.__setattr__(self, "jerks", _immutable(jerks))

    def sample(self, stamp):
        if not np.isfinite(stamp) or stamp < self.stamp or stamp >= self.stamp + self.dt * len(self.jerks):
            raise ValueError("CoM plan is future-dated or expired")
        elapsed = stamp - self.stamp
        index = min(int(elapsed / self.dt), len(self.jerks) - 1)
        t = elapsed - index * self.dt
        p, v, a = self.states[index]
        jerk = self.jerks[index]
        return p + t * v + .5 * t**2 * a + t**3 / 6 * jerk, v + t * a + .5 * t**2 * jerk, a + t * jerk

    @property
    def zmp(self):
        return self.states[:, 0] - self.height / self.gravity * self.states[:, 2]


@dataclass(frozen=True)
class _PreviewSettings:
    horizon: int
    dt: float
    height: float
    gravity: float
    velocity_limit: float
    acceleration_limit: float
    jerk_limit: float


class CoMPreviewMPC:
    horizon = property(lambda self: self._settings.horizon)
    dt = property(lambda self: self._settings.dt)
    height = property(lambda self: self._settings.height)
    gravity = property(lambda self: self._settings.gravity)
    velocity_limit = property(lambda self: self._settings.velocity_limit)
    acceleration_limit = property(lambda self: self._settings.acceleration_limit)
    jerk_limit = property(lambda self: self._settings.jerk_limit)

    def __init__(self, support_points, height, horizon=20, dt=.1, margin=.005,
                 velocity_limit=.2, acceleration_limit=.6, jerk_limit=3., gravity=9.81):
        points = array(support_points, 2, "support points")
        if points.shape[1:] != (2,) or len(points) < 3 or np.linalg.matrix_rank(points - points[0]) != 2:
            raise ValueError("support points must form a planar polygon")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or not 2 <= horizon <= 100:
            raise ValueError("horizon must be an integer in [2, 100]")
        for value in (height, dt, velocity_limit, acceleration_limit, jerk_limit, gravity):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("MPC height, timing and bounds must be finite and positive")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("support margin must be nonnegative")
        hull = ConvexHull(points)
        self.support_vertices = _immutable(points[hull.vertices])
        self.normals = _immutable(hull.equations[:, :2])
        self.offsets = _immutable(-hull.equations[:, 2] - margin)
        self._settings = _PreviewSettings(horizon, float(dt), float(height), float(gravity),
                                          float(velocity_limit), float(acceleration_limit), float(jerk_limit))
        self.solver = OSQPSolver(eps_abs=1e-7, eps_rel=1e-7, max_iter=50000)
        self.last_plan = None
        self.replans = 0
        self._prepare()

    def _maps(self, t, interval, start, control):
        transition = np.array([[1., t, .5 * t**2], [0., 1., t], [0., 0., 1.]])
        injection = np.array([t**3 / 6, .5 * t**2, t])
        next_start, next_control = transition @ start, transition @ control
        next_control[:, interval] += injection
        return next_start, next_control

    def _prepare(self):
        n = self.horizon
        start, control = np.eye(3), np.zeros((3, n))
        starts, controls, z_starts, z_controls = [], [], [], []
        v_starts, v_controls = [], []
        for k in range(n):
            # The four cubic Bernstein control points enclose the complete
            # ZMP curve. Constraining all four guarantees support containment
            # between knots too, rather than only at discrete samples.
            h = self.height / self.gravity
            for coefficients, jerk_coefficient in (
                    ([1., 0., -h], 0.),
                    ([1., self.dt / 3, -h], -h * self.dt / 3),
                    ([1., 2 * self.dt / 3, -h + self.dt**2 / 6], -2 * h * self.dt / 3),
                    ([1., self.dt, -h + self.dt**2 / 2], -h * self.dt + self.dt**3 / 6)):
                coefficients = np.asarray(coefficients)
                z_starts.append(coefficients @ start)
                z_map = coefficients @ control
                z_map[k] += jerk_coefficient
                z_controls.append(z_map)
            for coefficients, jerk_coefficient in (([0., 1., 0.], 0.),
                                                    ([0., 1., self.dt / 2], 0.),
                                                    ([0., 1., self.dt], .5 * self.dt**2)):
                coefficients = np.asarray(coefficients)
                v_starts.append(coefficients @ start)
                v_map = coefficients @ control
                v_map[k] += jerk_coefficient
                v_controls.append(v_map)
            start, control = self._maps(self.dt, k, start, control)
            starts.append(start.copy())
            controls.append(control.copy())
        self.state_start, self.state_control = np.asarray(starts), np.asarray(controls)
        self.z_start, self.z_control = np.asarray(z_starts), np.asarray(z_controls)
        self.v_start, self.v_control = np.asarray(v_starts), np.asarray(v_controls)
        self.position_map = np.kron(self.state_control[:, 0], np.eye(2))
        self.velocity_map = np.kron(self.state_control[:, 1], np.eye(2))
        self.acceleration_map = np.kron(self.state_control[:, 2], np.eye(2))
        self.zmp_map = np.kron(self.z_control, self.normals)
        hessian = (1000 * self.position_map.T @ self.position_map +
                   10 * self.velocity_map.T @ self.velocity_map + .01 * np.eye(2 * n))
        capture = np.array([1., np.sqrt(self.height / self.gravity), 0.])
        self.capture_start = capture @ self.state_start[-1]
        self.capture_control = np.kron((capture @ self.state_control[-1])[None], self.normals)
        constraint = np.vstack((np.eye(2 * n), np.kron(self.v_control, np.eye(2)), self.acceleration_map,
                                self.zmp_map, self.capture_control))
        self.problem = _FixedQP(hessian / 1000., constraint)

    def reset(self):
        self.solver.reset()
        self.last_plan = None
        self.replans = 0

    def plan(self, position, velocity, acceleration, targets, stamp=0., *, target_velocities=None):
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
        gradient = 1000 * self.position_map.T @ (nominal[:, 0] - targets).ravel()
        gradient += 10 * self.velocity_map.T @ (nominal[:, 1] - target_velocities).ravel()
        n = self.horizon * 2
        support = (self.z_start @ initial) @ self.normals.T
        velocity_offset = (self.v_start @ initial).ravel()
        lower = np.r_[np.full(n, -self.jerk_limit), -self.velocity_limit - velocity_offset,
                      -self.acceleration_limit - nominal[:, 2].ravel(), np.full(support.size + len(self.normals), -np.inf)]
        upper = np.r_[np.full(n, self.jerk_limit), self.velocity_limit - velocity_offset,
                      self.acceleration_limit - nominal[:, 2].ravel(), (self.offsets - support).ravel(),
                      self.offsets - self.normals @ (self.capture_start @ initial)]
        result = self.solver.solve(self.problem.problem(gradient / 1000., lower, upper))
        if not result.success:
            self.last_plan = None
            raise RuntimeError(f"CoM MPC failed: {result.status}: {result.detail}")
        jerks = result.x.reshape(self.horizon, 2)
        states = np.concatenate((initial[None], nominal + self.state_control @ jerks))
        # Own immutable snapshots: sampling must not depend on solver buffers.
        states.setflags(write=False)
        jerks = jerks.copy()
        jerks.setflags(write=False)
        plan = CoMPlan(states, jerks, float(stamp), self.dt, self.height, self.gravity)
        self.last_plan = plan
        self.replans += 1
        return plan
