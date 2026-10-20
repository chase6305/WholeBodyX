"""Condensed, constrained joint-velocity MPC for a fixed-base integrator."""

from dataclasses import dataclass
from math import ulp

import numpy as np

from .qp import OSQPSolver, QPResult, _FixedQP, _immutable
from .types import JointLimits, array, positive


def _time_window(stamp, dt, count):
    stamp, dt = float(stamp), float(dt)
    end = stamp + dt * count
    if not np.isfinite([stamp, end]).all() or stamp < 0 or end <= stamp or dt <= 8 * max(ulp(stamp), ulp(end)):
        raise ValueError("plan time window must be finite and resolve each time step")
    return end


def _interval(time, start, dt, count):
    # Clamp before conversion, so large forward clock jumps cannot overflow int().
    if time >= start + count * dt:
        return count
    elapsed = (time - start) / dt
    nearest = round(elapsed)
    knot = start + nearest * dt
    if abs(time - knot) <= 4 * max(ulp(time), ulp(knot)):
        return min(nearest, count - 1)
    return min(int(elapsed), count - 1)


@dataclass(frozen=True)
class _MPCSettings:
    limits: JointLimits
    horizon: int
    dt: float
    tracking_weight: float
    control_weight: float
    terminal_weight: float

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        object.__setattr__(self, "horizon", int(self.horizon))
        for name in ("dt", "tracking_weight", "control_weight", "terminal_weight"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        limits = JointLimits(self.limits.lower, self.limits.upper, self.limits.velocity, self.limits.acceleration)
        for name in ("lower", "upper", "velocity", "acceleration"):
            if getattr(limits, name) is not None:
                object.__setattr__(limits, name, _immutable(getattr(limits, name)))
        object.__setattr__(self, "limits", limits)
        _time_window(0.0, self.dt, self.horizon)


@dataclass(frozen=True)
class JointPlan:
    positions: np.ndarray
    velocities: np.ndarray
    stamp: float
    dt: float

    def __post_init__(self):
        q = array(self.positions, 2, "positions")
        v = array(self.velocities, 2, "velocities")
        object.__setattr__(self, "dt", positive(self.dt, "plan dt"))
        object.__setattr__(self, "stamp", float(self.stamp))
        if not v.shape[0] or not v.shape[1] or q.shape != (v.shape[0] + 1, v.shape[1]):
            raise ValueError("plan must contain N+1 positions and N velocities")
        _time_window(self.stamp, self.dt, len(v))
        if not np.allclose(np.diff(q, axis=0), self.dt * v, rtol=1e-7, atol=1e-8):
            raise ValueError("plan positions and velocities must satisfy integrator dynamics")
        object.__setattr__(self, "positions", q)
        object.__setattr__(self, "velocities", v)

    @property
    def valid_until(self):
        return self.stamp + self.dt * len(self.velocities)

    def sample(self, stamp):
        stamp = float(stamp)
        if not np.isfinite(stamp) or stamp < self.stamp or stamp >= self.valid_until:
            raise ValueError("reference is future-dated or expired")
        elapsed = stamp - self.stamp
        k = _interval(stamp, self.stamp, self.dt, len(self.velocities))
        return self.positions[k] + max(0.0, elapsed - k * self.dt) * self.velocities[k], self.velocities[k].copy()


@dataclass(frozen=True)
class PlanResult:
    solver: QPResult
    plan: JointPlan | None = None


class JointMPC:
    def __init__(self, limits, horizon=15, dt=0.1, tracking_weight=10.0,
                 control_weight=0.1, terminal_weight=30.0, solver=None):
        self._settings = _MPCSettings(limits, horizon, dt, tracking_weight, control_weight, terminal_weight)
        self.solver = solver if solver is not None else OSQPSolver()
        self._last_plan = None
        self._prepare()

    limits = property(lambda self: self._settings.limits)
    horizon = property(lambda self: self._settings.horizon)
    dt = property(lambda self: self._settings.dt)
    tracking_weight = property(lambda self: self._settings.tracking_weight)
    control_weight = property(lambda self: self._settings.control_weight)
    terminal_weight = property(lambda self: self._settings.terminal_weight)

    def reset(self):
        self.solver.reset()
        self._last_plan = None

    def _prepare(self):
        n, count = self.limits.lower.size, self.horizon
        size = n * count
        # Joint costs are separable: multiply the small temporal matrices before
        # expanding to joint space, instead of multiplying (N*n)-square arrays.
        temporal = self.dt * np.tril(np.ones((count, count)))
        weights = np.full(count, self.tracking_weight)
        weights[-1] = self.terminal_weight
        prediction = np.kron(temporal, np.eye(n))
        temporal_h = temporal.T @ (weights[:, None] * temporal)
        temporal_h.flat[::count + 1] += self.control_weight
        h = np.kron(temporal_h, np.eye(n))
        self._gradient_map = np.kron((temporal.T @ weights)[:, None], np.eye(n))
        rows = [np.eye(size), prediction]
        lo = [-np.tile(self.limits.velocity, count), np.tile(self.limits.lower, count)]
        hi = [np.tile(self.limits.velocity, count), np.tile(self.limits.upper, count)]
        if self.limits.acceleration is not None:
            difference = np.eye(size)
            difference[n:, :-n] -= np.eye(size - n)
            bound = self.dt * np.tile(self.limits.acceleration, count)
            rows.append(difference)
            lo.append(-bound)
            hi.append(bound)
        self._template = _FixedQP(h, np.vstack(rows))
        self._lower, self._upper = np.concatenate(lo), np.concatenate(hi)

    def formulate(self, state, goal):
        self.limits.check_state(state)
        _time_window(state.stamp, self.dt, self.horizon)
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        n, size = state.q.size, state.q.size * self.horizon
        lo, hi = self._lower.copy(), self._upper.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                lo[size:2 * size].reshape(self.horizon, n)[:] -= state.q
                hi[size:2 * size].reshape(self.horizon, n)[:] -= state.q
                if self.limits.acceleration is not None:
                    lo[2 * size:2 * size + n] += state.v
                    hi[2 * size:2 * size + n] += state.v
                gradient = self._gradient_map @ (state.q - goal)
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return self._template.problem(gradient, lo, hi)

    def solve(self, state, goal):
        problem = self.formulate(state, goal)
        warm = None
        if self._last_plan is not None and state.stamp >= self._last_plan.stamp:
            offset = _interval(state.stamp, self._last_plan.stamp, self.dt, self.horizon)
            old = self._last_plan.velocities
            shifted = np.empty_like(old)
            remaining = self.horizon - offset
            shifted[:remaining] = old[offset:]
            shifted[remaining:] = old[-1]
            warm = shifted.reshape(-1)
        result = self.solver.solve(problem, warm_start=warm)
        if not result.success:
            return PlanResult(result)
        velocities = result.x.reshape(self.horizon, state.q.size)
        positions = np.vstack((state.q, state.q + self.dt * np.cumsum(velocities, axis=0)))
        plan = JointPlan(positions, velocities, state.stamp, self.dt)
        self._last_plan = plan
        return PlanResult(result, plan)
