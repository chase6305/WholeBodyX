"""Condensed, constrained joint-velocity MPC for a fixed-base integrator."""

from dataclasses import dataclass
from math import ulp

import numpy as np
from scipy import sparse

from .qp import OSQPSolver, QPResult, _immutable
from .sparse_qp import _SparseFixedQP
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


def _joint_plan_violation(limits, positions, velocities, initial_velocity, dt):
    """Physical boxes/slew after integrating the exported velocity sequence."""
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        violation = max(np.max(limits.lower - positions[1:]),
                        np.max(positions[1:] - limits.upper),
                        np.max(np.abs(velocities) - limits.velocity), 0.)
        if limits.acceleration is not None:
            change = np.diff(np.vstack((initial_velocity, velocities)), axis=0)
            violation = max(violation, np.max(np.abs(change) - dt * limits.acceleration))
    return float(violation)


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


def _validated_joint_plan(plan, state, limits, control_dt):
    """Own and accept an integrator plan at either scheduling or IPC boundaries."""
    if not isinstance(plan, JointPlan):
        raise ValueError("planner must return a JointPlan value")
    plan = JointPlan(plan.positions, plan.velocities, plan.stamp, plan.dt)
    if plan.stamp != state.stamp:
        raise ValueError("planner returned a stale or future-dated plan")
    if plan.positions.shape[1] != state.q.size or not np.allclose(plan.positions[0], state.q, atol=1e-6, rtol=0):
        raise ValueError("plan does not start at the supplied measured state")
    if plan.valid_until < state.stamp + control_dt:
        raise ValueError("plan does not cover the complete control interval")
    tol = 1e-6
    if (np.any(plan.positions[1:] < limits.lower - tol) or np.any(plan.positions[1:] > limits.upper + tol)
            or np.any(np.abs(plan.velocities) > limits.velocity + tol)):
        raise ValueError("plan violates position or velocity limits")
    if limits.acceleration is not None:
        change = np.diff(np.vstack((state.v, plan.velocities)), axis=0)
        if np.any(np.abs(change) > limits.acceleration * plan.dt + tol):
            raise ValueError("plan violates velocity-slew limits")
    return plan


@dataclass(frozen=True)
class PlanResult:
    solver: QPResult
    plan: JointPlan | None = None


class _JointMPC:
    """Common configuration, warm history and physical plan acceptance.

    Condensed and uncondensed planners retain their own matrix layouts and
    solver policies. Neither may export a plan without this physical check.
    """
    limits = property(lambda self: self._settings.limits)
    horizon = property(lambda self: self._settings.horizon)
    dt = property(lambda self: self._settings.dt)
    tracking_weight = property(lambda self: self._settings.tracking_weight)
    control_weight = property(lambda self: self._settings.control_weight)
    terminal_weight = property(lambda self: self._settings.terminal_weight)

    def reset(self):
        self.solver.reset()
        self._last_plan = None

    def _warm_velocities(self, state):
        if self._last_plan is None or state.stamp < self._last_plan.stamp:
            return None
        offset = _interval(state.stamp, self._last_plan.stamp, self.dt, self.horizon)
        old = self._last_plan.velocities
        shifted = np.empty_like(old)
        remaining = self.horizon - offset
        shifted[:remaining] = old[offset:]
        shifted[remaining:] = old[-1]
        return shifted

    def _plan_result(self, state, result, velocity_offset=0, kind="MPC"):
        if not result.success:
            return PlanResult(result)
        size = self.horizon * state.q.size
        if (np.shape(result.x) != (velocity_offset + size,) or np.iscomplexobj(result.x)
                or not np.isfinite(result.x).all()):
            return PlanResult(QPResult("error", iterations=result.iterations, detail=f"invalid {kind} solution"))
        velocities = np.asarray(result.x, dtype=float)[velocity_offset:].reshape(self.horizon, state.q.size)
        with np.errstate(over="ignore", invalid="ignore"):
            positions = np.vstack((state.q, state.q + self.dt * np.cumsum(velocities, axis=0)))
        violation = _joint_plan_violation(self.limits, positions, velocities, state.v, self.dt)
        if violation > 1e-6:
            return PlanResult(QPResult("error", iterations=result.iterations, violation=violation,
                                       detail=f"reconstructed {kind} plan violates joint bounds"))
        plan = JointPlan(positions, velocities, state.stamp, self.dt)
        self._last_plan = plan
        return PlanResult(result, plan)


class JointMPC(_JointMPC):
    def __init__(self, limits, horizon=15, dt=0.1, tracking_weight=10.0,
                 control_weight=0.1, terminal_weight=30.0, solver=None):
        self._settings = _MPCSettings(limits, horizon, dt, tracking_weight, control_weight, terminal_weight)
        self.solver = solver if solver is not None else OSQPSolver()
        self._last_plan = None
        self._prepare()

    @property
    def fixed_matrix_bytes(self):
        """Owned fixed H/A buffers, including dense diagnostics if requested."""
        matrices = (self._template._hessian, self._template._constraint)
        total = sum(part.nbytes for matrix in matrices for part in
                    (matrix.data, matrix.indices, matrix.indptr))
        if self._dense_template is not None:
            total += self._dense_template._hessian.nbytes + self._dense_template._constraint.nbytes
        return total

    def _prepare(self):
        n, count = self.limits.lower.size, self.horizon
        size = n * count
        # Keep only nonzero joint/temporal terms. The condensed decision order
        # remains u_0..u_(N-1); this is not the uncondensed SparseJointMPC model.
        temporal = self.dt * np.tril(np.ones((count, count)))
        weights = np.full(count, self.tracking_weight)
        weights[-1] = self.terminal_weight
        prediction = sparse.kron(temporal, sparse.eye(n, format="csc"), format="csc")
        temporal_h = temporal.T @ (weights[:, None] * temporal)
        temporal_h.flat[::count + 1] += self.control_weight
        self._gradient_weights = temporal.T @ weights
        rows = [sparse.eye(size, format="csc"), prediction]
        lo = [-np.tile(self.limits.velocity, count), np.tile(self.limits.lower, count)]
        hi = [np.tile(self.limits.velocity, count), np.tile(self.limits.upper, count)]
        if self.limits.acceleration is not None:
            difference = sparse.eye(size, format="csc") - sparse.diags(
                np.ones(size - n), -n, shape=(size, size), format="csc")
            bound = self.dt * np.tile(self.limits.acceleration, count)
            rows.append(difference)
            lo.append(-bound)
            hi.append(bound)
        self._template = _SparseFixedQP.separable(temporal_h, n, sparse.vstack(rows, format="csc"))
        self._dense_template = None
        self._lower, self._upper = np.concatenate(lo), np.concatenate(hi)

    def _vectors(self, state, goal):
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
                gradient = (self._gradient_weights[:, None] * (state.q - goal)).ravel()
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return gradient, lo, hi

    def formulate(self, state, goal):
        """Return the public dense QP; allocate its fixed matrices on first use."""
        vectors = self._vectors(state, goal)
        if self._dense_template is None:
            self._dense_template = self._template.dense_template()
        return self._dense_template.problem(*vectors)

    def solve(self, state, goal):
        # Custom solvers and overridden formulation methods retain the public
        # dense contract. Only the built-in adapter uses the compressed model.
        if (type(self.solver) is OSQPSolver and
                getattr(self.solver.solve, "__func__", None) is OSQPSolver.solve and
                getattr(self.formulate, "__func__", None) is JointMPC.formulate):
            problem = self._template.problem(*self._vectors(state, goal))
        else:
            problem = self.formulate(state, goal)
        velocities = self._warm_velocities(state)
        warm = None if velocities is None else velocities.reshape(-1)
        result = self.solver.solve(problem, warm_start=warm)
        return self._plan_result(state, result)
