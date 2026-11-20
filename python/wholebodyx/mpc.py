"""Condensed, constrained joint-velocity MPC for a fixed-base integrator."""

from dataclasses import dataclass
from math import isfinite, ulp

import numpy as np
from scipy import sparse

from .qp import OSQPSolver, QPResult, _OSQP_SOLVE, _immutable
from .sparse_qp import _SparseFixedQP
from .types import JointLimits, array, positive


def _time_window(stamp, dt, count):
    stamp, dt = float(stamp), float(dt)
    end = stamp + dt * count
    if (
        not isfinite(stamp)
        or not isfinite(end)
        or stamp < 0
        or end <= stamp
        or dt <= 8 * max(ulp(stamp), ulp(end))
    ):
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


def _velocity_changes(velocities, initial_velocity):
    """Include the measured-to-first-command slew without stacking a history."""
    changes = np.empty_like(velocities)
    np.subtract(velocities[0], initial_velocity, out=changes[0])
    np.subtract(velocities[1:], velocities[:-1], out=changes[1:])
    return changes


def _integrate_positions(initial_position, velocities, dt, *, out):
    """Write predicted nodes after the initial state into the supplied buffer."""
    with np.errstate(over="ignore", invalid="ignore"):
        np.cumsum(velocities, axis=0, out=out)
        out *= dt
        out += initial_position


def _joint_plan_violation(limits, positions, velocities, initial_velocity, dt):
    """Physical boxes/slew after integrating the exported velocity sequence."""
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        violation = max(
            np.max(limits.lower - positions[1:]),
            np.max(positions[1:] - limits.upper),
            np.max(np.abs(velocities) - limits.velocity),
            0.0,
        )
        if limits.acceleration is not None:
            changes = _velocity_changes(velocities, initial_velocity)
            violation = max(violation, np.max(np.abs(changes) - dt * limits.acceleration))
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
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, (int, np.integer))
            or self.horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        object.__setattr__(self, "horizon", int(self.horizon))
        for name in ("dt", "tracking_weight", "control_weight", "terminal_weight"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        limits = JointLimits(
            self.limits.lower, self.limits.upper, self.limits.velocity, self.limits.acceleration
        )
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
        positions = array(self.positions, 2, "positions")
        velocities = array(self.velocities, 2, "velocities")
        object.__setattr__(self, "dt", positive(self.dt, "plan dt"))
        object.__setattr__(self, "stamp", float(self.stamp))
        if (
            not velocities.shape[0]
            or not velocities.shape[1]
            or positions.shape != (velocities.shape[0] + 1, velocities.shape[1])
        ):
            raise ValueError("plan must contain N+1 positions and N velocities")
        _time_window(self.stamp, self.dt, len(velocities))
        try:
            # Overflow must not make two inconsistent displacements compare
            # equal as infinities. Both temporary arrays are safe to reuse.
            with np.errstate(over="raise", invalid="raise"):
                error = np.diff(positions, axis=0)
                expected = self.dt * velocities
                error -= expected
                np.abs(error, out=error)
                tolerance = np.abs(expected, out=expected)
                tolerance *= 1e-7
                tolerance += 1e-8
                consistent = (error <= tolerance).all()
        except FloatingPointError:
            consistent = False
        if not consistent:
            raise ValueError("plan positions and velocities must satisfy integrator dynamics")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)

    @property
    def valid_until(self):
        return self.stamp + self.dt * len(self.velocities)

    def sample(self, stamp):
        stamp = float(stamp)
        if not isfinite(stamp) or stamp < self.stamp or stamp >= self.valid_until:
            raise ValueError("reference is future-dated or expired")
        elapsed = stamp - self.stamp
        index = _interval(stamp, self.stamp, self.dt, len(self.velocities))
        offset = max(0.0, elapsed - index * self.dt)
        velocity = self.velocities[index]
        return self.positions[index] + offset * velocity, velocity.copy()


def _validated_joint_plan(plan, state, limits, control_dt):
    """Own and accept an integrator plan at either scheduling or IPC boundaries."""
    if not isinstance(plan, JointPlan):
        raise ValueError("planner must return a JointPlan value")
    plan = JointPlan(plan.positions, plan.velocities, plan.stamp, plan.dt)
    if plan.stamp != state.stamp:
        raise ValueError("planner returned a stale or future-dated plan")
    if plan.positions.shape[1] != state.q.size or not np.allclose(
        plan.positions[0], state.q, atol=1e-6, rtol=0
    ):
        raise ValueError("plan does not start at the supplied measured state")
    if plan.valid_until < state.stamp + control_dt:
        raise ValueError("plan does not cover the complete control interval")
    tolerance = 1e-6
    if (
        np.any(plan.positions[1:] < limits.lower - tolerance)
        or np.any(plan.positions[1:] > limits.upper + tolerance)
        or np.any(np.abs(plan.velocities) > limits.velocity + tolerance)
    ):
        raise ValueError("plan violates position or velocity limits")
    if limits.acceleration is not None:
        changes = _velocity_changes(plan.velocities, state.v)
        if np.any(np.abs(changes) > limits.acceleration * plan.dt + tolerance):
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
        previous_velocities = self._last_plan.velocities
        shifted = np.empty_like(previous_velocities)
        remaining = self.horizon - offset
        shifted[:remaining] = previous_velocities[offset:]
        shifted[remaining:] = previous_velocities[-1]
        return shifted

    def _plan_result(self, state, result, velocity_offset=0, kind="MPC"):
        if not result.success:
            return PlanResult(result)
        joint_count = state.q.size
        velocity_count = self.horizon * joint_count
        if (
            np.shape(result.x) != (velocity_offset + velocity_count,)
            or np.iscomplexobj(result.x)
            or not np.isfinite(result.x).all()
        ):
            return PlanResult(
                QPResult("error", iterations=result.iterations, detail=f"invalid {kind} solution")
            )
        velocities = np.asarray(result.x, dtype=float)[velocity_offset:].reshape(
            self.horizon, joint_count
        )
        positions = np.empty((self.horizon + 1, joint_count))
        positions[0] = state.q
        _integrate_positions(state.q, velocities, self.dt, out=positions[1:])
        violation = _joint_plan_violation(self.limits, positions, velocities, state.v, self.dt)
        if violation > 1e-6:
            return PlanResult(
                QPResult(
                    "error",
                    iterations=result.iterations,
                    violation=violation,
                    detail=f"reconstructed {kind} plan violates joint bounds",
                )
            )
        plan = JointPlan(positions, velocities, state.stamp, self.dt)
        self._last_plan = plan
        return PlanResult(result, plan)


class JointMPC(_JointMPC):
    def __init__(
        self,
        limits,
        horizon=15,
        dt=0.1,
        tracking_weight=10.0,
        control_weight=0.1,
        terminal_weight=30.0,
        solver=None,
        *,
        unconstrained_fast_path=True,
    ):
        if not isinstance(unconstrained_fast_path, bool):
            raise ValueError("unconstrained_fast_path must be boolean")
        self._settings = _MPCSettings(
            limits, horizon, dt, tracking_weight, control_weight, terminal_weight
        )
        self.solver = solver if solver is not None else OSQPSolver()
        self._default_solver_token = None
        if solver is None and unconstrained_fast_path:
            # Identify the original adapter without retaining it after a
            # replacement. An unopened planner can still be serialized.
            self._default_solver_token = object()
            self.solver._joint_mpc_owner = self._default_solver_token
        self._last_plan = None
        self._prepare()

    @property
    def fixed_matrix_bytes(self):
        """Owned fixed H/A buffers, including dense diagnostics if requested."""
        matrices = (self._template._hessian, self._template._constraint)
        total = sum(
            part.nbytes
            for matrix in matrices
            for part in (matrix.data, matrix.indices, matrix.indptr)
        )
        if self._dense_template is not None:
            total += self._dense_template._hessian.nbytes + self._dense_template._constraint.nbytes
        return total

    def _prepare(self):
        joint_count, horizon = self.limits.lower.size, self.horizon
        decision_count = joint_count * horizon
        # Keep only nonzero joint/temporal terms. The condensed decision order
        # remains u_0..u_(N-1); this is not the uncondensed SparseJointMPC model.
        temporal = self.dt * np.tril(np.ones((horizon, horizon)))
        weights = np.full(horizon, self.tracking_weight)
        weights[-1] = self.terminal_weight
        prediction = sparse.kron(temporal, sparse.eye(joint_count, format="csc"), format="csc")
        temporal_hessian = temporal.T @ (weights[:, None] * temporal)
        temporal_hessian.flat[:: horizon + 1] += self.control_weight
        self._gradient_weights = temporal.T @ weights
        rows = [sparse.eye(decision_count, format="csc"), prediction]
        lower = [-np.tile(self.limits.velocity, horizon), np.tile(self.limits.lower, horizon)]
        upper = [np.tile(self.limits.velocity, horizon), np.tile(self.limits.upper, horizon)]
        if self.limits.acceleration is not None:
            difference = sparse.eye(decision_count, format="csc") - sparse.diags(
                np.ones(decision_count - joint_count),
                -joint_count,
                shape=(decision_count, decision_count),
                format="csc",
            )
            bound = self.dt * np.tile(self.limits.acceleration, horizon)
            rows.append(difference)
            lower.append(-bound)
            upper.append(bound)
        self._template = _SparseFixedQP.separable(
            temporal_hessian, joint_count, sparse.vstack(rows, format="csc")
        )
        self._dense_template = None
        self._lower, self._upper = np.concatenate(lower), np.concatenate(upper)
        self._unconstrained_gain = None
        if self._default_solver_token is not None and self._gradient_weights[0] > 0:
            self._prepare_unconstrained_gain(temporal_hessian)

    def _prepare_unconstrained_gain(self, hessian):
        # Each joint has the same temporal Hessian and a gradient proportional
        # to its first entry. Cache the response to a unit first entry.
        # When formation roundoff can dominate the control regularization,
        # let the QP solver handle the problem using its normal checks.
        with np.errstate(all="ignore"):
            largest_row_sum = np.linalg.norm(hessian, ord=np.inf)
            rounding = np.finfo(float).eps * (8 * self.horizon) * largest_row_sum
            if not max(np.finfo(float).tiny, rounding) < self.control_weight:
                return
            rhs = -self._gradient_weights / self._gradient_weights[0]
            # OSQP symmetrizes the upper triangle. Mirror it if matrix
            # multiplication produced roundoff asymmetry.
            if not np.array_equal(hessian, hessian.T):
                upper = np.triu(hessian, k=1)
                symmetric = upper + upper.T
                np.fill_diagonal(symmetric, hessian.diagonal())
                hessian = symmetric
            try:
                gain = np.linalg.solve(hessian, rhs)
            except np.linalg.LinAlgError:
                return
        if np.isfinite(gain).all():
            self._unconstrained_gain = _immutable(gain)

    def _vectors(self, state, goal):
        self.limits.check_state(state)
        _time_window(state.stamp, self.dt, self.horizon)
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        joint_count = state.q.size
        decision_count = joint_count * self.horizon
        lower, upper = self._lower.copy(), self._upper.copy()
        position_rows = slice(decision_count, 2 * decision_count)
        initial_slew_rows = slice(2 * decision_count, 2 * decision_count + joint_count)
        try:
            with np.errstate(over="raise", invalid="raise"):
                lower[position_rows].reshape(self.horizon, joint_count)[:] -= state.q
                upper[position_rows].reshape(self.horizon, joint_count)[:] -= state.q
                if self.limits.acceleration is not None:
                    lower[initial_slew_rows] += state.v
                    upper[initial_slew_rows] += state.v
                gradient = (self._gradient_weights[:, None] * (state.q - goal)).ravel()
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return gradient, lower, upper

    def formulate(self, state, goal):
        """Return the public dense QP; allocate its fixed matrices on first use."""
        vectors = self._vectors(state, goal)
        if self._dense_template is None:
            self._dense_template = self._template.dense_template()
        return self._dense_template.problem(*vectors)

    def _unconstrained_result(self, state, gradient, lower, upper):
        gain = self._unconstrained_gain
        joint_count = state.q.size
        first_gradient = gradient[:joint_count]
        with np.errstate(all="ignore"):
            first = gain[0] * first_gradient
            if self.limits.acceleration is not None:
                initial_slew = slice(2 * gradient.size, 2 * gradient.size + joint_count)
                if not (
                    (first >= lower[initial_slew]).all() and (first <= upper[initial_slew]).all()
                ):
                    return None
            if not ((first >= lower[:joint_count]).all() and (first <= upper[:joint_count]).all()):
                return None
            decision = (gain[:, None] * first_gradient).ravel()
            values = self._template._constraint @ decision
        if (
            not np.isfinite(values).all()
            or not (values >= lower).all()
            or not (values <= upper).all()
        ):
            return None
        decision.setflags(write=False)
        result = QPResult("solved", decision, violation=0.0, detail="unconstrained optimum")
        try:
            outcome = self._plan_result(state, result)
        except ValueError:
            return None
        return outcome if outcome.plan is not None else None

    def solve(self, state, goal):
        # Custom solvers and overridden formulation methods retain the public
        # dense contract. Only the built-in adapter uses the compressed model.
        if (
            type(self.solver) is OSQPSolver
            and getattr(self.solver.solve, "__func__", None) is _OSQP_SOLVE
            and getattr(self.formulate, "__func__", None) is _JOINT_MPC_FORMULATE
        ):
            vectors = self._vectors(state, goal)
            if (
                type(self) is JointMPC
                and getattr(self, "_unconstrained_gain", None) is not None
                and getattr(self.solver, "_joint_mpc_owner", None) is self._default_solver_token
                and getattr(self._vectors, "__func__", None) is _JOINT_MPC_VECTORS
            ):
                direct = self._unconstrained_result(state, *vectors)
                if direct is not None:
                    return direct
            problem = self._template.problem(*vectors)
        else:
            problem = self.formulate(state, goal)
        velocities = self._warm_velocities(state)
        warm = None if velocities is None else velocities.reshape(-1)
        result = self.solver.solve(problem, warm_start=warm)
        return self._plan_result(state, result)


_JOINT_MPC_FORMULATE = JointMPC.formulate
_JOINT_MPC_VECTORS = JointMPC._vectors
