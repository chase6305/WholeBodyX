"""Condensed, constrained joint-velocity MPC for a fixed-base integrator."""

from dataclasses import dataclass
from math import isfinite, ulp
from weakref import ref

import numpy as np
from scipy import sparse

from .qp import OSQPSolver, QPResult, _OSQP_SOLVE, _immutable, _validated_qp_solution
from .sparse_qp import _SparseFixedQP, _sparse_template_bytes
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


def _prefer_slew_variables(limits, dt, horizon):
    return (
        limits.acceleration is not None
        and dt <= .02
        and horizon >= 30
        and (dt * limits.acceleration <= .1 * limits.velocity).any()
    )


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
    _check_plan_acceptance(plan, state, limits, control_dt)
    return plan


def _check_plan_acceptance(plan, state, limits, control_dt):
    """Check a freshly constructed plan against the current execution context."""
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
        self._unconstrained_fast_path = unconstrained_fast_path
        self._default_solver_token = None
        if solver is None:
            # Identify the original adapter without retaining it after a
            # replacement. An unopened planner can still be serialized.
            self._default_solver_token = object()
            self.solver._joint_mpc_owner = self._default_solver_token
        self._last_plan = None
        self._last_qp_warm_start = self._last_direct_warm_start = None
        self._prefer_sparse = solver is None and _prefer_slew_variables(self.limits, self.dt, self.horizon)
        self._sparse_planner = None
        self._prepare()

    def reset(self):
        super().reset()
        self._last_qp_warm_start = self._last_direct_warm_start = None
        if getattr(self, "_sparse_planner", None) is not None:
            self._sparse_planner.reset()

    @property
    def fixed_matrix_bytes(self):
        """Owned fixed H/A buffers, including dense diagnostics if requested."""
        total = _sparse_template_bytes(self._template)
        if self._dense_template is not None:
            total += self._dense_template._hessian.nbytes + self._dense_template._constraint.nbytes
        if getattr(self, "_sparse_planner", None) is not None:
            total += self._sparse_planner.fixed_matrix_bytes
        return total

    def _prepare(self):
        joint_count, horizon = self.limits.lower.size, self.horizon
        decision_count = joint_count * horizon
        # Keep only nonzero joint/temporal terms. The condensed decision order
        # remains u_0..u_(N-1); this is not the uncondensed SparseJointMPC model.
        temporal = np.tri(horizon)
        temporal *= self.dt
        weights = np.full(horizon, self.tracking_weight)
        weights[-1] = self.terminal_weight
        prediction = sparse.kron(temporal, sparse.eye(joint_count, format="csc"), format="csc")
        # Two command columns overlap from their later interval to the end.
        self._gradient_weights = np.cumsum((self.dt * weights)[::-1])[::-1]
        position_cost = self.dt * self._gradient_weights
        temporal_hessian = np.minimum.outer(position_cost, position_cost)
        temporal_hessian.flat[:: horizon + 1] += self.control_weight
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
        if (
            self._default_solver_token is not None
            and self._unconstrained_fast_path
            and self._gradient_weights[0] > 0
        ):
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
            try:
                gain = np.linalg.solve(hessian, rhs)
            except np.linalg.LinAlgError:
                return
        if np.isfinite(gain).all():
            self._unconstrained_gain = _immutable(gain)

    def _vectors(self, state, goal, *, include_goal=False):
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
        vectors = gradient, lower, upper
        return (vectors, goal) if include_goal else vectors

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
            velocities = gain[:, None] * first_gradient
            decision = velocities.ravel()
            size = decision.size
            values = np.empty_like(lower)
            values[:size] = decision
            # The condensed position rows sum dt*u in interval order.
            # Accumulate those rows without visiting their triangular matrix.
            position = values[size:2 * size].reshape(velocities.shape)
            np.multiply(velocities, self.dt, out=position)
            np.cumsum(position, axis=0, out=position)
            if self.limits.acceleration is not None:
                slew = values[2 * size:].reshape(velocities.shape)
                slew[0] = first
                np.subtract(velocities[1:], velocities[:-1], out=slew[1:])
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

    def _reuse_workspace_warm_start(self, state, gradient):
        recent_qp = self._last_qp_warm_start
        recent_direct = self._last_direct_warm_start
        if recent_qp is None or recent_direct is None:
            return False
        workspace, stamp, qp_gradient = recent_qp
        direct_plan, direct_gradient = recent_direct
        if (
            self._last_plan is not direct_plan
            or self.solver._solver is None
            or workspace() is not self.solver._solver
            or state.stamp < stamp
            or _interval(state.stamp, stamp, self.dt, self.horizon) != 0
        ):
            return False
        # A direct solve leaves the OSQP primal and dual iterates in place.
        # Keep those iterates if the new target is closer to their problem.
        with np.errstate(over="ignore", invalid="ignore"):
            return np.linalg.norm(gradient - qp_gradient) < np.linalg.norm(
                gradient - direct_gradient
            )

    def _solve_sparse_model(self, state, goal, problem):
        from .sparse_mpc import SparseJointMPC

        try:
            if self._sparse_planner is None:
                self._sparse_planner = SparseJointMPC(
                    self.limits, self.horizon, self.dt, self.tracking_weight,
                    self.control_weight, self.terminal_weight,
                )
            outcome = self._sparse_planner.solve(state, goal)
        except ValueError as exc:
            return PlanResult(QPResult("error", detail=str(exc)))
        if outcome.plan is None:
            return outcome
        decision = outcome.solver.x[-problem.gradient.size:].copy()
        decision.setflags(write=False)
        result = _validated_qp_solution(
            decision, problem.constraint, problem.lower, problem.upper,
            iterations=outcome.solver.iterations, detail=outcome.solver.detail,
        )
        return self._plan_result(state, result)

    def solve(self, state, goal):
        # Custom solvers and overridden formulation methods retain the public
        # dense contract. Only the built-in adapter uses the compressed model.
        first_gradient = None
        sparse_attempt = None
        default_model = False
        if (
            type(self.solver) is OSQPSolver
            and getattr(self.solver.solve, "__func__", None) is _OSQP_SOLVE
            and getattr(self.formulate, "__func__", None) is _JOINT_MPC_FORMULATE
        ):
            default_model = (
                type(self) is JointMPC
                and getattr(self, "_default_solver_token", None) is not None
                and getattr(self.solver, "_joint_mpc_owner", None) is self._default_solver_token
                and getattr(self._vectors, "__func__", None) is _JOINT_MPC_VECTORS
            )
            if default_model:
                vectors, goal = self._vectors(state, goal, include_goal=True)
            else:
                vectors = self._vectors(state, goal)
            use_sparse = default_model and getattr(self, "_prefer_sparse", False)
            if default_model and self._unconstrained_gain is not None:
                first_gradient = vectors[0][:state.q.size]
                direct = self._unconstrained_result(state, *vectors)
                if direct is not None:
                    self._last_direct_warm_start = (direct.plan, first_gradient.copy())
                    return direct
            problem = self._template.problem(*vectors)
            if use_sparse:
                sparse_attempt = self._solve_sparse_model(state, goal, problem)
                if sparse_attempt.plan is not None or sparse_attempt.solver.status == "infeasible":
                    return sparse_attempt
        else:
            problem = self.formulate(state, goal)
        if first_gradient is not None and self._reuse_workspace_warm_start(state, first_gradient):
            velocities = None
        else:
            velocities = self._warm_velocities(state)
        warm = None if velocities is None else velocities.reshape(-1)
        self._last_qp_warm_start = None
        result = self.solver.solve(problem, warm_start=warm)
        if default_model and sparse_attempt is None and result.status == "iteration_limit":
            retry = self._solve_sparse_model(state, goal, problem)
            solver = retry.solver
            self._last_direct_warm_start = None
            result = QPResult(
                solver.status, solver.x, result.iterations + solver.iterations,
                solver.violation, "sparse retry after condensed iteration limit: " + solver.detail,
            )
            return PlanResult(result, retry.plan)
        if sparse_attempt is not None:
            result = QPResult(
                result.status, result.x, sparse_attempt.solver.iterations + result.iterations,
                result.violation, "condensed retry after sparse solve: " + result.detail,
            )
        outcome = self._plan_result(state, result)
        if outcome.plan is not None:
            self._last_direct_warm_start = None
            if first_gradient is not None and self.solver._solver is not None:
                self._last_qp_warm_start = (
                    ref(self.solver._solver), state.stamp, first_gradient.copy()
                )
        return outcome


_JOINT_MPC_FORMULATE = JointMPC.formulate
_JOINT_MPC_VECTORS = JointMPC._vectors
