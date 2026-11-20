"""Uncondensed sparse integrator MPC, with the same JointPlan contract."""

import numpy as np
from scipy import sparse

from .mpc import _JointMPC, _MPCSettings, _integrate_positions, _time_window, _velocity_changes
from .qp import OSQPSolver, QPResult, _OSQP_SOLVE, _immutable, _validated_qp_solution
from .sparse_qp import _SparseFixedQP
from .types import array


class SparseJointMPC(_JointMPC):
    """Optimize position nodes and interval velocities without dense prediction.

    Model, costs, bounds and returned plans match JointMPC. The solver decision
    order is [q_1..q_N, u_0..u_(N-1)]. Sparse storage grows linearly with joints
    and horizon. Instances remain single-owner and independently resettable.
    """

    def __init__(
        self,
        limits,
        horizon=15,
        dt=0.1,
        tracking_weight=10.0,
        control_weight=0.1,
        terminal_weight=30.0,
        solver=None,
    ):
        self._settings = _MPCSettings(
            limits, horizon, dt, tracking_weight, control_weight, terminal_weight
        )
        # x = scale*z puts position columns near one-step velocity units. Bound
        # the numerical transformation at unusual prediction time steps.
        # Public problems, results and custom solvers keep physical coordinates.
        trajectory_size = self.limits.lower.size * self.horizon
        scale = np.concatenate(
            (np.full(trajectory_size, np.clip(self.dt, 0.01, 1.0)), np.ones(trajectory_size))
        )
        self.solver = (
            OSQPSolver(variable_scaling=scale, adaptive_rho_interval=100)
            if solver is None
            else solver
        )
        self._default_solver = self.solver if solver is None else None
        # OSQP's relative tolerance can exceed the 1e-6 physical residual
        # tolerance at large absolute positions. Translate narrow, distant
        # joint ranges and keep the physical problem for result acceptance.
        half_lower, half_upper = .5 * self.limits.lower, .5 * self.limits.upper
        midpoint = half_lower + half_upper
        magnitude = np.abs(midpoint)
        offset = .125 * magnitude > half_upper - half_lower
        self._position_origin = (
            _immutable(np.where(offset, midpoint, 0.))
            if (offset & (magnitude > 100.)).any()
            else None
        )
        self._last_plan = None
        self._prepare()

    def _prepare(self):
        joint_count, horizon = self.limits.lower.size, self.horizon
        trajectory_size = joint_count * horizon
        self._position_weights = np.full(trajectory_size, self.tracking_weight)
        self._position_weights[-joint_count:] = self.terminal_weight
        diagonal = np.concatenate(
            (self._position_weights, np.full(trajectory_size, self.control_weight))
        )
        hessian = sparse.diags(diagonal, format="csc")
        difference = sparse.eye(trajectory_size, format="csc") - sparse.diags(
            np.ones(trajectory_size - joint_count),
            -joint_count,
            shape=(trajectory_size, trajectory_size),
            format="csc",
        )
        dynamics = sparse.hstack((difference, -self.dt * sparse.eye(trajectory_size)), format="csc")
        # Rows: integrator equalities, position boxes, velocity boxes, then slew.
        rows = [dynamics, sparse.eye(2 * trajectory_size, format="csc")]
        lower = [
            np.zeros(trajectory_size),
            np.tile(self.limits.lower, horizon),
            -np.tile(self.limits.velocity, horizon),
        ]
        upper = [
            np.zeros(trajectory_size),
            np.tile(self.limits.upper, horizon),
            np.tile(self.limits.velocity, horizon),
        ]
        if self.limits.acceleration is not None:
            rows.append(
                sparse.hstack(
                    (sparse.csc_matrix((trajectory_size, trajectory_size)), difference),
                    format="csc",
                )
            )
            bound = self.dt * np.tile(self.limits.acceleration, horizon)
            lower.append(-bound)
            upper.append(bound)
        constraint = sparse.vstack(rows, format="csc")
        self._template = _SparseFixedQP(hessian, constraint)
        self._lower, self._upper = np.concatenate(lower), np.concatenate(upper)
        self._slew_problem = None
        if (
            self._default_solver is not None
            and self.limits.acceleration is not None
            and self.dt <= .02
            and self.horizon >= 30
            and (self.dt * self.limits.acceleration <= .1 * self.limits.velocity).any()
        ):
            self._prepare_slew_problem(hessian, constraint, trajectory_size)

    def _prepare_slew_problem(self, hessian, constraint, size):
        # Add one decision per velocity difference. The original slew rows
        # become equalities and the new decisions receive separate box rows.
        identity = sparse.eye(size, format="csc")
        slew_columns = sparse.vstack((sparse.csc_matrix((3 * size, size)), -identity))
        equalities = sparse.hstack((constraint, slew_columns), format="csc")
        slew_boxes = sparse.hstack((sparse.csc_matrix((size, 2 * size)), identity))
        expanded_constraint = sparse.vstack((equalities, slew_boxes), format="csc")
        expanded_hessian = sparse.block_diag((hessian, sparse.csc_matrix((size, size))), format="csc")
        template = _SparseFixedQP(expanded_hessian, expanded_constraint)
        scale = np.concatenate((self._default_solver.variable_scaling, np.full(size, .1)))
        solver = OSQPSolver(variable_scaling=scale, adaptive_rho_interval=100)
        self._slew_problem = template, solver

    def reset(self):
        super().reset()
        if getattr(self, "_slew_problem", None) is not None:
            self._slew_problem[1].reset()

    def formulate(self, state, goal):
        return self._formulate(state, goal, position_origin=None)

    def _formulate(self, state, goal, position_origin):
        self.limits.check_state(state)
        _time_window(state.stamp, self.dt, self.horizon)
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        joint_count = state.q.size
        trajectory_size = joint_count * self.horizon
        lower, upper = self._lower.copy(), self._upper.copy()
        gradient = np.zeros(2 * trajectory_size)
        position_gradient = gradient[:trajectory_size].reshape(self.horizon, joint_count)
        initial_slew_rows = slice(3 * trajectory_size, 3 * trajectory_size + joint_count)
        try:
            with np.errstate(over="raise", invalid="raise"):
                initial_position = state.q
                position_target = -goal
                if position_origin is not None:
                    initial_position = state.q - position_origin
                    position_target += position_origin
                    position_rows = slice(trajectory_size, 2 * trajectory_size)
                    lower[position_rows].reshape(self.horizon, joint_count)[:] -= position_origin
                    upper[position_rows].reshape(self.horizon, joint_count)[:] -= position_origin
                lower[:joint_count] = upper[:joint_count] = initial_position
                np.multiply(
                    self._position_weights.reshape(self.horizon, joint_count),
                    position_target,
                    out=position_gradient,
                )
                if self.limits.acceleration is not None:
                    lower[initial_slew_rows] += state.v
                    upper[initial_slew_rows] += state.v
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return self._template.problem(gradient, lower, upper)

    def _physical_result(self, result, problem, origin, trajectory_size):
        if not result.success:
            return result
        decision = result.x.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            decision[:trajectory_size].reshape(self.horizon, origin.size)[:] += origin
            decision.setflags(write=False)
            return _validated_qp_solution(
                decision, problem.constraint, problem.lower, problem.upper,
                iterations=result.iterations, detail=result.detail,
            )

    def _solve_slew_problem(self, state, problem, warm, size):
        template, solver = self._slew_problem
        joints = state.q.size
        initial_difference = np.zeros(size)
        initial_difference[:joints] = state.v
        lower = np.concatenate((problem.lower[:3 * size], initial_difference, self._lower[3 * size:]))
        upper = np.concatenate((problem.upper[:3 * size], initial_difference, self._upper[3 * size:]))
        gradient = np.concatenate((problem.gradient, np.zeros(size)))
        if warm is not None:
            velocities = warm[size:].reshape(self.horizon, joints)
            with np.errstate(over="ignore", invalid="ignore"):
                differences = _velocity_changes(velocities, state.v)
            warm = (
                np.concatenate((warm, differences.ravel()))
                if np.isfinite(differences).all()
                else None
            )
        result = solver.solve(template.problem(gradient, lower, upper), warm_start=warm)
        if not result.success:
            return result
        decision = result.x[:2 * size].copy()
        decision.setflags(write=False)
        return _validated_qp_solution(
            decision, problem.constraint, problem.lower, problem.upper,
            iterations=result.iterations, detail=result.detail,
        )

    def solve(self, state, goal):
        physical_problem = self.formulate(state, goal)
        joint_count = state.q.size
        trajectory_size = joint_count * self.horizon
        default_adapter = (
            self.solver is self._default_solver
            and type(self.solver) is OSQPSolver
            and getattr(self.solver.solve, "__func__", None) is _OSQP_SOLVE
        )
        default_problem = (
            default_adapter
            and type(self) is SparseJointMPC
            and getattr(self.formulate, "__func__", None) is _SPARSE_MPC_FORMULATE
        )
        origin = getattr(self, "_position_origin", None)
        problem = physical_problem
        if origin is not None and default_problem:
            problem = self._formulate(state, goal, origin)
        else:
            origin = None
        warm = None
        velocities = self._warm_velocities(state)
        if velocities is not None:
            warm = np.empty(2 * trajectory_size)
            positions = warm[:trajectory_size].reshape(self.horizon, joint_count)
            initial = state.q if origin is None else state.q - origin
            _integrate_positions(initial, velocities, self.dt, out=positions)
            warm[trajectory_size:] = velocities.ravel()
        if default_problem and getattr(self, "_slew_problem", None) is not None:
            first = self._solve_slew_problem(state, problem, warm, trajectory_size)
            if origin is not None:
                first = self._physical_result(first, physical_problem, origin, trajectory_size)
            outcome = self._plan_result(state, first, velocity_offset=trajectory_size, kind="sparse MPC")
            if outcome.plan is not None or first.status == "infeasible":
                return outcome
            # Give a refused expanded solve one attempt in the original model.
            self.reset()
            fallback = self.solver.solve(problem)
            result = QPResult(
                fallback.status, fallback.x, outcome.solver.iterations + fallback.iterations,
                fallback.violation, "retry after velocity-difference solve: " + fallback.detail,
            )
        else:
            retryable_workspace = default_adapter and self.solver._solver is not None
            result = self.solver.solve(problem, warm_start=warm)
            if (
                retryable_workspace
                and result.x is None
                and (result.status == "iteration_limit" or result.detail.lower() == "solved inaccurate")
            ):
                first = result
                self.reset()
                cold = self.solver.solve(problem)
                result = QPResult(
                    cold.status,
                    cold.x,
                    first.iterations + cold.iterations,
                    cold.violation,
                    "cold retry after " + first.detail + "; " + cold.detail,
                )
        if origin is not None:
            result = self._physical_result(result, physical_problem, origin, trajectory_size)
        return self._plan_result(state, result, velocity_offset=trajectory_size, kind="sparse MPC")


_SPARSE_MPC_FORMULATE = SparseJointMPC.formulate
