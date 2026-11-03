"""Uncondensed sparse integrator MPC, with the same JointPlan contract."""

import numpy as np
from scipy import sparse

from .mpc import _JointMPC, _MPCSettings, _integrate_positions, _time_window
from .qp import OSQPSolver, QPResult
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
        self._template = _SparseFixedQP(hessian, sparse.vstack(rows, format="csc"))
        self._lower, self._upper = np.concatenate(lower), np.concatenate(upper)

    def formulate(self, state, goal):
        self.limits.check_state(state)
        _time_window(state.stamp, self.dt, self.horizon)
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        joint_count = state.q.size
        trajectory_size = joint_count * self.horizon
        lower, upper = self._lower.copy(), self._upper.copy()
        lower[:joint_count] = upper[:joint_count] = state.q
        gradient = np.zeros(2 * trajectory_size)
        position_gradient = gradient[:trajectory_size].reshape(self.horizon, joint_count)
        initial_slew_rows = slice(3 * trajectory_size, 3 * trajectory_size + joint_count)
        try:
            with np.errstate(over="raise", invalid="raise"):
                np.multiply(
                    self._position_weights.reshape(self.horizon, joint_count),
                    -goal,
                    out=position_gradient,
                )
                if self.limits.acceleration is not None:
                    lower[initial_slew_rows] += state.v
                    upper[initial_slew_rows] += state.v
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return self._template.problem(gradient, lower, upper)

    def solve(self, state, goal):
        problem = self.formulate(state, goal)
        joint_count = state.q.size
        trajectory_size = joint_count * self.horizon
        warm = None
        velocities = self._warm_velocities(state)
        if velocities is not None:
            warm = np.empty(2 * trajectory_size)
            positions = warm[:trajectory_size].reshape(self.horizon, joint_count)
            _integrate_positions(state.q, velocities, self.dt, out=positions)
            warm[trajectory_size:] = velocities.ravel()
        retryable_workspace = (
            self.solver is self._default_solver
            and type(self.solver) is OSQPSolver
            and "solve" not in self.solver.__dict__
            and self.solver._solver is not None
        )
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
        return self._plan_result(state, result, velocity_offset=trajectory_size, kind="sparse MPC")
