"""Uncondensed sparse integrator MPC, with the same JointPlan contract."""

import numpy as np
from scipy import sparse

from .mpc import _JointMPC, _MPCSettings, _time_window
from .qp import OSQPSolver, QPResult
from .sparse_qp import _SparseFixedQP
from .types import array


class SparseJointMPC(_JointMPC):
    """Optimize position nodes and interval velocities without dense prediction.

    Model, costs, bounds and returned plans match JointMPC. The solver decision
    order is [q_1..q_N, u_0..u_(N-1)]. Sparse storage grows linearly with joints
    and horizon. Instances remain single-owner and independently resettable.
    """

    def __init__(self, limits, horizon=15, dt=.1, tracking_weight=10.,
                 control_weight=.1, terminal_weight=30., solver=None):
        self._settings = _MPCSettings(limits, horizon, dt, tracking_weight,
                                      control_weight, terminal_weight)
        # q = scale*z puts position columns near one-step velocity units. Bound
        # the numerical transformation at unusual prediction time steps.
        # Public problems, results and custom solvers keep physical coordinates.
        size = self.limits.lower.size * self.horizon
        scale = np.r_[np.full(size, np.clip(self.dt, .01, 1.)), np.ones(size)]
        self.solver = (OSQPSolver(variable_scaling=scale, adaptive_rho_interval=100)
                       if solver is None else solver)
        self._default_solver = self.solver if solver is None else None
        self._last_plan = None
        self._prepare()

    def _prepare(self):
        n, count = self.limits.lower.size, self.horizon
        size = n * count
        self._position_weights = np.full(size, self.tracking_weight)
        self._position_weights[-n:] = self.terminal_weight
        h = sparse.diags(np.r_[self._position_weights, np.full(size, self.control_weight)], format="csc")
        difference = sparse.eye(size, format="csc") - sparse.diags(np.ones(size - n), -n,
                                                                    shape=(size, size), format="csc")
        dynamics = sparse.hstack((difference, -self.dt * sparse.eye(size)), format="csc")
        rows = [dynamics, sparse.eye(2 * size, format="csc")]
        lower = [np.zeros(size), np.tile(self.limits.lower, count), -np.tile(self.limits.velocity, count)]
        upper = [np.zeros(size), np.tile(self.limits.upper, count), np.tile(self.limits.velocity, count)]
        if self.limits.acceleration is not None:
            rows.append(sparse.hstack((sparse.csc_matrix((size, size)), difference), format="csc"))
            bound = self.dt * np.tile(self.limits.acceleration, count)
            lower.append(-bound)
            upper.append(bound)
        self._template = _SparseFixedQP(h, sparse.vstack(rows, format="csc"))
        self._lower, self._upper = np.concatenate(lower), np.concatenate(upper)

    def formulate(self, state, goal):
        self.limits.check_state(state)
        _time_window(state.stamp, self.dt, self.horizon)
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        n, size = state.q.size, state.q.size * self.horizon
        lower, upper = self._lower.copy(), self._upper.copy()
        lower[:n] = upper[:n] = state.q
        gradient = np.zeros(2 * size)
        try:
            with np.errstate(over="raise", invalid="raise"):
                gradient[:size] = -self._position_weights * np.tile(goal, self.horizon)
                if self.limits.acceleration is not None:
                    lower[3 * size:3 * size + n] += state.v
                    upper[3 * size:3 * size + n] += state.v
        except FloatingPointError as exc:
            raise ValueError("nonfinite MPC vectors") from exc
        return self._template.problem(gradient, lower, upper)

    def solve(self, state, goal):
        problem = self.formulate(state, goal)
        size = state.q.size * self.horizon
        warm = None
        velocities = self._warm_velocities(state)
        if velocities is not None:
            positions = state.q + self.dt * np.cumsum(velocities, axis=0)
            warm = np.r_[positions.ravel(), velocities.ravel()]
        retryable_workspace = (self.solver is self._default_solver
                               and type(self.solver) is OSQPSolver
                               and "solve" not in self.solver.__dict__
                               and self.solver._solver is not None)
        result = self.solver.solve(problem, warm_start=warm)
        if (retryable_workspace and result.x is None and
                (result.status == "iteration_limit" or result.detail.lower() == "solved inaccurate")):
            first = result
            self.reset()
            cold = self.solver.solve(problem)
            result = QPResult(cold.status, cold.x, first.iterations + cold.iterations,
                              cold.violation, "cold retry after " + first.detail + "; " + cold.detail)
        return self._plan_result(state, result, velocity_offset=size, kind="sparse MPC")
