"""Weighted velocity-level task control with hard joint bounds."""

from dataclasses import dataclass, replace

import numpy as np

from .qp import OSQPSolver, QPResult, QuadraticProgram, _FixedQP
from .types import VelocityCommand, array, positive


@dataclass(frozen=True)
class ControlResult:
    solver: QPResult
    command: VelocityCommand | None = None


class KinematicWBC:
    def __init__(self, limits, regularization=1e-6, smoothing=0.0, solver=None):
        self.limits = limits
        self.regularization = positive(regularization, "regularization")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and nonnegative")
        self.smoothing = smoothing
        self.solver = solver if solver is not None else OSQPSolver()
        self._template = None
        self._last_hessian = None

    def reset(self):
        self.solver.reset()

    def formulate(self, state, tasks, dt):
        lo, hi = self.limits.velocity_bounds(state, dt)
        n = state.q.size
        h = (self.regularization + self.smoothing) * np.eye(n)
        g = -self.smoothing * state.v
        for task in tasks:
            if task.jacobian.shape[1] != n:
                raise ValueError("task and state dimensions differ")
            weighted_j = task.weight[:, None] * task.jacobian
            h += task.jacobian.T @ weighted_j
            g -= task.jacobian.T @ (task.weight * task.target_velocity)
        # Posture tracking changes the target and bounds every cycle, while H
        # and the identity constraint stay fixed. Revalidate H whenever any
        # task Jacobian, weight, or controller setting changes its value.
        if self._last_hessian is None or not np.array_equal(h, self._last_hessian):
            problem = QuadraticProgram(h, g, np.eye(n), lo, hi)
            self._last_hessian = h.copy()
            self._template = None
            return problem
        if self._template is None:
            self._template = _FixedQP(h, np.eye(n))
        return self._template.problem(g, lo, hi)

    def step(self, state, tasks, dt):
        dt = positive(dt, "dt")
        problem = self.formulate(state, tasks, dt)
        result = self.solver.solve(problem)
        if result.success:
            if np.shape(result.x) != state.q.shape or not np.isfinite(result.x).all():
                return ControlResult(QPResult("error", iterations=result.iterations, detail="invalid WBC solution"))
            # OSQP accepts tiny feasibility residuals. Normalize only those
            # residuals to the exact box before integrating a command, so an
            # active braking bound remains feasible on the next cycle.
            velocity = np.clip(result.x, problem.lower, problem.upper)
            correction = float(np.max(np.abs(velocity - result.x)))
            if correction > 1e-6:
                return ControlResult(QPResult("error", iterations=result.iterations, violation=correction,
                                              detail="WBC solution exceeds velocity bounds"))
            if not np.array_equal(velocity, result.x):
                result = replace(result, x=array(velocity, 1, "velocity"), violation=0.)
        command = VelocityCommand(result.x, state.stamp, state.stamp + dt) if result.success else None
        return ControlResult(result, command)
