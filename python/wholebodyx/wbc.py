"""Weighted velocity-level task control with hard joint bounds."""

from dataclasses import dataclass, replace

import numpy as np

from .qp import OSQPSolver, QPResult, _FixedQP, _immutable
from .types import VelocityCommand, array, positive


@dataclass(frozen=True)
class ControlResult:
    solver: QPResult
    command: VelocityCommand | None = None


class KinematicWBC:
    def __init__(self, limits, regularization=1e-6, smoothing=0.0, solver=None, *, cache_task_hessian=False):
        self.limits = limits
        self.regularization = positive(regularization, "regularization")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and nonnegative")
        self.smoothing = smoothing
        self.solver = solver if solver is not None else OSQPSolver()
        self._template = None
        if not isinstance(cache_task_hessian, (bool, np.bool_)):
            raise TypeError("cache_task_hessian must be a boolean")
        self._cache_task_hessian = bool(cache_task_hessian)
        self._task_hessian_cache = None

    @property
    def cache_task_hessian(self):
        """Opt-in reuse for stable Jacobians/weights; targets remain dynamic."""
        return self._cache_task_hessian

    def reset(self):
        self.solver.reset()

    def formulate(self, state, tasks, dt):
        lo, hi = self.limits.velocity_bounds(state, dt)
        n = state.q.size
        diagonal = self.regularization + self.smoothing
        reuse = False
        if self._cache_task_hessian:
            tasks = tuple(tasks)
            cached = self._task_hessian_cache
            reuse = (cached is not None and cached[0] == diagonal and cached[2].shape == (n, n)
                     and len(tasks) == len(cached[1])
                     and all(np.array_equal(task.jacobian, j) and np.array_equal(task.weight, w)
                             for task, (j, w) in zip(tasks, cached[1])))
        h = cached[2] if reuse else diagonal * np.eye(n)
        g = -self.smoothing * state.v
        for task in tasks:
            if task.jacobian.shape[1] != n:
                raise ValueError("task and state dimensions differ")
            if not reuse:
                weighted_j = task.weight[:, None] * task.jacobian
                h += task.jacobian.T @ weighted_j
            g -= task.jacobian.T @ (task.weight * task.target_velocity)
        # Only an exactly equal Hessian reuses the validated immutable model.
        if self._template is None or not np.array_equal(h, self._template._hessian):
            if self._template is not None and self._template._constraint.shape == (n, n):
                template, problem = self._template.with_hessian(h, g, lo, hi)
            else:
                template, problem = _FixedQP.with_problem(h, g, np.eye(n), lo, hi)
        else:
            template, problem = self._template, self._template.problem(g, lo, hi)
        if self._cache_task_hessian and not reuse:
            # Keep only the latest validated batch. Do not trust array identity
            # or a reversible read-only flag, and do not cache dynamic vectors.
            inputs = tuple((_immutable(task.jacobian), _immutable(task.weight)) for task in tasks)
            self._task_hessian_cache = (diagonal, inputs, _immutable(h))
        self._template = template
        return problem

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
