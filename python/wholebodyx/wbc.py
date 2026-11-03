"""Weighted velocity-level task control with hard joint bounds."""

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from .qp import OSQPSolver, QPResult, _FixedQP, _immutable
from .types import VelocityCommand, array, positive


@dataclass(frozen=True)
class ControlResult:
    solver: QPResult
    command: VelocityCommand | None = None


class _TaskHessianCache(NamedTuple):
    diagonal: float
    inputs: tuple[tuple[np.ndarray, np.ndarray], ...]
    hessian: np.ndarray


def _validate_cost_weights(regularization, smoothing):
    """Share cost validation across frontends without creating a solver."""
    regularization = positive(regularization, "regularization")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative")
    return regularization, smoothing


class KinematicWBC:
    """Weighted least-squares joint velocities with hard box constraints.

    Smoothing penalizes deviation from the measured velocity in ``state.v``.
    Task-model caching is optional and compares Jacobians and weights by value.
    """

    def __init__(
        self, limits, regularization=1e-6, smoothing=0.0, solver=None, *, cache_task_hessian=False
    ):
        self.limits = limits
        self.regularization, self.smoothing = _validate_cost_weights(regularization, smoothing)
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
        """Build the velocity QP and retain only a successfully validated model."""
        velocity_lower, velocity_upper = self.limits.velocity_bounds(state, dt)
        joint_count = state.q.size
        diagonal_cost = self.regularization + self.smoothing
        reuse_hessian = False
        if self._cache_task_hessian:
            tasks = tuple(tasks)
            cached = self._task_hessian_cache
            reuse_hessian = (
                cached is not None
                and cached.diagonal == diagonal_cost
                and cached.hessian.shape == (joint_count, joint_count)
                and len(tasks) == len(cached.inputs)
                and all(
                    np.array_equal(task.jacobian, jacobian) and np.array_equal(task.weight, weight)
                    for task, (jacobian, weight) in zip(tasks, cached.inputs)
                )
            )
        hessian = cached.hessian if reuse_hessian else diagonal_cost * np.eye(joint_count)
        gradient = -self.smoothing * state.v
        for task in tasks:
            if task.jacobian.shape[1] != joint_count:
                raise ValueError("task and state dimensions differ")
            if not reuse_hessian:
                weighted_jacobian = task.weight[:, None] * task.jacobian
                hessian += task.jacobian.T @ weighted_jacobian
            gradient -= task.jacobian.T @ (task.weight * task.target_velocity)

        template, problem = self._validated_problem(
            hessian, gradient, velocity_lower, velocity_upper
        )
        # Replace caches only after validation succeeds. Invalid task changes
        # must leave the previous validated model available for recovery.
        if self._cache_task_hessian and not reuse_hessian:
            inputs = tuple((_immutable(task.jacobian), _immutable(task.weight)) for task in tasks)
            # Share certified bytes with the QP template, but give the cache
            # independent array metadata so retained diagnostics cannot alter it.
            self._task_hessian_cache = _TaskHessianCache(
                diagonal_cost, inputs, template._hessian.view()
            )
        self._template = template
        return problem

    def _validated_problem(self, hessian, gradient, lower, upper):
        """Reuse a certified model or replace only the matrix that changed."""
        template = self._template
        if template is not None:
            if np.array_equal(hessian, template._hessian):
                return template, template.problem(gradient, lower, upper)
            if template._constraint.shape == hessian.shape:
                return template.with_hessian(hessian, gradient, lower, upper)
        return _FixedQP.with_problem(hessian, gradient, np.eye(len(gradient)), lower, upper)

    def step(self, state, tasks, dt):
        dt = positive(dt, "dt")
        problem = self.formulate(state, tasks, dt)
        result = self.solver.solve(problem)
        if result.success:
            if (
                np.shape(result.x) != state.q.shape
                or np.iscomplexobj(result.x)
                or not np.isfinite(result.x).all()
            ):
                return ControlResult(
                    QPResult("error", iterations=result.iterations, detail="invalid WBC solution")
                )
            # OSQP accepts tiny feasibility residuals. Normalize only those
            # residuals to the exact box before integrating a command, so an
            # active braking bound remains feasible on the next cycle.
            velocity = np.clip(result.x, problem.lower, problem.upper)
            if not np.array_equal(velocity, result.x):
                correction = float(np.abs(velocity - result.x).max())
                if correction > 1e-6:
                    return ControlResult(
                        QPResult(
                            "error",
                            iterations=result.iterations,
                            violation=correction,
                            detail="WBC solution exceeds velocity bounds",
                        )
                    )
                result = replace(result, x=array(velocity, 1, "velocity"), violation=0.0)
        command = (
            VelocityCommand(result.x, state.stamp, state.stamp + dt) if result.success else None
        )
        return ControlResult(result, command)
