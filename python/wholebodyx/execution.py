"""Robot-independent composition of task control, MPC references and driver I/O.

One owner calls step/reset serially. The driver defines physical or simulation
time; VelocityControlLoop checks the execution boundary and optional wall budget.
"""

from dataclasses import dataclass

import numpy as np

from .hardware import CycleResult, VelocityControlLoop
from .runtime import MPCReferenceManager, ReferenceUpdate
from .tasks import PostureTask
from .types import positive


@dataclass(frozen=True)
class VelocityStep:
    cycle: CycleResult
    reference: ReferenceUpdate | None

    @property
    def command(self):
        return self.cycle.command


class _SessionController:
    """Reset the planner inside the loop's existing fault/stop boundary."""

    def __init__(self, controller, references):
        self.controller, self.references = controller, references

    def reset(self):
        if self.references is not None:
            self.references.reset()
        self.controller.reset()

    def step(self, state, tasks, dt):
        return self.controller.step(state, tasks, dt)


class VelocitySession:
    """Compose an injected model, driver and controller, with optional MPC.

    task_factory(model, state) produces additional LinearTask values every
    cycle. When a planner is supplied, step(goal) adds its sampled posture and
    velocity feedforward to those tasks. A goal may be an array or a callable
    goal(state), evaluated after the single driver read for that cycle.
    reference_manager accepts an MPCReferenceManager subclass in place of a
    planner, including explicitly seeded asynchronous references.

    Failures invalidate references and latch the underlying loop fault. reset()
    resets both algorithms before clearing that fault; it never resets a driver
    or rewinds its state. No history or unbounded diagnostics are retained.
    """

    def __init__(
        self,
        model,
        backend,
        controller,
        control_dt=0.02,
        *,
        planner=None,
        reference_manager=None,
        task_factory=None,
        limits=None,
        replan_period=0.1,
        tracking_tolerance=0.03,
        posture_gain=4.0,
        posture_weight=1.0,
        max_cycle_time=None,
    ):
        self.model, self.backend, self.controller, self.planner = (
            model,
            backend,
            controller,
            planner,
        )
        self.limits = model.limits if limits is None else limits
        if len(model.joint_names) != self.limits.lower.size:
            raise ValueError("model joint names and configured limits dimensions differ")
        if task_factory is not None and not callable(task_factory):
            raise TypeError("task_factory must be callable")
        self.task_factory = task_factory
        self.control_dt = positive(control_dt, "control dt")
        self.posture_gain = positive(posture_gain, "posture gain")
        self.posture_weight = float(posture_weight)
        if not np.isfinite(self.posture_weight) or self.posture_weight < 0:
            raise ValueError("posture weight must be finite and nonnegative")
        if reference_manager is not None:
            if planner is not None:
                raise ValueError("provide either planner or reference_manager")
            if not isinstance(reference_manager, MPCReferenceManager):
                raise TypeError("reference_manager must be an MPCReferenceManager")
            for name in ("lower", "upper", "velocity", "acceleration"):
                if not np.array_equal(
                    getattr(reference_manager.limits, name), getattr(self.limits, name)
                ):
                    raise ValueError("reference manager and session limits differ")
            self.planner = reference_manager.planner
            self.reference_manager = reference_manager
        elif planner is not None:
            self.reference_manager = MPCReferenceManager(
                planner, self.limits, replan_period, tracking_tolerance
            )
        else:
            self.reference_manager = None
        self.loop = VelocityControlLoop(
            backend,
            self.limits,
            _SessionController(controller, self.reference_manager),
            self.control_dt,
            max_cycle_time=max_cycle_time,
        )
        # Validate the actual controller too, rather than only the wrapper.
        if not all(callable(getattr(controller, name, None)) for name in ("reset", "step")):
            raise TypeError("controller must implement reset and step")
        if planner is not None and not all(
            callable(getattr(planner, name, None)) for name in ("reset", "solve")
        ):
            raise TypeError("planner must implement reset and solve")
        self.steps = 0
        self.last_reference = None
        self.failure = None

    @property
    def fault(self):
        return self.loop.fault

    @property
    def plan(self):
        return self.reference_manager.plan if self.reference_manager is not None else None

    def invalidate(self, reason="forced"):
        if self.reference_manager is not None:
            self.reference_manager.invalidate(reason)
        self.last_reference = None

    def reset(self):
        self.invalidate()
        try:
            self.loop.reset()
        except BaseException as exc:
            self.failure = str(exc) or type(exc).__name__
            raise
        self.steps = 0
        self.failure = None

    def step(self, goal=None):
        update = None

        def tasks(state):
            nonlocal update
            result = []
            if self.reference_manager is not None:
                target = goal(state) if callable(goal) else goal
                if target is None:
                    raise ValueError("MPC session requires a joint goal")
                update = self.reference_manager.update(state, target, self.control_dt)
                if not update.success:
                    raise RuntimeError(f"MPC: {update.solver.status} — {update.solver.detail}")
                result.append(
                    PostureTask(
                        update.position,
                        gain=self.posture_gain,
                        weight=self.posture_weight,
                        feedforward=update.velocity,
                    ).linearize(self.model, state)
                )
            elif goal is not None:
                raise ValueError(
                    "joint goals require a planner; provide WBC targets through task_factory"
                )
            if self.task_factory is not None:
                extra = self.task_factory(self.model, state)
                if not isinstance(extra, (list, tuple)):
                    raise ValueError(
                        "task_factory must return a list or tuple of LinearTask values"
                    )
                result.extend(extra)
            return result

        try:
            cycle = self.loop.step(tasks)
        except BaseException as exc:
            self.invalidate("retry")
            self.failure = str(exc) or type(exc).__name__
            raise
        self.steps += 1
        self.last_reference = update
        self.failure = None
        return VelocityStep(cycle, update)
