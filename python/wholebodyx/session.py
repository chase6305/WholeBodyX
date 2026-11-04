"""A display-independent demo session, also usable for replay and tests."""

from collections import deque
from time import perf_counter

import numpy as np

from .backends import BackendRegistry
from .experiments import (CONFIGURATION_OFFSETS, DEFAULT_GOAL, FRAMES, INITIAL_Q, SCENARIOS,
                          reference, reference_configuration, wbc_tasks)
from .model import PlanarDualArm
from .qp import QPResult
from .runtime import MPCReferenceManager
from .simulation import JointIntegrator
from .tasks import PostureTask
from .types import RobotState, VelocityCommand, array
from .wbc import ControlResult


class TrackingSession:
    dt = 0.02
    frames = FRAMES

    def __init__(self, backend="python", mode="MPC + WBC", scenario="tracking",
                 mpc_backend=None, *, registry=None):
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario")
        self.model = PlanarDualArm()
        self.registry = (registry if registry is not None else BackendRegistry()).copy()
        self.scenario = scenario
        self.goal = array(
            INITIAL_Q if scenario in ("cartesian_circle", "coordination") else DEFAULT_GOAL,
            1, "goal")
        self.configure(backend, SCENARIOS[scenario].mode or mode, mpc_backend)
        self.reset()

    def configure(self, backend, mode, mpc_backend=None):
        if mpc_backend is None:
            mpc_backend = backend if backend in self.registry.names("planner") else "python"
        if mode not in ("WBC", "MPC + WBC"):
            raise ValueError("unknown control mode")
        required_mode = SCENARIOS[self.scenario].mode
        if required_mode is not None and mode != required_mode:
            raise ValueError(f"{self.scenario} requires {required_mode}")
        controller = self.registry.create_controller(backend, self.model.limits)
        planner = (self.registry.create_planner(mpc_backend, self.model.limits)
                   if mode == "MPC + WBC" else None)
        manager = MPCReferenceManager(planner, self.model.limits) if planner is not None else None
        # Keep existing components until factory creation and reset both succeed.
        controller.reset()
        if manager is not None:
            manager.reset()
        self.controller, self.planner = controller, planner
        self.backend, self.mode = backend, mode
        self.mpc_backend = mpc_backend
        self.reference_manager = manager
        self._prediction_cache = {}
        self._cached_plan = None

    @property
    def plan(self):
        return self.reference_manager.plan if self.reference_manager is not None else None

    def _invalidate_plan(self, reason="forced"):
        if self.reference_manager is not None:
            self.reference_manager.invalidate(reason)
        self._prediction_cache = {}
        self._cached_plan = None

    def set_scenario(self, scenario):
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario")
        # New components own reset side effects; retain the old session
        # bindings until configuration and simulation reset both succeed.
        previous = self.__dict__.copy()
        try:
            self.scenario = scenario
            self.configure(self.backend, SCENARIOS[scenario].mode or self.mode, self.mpc_backend)
            self.goal = array(
                INITIAL_Q if scenario in ("cartesian_circle", "coordination") else DEFAULT_GOAL,
                1, "goal")
            self.reset()
        except BaseException:
            self.__dict__ = previous
            raise

    def reset(self):
        plant = JointIntegrator(self.model.limits, RobotState(INITIAL_Q, np.zeros(6)))
        self._invalidate_plan("retry")
        try:
            self.controller.reset()
            if self.reference_manager is not None:
                self.reference_manager.reset()
        except BaseException as exc:
            self.failure = str(exc) or type(exc).__name__
            raise
        self.plant = plant
        self._prediction_cache = {}
        self._cached_plan = None
        self.steps = 0
        self.history = deque(maxlen=300)
        self.trails = [deque(maxlen=200), deque(maxlen=200)]
        self.replans = 0
        self.goal_changes = 0
        self.replan_stamps = deque(maxlen=300)
        self.tracking_replans = 0
        self.disturbances = 0
        self.failure = None

    def set_goal(self, goal):
        goal = array(goal, 1, "goal")
        bounds = self.model.limits
        if (goal.shape != bounds.lower.shape
                or np.any(goal < bounds.lower) or np.any(goal > bounds.upper)):
            raise ValueError("goal is outside joint limits or has wrong dimensions")
        if not np.array_equal(goal, self.goal):
            self.goal = goal
            self._invalidate_plan("goal_changed")

    def reference(self, stamp=None):
        return reference(self.model, self.scenario, self.goal,
                         self.plant.read().stamp if stamp is None else stamp)

    def _reference_goal(self, stamp):
        return reference_configuration(self.model, self.scenario, self.goal, stamp)

    def step(self):
        try:
            result = self._step()
            self.failure = None
            return result
        except BaseException as exc:
            self._invalidate_plan("retry")
            self.failure = str(exc) or type(exc).__name__
            raise

    def _step(self):
        disturbance = np.zeros(6)
        if (self.scenario == "disturbance_recovery"
                and self.disturbances < len(CONFIGURATION_OFFSETS)):
            stamp, offset = CONFIGURATION_OFFSETS[self.disturbances]
            if self.plant.read().stamp >= stamp - 1e-9:
                self.plant.inject_configuration_offset(offset)
                disturbance = np.array(offset)
                self.disturbances += 1
        state = self.plant.read()
        mpc_ms = None
        update = None
        if self.mode == "MPC + WBC":
            update = self.reference_manager.update(
                state, self._reference_goal(state.stamp), self.dt)
            mpc_ms = update.solve_ms
            if not update.success:
                raise RuntimeError(f"MPC: {update.solver.status} — {update.solver.detail}")
            if update.replanned:
                self.replans += 1
                self.goal_changes += int(update.reason == "goal_changed")
                self.tracking_replans += int(update.reason == "tracking_error")
                self.replan_stamps.append(state.stamp)
            tasks = [PostureTask(update.position, feedforward=update.velocity).linearize(
                self.model, state)]
        else:
            target = self.reference(state.stamp)
            tasks = wbc_tasks(self.model, state, self.scenario, target)
        started = perf_counter()
        result = self.controller.step(state, tasks, self.dt)
        wbc_ms = (perf_counter() - started) * 1000
        if not isinstance(result, ControlResult) or not isinstance(result.solver, QPResult):
            raise ValueError("WBC adapter must return ControlResult with QPResult")
        if not result.solver.success or result.command is None:
            raise RuntimeError(f"WBC: {result.solver.status} — {result.solver.detail}")
        if not isinstance(result.command, VelocityCommand):
            raise ValueError("WBC adapter must return a VelocityCommand")
        if result.command.stamp != state.stamp:
            raise ValueError("WBC command stamp must match the input state")
        if (not np.isfinite(result.solver.violation)
                or result.solver.violation > 1e-6 or result.solver.violation < 0):
            raise ValueError("WBC adapter reported an invalid constraint violation")
        self.plant.write(result.command, self.dt)
        self.steps += 1
        state = self.plant.read()
        target = self.reference(state.stamp)
        if self.mode == "MPC + WBC":
            self.reference_manager.discard_outdated(state.stamp, target.q)
        errors = []
        for k, frame in enumerate(self.frames):
            actual = self.model.translations(state.q[None, :], frame)[0]
            errors.append(float(np.linalg.norm(actual - target.poses[k].translation)))
            self.trails[k].append(actual.copy())
        sample = {"time": state.stamp, "left_error": errors[0], "right_error": errors[1],
                  "control_stamp_s": result.command.stamp,
                  "wbc_ms": wbc_ms, "mpc_ms": mpc_ms, "violation": result.solver.violation,
                  "iterations": result.solver.iterations,
                  "replan_reason": update.reason if update else "disabled",
                  "tracking_error_rad": update.tracking_error if update else None,
                  "plan_age_s": state.stamp - self.plan.stamp if self.plan else None,
                  "plan_remaining_s": self.plan.valid_until - state.stamp if self.plan else None,
                  "disturbances": self.disturbances,
                  "relative_error": float(np.linalg.norm(
                      (self.trails[1][-1] - self.trails[0][-1])
                      - (target.poses[1].translation - target.poses[0].translation)))}
        for k, value in enumerate(disturbance):
            sample[f"offset_{k}_rad"] = float(value)
        self.history.append(sample)
        return sample

    def prediction(self, frame):
        if frame not in self.frames:
            raise ValueError(f"unknown frame: {frame}")
        if self.reference_manager is not None:
            stamp = self.plant.read().stamp
            self.reference_manager.discard_outdated(stamp, self._reference_goal(stamp))
        if self.mode != "MPC + WBC" or self.plan is None:
            self._prediction_cache = {}
            self._cached_plan = None
            return np.empty((0, 3))
        if self._cached_plan is not self.plan:
            self._prediction_cache = {f: self.model.translations(self.plan.positions, f)
                                      for f in self.frames}
            self._cached_plan = self.plan
        return self._prediction_cache[frame].copy()

    def links(self, q, frame):
        theta, offset = self.model._chain(q, frame)
        base = np.array([0.0, 0.4 if offset == 0 else -0.4, 0.0])
        link_vectors = np.column_stack((self.model.lengths * np.cos(theta),
                                        self.model.lengths * np.sin(theta), np.zeros(3)))
        return np.vstack((base, base + np.cumsum(link_vectors, axis=0)))
