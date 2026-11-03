"""Synchronous, clock-based acceptance and scheduling of CoM preview plans."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .centroidal import CoMPlan
from .dynamics_types import CoMReference, CoMState
from .types import array, positive
from .runtime import _deadline_reached


@dataclass(frozen=True)
class CoMReferenceUpdate:
    reference: CoMReference | None
    reason: str
    solve_ms: float | None = None
    tracking_error: float | None = None
    continuity_fallback: bool = False
    detail: str = ""

    @property
    def success(self):
        return self.reference is not None

    @property
    def replanned(self):
        return self.success and self.solve_ms is not None


class CoMReferenceManager:
    """Accept fixed-height preview plans and retain acceleration continuity.

    planner implements plan(), reset(), validate_plan(), horizon and dt. The
    validator returns the maximum physical constraint violation (<=1e-6).
    preview(stamps) optionally returns horizon-by-2 positions and velocities;
    it is called only when planning. Vertical reference components pass through.
    """

    def __init__(self, planner, *, replan_period=.1, tracking_tolerance=.03):
        if not all(callable(getattr(planner, name, None)) for name in ("plan", "reset", "validate_plan")):
            raise TypeError("CoM planner must implement plan, reset, and validate_plan")
        self.planner = planner
        if isinstance(planner.horizon, bool) or not isinstance(planner.horizon, (int, np.integer)) or planner.horizon < 1:
            raise ValueError("CoM planner horizon must be a positive integer")
        self._horizon, self._dt = int(planner.horizon), positive(planner.dt, "planner dt")
        self.replan_period = positive(replan_period, "replan period")
        self.tracking_tolerance = positive(tracking_tolerance, "tracking tolerance")
        self._clear()

    def _clear(self):
        self.plan = None
        self._last_stamp = self._next_replan = None
        self._clock_invalid = False
        self._pending_reason = "initial"
        self.continuity_fallbacks = 0

    def invalidate(self, reason="forced"):
        self.plan = None
        self._next_replan = None
        self._pending_reason = reason

    def reset(self):
        self.invalidate()
        self.planner.reset()
        self._clear()

    def _accept(self, plan, measured, seed, control_dt):
        if not isinstance(plan, CoMPlan):
            raise ValueError("CoM planner must return a CoMPlan")
        plan = CoMPlan(plan.states, plan.jerks, plan.stamp, plan.dt, plan.height, plan.gravity)
        if (plan.stamp != measured.stamp or plan.valid_until < measured.stamp + control_dt
                or plan.dt != self._dt or len(plan.jerks) != self._horizon):
            raise ValueError("CoM plan must cover the current interval and match planner timing")
        expected = np.stack((measured.position[:2], measured.velocity[:2], seed))
        if not np.allclose(plan.states[0], expected, rtol=0, atol=1e-6):
            raise ValueError("CoM plan does not start at the supplied state")
        violation = self.planner.validate_plan(plan)
        if not np.isfinite(violation) or not 0 <= violation <= 1e-6:
            raise ValueError("CoM plan violates physical preview bounds")
        return plan

    def update(self, measured, desired, control_dt, *, preview=None, force=False):
        if not isinstance(measured, CoMState) or not isinstance(desired, CoMReference):
            raise ValueError("CoM runtime requires CoMState and CoMReference values")
        control_dt = positive(control_dt, "control dt")
        stamp, end = measured.stamp, measured.stamp + control_dt
        if not desired.covers(stamp, control_dt):
            raise ValueError("desired CoM reference must cover the current control interval")
        if self._clock_invalid or (self._last_stamp is not None and stamp < self._last_stamp):
            self._clock_invalid = True
            self.invalidate("clock_rollback")
            return CoMReferenceUpdate(None, "clock_rollback", detail="clock moved backward; reset the runtime")
        self._last_stamp = stamp
        previous = self.plan
        error = None
        if previous is not None and previous.stamp <= stamp < previous.valid_until:
            error = float(np.max(np.abs(measured.position[:2] - previous.sample(stamp)[0])))
        if force:
            reason = "forced"
        elif previous is None:
            reason = self._pending_reason
        elif not previous.stamp <= stamp < previous.valid_until:
            reason = "expired"
        elif end > previous.valid_until:
            reason = "coverage"
        elif error > self.tracking_tolerance:
            reason = "tracking_error"
        elif _deadline_reached(stamp, self._next_replan, previous.stamp, control_dt):
            reason = "periodic"
        else:
            reason = "reused"
        solve_ms, fallback = None, False
        if reason != "reused":
            self.invalidate(reason)
            started = perf_counter()
            try:
                if self.planner.horizon != self._horizon or self.planner.dt != self._dt:
                    raise ValueError("CoM planner timing changed; construct a new reference manager")
                stamps = stamp + (np.arange(self._horizon) + 1) * self._dt
                if preview is None:
                    targets = np.tile(desired.position[:2], (self._horizon, 1))
                    velocities = np.tile(desired.velocity[:2], (self._horizon, 1))
                else:
                    targets, velocities = preview(stamps)
                    targets, velocities = array(targets, 2, "CoM preview targets"), array(velocities, 2, "CoM preview velocities")
                expected_shape = (self._horizon, 2)
                if targets.shape != expected_shape or velocities.shape != expected_shape:
                    raise ValueError("CoM preview must match the planner horizon")
                seed = measured.acceleration[:2]
                can_continue = previous is not None and previous.stamp <= stamp < previous.valid_until
                if can_continue:
                    seed = previous.sample(stamp)[2]
                try:
                    plan = self.planner.plan(measured.position[:2], measured.velocity[:2], seed,
                                              targets, stamp, target_velocities=velocities)
                except RuntimeError:
                    if not can_continue:
                        raise
                    fallback = True
                    self.continuity_fallbacks += 1
                    seed = measured.acceleration[:2]
                    plan = self.planner.plan(measured.position[:2], measured.velocity[:2], seed,
                                              targets, stamp, target_velocities=velocities)
                plan = self._accept(plan, measured, seed, control_dt)
                next_replan = stamp + self.replan_period
                if not np.isfinite(next_replan) or next_replan <= stamp:
                    raise ValueError("CoM replan deadline cannot be represented")
            except (ValueError, RuntimeError) as exc:
                return CoMReferenceUpdate(None, reason, (perf_counter() - started) * 1000,
                                           error, fallback, str(exc))
            self.plan, self._next_replan = plan, next_replan
            solve_ms = (perf_counter() - started) * 1000
        p, v, a = self.plan.sample(stamp)
        reference = CoMReference(np.r_[p, desired.position[2]], stamp, end,
                                  velocity=np.r_[v, desired.velocity[2]],
                                  acceleration=np.r_[a, desired.acceleration[2]])
        return CoMReferenceUpdate(reference, reason, solve_ms, error, fallback)
