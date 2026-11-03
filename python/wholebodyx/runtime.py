"""Synchronous MPC reference scheduling, independent of robot geometry and UI."""

from dataclasses import dataclass
from math import isfinite, ulp
from time import perf_counter

import numpy as np

from .mpc import PlanResult, _validated_joint_plan
from .qp import QPResult
from .types import array, positive


def _deadline_reached(stamp, deadline, origin, control_dt):
    """Allow roundoff from repeated clock additions across a planning period.

    A low-rate plan may span many high-rate ticks. Four ULPs of the final
    timestamp alone do not cover those additions (for example 20 x 5 ms).
    The interval count is local to this plan, not the lifetime of the process.
    Cap the allowance below one control tick so poor clock resolution cannot
    turn rounding tolerance into a materially early replan.
    """
    ticks = max(1.0, np.ceil(max(0.0, stamp - origin) / control_dt))
    tolerance = min((4 + ticks) * max(ulp(stamp), ulp(deadline)), 0.25 * control_dt)
    return stamp >= deadline - tolerance


@dataclass(frozen=True)
class ReferenceUpdate:
    position: np.ndarray | None
    velocity: np.ndarray | None
    reason: str
    solver: QPResult | None = None
    solve_ms: float | None = None
    tracking_error: float | None = None  # Max joint-position error from the latest tracking check.

    @property
    def success(self):
        return self.position is not None and self.velocity is not None

    @property
    def replanned(self):
        return self.success and self.solve_ms is not None


class MPCReferenceManager:
    """One owner thread; failed refreshes discard the old plan.

    Replanning follows elapsed simulation/state time, not wall time or step count.
    This does not provide asynchronous execution or a hard real-time deadline.
    """

    def __init__(self, planner, limits, replan_period=0.1, tracking_tolerance=0.03):
        self.planner, self.limits = planner, limits
        self.replan_period = positive(replan_period, "replan period")
        self.tracking_tolerance = positive(tracking_tolerance, "tracking tolerance")
        self._clear()

    def _clear(self):
        self.plan = None
        self._goal = None
        self._next_replan = None
        self._last_stamp = None
        self._clock_invalid = False
        self._pending_reason = "initial"

    def reset(self):
        self.invalidate()
        self.planner.reset()
        self._clear()

    def invalidate(self, reason="forced"):
        self.plan = None
        self._next_replan = None
        self._pending_reason = reason

    def discard_outdated(self, stamp, goal):
        """Hide outdated references at a display boundary without running a solver."""
        stamp = float(stamp)
        if self.plan is not None:
            if not np.array_equal(goal, self._goal):
                self.invalidate("goal_changed")
            elif stamp < self.plan.stamp or stamp >= self.plan.valid_until:
                self.invalidate("expired")

    def update(self, state, goal, control_dt):
        self.limits.check_state(state)
        control_dt = positive(control_dt, "control dt")
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        control_end = state.stamp + control_dt
        if not isfinite(control_end) or control_end <= state.stamp:
            raise ValueError("control interval cannot be represented")
        if self._clock_invalid or (self._last_stamp is not None and state.stamp < self._last_stamp):
            self._clock_invalid = True
            self.invalidate("clock_rollback")
            return ReferenceUpdate(
                None,
                None,
                "clock_rollback",
                QPResult("error", detail="clock moved backward; reset the runtime"),
            )
        self._last_stamp = state.stamp
        self.discard_outdated(state.stamp, goal)
        error = None
        if self.plan is None:
            reason = self._pending_reason
        elif control_end > self.plan.valid_until:
            reason = "coverage"
        else:
            sampled = self.plan.sample(state.stamp)
            predicted = sampled[0]
            error = float(np.max(np.abs(state.q - predicted)))
            if error > self.tracking_tolerance:
                reason = "tracking_error"
            elif _deadline_reached(state.stamp, self._next_replan, self.plan.stamp, control_dt):
                reason = "periodic"
            else:
                reason = "reused"
        result, solve_ms = None, None
        if reason != "reused":
            self.invalidate(reason)  # A failed refresh must not keep the old plan.
            started = perf_counter()
            try:
                outcome = self.planner.solve(state, goal)
                if not isinstance(outcome, PlanResult) or not isinstance(outcome.solver, QPResult):
                    raise ValueError("planner must return a PlanResult with a QPResult status")
                result = outcome.solver
                if not result.success or outcome.plan is None:
                    if result.status == "solved":
                        result = QPResult(
                            "error", detail="solved result has no valid solution or plan"
                        )
                    return ReferenceUpdate(
                        None, None, reason, result, (perf_counter() - started) * 1000, error
                    )
                plan = _validated_joint_plan(outcome.plan, state, self.limits, control_dt)
                next_replan = state.stamp + self.replan_period
                if not isfinite(next_replan) or next_replan <= state.stamp:
                    raise ValueError("replan deadline cannot be represented")
            except (ValueError, RuntimeError) as exc:
                return ReferenceUpdate(
                    None,
                    None,
                    reason,
                    QPResult("error", detail=str(exc)),
                    (perf_counter() - started) * 1000,
                    error,
                )
            solve_ms = (perf_counter() - started) * 1000
            self.plan, self._goal = plan, goal
            self._next_replan = next_replan
            sampled = plan.sample(state.stamp)
        position, velocity = sampled
        return ReferenceUpdate(
            array(position, 1, "reference position"),
            array(velocity, 1, "reference velocity"),
            reason,
            result,
            solve_ms,
            error,
        )
