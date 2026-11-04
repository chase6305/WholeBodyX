"""Future-state joint MPC references with explicit startup seeding."""

from math import ceil, ulp
from time import monotonic
from typing import NamedTuple

import numpy as np

from .mpc import JointPlan, PlanResult, _validated_joint_plan
from .process_mpc import PlanningReply, PlanningToken
from .qp import QPResult
from .runtime import MPCReferenceManager, ReferenceUpdate, _deadline_reached
from .types import RobotState, array, positive


class _PendingPlan(NamedTuple):
    token: PlanningToken
    state: RobotState
    goal: np.ndarray
    submitted_at: float  # Monotonic wall time; state.stamp uses the driver clock.
    reason: str
    nominal_start: float


class _CompletedPlan(NamedTuple):
    plan: JointPlan
    reply: PlanningReply
    reason: str
    nominal_start: float


def _forecast_window(stamp, control_dt, lookahead):
    """Predict a full-tick start without landing after either clock convention.

    Direct stamp + k*dt can land a few ULPs after the actual kth driver tick.
    Waiting for the next tick then skips a complete first velocity interval.
    Counter-based clocks may instead evaluate origin + tick*dt. Use the earlier
    of direct and repeated addition, with four ULPs for their final rounding.
    This chooses the forecast timestamp before solving; replies are never
    rebased and a genuinely future reference is still withheld.
    Bound the work explicitly; longer lookaheads need a coarser control clock
    or a separate forecasting implementation.
    """
    ratio = lookahead / control_dt
    if not np.isfinite(ratio) or ratio > 4096:
        raise ValueError("lookahead must span at most 4096 control ticks")
    nearest = round(ratio)
    ticks = max(1, nearest if abs(ratio - nearest) <= 8 * ulp(ratio) else ceil(ratio))
    start = stamp
    for _ in range(ticks):
        previous = start
        start += control_dt
    # RobotState.v is the command held during the completed interval. The
    # reference at the future tick instead contains the next interval's input.
    # Retain the previous nominal tick separately from the biased plan start;
    # subtracting dt from that start could cross a knot by several ULPs.
    previous = max(previous, stamp + (ticks - 1) * control_dt)
    start = min(start, stamp + ticks * control_dt)
    for _ in range(4):
        start = float(np.nextafter(start, -np.inf))
    if not np.isfinite(start) or start <= stamp or previous >= start:
        raise ValueError("future control timestamp cannot be represented")
    return start, previous


def _future_stamp(stamp, control_dt, lookahead):
    return _forecast_window(stamp, control_dt, lookahead)[0]


class AsyncMPCReferenceManager(MPCReferenceManager):
    """Reuse a valid plan while a process plans from its predicted future state.

    The owner must reset and seed an approved trajectory before stepping. A
    completed plan keeps its original timestamp: it is never rebased to hide
    latency. Goal changes fence older requests but keep the active trajectory
    until its replacement passes tracking and coverage checks. Failures clear
    references and require an explicit reset/seed through the session owner.
    Startup, worker shutdown and readiness waits belong outside the cycle.
    max_adoption_delay optionally bounds model-time lateness at adoption. A
    late result is discarded and replanned while the active plan stays valid;
    None retains the original sampling policy without a lateness bound.
    """

    def __init__(
        self,
        worker,
        *,
        replan_period=0.1,
        tracking_tolerance=0.03,
        lookahead=0.1,
        max_pending_time=1.0,
        max_adoption_delay=None,
    ):
        for method in ("submit", "poll", "cancel", "reset"):
            if not callable(getattr(worker, method, None)):
                raise TypeError("worker must implement submit, poll, cancel and reset")
        self.lookahead = positive(lookahead, "planning lookahead")
        self.max_pending_time = positive(max_pending_time, "maximum pending time")
        if max_adoption_delay is not None:
            max_adoption_delay = float(max_adoption_delay)
            if not np.isfinite(max_adoption_delay) or max_adoption_delay < 0:
                raise ValueError("maximum adoption delay must be finite and nonnegative")
        self.max_adoption_delay = max_adoption_delay
        if self.lookahead >= worker.horizon * worker.dt:
            raise ValueError("lookahead must be shorter than the planning horizon")
        super().__init__(worker, worker.limits, replan_period, tracking_tolerance)

    def _clear(self):
        super()._clear()
        self._pending = self._future = None
        self._active_goal = None
        self._needs_reset = False

    @property
    def pending(self):
        return self._pending is not None or self._future is not None

    @property
    def active_goal(self):
        return None if self._active_goal is None else self._active_goal.copy()

    def invalidate(self, reason="forced"):
        super().invalidate(reason)
        self._pending = self._future = None
        self._active_goal = None
        self._needs_reset = True
        self.planner.cancel()

    def seed(self, plan, state, goal, control_dt):
        """Install an explicitly approved initial plan outside the control cycle.

        Call session.reset() first, then seed from the driver's current state.
        A stationary seed is appropriate only when validated against that state
        and its limits. This method does not clear a latched clock/session fault.
        """
        if self._clock_invalid or self._needs_reset:
            raise RuntimeError(
                "reset the runtime before seeding after invalidation or clock rollback"
            )
        if self.plan is not None or self.pending:
            raise RuntimeError("reset the runtime before replacing its seed")
        self.limits.check_state(state)
        control_dt = positive(control_dt, "control dt")
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        plan = _validated_joint_plan(plan, state, self.limits, control_dt)
        if _future_stamp(state.stamp, control_dt, self.lookahead) + control_dt > plan.valid_until:
            raise ValueError("seed does not cover lookahead and a full control interval")
        self.plan, self._goal, self._active_goal = plan, goal, goal.copy()
        self._last_stamp = state.stamp
        self._next_replan = state.stamp
        self._pending_reason = "seeded"

    def discard_outdated(self, stamp, goal):
        # Goal changes are handled by update, where they fence pending work.
        # A display boundary may hide expired data but must never run a solver.
        stamp = float(stamp)
        if self.plan is not None and (
            not np.isfinite(stamp) or stamp < self.plan.stamp or stamp >= self.plan.valid_until
        ):
            self.invalidate("expired")

    def _failure(self, reason, detail, result=None, solve_ms=None, error=None):
        self.invalidate(reason)
        return ReferenceUpdate(
            None,
            None,
            reason,
            QPResult("error", detail=detail) if result is None else result,
            solve_ms,
            error,
        )

    def _receive(self, control_dt):
        reply = self.planner.poll()
        if reply is None:
            return
        if not isinstance(reply, PlanningReply) or self._pending is None:
            raise ValueError("unexpected planning reply")
        pending = self._pending
        if (
            reply.token != pending.token
            or reply.state.stamp != pending.state.stamp
            or not np.array_equal(reply.state.q, pending.state.q)
            or not np.array_equal(reply.state.v, pending.state.v)
            or not np.array_equal(reply.goal, pending.goal)
        ):
            raise ValueError("planning reply differs from the current request")
        outcome = reply.outcome
        if not isinstance(outcome, PlanResult) or not isinstance(outcome.solver, QPResult):
            raise ValueError("worker must return a PlanResult with a QPResult status")
        if not np.isfinite(reply.solve_ms) or reply.solve_ms < 0:
            raise ValueError("invalid worker solve duration")
        if not outcome.solver.success or outcome.plan is None:
            raise RuntimeError(
                f"worker planning failed: {outcome.solver.status}: {outcome.solver.detail}"
            )
        solution = array(outcome.solver.x, 1, "worker solution")
        expected_solution_size = (
            self.planner.horizon
            * pending.state.q.size
            * (2 if "sparse" in self.planner.backend else 1)
        )
        if solution.shape != (expected_solution_size,):
            raise ValueError("worker solution dimensions differ")
        plan = _validated_joint_plan(outcome.plan, pending.state, self.limits, control_dt)
        if plan.dt != self.planner.dt or len(plan.velocities) != self.planner.horizon:
            raise ValueError("worker returned a different planning model")
        self._future = _CompletedPlan(plan, reply, pending.reason, pending.nominal_start)
        self._pending = None

    def _submit_future(self, state, goal, control_dt, reason):
        """Forecast and submit once; a busy relay leaves the request due for retry."""
        try:
            start, previous = _forecast_window(state.stamp, control_dt, self.lookahead)
        except ValueError as exc:
            return self._failure("coverage", str(exc))
        if start + control_dt > self.plan.valid_until:
            return self._failure(
                "coverage", "active plan cannot predict the requested future interval"
            )
        position, _ = self.plan.sample(start)
        _, held_velocity = self.plan.sample(previous)
        predicted_state = RobotState(position, held_velocity, start)
        try:
            token = self.planner.submit(predicted_state, goal)
        except (ValueError, RuntimeError) as exc:
            return self._failure("worker_error", str(exc))
        if token is not None:
            deadline = state.stamp + self.replan_period
            if not np.isfinite(deadline) or deadline <= state.stamp:
                return self._failure("coverage", "replan deadline cannot be represented")
            self._pending = _PendingPlan(
                token,
                predicted_state,
                goal,
                monotonic(),
                reason,
                max(start, previous + control_dt),
            )
            self._next_replan = deadline
        return None

    def update(self, state, goal, control_dt):
        self.limits.check_state(state)
        control_dt = positive(control_dt, "control dt")
        goal = array(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        if not np.isfinite(state.stamp + control_dt) or state.stamp + control_dt <= state.stamp:
            raise ValueError("control interval cannot be represented")
        if self._clock_invalid or (self._last_stamp is not None and state.stamp < self._last_stamp):
            self._clock_invalid = True
            return self._failure("clock_rollback", "clock moved backward; reset the runtime")
        self._last_stamp = state.stamp
        if self.planner.error is not None:
            return self._failure("worker_error", self.planner.error)
        if self.plan is None:
            return self._failure(
                "unseeded", "asynchronous references require an approved seed after reset"
            )
        changed = not np.array_equal(goal, self._goal)
        if changed:
            self.planner.cancel()
            self._pending = self._future = None
            self._goal = goal
            self._next_replan = state.stamp
        try:
            self._receive(control_dt)
        except (ValueError, RuntimeError) as exc:
            return self._failure("worker_result", str(exc))
        result = solve_ms = None
        reason = "goal_changed" if changed else "reused"
        if self._future is not None and state.stamp >= self._future.plan.stamp:
            completed = self._future
            plan = completed.plan
            # Use the latest forecast clock convention, rather than the
            # downward-biased plan stamp, for the optional adoption deadline.
            # A stale candidate never extends the active plan's lifetime.
            rounding = min(8 * ulp(state.stamp), control_dt * 0.25)
            late = (
                self.max_adoption_delay is not None
                and state.stamp - completed.nominal_start > self.max_adoption_delay + rounding
            )
            if late:
                self._future = None
                self._next_replan = state.stamp
                reason = "late_reply"
            else:
                if state.stamp >= plan.valid_until or state.stamp + control_dt > plan.valid_until:
                    return self._failure("coverage", "completed plan expired before adoption")
                predicted, _ = plan.sample(state.stamp)
                error = float(np.max(np.abs(state.q - predicted)))
                if error > self.tracking_tolerance:
                    return self._failure(
                        "tracking_error",
                        "measured state diverged from the future plan",
                        error=error,
                    )
                self.plan, self._active_goal = plan, completed.reply.goal.copy()
                self._future = None
                result = completed.reply.outcome.solver
                solve_ms = completed.reply.solve_ms
                reason = completed.reason
        if (
            state.stamp < self.plan.stamp
            or state.stamp >= self.plan.valid_until
            or state.stamp + control_dt > self.plan.valid_until
        ):
            return self._failure(
                "coverage", "active plan does not cover the complete control interval"
            )
        position, velocity = self.plan.sample(state.stamp)
        error = float(np.max(np.abs(state.q - position)))
        if error > self.tracking_tolerance:
            return self._failure(
                "tracking_error", "measured state diverged from the active plan", error=error
            )
        if (
            self._pending is not None
            and monotonic() - self._pending.submitted_at > self.max_pending_time
        ):
            return self._failure("worker_timeout", "planning request exceeded its wall-time budget")
        if not self.pending:
            due = _deadline_reached(state.stamp, self._next_replan, self.plan.stamp, control_dt)
            coverage = (
                state.stamp + self.lookahead + self.replan_period + control_dt
                >= self.plan.valid_until
            )
            if due or coverage:
                request_reason = (
                    "goal_changed" if changed else "coverage" if coverage else "periodic"
                )
                failure = self._submit_future(state, goal, control_dt, request_reason)
                if failure is not None:
                    return failure
        if result is None and self.pending and not changed and reason != "late_reply":
            reason = "pending"
        return ReferenceUpdate(
            array(position, 1, "reference position"),
            array(velocity, 1, "reference velocity"),
            reason,
            result,
            solve_ms,
            error,
        )
