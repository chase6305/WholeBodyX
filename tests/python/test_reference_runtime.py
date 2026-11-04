from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import JointLimits, JointPlan, MPCReferenceManager, PlanResult, QPResult, RobotState


def bounds():
    return JointLimits([-2], [2], [1], [4])


def at(time, q=0.0, v=0.0):
    return RobotState([q], [v], time)


def solved(plan):
    return PlanResult(QPResult("solved", np.zeros(1)), plan)


class HoldPlanner:
    """Deliberately simple planner to isolate runtime scheduling and acceptance."""

    def __init__(self, dt=.1):
        self.dt, self.calls = dt, []

    def reset(self):
        self.calls.clear()

    def solve(self, state, goal):
        self.calls.append(state.stamp)
        return solved(JointPlan(np.tile(state.q, (3, 1)), np.zeros((2, state.q.size)), state.stamp, self.dt))


def test_time_based_period_and_off_grid_goal_change():
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds())
    assert runtime.update(at(0), [.3], .02).reason == "initial"
    assert runtime.update(at(.07), [.3], .02).reason == "reused"
    assert runtime.update(at(.08), [.4], .02).reason == "goal_changed"
    assert runtime.update(at(.1), [.4], .02).reason == "reused"
    assert runtime.update(at(.18), [.4], .02).reason == "periodic"
    np.testing.assert_allclose(planner.calls, [0, .08, .18])


def test_full_interval_coverage_and_expiry_override_period():
    planner = HoldPlanner(dt=.025)
    runtime = MPCReferenceManager(planner, bounds(), replan_period=10)
    assert runtime.update(at(0), [.3], .02).success  # Plan is valid through 0.05.
    assert runtime.update(at(.02), [.3], .02).reason == "reused"
    update = runtime.update(at(.04), [.3], .02)  # Current time valid, next command end invalid.
    assert update.replanned and update.reason == "coverage"
    assert runtime.update(at(.2), [.3], .02).reason == "expired"


def test_tracking_deviation_triggers_early_replan():
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds(), replan_period=1, tracking_tolerance=.03)
    runtime.update(at(0), [.3], .02)
    assert runtime.update(at(.02, q=.01), [.3], .02).reason == "reused"
    update = runtime.update(at(.04, q=.1), [.3], .02)
    assert update.reason == "tracking_error" and update.replanned
    assert update.tracking_error == pytest.approx(.1)
    np.testing.assert_allclose(runtime.plan.positions[0], [.1])


@pytest.mark.parametrize("kind", ["status", "exception", "malformed", "missing_plan", "inconsistent_status"])
def test_failed_refresh_discards_old_plan_then_retries(kind):
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds())
    runtime.update(at(0), [.3], .02)
    good_solve = planner.solve

    def fail(state, goal):
        if kind == "exception":
            raise RuntimeError("backend unavailable")
        if kind == "malformed":
            return object()
        if kind == "missing_plan":
            return solved(None)
        result = good_solve(state, goal)
        if kind == "inconsistent_status":
            return PlanResult(QPResult("infeasible"), result.plan)
        return PlanResult(QPResult("iteration_limit"))

    planner.solve = fail
    result = runtime.update(at(.1), [.3], .02)
    assert not result.success and result.position is None and result.velocity is None
    assert runtime.plan is None and result.solve_ms is not None
    planner.solve = good_solve
    result = runtime.update(at(.1), [.3], .02)
    assert result.replanned and runtime.plan.stamp == .1


@pytest.mark.parametrize("kind", ["stale", "future", "wrong_start", "dimension", "short",
                                  "position", "velocity", "slew"])
def test_rejects_invalid_adapter_plan(kind):
    def solve(state, goal):
        stamp, dt, q, v = state.stamp, .1, state.q.copy(), np.zeros((2, 1))
        if kind == "stale": stamp -= .1
        if kind == "future": stamp += .1
        if kind == "wrong_start": q += .1
        if kind == "dimension": q, v = np.zeros(2), np.zeros((2, 2))
        if kind == "short": dt = .005
        if kind == "position": dt, v[:] = 3., 1.
        if kind == "velocity": v[:] = 1.1
        if kind == "slew": v[:] = .5
        positions = np.vstack((q, q + dt * np.cumsum(v, axis=0)))
        return solved(JointPlan(positions, v, stamp, dt))

    runtime = MPCReferenceManager(SimpleNamespace(solve=solve), bounds())
    result = runtime.update(at(1), [.3], .02)
    assert not result.success and result.solver.status == "error" and runtime.plan is None


def test_clock_rollback_requires_reset_even_after_clock_catches_up():
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds())
    runtime.update(at(1), [.3], .02)
    assert runtime.update(at(.9), [.3], .02).reason == "clock_rollback"
    assert not runtime.update(at(2), [.3], .02).success
    assert runtime.plan is None and len(planner.calls) == 1
    runtime.reset()
    assert runtime.update(at(0), [.3], .02).success


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_planner_reset_discards_old_reference_and_requires_fresh_solve(exception_type):
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds(), replan_period=1.)
    runtime.update(at(0), [.3], .02)
    failure = exception_type("planner reset failed")
    def fail_reset():
        raise failure
    planner.reset = fail_reset
    with pytest.raises(exception_type, match="planner reset failed") as caught:
        runtime.reset()
    assert caught.value is failure
    assert runtime.plan is None
    result = runtime.update(at(.02), [.3], .02)
    assert result.replanned
    np.testing.assert_allclose(planner.calls, [0., .02])


def test_failed_planner_reset_preserves_clock_rollback_latch():
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds())
    runtime.update(at(1), [.3], .02)
    assert runtime.update(at(.9), [.3], .02).reason == "clock_rollback"
    reset = planner.reset
    def fail_reset():
        raise RuntimeError("planner reset failed")
    planner.reset = fail_reset
    with pytest.raises(RuntimeError):
        runtime.reset()
    assert runtime.update(at(2), [.3], .02).reason == "clock_rollback"
    planner.reset = reset
    runtime.reset()
    assert runtime.update(at(0), [.3], .02).success


def test_manager_owns_adapter_plan_snapshot():
    source = JointPlan([[0], [0], [0]], [[0], [0]], 0, .1)
    runtime = MPCReferenceManager(SimpleNamespace(solve=lambda *args: solved(source)), bounds())
    runtime.update(at(0), [.3], .02)
    source.positions.setflags(write=True)
    source.positions[:] = 9
    result = runtime.update(at(.02), [.3], .02)
    np.testing.assert_array_equal(result.position, [0])


def test_single_precision_display_stamp_keeps_a_current_plan():
    planner = HoldPlanner()
    runtime = MPCReferenceManager(planner, bounds())
    runtime.update(at(1e8), [.3], .02)
    plan = runtime.plan
    runtime.discard_outdated(np.float32(1e8), [.3])
    assert runtime.plan is plan
    runtime.discard_outdated(np.nextafter(np.float32(1e8), np.float32(np.inf)), [.3])
    assert runtime.plan is None


@pytest.mark.parametrize("period,tolerance", [(0, .03), (.1, 0), (np.inf, .03), (.1, np.nan)])
def test_invalid_runtime_settings(period, tolerance):
    with pytest.raises(ValueError):
        MPCReferenceManager(HoldPlanner(), bounds(), period, tolerance)


def test_clock_roundoff_allowance_cannot_advance_replan_by_half_a_period():
    runtime = MPCReferenceManager(HoldPlanner(), bounds())
    origin = 1e8
    tick = float(np.spacing(origin))
    assert runtime.update(at(origin), [0], tick).replanned
    update = runtime.update(at(origin + .05), [0], tick)
    assert update.success and update.reason == "reused"
    assert runtime.update(at(origin + .1), [0], tick).reason == "periodic"
