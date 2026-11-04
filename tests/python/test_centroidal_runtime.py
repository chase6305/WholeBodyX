"""CoM references are accepted, scheduled, expired and invalidated explicitly."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import CoMReference, CoMReferenceManager, CoMState
from wholebodyx.centroidal import CoMPlan, CoMPreviewMPC


def planner():
    return CoMPreviewMPC([[-.1, -.2], [.1, -.2], [.1, .2], [-.1, .2]], .95)


def measured(stamp=0., position=None, velocity=None, acceleration=None):
    return CoMState([0, 0, .95] if position is None else position,
                     np.zeros(3) if velocity is None else velocity,
                     np.zeros(3) if acceleration is None else acceleration, stamp)


def desired(stamp, dt=.005):
    return CoMReference([.02, .03, .95], stamp, stamp + dt)


@pytest.mark.parametrize("start", [0., 1e8])
def test_clock_schedule_handles_twenty_small_ticks_per_replan_without_step_drift(start):
    mpc = planner()
    manager = CoMReferenceManager(mpc)
    state = measured(start)
    replan_ticks = []
    preview_calls = []

    def preview(stamps):
        preview_calls.append(stamps.copy())
        return np.tile([.02, .03], (mpc.horizon, 1)), np.zeros((mpc.horizon, 2))

    for tick in range(601):
        update = manager.update(state, desired(state.stamp), .005, preview=preview)
        assert update.success, update.detail
        if update.replanned:
            replan_ticks.append(tick)
        stamp = state.stamp + .005
        p, v, a = manager.plan.sample(stamp)
        state = measured(stamp, np.r_[p, .95], np.r_[v, 0], np.r_[a, 0])
    assert replan_ticks == list(range(0, 601, 20))
    assert len(preview_calls) == len(replan_ticks)
    assert mpc.replans == len(replan_ticks)


def test_replanning_preserves_acceleration_and_forwards_vertical_reference():
    manager = CoMReferenceManager(planner())
    manager.update(measured(), desired(0), .005)
    p, v, a = manager.plan.sample(.1)
    state = measured(.1, np.r_[p, .95], np.r_[v, 0], [0, 0, 0])
    command = CoMReference([.02, .03, .96], .1, .105, [0, 0, .01], [0, 0, .02])
    update = manager.update(state, command, .005)
    assert update.replanned and update.reason == "periodic"
    np.testing.assert_allclose(manager.plan.states[0, 2], a, atol=1e-12)
    assert update.reference.position[2] == .96
    assert update.reference.velocity[2] == .01 and update.reference.acceleration[2] == .02


def test_new_plan_is_a_snapshot_and_stale_or_wrong_start_plans_are_rejected(monkeypatch):
    mpc = planner()
    manager = CoMReferenceManager(mpc)
    original = mpc.plan
    for kind in ("stamp", "start", "height", "coverage"):
        manager.reset()

        def malformed(*args, **kwargs):
            plan = original(*args, **kwargs)
            if kind == "stamp":
                return replace(plan, stamp=.1)
            if kind == "start":
                states = plan.states.copy()
                states[:, 0, 0] += .01
                return replace(plan, states=states)
            if kind == "height":
                return replace(plan, height=.7)
            return CoMPlan(plan.states[:2], plan.jerks[:1], 0., .1, .95, 9.81)

        monkeypatch.setattr(mpc, "plan", malformed)
        update = manager.update(measured(), desired(0), .005)
        assert not update.success and manager.plan is None and update.detail
        monkeypatch.setattr(mpc, "plan", original)
    manager.reset()
    assert manager.update(measured(), desired(0), .005).success
    assert manager.plan is not mpc.last_plan
    np.testing.assert_array_equal(manager.plan.states, mpc.last_plan.states)


def test_infeasible_continuity_seed_retries_once_from_measurement():
    mpc = planner()
    manager = CoMReferenceManager(mpc)
    manager.plan = SimpleNamespace(stamp=0., valid_until=1.,
                                   sample=lambda t: (np.zeros(2), np.zeros(2), np.full(2, 100.)))
    update = manager.update(measured(), desired(0), .005, force=True)
    assert update.success and update.continuity_fallback
    assert manager.continuity_fallbacks == 1
    np.testing.assert_array_equal(manager.plan.states[0, 2], [0, 0])
    manager.plan = SimpleNamespace(stamp=0., valid_until=1.,
                                   sample=lambda t: (np.zeros(2), np.zeros(2), np.full(2, 100.)))
    update = manager.update(measured(acceleration=[100, 100, 0]), desired(0), .005, force=True)
    assert not update.success and update.continuity_fallback and manager.plan is None
    assert mpc.last_plan is None


def test_rollback_is_latched_until_successful_reset_and_reset_failure_discards_plan(monkeypatch):
    mpc = planner()
    manager = CoMReferenceManager(mpc)
    assert manager.update(measured(1), desired(1), .005).success
    assert manager.update(measured(.9), desired(.9), .005).reason == "clock_rollback"
    assert not manager.update(measured(2), desired(2), .005).success
    original = mpc.reset

    def fail():
        raise RuntimeError("external reset failed")

    monkeypatch.setattr(mpc, "reset", fail)
    with pytest.raises(RuntimeError, match="external reset"):
        manager.reset()
    assert manager.plan is None
    assert not manager.update(measured(2), desired(2), .005).success
    monkeypatch.setattr(mpc, "reset", original)
    manager.reset()
    assert manager.update(measured(2), desired(2), .005).success


def test_continuous_support_validator_is_independent_of_qp_matrices():
    mpc = planner()
    valid = mpc.plan([0, 0], [0, 0], [0, 0], np.tile([.02, .03], (20, 1)))
    assert mpc.validate_plan(valid) <= 1e-6
    states = valid.states.copy()
    states[:, 0] += [1, 1]
    outside = replace(valid, states=states)
    # A consistent integrator rollout can still violate the support polygon.
    assert mpc.validate_plan(outside) > .5
    mpc.problem.hessian = np.eye(1)
    assert mpc.validate_plan(valid) <= 1e-6
