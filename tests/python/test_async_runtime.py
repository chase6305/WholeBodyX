"""Future reference scheduling without wall-time dependent solver tests."""

from dataclasses import replace
from math import ulp
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (AsyncMPCReferenceManager, JointLimits, JointPlan, KinematicWBC,
                        PlanningReply, PlanningToken, PlanResult, QPResult, RobotState,
                        VelocitySession)
from wholebodyx.simulation import JointIntegrator


class Worker:
    backend, horizon, dt, error = "sparse", 20, .1, None

    def __init__(self):
        self.limits = JointLimits([-1.], [1.], [.8], [2.])
        self.generation = self.sequence = 0
        self.request = self.reply = None
        self.busy = False

    def cancel(self):
        self.generation += 1
        self.request = self.reply = None

    def reset(self):
        if self.error:
            raise RuntimeError(self.error)
        self.cancel()

    def submit(self, state, goal):
        if self.busy:
            return None
        self.sequence += 1
        token = PlanningToken(self.generation, self.sequence)
        self.request = token, state, goal
        return token

    def poll(self):
        reply, self.reply = self.reply, None
        return reply

    def finish(self, velocity=.1):
        token, state, goal = self.request
        v = np.full((self.horizon, 1), velocity)
        q = np.vstack((state.q, state.q + np.cumsum(v * self.dt, axis=0)))
        plan = JointPlan(q, v, state.stamp, self.dt)
        result = PlanResult(QPResult("solved", np.zeros(2 * self.horizon)), plan)
        self.reply = PlanningReply(token, state, goal, result, 12.)
        self.request = None
        return self.reply


def state(stamp=0., q=0., v=0.):
    return RobotState([q], [v], stamp)


def runtime(**kwargs):
    worker = Worker()
    manager = AsyncMPCReferenceManager(worker, lookahead=.2, **kwargs)
    hold = JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 0., .1)
    manager.seed(hold, state(), [.3], .02)
    return manager, worker


def test_future_result_is_withheld_until_original_stamp_then_adopted():
    manager, worker = runtime()
    initial = manager.update(state(), [.3], .02)
    assert initial.success and not initial.replanned and initial.reason == "pending"
    assert worker.request[1].stamp == pytest.approx(.2)
    worker.finish()
    early = manager.update(state(.1), [.3], .02)
    assert early.success and not early.replanned
    np.testing.assert_array_equal(early.velocity, [0.])
    adopted = manager.update(state(.2), [.3], .02)
    assert adopted.success and adopted.replanned and adopted.solve_ms == 12.
    np.testing.assert_array_equal(adopted.velocity, [.1])
    assert manager.plan.stamp == pytest.approx(.2)
    np.testing.assert_array_equal(manager.active_goal, [.3])
    assert worker.request[1].stamp == pytest.approx(.4)  # Subsequent work predicts the new plan.


def test_late_reply_is_sampled_at_current_time_without_rebasing():
    manager, worker = runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    update = manager.update(state(.26, q=.006, v=.1), [.3], .02)
    assert update.replanned and manager.plan.stamp == pytest.approx(.2)
    np.testing.assert_allclose(update.position, [.006], atol=1e-15)


def test_bounded_adoption_discards_late_candidate_and_replans_from_valid_active_plan():
    manager, worker = runtime(max_adoption_delay=0.)
    original = manager.plan
    manager.update(state(), [.3], .02)
    worker.finish()
    update = manager.update(state(.22), [.3], .02)
    assert update.success and not update.replanned and update.reason == "late_reply"
    assert manager.plan is original and manager.pending
    np.testing.assert_array_equal(update.velocity, [0.])
    assert worker.request[1].stamp == pytest.approx(.42)
    worker.finish()
    update = manager.update(state(.42), [.3], .02)
    assert update.replanned and manager.plan.stamp == pytest.approx(.42)


@pytest.mark.parametrize("limit,stamp,adopt", [(0., .2, True), (.04, .24, True), (.04, .26, False)])
def test_adoption_budget_boundary_keeps_original_plan_timestamp(limit, stamp, adopt):
    manager, worker = runtime(max_adoption_delay=limit)
    manager.update(state(), [.3], .02)
    worker.finish()
    update = manager.update(state(stamp), [.3], .02)
    assert update.success and update.replanned == adopt
    if adopt:
        assert manager.plan.stamp == pytest.approx(.2)


@pytest.mark.parametrize("failure", ["tracking", "coverage", "malformed"])
def test_late_candidate_cannot_hide_active_failure_or_invalid_worker_result(failure):
    manager, worker = runtime(max_adoption_delay=0.)
    manager.update(state(), [.3], .02)
    reply = worker.finish()
    if failure == "malformed":
        worker.reply = replace(reply, token=PlanningToken(99, 99))
    current = state(2. if failure == "coverage" else .22, .04 if failure == "tracking" else 0.)
    update = manager.update(current, [.3], .02)
    assert not update.success and manager.plan is None and not manager.pending
    assert update.reason == {"tracking": "tracking_error", "coverage": "coverage", "malformed": "worker_result"}[failure]


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_adoption_budget_is_rejected(value):
    with pytest.raises(ValueError, match="adoption"):
        runtime(max_adoption_delay=value)


def _soak_tool():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts/soak_async_mpc.py"
    spec = importlib.util.spec_from_file_location("wholebodyx_soak_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_diagnostic_retains_plans_after_runtime_invalidation():
    import json
    worker = Worker()
    manager = _soak_tool().DiagnosticReferences(worker, lookahead=.2)
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 0., .1), state(), [.3], .02)
    manager.update(state(), [.3], .02)
    worker.finish()
    update = manager.update(state(.26, .04), [.3], .02)
    assert not update.success and manager.plan is None
    record = manager.diagnostic_snapshot()
    assert record["reference"]["reason"] == "tracking_error"
    assert record["reference"]["tracking_error"] == pytest.approx(.034)
    assert len(record["plans"]) == 2
    assert record["failure"]["future"]["solve_ms"] == 12.
    assert record["failure"]["future"]["solver"]["violation"] is None
    assert not record["failure"]["future"]["solver"]["violation_finite"]
    assert record["after"]["active"] is None
    json.dumps(record, allow_nan=False)


def test_recent_diagnostics_keep_rare_causes_after_many_injected_faults():
    tool = _soak_tool()
    case = object.__new__(tool.Case)
    case.counts = dict(failure_samples=[], failure_samples_by_cause={})
    case.references = SimpleNamespace(diagnostic_snapshot=lambda **kw: {"stamp": 1.})
    case.session = SimpleNamespace(loop=SimpleNamespace(gate=SimpleNamespace(_observed_state=None)))
    case._goal, case.dt = None, .02
    case.worker = SimpleNamespace(statistics={"completed": 1})
    case.record_event(0, "tracking_error", "rare event")
    for i in range(1, 101):
        case.record_event(i, "tracking_error", "injected", "tracking_error")
    assert len(case.counts["failure_samples"]) == 10
    assert case.counts["failure_samples"][0]["attempt"] == 91
    groups = case.counts["failure_samples_by_cause"]
    assert groups["tracking_error/uninjected"][0]["attempt"] == 0
    assert len(groups["tracking_error/injected"]) == 4
    assert case.take_event()["attempt"] == 100
    assert case.take_event() is None


def _diagnostic_runtime(**kwargs):
    worker = Worker()
    manager = _soak_tool().DiagnosticReferences(worker, lookahead=.2, **kwargs)
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 0., .1), state(), [.3], .02)
    return manager, worker


def test_pre_reference_fault_snapshot_uses_actual_observation_without_stale_reference():
    manager, worker = _diagnostic_runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.2), [.3], .02)
    manager.begin_attempt()
    measured = state(.22, .002, .1)
    manager.invalidate("retry")
    record = manager.diagnostic_snapshot(observed=measured, goal=[.3], control_dt=.02)
    assert not record["reference_attempted"] and record["reference"] is None and record["after"] is None
    assert record["stamp"] == .22 and record["q"] == [.002] and record["v"] == [.1]
    assert record["before"]["active"] is not None
    assert record["last_adoption"]["stamp"] == .2


def test_last_adoption_survives_fault_and_clears_on_reset():
    manager, worker = _diagnostic_runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.2), [.3], .02)
    entry = manager.diagnostic_adoption
    manager.update(state(.22, .04), [.3], .02)
    assert manager.diagnostic_adoption is entry
    record = manager.diagnostic_snapshot()
    adoption = record["last_adoption"]
    assert adoption["stamp"] == .2 and adoption["lateness_s"] < 1e-15
    assert adoption["measured_q"] == [0.] and adoption["forecast_v"] == [0.]
    assert adoption["reference_v"] == [.1] and adoption["solve_ms"] == 12.
    assert record["plans"][adoption["plan"]]["stamp"] == pytest.approx(.2)
    manager.reset()
    assert manager.diagnostic_adoption is None


def test_withheld_and_discarded_candidates_are_not_recorded_as_adoptions():
    manager, worker = _diagnostic_runtime(max_adoption_delay=0.)
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.1), [.3], .02)
    assert manager.diagnostic_snapshot()["last_adoption"] is None
    result = manager.update(state(.22), [.3], .02)
    assert result.reason == "late_reply"
    assert manager.diagnostic_snapshot()["last_adoption"] is None


def test_last_adoption_replaces_previous_entry_without_retaining_a_chain():
    import gc
    import weakref
    manager, worker = _diagnostic_runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.2), [.3], .02)
    old_state = weakref.ref(manager.diagnostic_adoption[0])
    worker.finish()
    manager.update(state(.4, .02, .1), [.3], .02)
    assert manager.diagnostic_snapshot()["last_adoption"]["stamp"] == .4
    gc.collect()
    assert old_state() is None


def test_event_hash_uses_bounded_reads(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    tool = _soak_tool()
    payload = b"event\n" * 600000
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)
    def forbid_read_bytes(self):
        raise AssertionError("whole-file reads are not allowed")
    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    assert tool.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_future_start_uses_repeated_driver_ticks_without_skipping_first_velocity():
    worker = Worker()
    manager = AsyncMPCReferenceManager(worker, lookahead=1.)
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 0., .1), state(), [.3], .1)
    manager.update(state(), [.3], .1)
    stamp = 0.
    for _ in range(10):
        stamp += .1
    forecast = worker.request[1].stamp
    assert stamp < 1. and 0 <= stamp - forecast <= 4 * ulp(stamp)
    worker.finish()
    current = 0.
    for k in range(10):
        current += .1
        update = manager.update(state(current), [.3], .1)
        assert update.replanned == (k == 9)
    assert update.velocity[0] == .1 and manager.plan.stamp == forecast


def test_forecast_tick_work_is_bounded_and_lookahead_rounds_to_full_ticks():
    from wholebodyx.async_runtime import _future_stamp
    nominal = .1 + .1 + .1
    assert 0 <= nominal - _future_stamp(0., .1, .21) <= 4 * ulp(nominal)
    repeated = 0.
    for _ in range(6):
        repeated += .1
    assert 0 <= repeated - _future_stamp(0., .1, .6) <= 4 * ulp(repeated)
    with pytest.raises(ValueError, match="4096"):
        _future_stamp(0., 1e-9, 1.)


def test_counter_based_driver_adopts_on_the_first_eligible_tick():
    worker = Worker()
    manager = AsyncMPCReferenceManager(worker, lookahead=1.)
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 1., .1), state(1.), [.3], .1)
    manager.update(state(1.), [.3], .1)
    worker.finish()
    for k in range(1, 11):
        update = manager.update(state(1. + k * .1), [.3], .1)
        assert update.replanned == (k == 10)
    assert update.velocity[0] == .1


def test_forecast_uses_velocity_from_previous_control_interval():
    worker = Worker()
    manager = AsyncMPCReferenceManager(worker, lookahead=.2)
    velocities = np.array([[.1], [.2]] + [[.3]] * 18)
    positions = np.vstack(([0.], np.cumsum(.1 * velocities, axis=0)))
    manager.seed(JointPlan(positions, velocities, 0., .1), state(), [.7], .1)
    assert manager.update(state(), [.7], .1).success
    predicted = worker.request[1]
    np.testing.assert_allclose(predicted.q, positions[2], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(predicted.v, velocities[1])


def test_goal_change_cancels_pending_and_early_completed_results_but_reuses_active_plan():
    manager, worker = runtime()
    manager.update(state(), [.3], .02)
    old_token = worker.request[0]
    worker.finish()
    manager.update(state(.1), [.3], .02)  # Received but too early to adopt.
    update = manager.update(state(.12), [-.2], .02)
    assert update.success and not update.replanned and update.reason == "goal_changed"
    assert worker.request[0].generation > old_token.generation
    assert worker.request[1].stamp == pytest.approx(.32)
    np.testing.assert_array_equal(worker.request[2], [-.2])
    np.testing.assert_array_equal(manager.active_goal, [.3])
    worker.finish(-.1)
    update = manager.update(state(.32), [-.2], .02)
    assert update.replanned and update.velocity[0] == -.1
    np.testing.assert_array_equal(manager.active_goal, [-.2])


@pytest.mark.parametrize("failure", ["token", "stamp", "goal", "state", "status", "missing", "nan_x", "short_x", "physical", "duration", "model"])
def test_malformed_or_failed_replies_clear_references_and_require_reset(failure):
    manager, worker = runtime()
    manager.update(state(), [.3], .02)
    reply = worker.finish()
    if failure == "token":
        reply = replace(reply, token=PlanningToken(99, 99))
    elif failure == "stamp":
        reply = replace(reply, state=state(.3))
    elif failure == "goal":
        reply = replace(reply, goal=np.array([-.4]))
    elif failure == "state":
        reply = replace(reply, state=state(.2, .001))
    elif failure in ("status", "missing", "nan_x", "short_x"):
        result = (QPResult("solved inaccurate", np.zeros(40)) if failure == "status" else
                  QPResult("solved", None) if failure == "missing" else
                  QPResult("solved", np.full(40, np.nan)) if failure == "nan_x" else
                  QPResult("solved", np.zeros(1)))
        reply = replace(reply, outcome=PlanResult(result, reply.outcome.plan))
    elif failure == "physical":
        v = np.full((20, 1), .3)  # Exceeds first slew of .2; boxes remain feasible.
        q = np.vstack(([0.], np.cumsum(.1 * v, axis=0)))
        reply = replace(reply, outcome=PlanResult(reply.outcome.solver, JointPlan(q, v, reply.state.stamp, .1)))
    elif failure == "duration":
        reply = replace(reply, solve_ms=float("nan"))
    else:
        reply = replace(reply, outcome=PlanResult(reply.outcome.solver,
                        JointPlan(np.zeros((11, 1)), np.zeros((10, 1)), reply.state.stamp, .2)))
    worker.reply = reply
    update = manager.update(state(.2), [.3], .02)
    assert not update.success and update.reason == "worker_result"
    assert manager.plan is None and not manager.pending and manager.active_goal is None
    with pytest.raises(RuntimeError, match="reset"):
        manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), .2, .1), state(.2), [.3], .02)
    manager.reset()
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), .2, .1), state(.2), [.3], .02)
    assert manager.update(state(.2), [.3], .02).success


def test_wall_timeout_and_completed_reply_order(monkeypatch):
    clock = [10.]
    monkeypatch.setattr("wholebodyx.async_runtime.monotonic", lambda: clock[0])
    manager, worker = runtime(max_pending_time=.5)
    manager.update(state(), [.3], .02)
    clock[0] = 10.6
    failed = manager.update(state(.02), [.3], .02)
    assert not failed.success and failed.reason == "worker_timeout" and manager.plan is None
    manager, worker = runtime(max_pending_time=.5)
    manager.update(state(), [.3], .02)
    worker.finish()
    clock[0] += .6
    # A result delivered during a paused simulation remains usable at its
    # original future timestamp; elapsed wall time alone never rebases it.
    assert manager.update(state(.1), [.3], .02).success
    assert manager.update(state(.2), [.3], .02).replanned


@pytest.mark.parametrize("failure", ["old_tracking", "new_tracking", "coverage", "worker", "rollback"])
def test_runtime_failures_clear_plan_and_pending_work(failure):
    manager, worker = runtime()
    manager.update(state(), [.3], .02)
    if failure == "old_tracking":
        current, expected = state(.02, .04), "tracking_error"
    elif failure == "new_tracking":
        worker.finish()
        current, expected = state(.3, -.04), "tracking_error"
    elif failure == "coverage":
        current, expected = state(2.), "coverage"
    elif failure == "worker":
        worker.error = "worker crashed"
        current, expected = state(.02), "worker_error"
    else:
        manager.update(state(.1), [.3], .02)
        current, expected = state(.05), "clock_rollback"
    update = manager.update(current, [.3], .02)
    assert not update.success and update.reason == expected
    assert manager.plan is None and not manager.pending
    if failure == "rollback":
        assert manager.update(state(.2), [.3], .02).reason == "clock_rollback"


def test_busy_mailbox_retries_without_waiting_and_cannot_extend_seed_validity():
    manager, worker = runtime(replan_period=10.)
    worker.busy = True
    assert manager.update(state(), [.3], .02).success and not manager.pending
    worker.busy = False
    assert manager.update(state(.02), [.3], .02).success and manager.pending
    for k in range(1, 90):
        assert manager.update(state(k * .02), [k * .001], .02).success
    failed = manager.update(state(1.8), [-.2], .02)
    assert not failed.success and failed.reason == "coverage"


def test_unseeded_and_invalid_seed_are_rejected():
    worker = Worker()
    manager = AsyncMPCReferenceManager(worker)
    assert not manager.update(state(), [.3], .02).success
    manager.reset()
    with pytest.raises(ValueError, match="cover"):
        manager.seed(JointPlan([[0.], [0.]], [[0.]], 0., .1), state(), [.3], .02)
    with pytest.raises(ValueError, match="lookahead"):
        AsyncMPCReferenceManager(worker, lookahead=2.)


class Driver(JointIntegrator):
    def __init__(self, bounds):
        super().__init__(bounds, state(), control_dt=.02)
        self.reads = self.writes = self.stops = 0

    def read(self):
        self.reads += 1
        return super().read()

    def write(self, command):
        self.writes += 1
        super().write(command)

    def stop(self):
        self.stops += 1
        super().stop()


def test_session_injection_retains_single_read_fault_latch_and_explicit_recovery():
    manager, worker = runtime()
    model = SimpleNamespace(joint_names=("j0",), limits=worker.limits)
    driver = Driver(worker.limits)
    session = VelocitySession(model, driver, KinematicWBC(worker.limits), reference_manager=manager)
    session.reset()
    with pytest.raises(RuntimeError, match="seed"):
        session.step([.3])
    assert driver.stops == 1 and driver.reads == 1 and driver.writes == 0
    with pytest.raises(RuntimeError, match="faulted"):
        session.step([.3])
    assert driver.reads == 1
    session.reset()
    manager.seed(JointPlan(np.zeros((21, 1)), np.zeros((20, 1)), 0., .1), state(), [.3], .02)
    for _ in range(11):
        if worker.request is not None:
            worker.finish(0.)
        assert session.step([.3]).reference.success
    assert driver.reads == 12 and driver.writes == 11
    worker.error = "process lost"
    with pytest.raises(RuntimeError, match="process lost"):
        session.step([.3])
    assert driver.stops == 2 and session.plan is None
    before = driver.reads, driver.writes, driver.stops
    with pytest.raises(RuntimeError, match="faulted"):
        session.step([.3])
    assert before == (driver.reads, driver.writes, driver.stops)


def test_session_rejects_ambiguous_or_mismatched_reference_manager():
    manager, worker = runtime()
    model = SimpleNamespace(joint_names=("j0",), limits=worker.limits)
    args = model, Driver(worker.limits), KinematicWBC(worker.limits)
    with pytest.raises(ValueError, match="either"):
        VelocitySession(*args, planner=object(), reference_manager=manager)
    with pytest.raises(TypeError, match="MPCReferenceManager"):
        VelocitySession(*args, reference_manager=object())
    manager.limits = JointLimits([-2.], [2.], [.8], [2.])
    with pytest.raises(ValueError, match="limits differ"):
        VelocitySession(*args, reference_manager=manager)
