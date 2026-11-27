"""Real spawned planners, bounded mailboxes and process lifetime."""

import os
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from threading import Condition, Event
import time

import numpy as np
import pytest

from wholebodyx import AsyncMPCReferenceManager, JointLimits, JointPlan, ProcessMPCWorker, RobotState


def limits(n=3):
    return JointLimits([-1.] * n, [1.] * n, [.8] * n, [2.] * n)


def receive(worker, timeout=10.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.error:
            raise RuntimeError(worker.error)
        reply = worker.poll()
        if reply is not None:
            return reply
        time.sleep(.005)
    raise TimeoutError("test did not receive a planning reply")


@pytest.mark.parametrize("backend", ["python", "sparse", "native", "native-sparse"])
def test_spawned_backend_returns_validated_current_request_and_closes(backend):
    if backend.startswith("native"):
        pytest.importorskip("wholebodyx_native")
    worker = ProcessMPCWorker(limits(), backend=backend, horizon=10)
    pid = worker.pid
    with worker:
        assert pid != os.getpid() and Path(f"/proc/{pid}").exists()
        info = worker.runtime_info
        assert info["backend"] == backend and "process_mpc.py" in info["source_sha256"]
        info["source_sha256"].clear()
        assert worker.runtime_info["source_sha256"]
        goal = np.full(3, .3)
        token = worker.submit(RobotState(np.zeros(3), np.zeros(3), 4.), goal)
        goal[:] = -.8
        reply = receive(worker)
        assert reply.token == token and reply.state.stamp == 4.
        np.testing.assert_array_equal(reply.goal, [.3] * 3)
        assert reply.outcome.solver.success and reply.outcome.plan.stamp == 4.
        assert reply.solve_ms >= 0 and worker.poll() is None
        for value in (reply.state.q, reply.state.v, reply.goal, reply.outcome.solver.x):
            with pytest.raises(ValueError):
                value.setflags(write=True)
        assert worker.statistics["completed"] == 1
    assert not Path(f"/proc/{pid}").exists() and not worker._relay.is_alive()
    worker.close()
    worker.cancel()
    assert worker.error
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(RobotState([0.] * 3, [0.] * 3), [0.] * 3)


def test_latest_request_and_generation_fence_discard_old_work():
    with ProcessMPCWorker(limits(), horizon=10) as worker:
        state = RobotState([0.] * 3, [0.] * 3)
        for k in range(50):
            token = worker.submit(state, [.001 * k] * 3)
            if token is not None:
                latest = token
        reply = receive(worker)
        assert reply.token == latest
        worker.cancel()
        assert worker.poll() is None
        next_token = worker.submit(state, [-.2] * 3)
        reply = receive(worker)
        assert reply.token == next_token and next_token.generation > latest.generation
        np.testing.assert_array_equal(reply.goal, [-.2] * 3)
        assert worker.statistics["submitted"] > 1


def test_submit_poll_cancel_never_wait_on_held_relay_lock():
    with ProcessMPCWorker(limits()) as worker:
        with worker._lock:
            started = time.monotonic()
            assert worker.submit(RobotState([0.] * 3, [0.] * 3), [.2] * 3) is None
            assert worker.poll() is None
            worker.cancel()
            assert time.monotonic() - started < .1
        token = worker.submit(RobotState([0.] * 3, [0.] * 3), [.2] * 3)
        assert receive(worker).token == token


def test_submission_notifies_an_idle_relay(monkeypatch):
    from wholebodyx import process_mpc

    waiting = Event()

    class SlowHealthCheck(Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout=10.)

    monkeypatch.setattr(process_mpc, "Condition", SlowHealthCheck)
    with ProcessMPCWorker(limits(), horizon=5) as worker:
        assert waiting.wait(5.)
        state = RobotState([0.] * 3, [0.] * 3)
        deadline = time.monotonic() + 2.
        token = None
        while token is None and time.monotonic() < deadline:
            token = worker.submit(state, [.01] * 3)
            if token is None:
                time.sleep(.001)
        assert token is not None
        assert receive(worker, timeout=2.).token == token


def test_worker_death_is_reported_and_can_be_reaped():
    with ProcessMPCWorker(limits()) as worker:
        worker._process.kill()
        deadline = time.monotonic() + 5
        while worker.error is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.error
        with pytest.raises(RuntimeError):
            worker.reset()


def test_worker_configuration_rejects_invalid_values_before_spawning():
    for kwargs in ({"backend": "other"}, {"horizon": True}, {"dt": float("nan")}, {"horizon": 0}):
        with pytest.raises(ValueError):
            ProcessMPCWorker(limits(), **kwargs)


def test_relay_start_failure_reaps_the_already_spawned_child(monkeypatch):
    from wholebodyx import process_mpc
    before = {p.pid for p in mp.active_children()}

    class BrokenThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("relay start failed")

    monkeypatch.setattr(process_mpc, "Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="relay start failed"):
        ProcessMPCWorker(limits())
    assert {p.pid for p in mp.active_children()} == before


def test_close_before_readiness_is_safe_and_future_wait_fails_immediately():
    worker = ProcessMPCWorker(limits())
    worker.close()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="closed"):
        worker.wait_ready()
    assert time.monotonic() - started < .1
    assert not worker._relay.is_alive() and not Path(f"/proc/{worker.pid}").exists()


def test_invalid_submission_cannot_replace_the_latest_valid_request():
    with ProcessMPCWorker(limits()) as worker:
        token = worker.submit(RobotState([0.] * 3, [0.] * 3), [.1] * 3)
        with pytest.raises(ValueError):
            worker.submit(RobotState([0.] * 3, [0.] * 3), [float("nan")] * 3)
        with pytest.raises(ValueError):
            worker.submit(RobotState([0.] * 2, [0.] * 2), [.1] * 2)
        assert receive(worker).token == token


@pytest.mark.parametrize("case", ["transfer", "limits", "interval", "replaced_poll"])
def test_reference_receipt_reuses_only_applicable_relay_validation(case, monkeypatch):
    from wholebodyx import async_runtime

    with ProcessMPCWorker(limits(), horizon=8, dt=.1) as worker:
        manager = AsyncMPCReferenceManager(worker, lookahead=.2)
        state = RobotState(np.zeros(3), np.zeros(3))
        goal = np.full(3, .3)
        seed = JointPlan(np.zeros((9, 3)), np.zeros((8, 3)), 0., .1)
        manager.seed(seed, state, goal, .02)
        assert manager.update(state, goal, .02).success
        deadline = time.monotonic() + 3.
        while worker._reply is None and time.monotonic() < deadline and not worker.error:
            time.sleep(.001)
        assert worker._reply is not None, worker.error
        control_dt = .02
        if case == "transfer":
            def unexpected_validation(*args, **kwargs):
                raise AssertionError("the reply has already passed relay validation")

            monkeypatch.setattr(async_runtime, "_validated_joint_plan", unexpected_validation)
        elif case == "limits":
            manager.limits.velocity.setflags(write=True)
            manager.limits.velocity[:] = 1e-4
        elif case == "interval":
            control_dt = .9
        else:
            poll = worker.poll

            def replaced_poll():
                reply = poll()
                outcome = replace(reply.outcome, solver=replace(reply.outcome.solver, x=np.zeros(1)))
                return replace(reply, outcome=outcome)

            monkeypatch.setattr(worker, "poll", replaced_poll)
        result = manager.update(RobotState(np.zeros(3), np.zeros(3), .2), goal, control_dt)
        if case == "transfer":
            assert result.success and result.replanned
        else:
            assert not result.success and result.reason == "worker_result"
            assert manager.plan is None


def test_close_during_real_pending_solve_reaps_child_and_relay():
    worker = ProcessMPCWorker(limits(30), horizon=60, backend="native-sparse")
    try:
        try:
            worker.wait_ready()
        except RuntimeError as exc:
            if "wholebodyx_native" in str(exc):
                pytest.skip("native extension unavailable")
            raise
        worker.submit(RobotState([0.] * 30, [0.] * 30), [.7] * 30)
        time.sleep(.005)
    finally:
        worker.close()
    assert not worker._relay.is_alive() and not Path(f"/proc/{worker.pid}").exists()
