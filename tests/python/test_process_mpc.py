"""Real spawned planners, bounded mailboxes and process lifetime."""

import os
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pytest

from wholebodyx import JointLimits, ProcessMPCWorker, RobotState


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
