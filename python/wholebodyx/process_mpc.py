"""Bounded latest-request MPC in a spawned process.

Only the relay thread touches IPC. submit/poll/cancel never wait for a solver,
pipe transfer or a held relay lock. Construction, wait_ready and close belong
outside the control cycle. Python scheduling still prevents hard-RT guarantees.
"""

from dataclasses import dataclass, replace
import hashlib
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
from threading import Condition, Event, Lock, Thread
from time import perf_counter
from typing import NamedTuple

import numpy as np

from .mpc import (
    JointMPC,
    JointPlan,
    PlanResult,
    _check_plan_acceptance,
    _MPCSettings,
    _time_window,
    _validated_joint_plan,
)
from .qp import QPResult, _immutable, _immutable_array_key
from .types import JointLimits, RobotState, _snapshot, positive


@dataclass(frozen=True)
class PlanningToken:
    generation: int
    sequence: int


@dataclass(frozen=True)
class PlanningReply:
    token: PlanningToken
    state: RobotState
    goal: np.ndarray
    outcome: PlanResult
    solve_ms: float


class _CompactPlan(NamedTuple):
    solver: QPResult
    positions: np.ndarray
    stamp: float
    dt: float
    velocity_shape: tuple[int, int]


_LARGE_DECISION_BYTES = 256 * 1024


def _pack_outcome(outcome):
    """Encode the velocity only once when both arrays contain the same bits."""
    if not isinstance(outcome, PlanResult) or not isinstance(outcome.solver, QPResult):
        return outcome
    solver, plan = outcome.solver, outcome.plan
    if not solver.success or not isinstance(plan, JointPlan):
        return outcome
    decision = solver.x
    velocity = plan.velocities
    if (
        type(decision) is np.ndarray
        and type(velocity) is np.ndarray
        and decision.ndim == 1
        and velocity.ndim == 2
        and velocity.size
        and decision.dtype == velocity.dtype == np.dtype(float)
        and decision.size in (velocity.size, 2 * velocity.size)
        and np.array_equal(
            decision[-velocity.size:].reshape(velocity.shape).view(np.uint64),
            velocity.view(np.uint64),
        )
    ):
        if decision.nbytes >= _LARGE_DECISION_BYTES and decision.flags.c_contiguous:
            # With protocol 5 a contiguous read-only view can arrive backed
            # by bytes. The solver keeps its original array and flags.
            decision = decision.view()
            decision.setflags(write=False)
            solver = replace(solver, x=decision)
        return _CompactPlan(solver, plan.positions, plan.stamp, plan.dt, velocity.shape)
    return outcome


def _snapshot_solution(value):
    """Validate the decision and reuse large bytes-backed buffers."""
    if (
        type(value) is np.ndarray
        and not value.flags.writeable
        and value.flags.c_contiguous
        and value.flags.aligned
        and value.nbytes >= _LARGE_DECISION_BYTES
        and value.dtype == np.dtype(float)
        and value.ndim == 1
        and _immutable_array_key(value) is not None
    ):
        if not np.isfinite(value).all():
            raise ValueError("worker solution must be a finite 1-D array")
        return value.view()
    return _snapshot(value, 1, "worker solution")


def _child_main(connection, backend, settings):
    try:
        if backend == "python":
            factory = JointMPC
        elif backend == "sparse":
            from .sparse_mpc import SparseJointMPC

            factory = SparseJointMPC
        else:
            from .native import NativeMPC, NativeSparseMPC

            factory = NativeMPC if backend == "native" else NativeSparseMPC
        planner = factory(
            settings.limits,
            horizon=settings.horizon,
            dt=settings.dt,
            tracking_weight=settings.tracking_weight,
            control_weight=settings.control_weight,
            terminal_weight=settings.terminal_weight,
        )
        package = Path(__file__).resolve().parent
        import osqp
        import scipy

        info = dict(
            backend=backend,
            package=str(package),
            python=sys.version.split()[0],
            numpy=np.__version__,
            scipy=scipy.__version__,
            osqp=osqp.__version__,
            source_sha256={
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.rglob("*.py")
            },
        )
        if backend.startswith("native"):
            import wholebodyx_native

            binary = Path(wholebodyx_native.__file__).resolve()
            info.update(
                native_file=str(binary),
                native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            )
        connection.send(("ready", info))
        generation = None
        while True:
            message = connection.recv()
            if message == ("close",):
                break
            tag, token, q, v, stamp, goal = message
            if tag != "solve" or not isinstance(token, PlanningToken):
                raise ValueError("invalid planning request")
            if token.generation != generation:
                planner.reset()
                generation = token.generation
            started = perf_counter()
            try:
                outcome = planner.solve(RobotState(q, v, stamp), goal)
            except (ValueError, RuntimeError) as exc:
                outcome = PlanResult(QPResult("error", detail=str(exc)[:1000]))
            solve_ms = (perf_counter() - started) * 1000
            message = ("result", token, _pack_outcome(outcome), solve_ms)
            # Protocol 5 can serialize contiguous NumPy buffers with fewer
            # copies while keeping the reply in a single pipe message.
            connection.send_bytes(pickle.dumps(message, protocol=5))
    except (EOFError, BrokenPipeError, ConnectionError):
        pass
    except BaseException as exc:
        try:
            connection.send(("fatal", f"{type(exc).__name__}: {exc}"[:1000]))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ProcessMPCWorker:
    """One control owner, one relay, one spawned built-in joint MPC planner.

    Storage is limited to one in-flight request, one replaceable pending request
    and one completed reply. A new submission supersedes older replies. cancel
    immediately fences all older generations without waiting for child work.
    """

    def __init__(
        self,
        limits,
        *,
        backend="sparse",
        horizon=30,
        dt=0.02,
        tracking_weight=10.0,
        control_weight=0.1,
        terminal_weight=30.0,
    ):
        if backend not in ("python", "native", "sparse", "native-sparse"):
            raise ValueError("process planner requires a built-in joint MPC backend")
        self._settings = _MPCSettings(
            limits, horizon, dt, tracking_weight, control_weight, terminal_weight
        )
        self._backend = backend
        self._generation = self._sequence = 0
        self._latest = self._outbox = self._reply = None
        self._closed = False
        self._shutdown_complete = False
        self._error = None
        self._info = None
        self._lock = Lock()
        self._request_available = Condition(self._lock)
        self._stop, self._ready = Event(), Event()
        self._counts = dict(
            submitted=0, coalesced=0, completed=0, discarded=0, max_solve_ms=0.0, mean_solve_ms=0.0
        )
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_child_main, args=(child, backend, self._settings), daemon=True
        )
        try:
            self._process.start()
        except BaseException:
            self._connection.close()
            child.close()
            raise
        child.close()
        self._pid = self._process.pid
        self._relay = Thread(target=self._relay_main, name="wholebodyx-mpc-relay", daemon=True)
        try:
            self._relay.start()
        except BaseException:
            self._closed = True
            self._process.terminate()
            self._process.join(1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.0)
            self._connection.close()
            if not self._process.is_alive():
                self._process.close()
            raise

    @property
    def limits(self):
        limits = self._settings.limits
        return JointLimits(limits.lower, limits.upper, limits.velocity, limits.acceleration)

    def _matches_plan_limits(self, limits):
        """Check the limits used to validate replies before publishing them."""
        if type(limits) is not JointLimits:
            return False
        checked = self._settings.limits
        for name in ("lower", "upper", "velocity", "acceleration"):
            actual, expected = getattr(limits, name), getattr(checked, name)
            if actual is expected:
                continue
            if (
                type(actual) is not np.ndarray or expected is None
                or actual.dtype != expected.dtype or not np.array_equal(actual, expected)
            ):
                return False
        return True

    horizon = property(lambda self: self._settings.horizon)
    backend = property(lambda self: self._backend)
    dt = property(lambda self: self._settings.dt)
    pid = property(lambda self: self._pid)
    error = property(lambda self: "planner worker is closed" if self._closed else self._error)
    ready = property(lambda self: self._ready.is_set() and self.error is None)

    @property
    def runtime_info(self):
        if self._info is None:
            return None
        return {**self._info, "source_sha256": dict(self._info["source_sha256"])}

    @property
    def statistics(self):
        # Advisory counters; reading them never waits for the relay lock.
        return {
            **self._counts,
            "generation": self._generation,
            "pid": self._pid,
            "ready": self.ready,
            "closed": self._closed,
        }

    def wait_ready(self, timeout=10.0):
        """Startup barrier for use before starting the control cycle."""
        if self.error is not None:
            raise RuntimeError(self.error)
        if not self._ready.wait(positive(timeout, "startup timeout")):
            raise TimeoutError("planner worker startup timed out")
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.runtime_info

    def submit(self, state, goal):
        """Snapshot inputs; return a token, or None if the relay slot is busy."""
        if self.error is not None:
            raise RuntimeError(self.error)
        self._settings.limits.check_state(state)
        state = RobotState(state.q, state.v, state.stamp)
        object.__setattr__(state, "q", _immutable(state.q))
        object.__setattr__(state, "v", _immutable(state.v))
        goal = _snapshot(goal, 1, "goal")
        if goal.shape != state.q.shape:
            raise ValueError("goal and state dimensions differ")
        _time_window(state.stamp, self.dt, self.horizon)
        if not self._lock.acquire(blocking=False):
            return None
        try:
            self._sequence += 1
            token = PlanningToken(self._generation, self._sequence)
            if self._outbox is not None:
                self._counts["coalesced"] += 1
            self._outbox = (token, state, goal)
            self._latest = token
            self._reply = None
            self._counts["submitted"] += 1
            self._request_available.notify()
            return token
        finally:
            self._lock.release()

    def poll(self):
        """Take the latest completed reply without waiting on IPC or a lock."""
        if self.error is not None:
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            reply, self._reply = self._reply, None
            if (
                reply is not None
                and reply.token == self._latest
                and reply.token.generation == self._generation
            ):
                return reply
            return None
        finally:
            self._lock.release()

    def cancel(self):
        """Nonblocking generation fence, including on a closed worker."""
        self._generation += 1
        self._latest = None
        if self._lock.acquire(blocking=False):
            try:
                self._outbox = self._reply = None
                self._request_available.notify()
            finally:
                self._lock.release()

    def reset(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        self.cancel()

    def _snapshot(self, outcome, state):
        if not isinstance(outcome, (PlanResult, _CompactPlan)) or not isinstance(outcome.solver, QPResult):
            raise ValueError("worker returned an invalid planning result")
        result = outcome.solver
        if not result.success:
            return PlanResult(
                QPResult(
                    result.status,
                    iterations=result.iterations,
                    violation=result.violation,
                    detail=result.detail,
                )
            )
        shape = (self.horizon, state.q.size)
        velocity_count = shape[0] * shape[1]
        size = velocity_count * (2 if "sparse" in self.backend else 1)
        x = _snapshot_solution(result.x)
        if x.shape != (size,):
            raise ValueError("worker returned an invalid solution shape or missing plan")
        if isinstance(outcome, _CompactPlan):
            if outcome.velocity_shape != shape:
                raise ValueError("worker returned a different planning model")
            velocities = x[-velocity_count:].reshape(shape)
            plan = JointPlan(outcome.positions, velocities, outcome.stamp, outcome.dt)
            _check_plan_acceptance(plan, state, self._settings.limits, self.dt)
        else:
            if outcome.plan is None:
                raise ValueError("worker returned an invalid solution shape or missing plan")
            plan = _validated_joint_plan(outcome.plan, state, self._settings.limits, self.dt)
        if plan.velocities.shape != shape or plan.dt != self.dt:
            raise ValueError("worker returned a different planning model")
        return PlanResult(
            QPResult("solved", x, result.iterations, result.violation, result.detail), plan
        )

    def _relay_main(self):
        sent = None
        try:
            while not self._stop.is_set():
                waiting_for_child = sent is not None or self._info is None
                if self._connection.poll(0.001 if waiting_for_child else 0):
                    message = self._connection.recv()
                    if not isinstance(message, tuple) or not message:
                        raise ValueError("invalid planner worker message")
                    if message[0] == "ready" and len(message) == 2 and self._info is None:
                        self._info = message[1]
                        self._ready.set()
                    elif message[0] == "fatal" and len(message) == 2:
                        raise RuntimeError(str(message[1]))
                    elif message[0] == "result" and len(message) == 4 and sent is not None:
                        _, token, outcome, solve_ms = message
                        expected, state, goal = sent
                        if token != expected:
                            raise ValueError("planner worker reply token differs")
                        solve_ms = float(solve_ms)
                        if not np.isfinite(solve_ms) or solve_ms < 0:
                            raise ValueError("invalid planner worker duration")
                        reply = PlanningReply(
                            token, state, goal, self._snapshot(outcome, state), solve_ms
                        )
                        with self._lock:
                            self._counts["completed"] += 1
                            self._counts["max_solve_ms"] = max(
                                self._counts["max_solve_ms"], solve_ms
                            )
                            self._counts["mean_solve_ms"] += (
                                solve_ms - self._counts["mean_solve_ms"]
                            ) / self._counts["completed"]
                            if token == self._latest and token.generation == self._generation:
                                self._reply = reply
                            else:
                                self._counts["discarded"] += 1
                        # The mailbox now owns the accepted reply. Release
                        # the wire data and local references before idling.
                        del message, outcome, reply
                        sent = None
                    else:
                        raise ValueError("unexpected planner worker message")
                if self._info is not None and sent is None:
                    with self._request_available:
                        if self._outbox is None and not self._stop.is_set():
                            # An idle relay wakes on submit. Poll the child's
                            # health periodically even without new requests.
                            self._request_available.wait(timeout=0.02)
                        request, self._outbox = self._outbox, None
                    if request is not None and not self._stop.is_set():
                        token, state, goal = request
                        if token == self._latest and token.generation == self._generation:
                            self._connection.send(
                                ("solve", token, state.q, state.v, state.stamp, goal)
                            )
                            sent = request
                if not self._process.is_alive():
                    raise RuntimeError("planner worker exited unexpectedly")
        except BaseException as exc:
            if not self._closed:
                self._error = f"{type(exc).__name__}: {exc}"[:1000]
                self._ready.set()

    def close(self, timeout=1.0):
        """Teardown outside the cycle; at most three timeout waits, idempotent."""
        timeout = positive(timeout, "close timeout")
        if self._shutdown_complete:
            return
        self._closed = True
        self._stop.set()
        self.cancel()
        # Termination also unblocks a relay receiving a partial IPC frame.
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        self._relay.join(timeout)
        if self._process.is_alive() or self._relay.is_alive():
            raise RuntimeError(
                "planner worker or relay did not stop within close timeouts; retry close"
            )
        self._connection.close()
        self._process.close()
        self._shutdown_complete = True

    def __enter__(self):
        try:
            self.wait_ready()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()
