"""Bounded latest-request MPC in a spawned process.

Only the relay thread touches IPC. submit/poll/cancel never wait for a solver,
pipe transfer or a held relay lock. Construction, wait_ready and close belong
outside the control cycle. Python scheduling still prevents hard-RT guarantees.
"""

from dataclasses import dataclass
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np

from .mpc import JointMPC, JointPlan, PlanResult, _MPCSettings, _time_window, _validated_joint_plan
from .qp import QPResult, _immutable
from .types import RobotState, _snapshot, positive


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
        planner = factory(settings.limits, horizon=settings.horizon, dt=settings.dt,
                          tracking_weight=settings.tracking_weight,
                          control_weight=settings.control_weight,
                          terminal_weight=settings.terminal_weight)
        package = Path(__file__).resolve().parent
        import osqp
        import scipy
        info = dict(backend=backend, package=str(package), python=sys.version.split()[0],
                    numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__, source_sha256={
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob("*.py")})
        if backend.startswith("native"):
            import wholebodyx_native
            binary = Path(wholebodyx_native.__file__).resolve()
            info.update(native_file=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
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
            connection.send(("result", token, outcome, (perf_counter() - started) * 1000))
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

    def __init__(self, limits, *, backend="sparse", horizon=30, dt=.02,
                 tracking_weight=10., control_weight=.1, terminal_weight=30.):
        if backend not in ("python", "native", "sparse", "native-sparse"):
            raise ValueError("process planner requires a built-in joint MPC backend")
        self._settings = _MPCSettings(limits, horizon, dt, tracking_weight,
                                      control_weight, terminal_weight)
        self._backend = backend
        self._generation = self._sequence = 0
        self._latest = self._outbox = self._reply = None
        self._closed = False
        self._shutdown_complete = False
        self._error = None
        self._info = None
        self._lock = Lock()
        self._stop, self._ready = Event(), Event()
        self._counts = dict(submitted=0, coalesced=0, completed=0, discarded=0,
                            max_solve_ms=0., mean_solve_ms=0.)
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_child_main,
                                        args=(child, backend, self._settings), daemon=True)
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
            self._process.join(1.)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.)
            self._connection.close()
            if not self._process.is_alive():
                self._process.close()
            raise

    @property
    def limits(self):
        from .types import JointLimits
        s = self._settings.limits
        return JointLimits(s.lower, s.upper, s.velocity, s.acceleration)

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
        return {**self._counts, "generation": self._generation, "pid": self._pid,
                "ready": self.ready, "closed": self._closed}

    def wait_ready(self, timeout=10.):
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
            if reply is not None and reply.token == self._latest and reply.token.generation == self._generation:
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
            finally:
                self._lock.release()

    def reset(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        self.cancel()

    def _snapshot(self, outcome, state):
        if not isinstance(outcome, PlanResult) or not isinstance(outcome.solver, QPResult):
            raise ValueError("worker returned an invalid planning result")
        result = outcome.solver
        if not result.success:
            return PlanResult(QPResult(result.status, iterations=result.iterations,
                                       violation=result.violation, detail=result.detail))
        size = self.horizon * state.q.size * (2 if "sparse" in self.backend else 1)
        x = _snapshot(result.x, 1, "worker solution")
        if x.shape != (size,) or outcome.plan is None:
            raise ValueError("worker returned an invalid solution shape or missing plan")
        plan = _validated_joint_plan(outcome.plan, state, self._settings.limits, self.dt)
        if plan.velocities.shape != (self.horizon, state.q.size) or plan.dt != self.dt:
            raise ValueError("worker returned a different planning model")
        return PlanResult(QPResult("solved", x, result.iterations,
                                  result.violation, result.detail), plan)

    def _relay_main(self):
        sent = None
        try:
            while not self._stop.is_set():
                if self._connection.poll(.001):
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
                        reply = PlanningReply(token, state, goal, self._snapshot(outcome, state), solve_ms)
                        with self._lock:
                            self._counts["completed"] += 1
                            self._counts["max_solve_ms"] = max(self._counts["max_solve_ms"], solve_ms)
                            self._counts["mean_solve_ms"] += (solve_ms - self._counts["mean_solve_ms"]) / self._counts["completed"]
                            if token == self._latest and token.generation == self._generation:
                                self._reply = reply
                            else:
                                self._counts["discarded"] += 1
                        sent = None
                    else:
                        raise ValueError("unexpected planner worker message")
                if self._info is not None and sent is None:
                    with self._lock:
                        request, self._outbox = self._outbox, None
                    if request is not None:
                        token, state, goal = request
                        if token == self._latest and token.generation == self._generation:
                            self._connection.send(("solve", token, state.q, state.v, state.stamp, goal))
                            sent = request
                if not self._process.is_alive():
                    raise RuntimeError("planner worker exited unexpectedly")
        except BaseException as exc:
            if not self._closed:
                self._error = f"{type(exc).__name__}: {exc}"[:1000]
                self._ready.set()

    def close(self, timeout=1.):
        """Teardown outside the cycle; at most three timeout waits, idempotent."""
        timeout = positive(timeout, "close timeout")
        if self._shutdown_complete:
            return
        self._closed = True
        self.cancel()
        self._stop.set()
        # Termination also unblocks a relay receiving a partial IPC frame.
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        self._relay.join(timeout)
        if self._process.is_alive() or self._relay.is_alive():
            raise RuntimeError("planner worker or relay did not stop within close timeouts; retry close")
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
