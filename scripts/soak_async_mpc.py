"""Paced process MPC sessions with bounded statistics and fault recovery.

Each case has its own simulation clock. Pacing and Python soft deadlines are
measurements, not a guarantee of hardware scheduling. Startup, seed creation,
worker restart and shutdown are explicitly outside measured control cycles.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from wholebodyx import (AsyncMPCReferenceManager, ExecutionDeadlineError, JointLimits, JointPlan, KinematicWBC,
                        ProcessMPCWorker, RobotState, VelocitySession)
from wholebodyx.simulation import JointIntegrator

from _measurement import rss_kib as rss, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticReferences(AsyncMPCReferenceManager):
    """Retain the current transition and last adoption, without a history."""

    def _clear(self):
        super()._clear()
        self.diagnostic_adoption = None
        self.begin_attempt()

    def begin_attempt(self):
        self.diagnostic_input = self.diagnostic_result = None
        self.diagnostic_before = (self.plan, self._pending, self._future)
        self.diagnostic_failure = self.diagnostic_received = None
        self.diagnostic_after = None

    def update(self, state, goal, control_dt):
        self.diagnostic_input = (state, goal, control_dt)
        self.diagnostic_before = (self.plan, self._pending, self._future)
        self.diagnostic_failure = self.diagnostic_received = None
        result = super().update(state, goal, control_dt)
        self.diagnostic_after = (self.plan, self._pending, self._future)
        self.diagnostic_result = result
        if result.replanned:
            candidate = self.diagnostic_received or self.diagnostic_before[2]
            # Keep references only. Array conversion belongs outside the cycle.
            self.diagnostic_adoption = (state, result, candidate)
        return result

    def _receive(self, control_dt):
        previous = self._future
        super()._receive(control_dt)
        if self._future is not previous:
            self.diagnostic_received = self._future

    def _failure(self, *args, **kwargs):
        self.diagnostic_failure = (self.plan, self._pending, self._future)
        return super()._failure(*args, **kwargs)

    def diagnostic_snapshot(self, observed=None, goal=None, control_dt=None):
        attempted = self.diagnostic_input is not None
        if attempted:
            state, goal, dt = self.diagnostic_input
        else:
            if observed is None:
                raise ValueError("an observed state is required before reference evaluation")
            state, dt = observed, control_dt
        plans = {}

        def plan(value):
            if value is None:
                return None
            for name, entry in plans.items():
                if entry[0] is value:
                    return name
            name = str(len(plans))
            plans[name] = (value, dict(stamp=value.stamp, dt=value.dt,
                                      positions=value.positions.tolist(), velocities=value.velocities.tolist()))
            return name

        def pending(value):
            if value is None:
                return None
            token, predicted, target, started, reason, nominal_start = value
            return dict(token=[token.generation, token.sequence], q=predicted.q.tolist(),
                        v=predicted.v.tolist(), stamp=predicted.stamp, goal=target.tolist(),
                        wall_age_s=time.monotonic() - started, reason=reason, nominal_start=nominal_start)

        def future(value):
            if value is None:
                return None
            trajectory, reply, reason, nominal_start = value
            return dict(plan=plan(trajectory), solve_ms=reply.solve_ms, reason=reason,
                        token=[reply.token.generation, reply.token.sequence], nominal_start=nominal_start,
                        initial_q=reply.state.q.tolist(), initial_v=reply.state.v.tolist(),
                        initial_stamp=reply.state.stamp, goal=reply.goal.tolist(),
                        solver=dict(status=reply.outcome.solver.status,
                                    iterations=int(reply.outcome.solver.iterations),
                                    violation=(float(reply.outcome.solver.violation)
                                               if np.isfinite(reply.outcome.solver.violation) else None),
                                    violation_finite=bool(np.isfinite(reply.outcome.solver.violation))))

        def transition(value):
            if value is None:
                return None
            active, request, candidate = value
            return dict(active=plan(active), pending=pending(request), future=future(candidate))

        result = self.diagnostic_result
        adoption = None
        if self.diagnostic_adoption is not None:
            measured, reference, candidate = self.diagnostic_adoption
            trajectory, reply, _, nominal_start = candidate
            adoption = dict(stamp=measured.stamp, plan=plan(trajectory), plan_stamp=trajectory.stamp,
                            nominal_start=nominal_start,
                            lateness_s=max(0., measured.stamp - nominal_start),
                            solve_ms=reference.solve_ms, measured_q=measured.q.tolist(),
                            measured_v=measured.v.tolist(), forecast_v=reply.state.v.tolist(),
                            reference_q=reference.position.tolist(), reference_v=reference.velocity.tolist(),
                            tracking_error=reference.tracking_error, goal=reply.goal.tolist())
        snapshot = dict(q=state.q.tolist(), v=state.v.tolist(), stamp=state.stamp,
                        goal=np.asarray(goal).tolist(), control_dt=dt,
                        before=transition(self.diagnostic_before), failure=transition(self.diagnostic_failure),
                        after=transition(self.diagnostic_after) if attempted else None,
                        reference_attempted=attempted, received=future(self.diagnostic_received),
                        last_adoption=adoption,
                        reference=(dict(reason=result.reason, success=result.success,
                                        tracking_error=result.tracking_error, solve_ms=result.solve_ms)
                                   if result is not None else None))
        snapshot["plans"] = {name: entry[1] for name, entry in plans.items()}
        return snapshot


def file_sha256(path):
    """Hash large event streams with fixed-size reads, including Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Driver(JointIntegrator):
    def __init__(self, limits, dt):
        super().__init__(limits, RobotState(np.zeros(len(limits.lower)), np.zeros(len(limits.lower))), control_dt=dt)
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


class Case:
    def __init__(self, name, joints, horizon, dt, lookahead, max_cycle_time, max_adoption_delay=0.):
        self.name, self.horizon, self.dt, self.lookahead = name, horizon, dt, lookahead
        self.limits = JointLimits([-1.] * joints, [1.] * joints, [.8] * joints, [2.] * joints)
        self.model = SimpleNamespace(joint_names=tuple(f"joint_{i}" for i in range(joints)), limits=self.limits)
        self.driver = Driver(self.limits, dt)
        if name.startswith("native"):
            from wholebodyx.native import NativeWBC
            self.controller = NativeWBC(self.limits)
        else:
            # These sessions use posture tasks with a fixed identity Jacobian.
            self.controller = KinematicWBC(self.limits, cache_task_hessian=True)
        self.max_cycle_time = max_cycle_time
        self.max_adoption_delay = max_adoption_delay
        self._event = None
        self.worker = None
        self.counts = dict(attempts=0, accepted=0, adopted=0, goal_changes=0,
                           injected_tracking=0, injected_worker_deaths=0, manual_resets=0,
                           locked_checks=0, worker_restarts=0, child_completed=0,
                           preempted_injections=0, uninjected_tracking_faults=0,
                           child_discarded=0, child_submitted=0, child_coalesced=0,
                           max_physical_violation=0., max_accepted_cycle_ms=0.,
                           max_attempt_ms=0., mean_accepted_cycle_ms=0., max_child_solve_ms=0.,
                           max_recovery_ms=0.,
                           max_parent_rss_kib=0, max_child_rss_kib=0, late_replies=0,
                           failures={}, failure_samples=[], failure_samples_by_cause={})
        self.histogram = np.zeros(10001, dtype=np.int64)  # .01 ms bins, final bin >=100 ms.
        self.child_info = None
        self._goal = None
        self.start_worker()

    def start_worker(self):
        self.worker = ProcessMPCWorker(self.limits, backend=self.name, horizon=self.horizon, dt=self.dt)
        try:
            info = self.worker.wait_ready()
            if self.child_info is not None:
                assert info == self.child_info, "worker restart loaded different sources"
            self.child_info = info
            self.references = DiagnosticReferences(self.worker, lookahead=self.lookahead,
                                                    replan_period=.1, max_pending_time=5.,
                                                    max_adoption_delay=self.max_adoption_delay)
            self.session = VelocitySession(self.model, self.driver, self.controller, self.dt,
                                            reference_manager=self.references, max_cycle_time=self.max_cycle_time)
            self.reset()
        except BaseException:
            self.worker.close()
            raise

    def close_worker(self):
        if self.worker is not None:
            pid = self.worker.pid
            self.worker.close()
            stats = self.worker.statistics
            for field in ("completed", "discarded", "submitted", "coalesced"):
                self.counts["child_" + field] += stats[field]
            self.counts["max_child_solve_ms"] = max(self.counts["max_child_solve_ms"], stats["max_solve_ms"])
            assert not Path(f"/proc/{pid}").exists() and not self.worker._relay.is_alive()
            self.worker = None

    def reset(self):
        self.driver.stop()  # Simulation-only quiescence before approving a hold seed.
        self.session.reset()
        measured = self.driver.read()
        seed = JointPlan(np.tile(measured.q, (self.horizon + 1, 1)),
                         np.zeros((self.horizon, measured.q.size)), measured.stamp, self.dt)
        self.references.seed(seed, measured, self.goal(), self.dt)

    def goal(self):
        phase = int(self.driver._state.stamp // 8.)
        return .55 * np.sin(np.arange(len(self.limits.lower)) * .37 + phase * 1.9)

    def record_event(self, index, category, detail, injected=None, deadline=None):
        # Called after the attempt timer, before recovery changes the driver.
        event = dict(attempt=index, category=category, detail=detail, injected=injected,
                     context=self.references.diagnostic_snapshot(observed=self.session.loop.gate._observed_state,
                                                                 goal=self._goal, control_dt=self.dt),
                     worker=self.worker.statistics,
                     execution_deadline=deadline.as_dict() if deadline is not None else None)
        self.counts["failure_samples"].append(event)
        del self.counts["failure_samples"][:-10]
        cause = category + ("/injected" if injected is not None else "/uninjected")
        samples = self.counts["failure_samples_by_cause"].setdefault(cause, [])
        samples.append(event)
        del samples[:-4]
        self._event = event

    def take_event(self):
        event, self._event = self._event, None
        return event

    def step(self):
        index = self.counts["attempts"]
        injected = None
        if index and index % 29989 == 0:
            self.worker._process.kill()
            deadline = time.monotonic() + 5
            while self.worker.error is None and time.monotonic() < deadline:
                time.sleep(.005)
            assert self.worker.error is not None
            self.counts["injected_worker_deaths"] += 1
            injected = "worker_error"
        elif index and index % 9697 == 0:
            measured = self.driver._state
            self.driver.inject_configuration_offset(np.where(measured.q > 0, -.08, .08))
            self.counts["injected_tracking"] += 1
            injected = "tracking_error"
        elif index and index % 12799 == 0:
            self.reset()
            self.counts["manual_resets"] += 1
        goal = self.goal()
        if self._goal is not None and not np.array_equal(goal, self._goal):
            self.counts["goal_changes"] += 1
        self._goal = goal
        reads, writes, stops = self.driver.reads, self.driver.writes, self.driver.stops
        self.references.begin_attempt()
        before = time.perf_counter()
        attempt_ms = None
        self.counts["attempts"] += 1
        try:
            update = self.session.step(goal)
            attempt_ms = (time.perf_counter() - before) * 1000
            assert injected is None, "injected fault was accepted"
            assert self.driver.reads == reads + 1 and self.driver.writes == writes + 1
            assert self.driver.stops == stops and self.session.fault is None
            current, command = update.cycle.state, update.command
            after = current.q + self.dt * command.velocity
            violation = float(max(0., np.max(self.limits.lower - after), np.max(after - self.limits.upper),
                                  np.max(np.abs(command.velocity) - self.limits.velocity),
                                  np.max(np.abs(command.velocity - current.v) - self.dt * self.limits.acceleration)))
            assert violation <= 1e-6
            assert update.reference.success and self.session.plan.stamp <= current.stamp
            assert current.stamp + self.dt <= self.session.plan.valid_until
            self.counts["accepted"] += 1
            self.counts["adopted"] += int(update.reference.replanned)
            self.counts["max_physical_violation"] = max(self.counts["max_physical_violation"], violation)
            if update.reference.reason == "late_reply":
                self.counts["late_replies"] += 1
                self.record_event(index, "late_reply", "candidate exceeded adoption deadline")
            elapsed = update.cycle.cycle_ms
            self.counts["max_accepted_cycle_ms"] = max(self.counts["max_accepted_cycle_ms"], elapsed)
            self.counts["mean_accepted_cycle_ms"] += (elapsed - self.counts["mean_accepted_cycle_ms"]) / self.counts["accepted"]
            self.histogram[min(10000, int(elapsed * 100))] += 1
            if update.reference.replanned:
                self.counts["max_child_solve_ms"] = max(self.counts["max_child_solve_ms"], update.reference.solve_ms)
        except RuntimeError as exc:
            attempt_ms = (time.perf_counter() - before) * 1000
            recovery_started = time.perf_counter()
            detail = str(exc)
            category = ("soft_deadline" if "execution deadline" in detail else
                        "worker_error" if injected == "worker_error" and self.worker.error in detail else
                        "planning_rejection" if "worker planning failed" in detail else
                        "tracking_error" if "measured state diverged" in detail else
                        "planning_timeout" if "wall-time budget" in detail else
                        "coverage" if any(message in detail for message in
                                          ("does not cover", "cannot predict", "expired before adoption")) else None)
            if category is None:
                raise
            if injected is not None and category != injected:
                self.counts["preempted_injections"] += 1
            if category == "tracking_error" and injected != "tracking_error":
                self.counts["uninjected_tracking_faults"] += 1
            assert self.driver.reads == reads + 1 and self.driver.stops == stops + 1
            assert self.session.fault and self.session.plan is None and not self.references.pending
            self.counts["failures"][category] = self.counts["failures"].get(category, 0) + 1
            deadline = exc.__cause__ if isinstance(exc.__cause__, ExecutionDeadlineError) else None
            self.record_event(index, category, detail, injected, deadline)
            latched = self.driver.reads, self.driver.writes, self.driver.stops
            try:
                self.session.step(goal)
                raise AssertionError("latched session accepted a command")
            except RuntimeError as locked:
                assert "faulted" in str(locked)
            assert latched == (self.driver.reads, self.driver.writes, self.driver.stops)
            self.counts["locked_checks"] += 1
            if injected == "worker_error":
                self.close_worker()
                self.start_worker()
                self.counts["worker_restarts"] += 1
            else:
                self.reset()
            self.counts["max_recovery_ms"] = max(self.counts["max_recovery_ms"], (time.perf_counter() - recovery_started) * 1000)
        finally:
            if attempt_ms is not None:
                self.counts["max_attempt_ms"] = max(self.counts["max_attempt_ms"], attempt_ms)

    def live_counts(self):
        result = {**self.counts, "simulation_s": self.driver._state.stamp}
        if self.worker is not None:
            stats = self.worker.statistics
            result["max_child_solve_ms"] = max(result["max_child_solve_ms"], stats["max_solve_ms"])
            for field in ("completed", "discarded", "submitted", "coalesced"):
                result["child_" + field] += stats[field]
        return result

    def snapshot(self):
        result = {**self.live_counts(), "child_runtime": self.child_info,
                  "cache_task_hessian": bool(getattr(self.controller, "cache_task_hessian", False)),
                  "cycle_histogram_bin_ms": .01, "cycle_histogram": self.histogram.tolist()}
        cumulative = np.cumsum(self.histogram)
        result["cycle_percentile_upper_ms"] = {
            str(p): float((np.searchsorted(cumulative, max(1, p / 100 * cumulative[-1])) + 1) / 100)
            for p in (50, 95, 99)} if cumulative[-1] else {}
        return result


def run(seconds, output, *, backends, joints=3, horizon=60, control_dt=.02, lookahead=.2, max_cycle_time=.005,
        max_adoption_delay=0.):
    import os
    if (not np.isfinite([seconds, control_dt, lookahead, max_cycle_time]).all() or
            min(seconds, control_dt, lookahead, max_cycle_time) <= 0 or joints < 1 or horizon < 1 or not backends):
        raise ValueError("duration, dimensions, intervals and backend selection must be positive")
    sources = list((ROOT / "python/wholebodyx").rglob("*.py")) + [Path(__file__).resolve()]
    if any(name.startswith("native") for name in backends):
        import wholebodyx_native
        sources.append(Path(wholebodyx_native.__file__).resolve())
    hashes = source_hashes(ROOT, sources)
    report = dict(status="running", diagnostic_schema=3,
                  deadline_scope="checked boundary and total elapsed time; not per-stage duration or actuation acknowledgement",
                  adoption_scope="reference update accepted; does not imply a command was written",
                  requested_wall_s=seconds, joints=joints, horizon=horizon,
                  control_dt=control_dt, lookahead=lookahead, max_cycle_time=max_cycle_time,
                  max_adoption_delay=max_adoption_delay,
                  source_sha256=hashes, initial_parent_rss_kib=rss(os.getpid()),
                  rss_sample_period_s=30., cases={})
    cases = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events_path = output.with_suffix(".events.jsonl")
    events = events_path.open("w")
    event_count = 0
    started = heartbeat = time.monotonic()
    try:
        for name in backends:
            cases.append(Case(name, joints, horizon, control_dt, lookahead, max_cycle_time,
                              max_adoption_delay=max_adoption_delay))
        # All startup barriers finish before the endurance timer and pacing begin.
        started = heartbeat = time.monotonic()
        due = started
        index = 0
        while time.monotonic() - started < seconds:
            case = cases[index % len(cases)]
            case.step()
            event = case.take_event()
            if event is not None:
                events.write(json.dumps(dict(backend=case.name, wall_s=time.monotonic() - started,
                                             **event), allow_nan=False) + "\n")
                events.flush()
                event_count += 1
            index += 1
            due += control_dt / len(cases)
            now = time.monotonic()
            if due < now - control_dt:
                due = now  # Do not create a catch-up burst after an OS pause.
            if due > now:
                time.sleep(min(due - now, control_dt))
            if time.monotonic() - heartbeat >= 30:
                heartbeat = time.monotonic()
                parent_rss = rss(os.getpid()) or 0
                for c in cases:
                    c.counts["max_parent_rss_kib"] = max(c.counts["max_parent_rss_kib"], parent_rss)
                    c.counts["max_child_rss_kib"] = max(c.counts["max_child_rss_kib"], rss(c.worker.pid) or 0)
                print(json.dumps(dict(wall_s=heartbeat - started, attempts=sum(c.counts["attempts"] for c in cases),
                                      accepted=sum(c.counts["accepted"] for c in cases),
                                      adopted=sum(c.counts["adopted"] for c in cases), parent_rss_kib=parent_rss,
                                      cases={c.name: {k: v for k, v in c.live_counts().items()
                                                       if k not in ("failure_samples", "failure_samples_by_cause")}
                                             for c in cases})), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            events.close()
        finally:
            report["wall_s"] = time.monotonic() - started
            close_errors = []
            for case in cases:
                try:
                    case.close_worker()
                except Exception as exc:
                    close_errors.append(f"{case.name}: {type(exc).__name__}: {exc}")
        if close_errors:
            report.update(status="failed", close_errors=close_errors)
        report.update(cases={c.name: c.snapshot() for c in cases}, final_parent_rss_kib=rss(os.getpid()),
                      events_file=str(events_path), events_written=event_count,
                      events_sha256=file_sha256(events_path),
                      all_workers_reaped=all(c.worker is None for c in cases),
                      source_unchanged=sources_unchanged(ROOT, hashes))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status=report["status"], wall_s=report["wall_s"], output=str(output))), flush=True)
        if close_errors:
            raise RuntimeError("; ".join(close_errors))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600.)
    parser.add_argument("--backend", choices=("python", "native", "sparse", "native-sparse", "all"), default="all")
    parser.add_argument("--joints", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--control-dt", type=float, default=.02)
    parser.add_argument("--lookahead", type=float, default=.2)
    parser.add_argument("--max-cycle-time", type=float, default=.005)
    adoption = parser.add_mutually_exclusive_group()
    adoption.add_argument("--max-adoption-delay", type=float, default=0.,
                          help="maximum model-time lateness; default zero requires the forecast tick")
    adoption.add_argument("--allow-late-adoption", action="store_true",
                          help="use the original sampling policy with no adoption lateness bound")
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-async-mpc-soak.json"))
    args = parser.parse_args()
    backends = ("python", "native", "sparse", "native-sparse") if args.backend == "all" else (args.backend,)
    run(args.seconds, args.output, backends=backends, joints=args.joints, horizon=args.horizon,
        control_dt=args.control_dt, lookahead=args.lookahead, max_cycle_time=args.max_cycle_time,
        max_adoption_delay=None if args.allow_late_adoption else args.max_adoption_delay)
