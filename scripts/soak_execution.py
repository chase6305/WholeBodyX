"""Paced endurance of injected velocity/torque pipelines and fault recovery.

Each case owns its algorithms, driver and simulated clock. Wall pacing rotates
cases; the reported simulation clocks are not a real-time scheduling claim.
"""

import argparse
import heapq
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np

from wholebodyx import (BackendRegistry, CoMReference, ContactSpec, DynamicsState, ExecutionDeadlineError,
                        FixedContactSimulator, JointLimits, LinearTask, RobotState,
                        TorqueControlLoop, VelocitySession)
from wholebodyx.simulation import JointIntegrator

from _measurement import rss_kib as current_rss_kib, source_hashes, sources_unchanged

ROOT = Path(__file__).resolve().parents[1]


class InjectedCancellation(BaseException):
    pass


class Driver:
    def __init__(self, plant):
        self.plant = plant
        self.failure = None
        self.reads = self.writes = self.stops = 0

    def read(self):
        self.reads += 1
        if self.failure == "read":
            raise OSError("injected read failure")
        return self.plant.read()

    def write(self, command):
        self.writes += 1
        self.plant.write(command)
        if self.failure == "write":
            raise OSError("injected failure after write")

    def stop(self):
        self.stops += 1
        self.plant.stop()
        if self.failure == "stop":
            raise OSError("injected stop failure")


class Controller:
    def __init__(self, implementation):
        self.implementation, self.failure = implementation, None

    def reset(self):
        if self.failure == "reset":
            raise RuntimeError("injected controller reset failure")
        self.implementation.reset()

    def step(self, *args):
        if self.failure == "solve":
            raise RuntimeError("injected control failure")
        return self.implementation.step(*args)


class Planner:
    def __init__(self, implementation):
        self.implementation, self.failure = implementation, None

    def reset(self):
        if self.failure == "planner_reset":
            raise RuntimeError("injected planner reset failure")
        self.implementation.reset()

    def solve(self, *args):
        if self.failure == "planner":
            raise RuntimeError("injected planning failure")
        return self.implementation.solve(*args)


class Case:
    def __init__(self, name, kind, driver, controller, *, model=None, planner=None, target=None,
                 dynamic_tasks=False):
        self.name, self.kind = name, kind
        self.driver, self.controller, self.planner = driver, controller, planner
        self.failure = None
        self.target = target
        self.dynamic_tasks = dynamic_tasks
        self.interruption = None
        if kind == "velocity":
            self.runner = VelocitySession(model, driver, controller, .005, planner=planner,
                                           task_factory=self.tasks, max_cycle_time=.005)
        else:
            self.runner = TorqueControlLoop(driver, model, controller, .005, max_cycle_time=.005)
        self.failures = ("read", "task", "solve", "write", "reset", "cancel", "stop")
        if planner is not None:
            self.failures += ("planner", "planner_reset")
        self.report = dict(attempts=0, successful_cycles=0, replans=0, deadline_faults=0,
                           deadline_stages={}, deadline_samples=[],
                           injected_faults={failure: 0 for failure in self.failures},
                           blocked_faulted_calls=0, max_cycle_ms=0., max_abs_position=0.,
                           max_constraint_violation=0.)

    def _factory_check(self):
        if self.failure == "cancel":
            raise self.interruption
        if self.failure in ("task", "stop"):
            raise RuntimeError("injected task/reference failure")

    def tasks(self, model, state):
        self._factory_check()
        jacobian = np.eye(3)
        if self.dynamic_tasks:
            jacobian += .1 * np.sin(state.stamp) * np.ones((3, 3)) / 3
        return [LinearTask(jacobian, .01 * np.cos(state.stamp) * np.array([1., -1., 1.]), 1e-4)]

    def reference(self, state):
        self._factory_check()
        return CoMReference(self.target, state.stamp, state.stamp + .005)

    def step(self):
        if self.kind == "velocity":
            return self.runner.step(lambda state: (.25 if int(state.stamp / 2) % 2 == 0 else -.25)
                                     * np.array([1., -1., 1.]))
        return self.runner.step(self.reference)

    def attempt(self):
        report = self.report
        report["attempts"] += 1
        attempt = report["attempts"]
        failure = self.failures[(attempt // 200 - 1) % len(self.failures)] if attempt % 200 == 0 else None
        self.failure = self.driver.failure = self.controller.failure = failure
        if self.planner is not None:
            self.planner.failure = failure
        if failure == "planner":
            self.runner.invalidate("injected")
        self.interruption = InjectedCancellation("injected cancellation")
        stops = self.driver.stops
        try:
            if failure in ("reset", "planner_reset"):
                self.runner.reset()
                raise AssertionError("injected reset unexpectedly succeeded")
            result = self.step()
            if failure is not None:
                raise AssertionError("injected cycle unexpectedly succeeded")
            assert self.driver.stops == stops
            state = self.driver.plant.read()
            if self.kind == "velocity":
                cycle = result.cycle
                report["replans"] += int(result.reference.replanned)
                assert self.runner.plan is not None
                assert np.max(np.abs(state.q)) <= 1 + 1e-6
                assert np.max(np.abs(state.v)) <= .5 + 1e-6
            else:
                cycle = result
                assert np.max(np.abs(state.q[7:])) <= 1 + 1e-6
                assert np.max(np.abs(state.v[6:])) <= 2 + 1e-6
            report["successful_cycles"] += 1
            report["max_cycle_ms"] = max(report["max_cycle_ms"], cycle.cycle_ms)
            report["max_constraint_violation"] = max(report["max_constraint_violation"], cycle.control.solver.violation)
            q = state.q if self.kind == "velocity" else state.q[7:]
            report["max_abs_position"] = max(report["max_abs_position"], float(np.max(np.abs(q))))
        except (RuntimeError, InjectedCancellation) as exc:
            assert self.runner.fault is not None and self.driver.stops == stops + 1
            if isinstance(exc, InjectedCancellation):
                assert exc is self.interruption and failure == "cancel"
            if "execution deadline" in str(exc):
                report["deadline_faults"] += 1
                deadline = exc.__cause__
                assert isinstance(deadline, ExecutionDeadlineError)
                stages = report["deadline_stages"]
                stages[deadline.stage] = stages.get(deadline.stage, 0) + 1
                report["deadline_samples"].append(dict(attempt=attempt, **deadline.as_dict()))
                del report["deadline_samples"][:-4]
            else:
                assert failure is not None and "injected" in str(exc), str(exc)
                report["injected_faults"][failure] += 1
            if self.kind == "velocity":
                assert self.runner.plan is None and self.runner.last_reference is None
            counters = self.driver.reads, self.driver.writes, self.driver.stops
            try:
                self.step()
            except RuntimeError as blocked:
                assert "faulted" in str(blocked)
            else:
                raise AssertionError("faulted pipeline accepted another cycle")
            assert counters == (self.driver.reads, self.driver.writes, self.driver.stops)
            report["blocked_faulted_calls"] += 1
            self.failure = self.driver.failure = self.controller.failure = None
            if self.planner is not None:
                self.planner.failure = None
            self.runner.reset()
            assert self.runner.fault is None


def cases(native=False, torque=False, dynamic_tasks=False, cache_task_hessian=False):
    registry = BackendRegistry()
    pairs = [("python", "python"), ("python", "sparse")]
    if native:
        pairs += [("native", "sparse"), ("native", "native"),
                  ("python", "native-sparse"), ("native", "native-sparse")]
    if cache_task_hessian:
        from wholebodyx import KinematicWBC
        registry.register("python-cached", controller=lambda limits: KinematicWBC(limits, cache_task_hessian=True))
        pairs = [("python-cached" if c == "python" else c, p) for c, p in pairs]
    result = []
    for controller_name, planner_name in pairs:
        limits = JointLimits([-1] * 3, [1] * 3, [.5] * 3, [1] * 3)
        model = SimpleNamespace(joint_names=("a", "b", "c"), limits=limits)
        plant = JointIntegrator(limits, RobotState(np.zeros(3), np.zeros(3)), control_dt=.005)
        result.append(Case(controller_name + "/" + planner_name, "velocity", Driver(plant),
                            Controller(registry.create_controller(controller_name, limits)), model=model,
                            planner=Planner(registry.create_planner(planner_name, limits)),
                            dynamic_tasks=dynamic_tasks))
    if torque:
        from wholebodyx.dynamics import ContactWBC, FloatingBaseDynamics
        robot = FloatingBaseDynamics(ROOT / "examples/models/standing_robot.urdf", contact_specs=[ContactSpec("base")])
        q = robot.pin.neutral(robot.model)
        q[2], q[-1] = .07, .2
        state = DynamicsState(q, np.zeros(robot.nv))
        plant = FixedContactSimulator(robot, state)
        result.append(Case("torque", "torque", Driver(plant), Controller(ContactWBC(robot, state)),
                            model=robot, target=robot.terms(state).com.copy()))
    return result


def run(seconds, output, *, native=False, torque=False, dynamic_tasks=False, cache_task_hessian=False):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    suite = cases(native, torque, dynamic_tasks, cache_task_hessian)
    sources = [path for path in (ROOT / "python/wholebodyx").glob("*.py")]
    sources += [Path(__file__).resolve(), ROOT / "examples/models/standing_robot.urdf"]
    sources += list((ROOT / "cpp").glob("*.*")) + list((ROOT / "include/wholebodyx").glob("*.hpp"))
    sources += [ROOT / "bindings/module.cpp", ROOT / "CMakeLists.txt"]
    if native:
        import wholebodyx_native
        sources.append(Path(wholebodyx_native.__file__))
    hashes = source_hashes(ROOT, sources)
    report = dict(status="running", diagnostic_schema=1,
                  deadline_scope="checked boundary and total elapsed time; not per-stage duration or actuation acknowledgement",
                  requested_wall_s=seconds, dynamic_tasks=dynamic_tasks,
                  cache_task_hessian=cache_task_hessian,
                  cases={}, source_sha256=hashes,
                  initial_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report.update(initial_rss_kib=current_rss_kib(), max_sampled_rss_kib=current_rss_kib(),
                  rss_sample_period_s=30.)
    started = heartbeat = tick = time.monotonic()
    attempts, slowest = 0, []
    try:
        while time.monotonic() - started < seconds:
            case = suite[attempts % len(suite)]
            before = time.perf_counter()
            case.attempt()
            elapsed = (time.perf_counter() - before) * 1000
            attempts += 1
            sample = (elapsed, attempts, case.name)
            if len(slowest) < 10:
                heapq.heappush(slowest, sample)
            elif elapsed > slowest[0][0]:
                heapq.heapreplace(slowest, sample)
            tick = max(tick + .005, time.monotonic())
            time.sleep(max(0., tick - time.monotonic()))
            if time.monotonic() - heartbeat >= 30:
                heartbeat = time.monotonic()
                rss = current_rss_kib()
                report["max_sampled_rss_kib"] = max(report["max_sampled_rss_kib"] or 0, rss or 0) or None
                print(json.dumps(dict(wall_s=heartbeat - started, attempts=attempts,
                                      faults=sum(item.report["blocked_faulted_calls"] for item in suite),
                                      current_rss_kib=rss,
                                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for case in suite:
            case.driver.failure = None
            case.driver.stop()
            case.report.update(reads=case.driver.reads, writes=case.driver.writes, stops=case.driver.stops,
                               simulation_s=case.driver.plant.read().stamp)
            report["cases"][case.name] = case.report
        report.update(wall_s=time.monotonic() - started, attempts=attempts,
                      final_rss_kib=current_rss_kib(),
                      slowest_attempts=[dict(wall_ms=wall, attempt=index, case=name)
                                        for wall, index, name in sorted(slowest, reverse=True)],
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_unchanged=sources_unchanged(ROOT, hashes))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--torque", action="store_true")
    parser.add_argument("--dynamic-tasks", action="store_true", help="vary coupled velocity-task Jacobians")
    parser.add_argument("--cache-task-hessian", action="store_true", help="opt in for the Python WBC cases")
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-execution-soak.json"))
    args = parser.parse_args()
    run(args.seconds, args.output, native=args.native, torque=args.torque, dynamic_tasks=args.dynamic_tasks,
        cache_task_hessian=args.cache_task_hessian)
