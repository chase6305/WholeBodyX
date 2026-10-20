"""Paced, bounded-memory fault/recovery checks of the velocity driver boundary."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from wholebodyx import JointLimits, KinematicWBC, LinearTask, RobotState, VelocityControlLoop


class InjectedCancellation(BaseException):
    pass


class Driver:
    def __init__(self, dt):
        self.dt = dt
        self.state = RobotState(np.zeros(2), np.zeros(2))
        self.reads = self.writes = self.stops = 0
        self.failure = None

    def read(self):
        self.reads += 1
        if self.failure == "read":
            raise OSError("injected read failure")
        return self.state

    def write(self, command):
        self.writes += 1
        self.state = RobotState(self.state.q + self.dt * command.velocity,
                                command.velocity, self.state.stamp + self.dt)
        if self.failure == "write":
            raise OSError("injected failure after write")

    def stop(self):
        self.stops += 1
        self.state = RobotState(self.state.q, np.zeros(2), self.state.stamp)
        if self.failure == "stop":
            raise OSError("injected stop failure")


class Controller(KinematicWBC):
    failure = None

    def reset(self):
        if self.failure == "reset":
            raise RuntimeError("injected reset failure")
        super().reset()

    def step(self, *args):
        if self.failure == "solve":
            raise RuntimeError("injected control failure")
        return super().step(*args)


def run(seconds, output):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    dt = .005
    limits = JointLimits([-1., -1.], [1., 1.], [.5, .5], [1., 1.])
    driver, controller = Driver(dt), Controller(limits)
    loop = VelocityControlLoop(driver, limits, controller, dt, max_cycle_time=dt)
    failures = ("read", "task", "solve", "write", "reset", "cancel", "stop")
    root = Path(__file__).resolve().parents[1]
    report = dict(status="running", requested_wall_s=seconds, attempts=0, successful_cycles=0,
                  injected_faults={name: 0 for name in failures}, deadline_faults=0,
                  blocked_faulted_calls=0, max_cycle_ms=0., max_position=0.,
                  initial_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ("python/wholebodyx/hardware.py", "python/wholebodyx/wbc.py",
                                              "python/wholebodyx/types.py", "scripts/soak_velocity_loop.py")})
    started = heartbeat = tick = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            report["attempts"] += 1
            attempt = report["attempts"]
            failure = failures[(attempt // 200 - 1) % len(failures)] if attempt % 200 == 0 else None
            driver.failure = controller.failure = failure
            before_stops = driver.stops
            interruption = InjectedCancellation("injected cancellation")

            def tasks(state):
                if failure == "cancel":
                    raise interruption
                if failure in ("task", "stop"):
                    raise RuntimeError("injected task failure")
                return [LinearTask(np.eye(2), .3 * np.cos(state.stamp) * np.array([1., -1.]))]

            try:
                if failure == "reset":
                    loop.reset()
                    raise AssertionError("injected reset unexpectedly succeeded")
                result = loop.step(tasks)
                if failure is not None:
                    raise AssertionError("injected cycle unexpectedly succeeded")
                assert driver.stops == before_stops
                assert np.max(np.abs(result.command.velocity - result.state.v)) <= dt + 1e-6
                assert np.max(np.abs(driver.state.q)) <= 1. + 1e-6
                assert np.max(np.abs(driver.state.v)) <= .5 + 1e-6
                report["successful_cycles"] += 1
                report["max_cycle_ms"] = max(report["max_cycle_ms"], result.cycle_ms)
                report["max_position"] = max(report["max_position"], float(np.max(np.abs(driver.state.q))))
            except (RuntimeError, InjectedCancellation) as exc:
                assert loop.fault is not None and driver.stops == before_stops + 1
                if isinstance(exc, InjectedCancellation):
                    assert exc is interruption and failure == "cancel"
                if "execution deadline" in str(exc):
                    report["deadline_faults"] += 1
                else:
                    assert failure is not None and "injected" in str(exc)
                    report["injected_faults"][failure] += 1
                counters = driver.reads, driver.writes, driver.stops
                try:
                    loop.step(tasks)
                except RuntimeError as blocked:
                    assert "faulted" in str(blocked)
                else:
                    raise AssertionError("faulted loop accepted another cycle")
                assert counters == (driver.reads, driver.writes, driver.stops)
                report["blocked_faulted_calls"] += 1
                driver.failure = controller.failure = None
                loop.reset()
                assert loop.fault is None
            tick = max(tick + dt, time.monotonic())
            time.sleep(max(0., tick - time.monotonic()))
            if time.monotonic() - heartbeat >= 30.:
                heartbeat = time.monotonic()
                print(json.dumps(dict(wall_s=heartbeat - started, attempts=attempt,
                                      faults=report["blocked_faulted_calls"],
                                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        driver.failure = None
        driver.stop()
        report.update(wall_s=time.monotonic() - started, reads=driver.reads, writes=driver.writes,
                      stop_attempts=driver.stops, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=7200.)
    parser.add_argument("--output", type=Path, default=Path("/tmp/wholebodyx-velocity-soak.json"))
    args = parser.parse_args()
    run(args.seconds, args.output)
