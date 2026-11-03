"""Velocity I/O checks remain outside algorithms and fail closed on bad cycles."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (CommandGate, ControlResult, JointLimits, JointOrder, KinematicWBC, LinearTask,
                        ModelOrderRobotBackend, QPResult, RobotState, VelocityCommand, VelocityControlLoop)


class MockBackend:
    def __init__(self, state, dt=.02):
        self.state, self.dt = state, dt
        self.reads = self.writes = self.stops = 0
        self.commands = []
        self.fail_write = self.fail_stop = False

    def read(self):
        self.reads += 1
        return self.state

    def write(self, command):
        self.writes += 1
        self.commands.append(command)
        if self.fail_write:
            raise OSError("transport disconnected")
        self.state = RobotState(self.state.q + self.dt * command.velocity, command.velocity,
                                self.state.stamp + self.dt)

    def stop(self):
        self.stops += 1
        if self.fail_stop:
            raise OSError("brake relay unavailable")
        self.state = RobotState(self.state.q, np.zeros_like(self.state.v), self.state.stamp)


def limits():
    return JointLimits([-1, -1], [1, 1], [.4, .8], [2, 3])


def tasks(_):
    return [LinearTask(np.eye(2), [.3, -.4])]


def solved(velocity, stamp, valid_until):
    return ControlResult(QPResult("solved", velocity, iterations=4, violation=0, detail="mock"),
                         VelocityCommand(velocity, stamp, valid_until))


def test_gate_accepts_only_fresh_covered_bounded_commands():
    gate = CommandGate(limits(), .02)
    state = RobotState([0, 0], [0, 0], 1.)
    gate.observe(state)
    accepted = gate.validate(state, VelocityCommand([.02, -.03], 1., 1.02))
    np.testing.assert_allclose(accepted.velocity, [.02, -.03])
    for command in (VelocityCommand([0, 0], .9, 1.02), VelocityCommand([0, 0], 1., 1.01),
                    VelocityCommand([1, 0], 1., 1.02), VelocityCommand([0], 1., 1.02)):
        with pytest.raises(ValueError):
            gate.validate(state, command)
    with pytest.raises(ValueError, match="did not advance"):
        gate.observe(state)
    gate.reset()
    assert gate.observe(state).stamp == 1


def test_gate_rejects_out_of_range_sensor_values_before_command():
    for state in (RobotState([1.1, 0], [0, 0], 1), RobotState([0, 0], [.5, 0], 1)):
        with pytest.raises(ValueError):
            CommandGate(limits(), .02).observe(state)


@pytest.mark.parametrize("component", ["q", "v"])
def test_gate_binds_command_to_observed_values_not_only_timestamp(component):
    gate = CommandGate(limits(), .02)
    state = RobotState([0, 0], [0, 0], 1.)
    gate.observe(state)
    getattr(state, component).setflags(write=True)
    getattr(state, component)[0] = .1
    with pytest.raises(ValueError, match="differs from the latest observed state"):
        gate.validate(state, VelocityCommand([0, 0], 1., 1.02))


def test_loop_executes_serial_cycles_with_real_wbc():
    backend = MockBackend(RobotState([0, 0], [0, 0]))
    loop = VelocityControlLoop(backend, limits(), KinematicWBC(limits()), .02)
    results = [loop.step(tasks) for _ in range(10)]
    assert backend.writes == 10 and backend.stops == 0 and backend.state.stamp == pytest.approx(.2)
    assert all(result.command is not None and result.control_ms >= 0 for result in results)
    assert results[-1].command.stamp == pytest.approx(.18)
    assert all(result.cycle_ms >= result.control_ms for result in results)


def test_numpy_period_and_large_state_clock_work_through_driver_gate():
    dt = np.float32(.005)
    backend = MockBackend(RobotState([0, 0], [0, 0], np.float32(1e8)), dt=float(dt))
    loop = VelocityControlLoop(backend, limits(), KinematicWBC(limits()), dt)
    assert isinstance(loop.control_dt, float)
    assert loop.step(tasks).command is not None
    assert backend.state.stamp == 1e8 + float(dt)


@pytest.mark.parametrize("stage", ["read", "task", "control", "write"])
def test_execution_deadline_covers_each_stage_and_latches_fault(monkeypatch, stage):
    import wholebodyx.hardware as hardware

    now = [0.]
    monkeypatch.setattr(hardware, "perf_counter", lambda: now[0])
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    controller = SimpleNamespace(reset=lambda: None,
                                 step=lambda *args: solved(np.zeros(2), 3, 3.02))
    factory = tasks
    owner, name = ((backend, "read") if stage == "read" else
                   (controller, "step") if stage == "control" else (backend, "write"))
    if stage == "task":
        def factory(state):
            now[0] += .03
            return tasks(state)
    else:
        original = getattr(owner, name)
        def delayed(*args):
            now[0] += .03
            return original(*args)
        setattr(owner, name, delayed)
    loop = VelocityControlLoop(backend, limits(), controller, .02, max_cycle_time=.01)
    with pytest.raises(RuntimeError, match="execution deadline"):
        loop.step(factory)
    assert backend.writes == (1 if stage == "write" else 0)
    assert backend.stops == 1 and loop.fault
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(tasks)
    assert backend.stops == 1


def test_execution_budget_cannot_extend_command_validity_and_reset_recovers(monkeypatch):
    import wholebodyx.hardware as hardware

    now = [0.]
    monkeypatch.setattr(hardware, "perf_counter", lambda: now[0])
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    def step(*args):
        now[0] += .03
        return solved(np.zeros(2), 3, 3.02)
    controller = SimpleNamespace(reset=lambda: None, step=step)
    loop = VelocityControlLoop(backend, limits(), controller, .02, max_cycle_time=1.)
    with pytest.raises(RuntimeError, match="execution deadline"):
        loop.step(tasks)
    assert backend.writes == 0
    loop.reset()
    controller.step = lambda *args: solved(np.zeros(2), 3, 3.02)
    assert loop.step(tasks).cycle_ms == 0


@pytest.mark.parametrize("budget", [0, -1., np.nan, np.inf])
def test_execution_budget_rejects_invalid_values(budget):
    with pytest.raises(ValueError, match="max cycle time"):
        VelocityControlLoop(MockBackend(RobotState([0, 0], [0, 0])), limits(),
                            KinematicWBC(limits()), .02, max_cycle_time=budget)


@pytest.mark.parametrize("kind", ["solver", "command_stamp", "command_coverage", "command_limit", "write", "task", "state"])
def test_bad_cycle_stops_once_latches_fault_and_never_writes_another_command(kind):
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    if kind == "state":
        backend.state = RobotState([2, 0], [0, 0], 3)
    velocity = np.zeros(2) if kind != "command_limit" else np.ones(2)
    controller_result = (ControlResult(QPResult("infeasible", detail="bounds")) if kind == "solver"
                         else solved(velocity, 0 if kind == "command_stamp" else 3,
                                     3.01 if kind == "command_coverage" else 3.02))
    controller = SimpleNamespace(reset=lambda: None, step=lambda *args: controller_result)
    if kind == "write":
        backend.fail_write = True
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    factory = (lambda _: (_ for _ in ()).throw(ValueError("task source failed"))) if kind == "task" else tasks
    with pytest.raises(RuntimeError, match="velocity control cycle fault"):
        loop.step(factory)
    writes = 1 if kind == "write" else 0
    assert backend.writes == writes and backend.stops == 1 and loop.fault
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(tasks)
    assert backend.stops == 1 and backend.writes == writes


def test_stop_failure_keeps_original_fault_and_prevents_retry():
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    backend.fail_stop = True
    controller = SimpleNamespace(reset=lambda: None, step=lambda *args: ControlResult(QPResult("infeasible", detail="bounds")))
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    with pytest.raises(RuntimeError, match="brake relay unavailable"):
        loop.step(tasks)
    assert "WBC rejected command" in loop.fault and backend.stops == 1


@pytest.mark.parametrize("stop_fails", [False, True])
def test_cancellation_stops_once_and_preserves_interrupt(stop_fails):
    backend = MockBackend(RobotState([0, 0], [0, 0]))
    backend.fail_stop = stop_fails
    loop = VelocityControlLoop(backend, limits(), KinematicWBC(limits()), .02)
    interruption = KeyboardInterrupt()
    def interrupt(state):
        raise interruption
    with pytest.raises(KeyboardInterrupt) as caught:
        loop.step(interrupt)
    assert caught.value is interruption
    assert backend.writes == 0 and backend.stops == 1 and loop.fault == "KeyboardInterrupt"
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(tasks)
    assert backend.stops == 1


def test_reset_forwards_and_allows_only_a_fresh_state_after_recovery():
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    resets = []
    controller = SimpleNamespace(reset=lambda: resets.append(1), step=lambda *args: solved(np.zeros(2), 0, 3.02))
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    with pytest.raises(RuntimeError):
        loop.step(tasks)
    controller.step = lambda *args: solved(np.zeros(2), 3, 3.02)
    loop.reset()
    assert loop.step(tasks).command is not None and resets == [1]


@pytest.mark.parametrize("stop_fails", [False, True])
def test_failed_controller_reset_stops_driver_and_blocks_further_io(stop_fails):
    backend = MockBackend(RobotState([0, 0], [0, 0], 3))
    def fail_reset():
        raise RuntimeError("controller reset failed")
    controller = SimpleNamespace(reset=fail_reset, step=lambda *args: solved(np.zeros(2), 3, 3.02))
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    assert loop.step(tasks).command is not None
    backend.fail_stop = stop_fails
    with pytest.raises(RuntimeError, match="controller reset failed"):
        loop.reset()
    assert backend.stops == 1 and loop.fault
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(tasks)
    assert backend.reads == backend.writes == 1 and backend.stops == 1
    backend.fail_stop = False
    controller.reset = lambda: None
    controller.step = lambda *args: solved(np.zeros(2), 3.02, 3.04)
    loop.reset()
    assert loop.fault is None and loop.step(tasks).command is not None


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_cancelled_controller_reset_stops_and_preserves_interrupt(interrupt_type, stop_fails):
    backend = MockBackend(RobotState([0, 0], [0, 0]))
    backend.fail_stop = stop_fails
    interruption = interrupt_type()
    def interrupt():
        raise interruption
    controller = SimpleNamespace(reset=interrupt, step=lambda *args: pytest.fail("faulted loop solved again"))
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    with pytest.raises(interrupt_type) as caught:
        loop.reset()
    assert caught.value is interruption
    assert loop.fault == interrupt_type.__name__ and backend.stops == 1
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(tasks)
    assert backend.reads == backend.writes == 0 and backend.stops == 1


@pytest.mark.parametrize("operation", ["cycle", "reset"])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stop_type", [KeyboardInterrupt, SystemExit])
def test_secondary_stop_cancellation_preserves_the_original_interrupt(operation, interrupt_type, stop_type):
    backend = MockBackend(RobotState([0, 0], [0, 0]))
    interruption, stop_failure = interrupt_type("original"), stop_type("secondary")
    def interrupt(*args):
        raise interruption
    def stop():
        backend.stops += 1
        raise stop_failure
    backend.stop = stop
    controller = SimpleNamespace(reset=interrupt, step=interrupt)
    loop = VelocityControlLoop(backend, limits(), controller, .02)
    with pytest.raises(BaseException) as caught:
        loop.reset() if operation == "reset" else loop.step(tasks)
    assert caught.value is interruption and caught.value.__cause__ is stop_failure
    assert loop.fault == interrupt_type.__name__ and backend.stops == 1


def test_model_order_backend_maps_only_driver_state_and_command_data():
    model = ("shoulder", "elbow", "wrist")
    hardware = ("wrist", "shoulder", "elbow")
    raw = MockBackend(RobotState([.3, .1, -.2], [3, 1, -2], 7))
    adapted = ModelOrderRobotBackend(raw, model, hardware)
    state = adapted.read()
    np.testing.assert_allclose(state.q, [.1, -.2, .3])
    np.testing.assert_allclose(state.v, [1, -2, 3])
    adapted.write(VelocityCommand([.4, -.5, .6], 7, 7.02))
    np.testing.assert_allclose(raw.commands[-1].velocity, [.6, .4, -.5])
    assert raw.commands[-1].stamp == 7 and raw.commands[-1].valid_until == 7.02
    adapted.stop()
    assert raw.stops == 1 and adapted.joint_names == model
    with pytest.raises(ValueError):
        ModelOrderRobotBackend(raw, model, ("wrist", "shoulder", "extra"))


def test_driver_snapshots_cannot_be_changed_by_caller_arrays():
    raw = MockBackend(RobotState([0, 0], [0, 0], 1))
    loop = VelocityControlLoop(raw, limits(), KinematicWBC(limits()), .02)
    result = loop.step(tasks)
    command = raw.commands[-1]
    command.velocity.setflags(write=True)
    command.velocity[:] = 99
    np.testing.assert_allclose(result.command.velocity, raw.commands[-1].velocity)
    # The driver mutation does not alter loop/gate state; a fresh state still works.
    loop.step(tasks)
