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


def test_loop_executes_serial_cycles_with_real_wbc():
    backend = MockBackend(RobotState([0, 0], [0, 0]))
    loop = VelocityControlLoop(backend, limits(), KinematicWBC(limits()), .02)
    results = [loop.step(tasks) for _ in range(10)]
    assert backend.writes == 10 and backend.stops == 0 and backend.state.stamp == pytest.approx(.2)
    assert all(result.command is not None and result.control_ms >= 0 for result in results)
    assert results[-1].command.stamp == pytest.approx(.18)


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
