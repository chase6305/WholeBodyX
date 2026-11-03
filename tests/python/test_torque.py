"""Torque execution validates model equations and driver failure boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (CoMReference, FixedContactSimulator, TorqueCommand,
                        TorqueCommandGate, TorqueControlLoop)
from wholebodyx.dynamics import ContactWBC


def reference(robot, state, dt=.005):
    return CoMReference(robot.terms(state).com, state.stamp, state.stamp + dt)


def test_torque_loop_and_independent_plant_hold_static_equilibrium(floating_robot):
    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    controller = ContactWBC(robot, state)
    loop = TorqueControlLoop(plant, robot, controller)
    loop.reset()
    for _ in range(100):
        result = loop.step(lambda s: reference(robot, s))
        assert result.cycle_ms >= result.control_ms >= 0
        assert result.command.mode == "torque"
        np.testing.assert_allclose(plant.acceleration, result.control.acceleration, atol=1e-8)
    assert plant.steps == 100 and plant.stops == 0 and loop.fault is None
    assert plant.read().stamp == pytest.approx(.5)
    np.testing.assert_allclose(plant.read().q, state.q, atol=1e-8)
    assert plant.wrench[2] == pytest.approx(11 * 9.81, abs=1e-6)


@pytest.mark.parametrize("kind", ["effort", "equations", "contact", "stamp", "coverage", "type"])
def test_invalid_controller_output_never_reaches_driver_and_latches_fault(floating_robot, kind):
    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    control = ContactWBC(robot, state).step(state, reference(robot, state))
    if kind in ("effort", "equations"):
        command = replace(control.command, torque=[100 if kind == "effort" else 1])
        bad = replace(control, command=command)
    elif kind == "contact":
        bad = replace(control, wrench=[0, 0, -1, 0, 0, 0])
    elif kind == "stamp":
        bad = replace(control, command=TorqueCommand(control.torque, .01, .02))
    elif kind == "coverage":
        bad = replace(control, command=TorqueCommand(control.torque, 0, .004))
    else:
        bad = object()
    loop = TorqueControlLoop(plant, robot, SimpleNamespace(reset=lambda: None, step=lambda *a: bad))
    with pytest.raises(RuntimeError, match="torque control cycle fault"):
        loop.step(reference(robot, state))
    assert plant.read() is not state  # Plant owns its initial snapshot.
    np.testing.assert_array_equal(plant.read().q, state.q)
    assert plant.steps == 0 and plant.stops == 1 and loop.fault
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(reference(robot, state))
    assert plant.stops == 1


def test_gate_rejects_stale_state_and_does_not_trust_reported_residuals(floating_robot):
    robot, state = floating_robot
    gate = TorqueCommandGate(robot, .005)
    gate.observe(state)
    result = ContactWBC(robot, state).step(state, reference(robot, state))
    changed = replace(result, acceleration=np.r_[1., np.zeros(robot.nv - 1)],
                       dynamics_residual=0., contact_residual=0.)
    with pytest.raises(ValueError, match="independently evaluated"):
        gate.validate(state, changed)
    gate.validate(state, result)
    with pytest.raises(ValueError, match="clock did not advance"):
        gate.observe(state)
    replacement = state.q.copy()
    replacement[-1] += .01
    with pytest.raises(ValueError, match="latest observed state"):
        gate.validate(replace(state, q=replacement), result)


def test_gate_enforces_optional_actuated_acceleration_limits(floating_robot):
    from wholebodyx import JointLimits

    robot, initial = floating_robot
    q = initial.q.copy()
    q[-1] = .1
    state = replace(initial, q=q)
    control = ContactWBC(robot, initial).step(state, reference(robot, state))
    assert abs(control.acceleration[-1]) > .1
    gate = TorqueCommandGate(robot, .005, limits=JointLimits([-1], [1], [2], [.1]))
    gate.observe(state)
    with pytest.raises(ValueError, match="acceleration limits"):
        gate.validate(state, control)


@pytest.mark.parametrize("wrench", [np.full(6, np.nan), np.zeros(0), np.full(6, np.inf)])
def test_plant_rejects_invalid_forward_result_before_advancing(floating_robot, monkeypatch, wrench):
    robot, initial = floating_robot
    plant = FixedContactSimulator(robot, initial)
    before = plant.read()
    monkeypatch.setattr(robot, "forward", lambda *a, **k: (np.zeros(robot.nv), wrench))
    with pytest.raises(RuntimeError, match="invalid forward dynamics"):
        plant.write(TorqueCommand([0], 0, .005))
    assert plant.read() is before and plant.steps == 0


@pytest.mark.parametrize("stage", ["reference", "control", "write", "reset"])
def test_each_failure_stops_once_and_successful_reset_recovers(floating_robot, stage):
    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    real = ContactWBC(robot, state)
    controller = SimpleNamespace(reset=real.reset, step=real.step)
    loop = TorqueControlLoop(plant, robot, controller)
    failure = OSError("injected failure")

    def fail(*args):
        raise failure

    factory = lambda s: reference(robot, s)
    original_write = plant.write
    if stage == "reference":
        factory = fail
    elif stage == "control":
        controller.step = fail
    elif stage == "write":
        plant.write = fail
    else:
        controller.reset = fail
    with pytest.raises(RuntimeError, match="injected failure"):
        loop.reset() if stage == "reset" else loop.step(factory)
    assert plant.steps == 0 and plant.stops == 1 and loop.fault
    controller.reset, controller.step, plant.write = real.reset, real.step, original_write
    loop.reset()
    assert loop.step(lambda s: reference(robot, s)).command is not None
    assert plant.steps == 1 and plant.stops == 1


def test_reference_deadline_and_cancellation_preserve_stop_behavior(floating_robot, monkeypatch):
    import wholebodyx.torque as torque

    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    loop = TorqueControlLoop(plant, robot, ContactWBC(robot, state), max_cycle_time=.004)
    now = [0.]
    monkeypatch.setattr(torque, "perf_counter", lambda: now[0])

    def delayed(s):
        now[0] += .006
        return reference(robot, s)

    with pytest.raises(RuntimeError, match="execution deadline"):
        loop.step(delayed)
    assert plant.steps == 0 and plant.stops == 1
    loop.reset()
    interruption = KeyboardInterrupt("cancelled")

    def cancel(s):
        raise interruption

    with pytest.raises(KeyboardInterrupt) as caught:
        loop.step(cancel)
    assert caught.value is interruption and plant.stops == 2
    assert plant.steps == 0


@pytest.mark.parametrize("stage", ["read", "reference", "control", "write"])
def test_torque_deadline_boundary_details_preserve_write_and_stop_counts(floating_robot, monkeypatch, stage):
    import wholebodyx.torque as torque
    from wholebodyx import ExecutionDeadlineError

    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    real = ContactWBC(robot, state)
    controller = SimpleNamespace(reset=real.reset, step=real.step)
    now = [0.]
    monkeypatch.setattr(torque, "perf_counter", lambda: now[0])
    factory = lambda s: reference(robot, s)
    if stage == "reference":
        def factory(s):
            now[0] += .006
            return reference(robot, s)
    else:
        owner, name = (controller, "step") if stage == "control" else (plant, stage)
        original = getattr(owner, name)
        def delayed(*args):
            result = original(*args)
            now[0] += .006
            return result
        setattr(owner, name, delayed)
    loop = TorqueControlLoop(plant, robot, controller, max_cycle_time=.004)
    with pytest.raises(RuntimeError, match="execution deadline") as caught:
        loop.step(factory)
    reason = caught.value.__cause__
    assert isinstance(reason, ExecutionDeadlineError) and reason.mode == "torque"
    assert reason.stage == {"read": "after_read", "reference": "after_reference",
                            "control": "before_write", "write": "after_write"}[stage]
    assert reason.elapsed_s == .006 and reason.budget_s == .004
    assert reason.write_completed == (stage == "write")
    assert plant.steps == int(stage == "write") and plant.stops == 1 and loop.fault
    with pytest.raises(RuntimeError, match="faulted"):
        loop.step(factory)
    assert plant.stops == 1


def test_plant_uses_torque_and_own_contact_dynamics_and_rolls_back_bad_wrench(floating_robot, monkeypatch):
    robot, state = floating_robot
    plant = FixedContactSimulator(robot, state)
    control = ContactWBC(robot, state).step(state, reference(robot, state))
    plant.write(replace(control.command, torque=control.torque + .1))
    assert np.linalg.norm(plant.acceleration - control.acceleration) > .1
    before = plant.read()
    monkeypatch.setattr(robot, "forward", lambda *a, **k: (np.zeros(robot.nv), np.array([0, 0, -1, 0, 0, 0])))
    with pytest.raises(RuntimeError, match="physical support"):
        plant.write(TorqueCommand([0], before.stamp, before.stamp + .005))
    assert plant.read() is before and plant.steps == 1


def test_self_contained_torque_example_rejects_disturbance_and_recovers(tmp_path):
    import csv
    import importlib.util
    from pathlib import Path

    pytest.importorskip("pinocchio")
    path = Path(__file__).resolve().parents[2] / "examples/python/torque_control.py"
    spec = importlib.util.spec_from_file_location("torque_example_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "torque.csv"
    result = module.run(csv_path=output)
    assert result["actuated_joints"] == 1
    assert result["max_posture_error_rad"] > .001
    assert result["final_posture_error_rad"] < 1e-5
    assert result["max_constraint_violation"] <= 1e-5
    assert result["max_predicted_acceleration_difference"] > 1.
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == result["steps"]


def test_h1_torque_pipeline_matches_direct_demo_trajectory():
    import importlib.util
    from pathlib import Path

    pytest.importorskip("pinocchio")
    path = Path(__file__).resolve().parents[2] / "examples/python/h1_dynamics_wbc.py"
    spec = importlib.util.spec_from_file_location("h1_torque_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 assets are absent")
    direct, reference_demo = module.DynamicsDemo(), module.DynamicsDemo()
    robot = reference_demo.robot
    plant = FixedContactSimulator(robot, reference_demo.state)
    loop = TorqueControlLoop(plant, robot, reference_demo.controller)
    for _ in range(200):
        state = plant.read()
        velocity, acceleration = reference_demo.target_derivatives(state.stamp)
        result = loop.step(CoMReference(reference_demo.target(state.stamp), state.stamp,
                                         state.stamp + .005, velocity, acceleration))
        direct.step()
        np.testing.assert_allclose(plant.read().q, direct.state.q, atol=1e-10, rtol=0)
        np.testing.assert_allclose(plant.acceleration, result.control.acceleration, atol=1e-6)
    assert plant.stops == 0

