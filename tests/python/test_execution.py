"""Injected robots and drivers share reference validation and command gating."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (JointLimits, LinearTask, PlanResult, QPResult, RobotState,
                        VelocitySession)
from wholebodyx.simulation import JointIntegrator


class Driver(JointIntegrator):
    def __init__(self, limits, initial, dt=.02):
        super().__init__(limits, initial, control_dt=dt)
        self.reads = self.writes = self.stops = 0
        self.fail_write = False

    def read(self):
        self.reads += 1
        return super().read()

    def write(self, command):
        self.writes += 1
        if self.fail_write:
            raise OSError("transport unavailable")
        super().write(command)

    def stop(self):
        self.stops += 1
        super().stop()


def setup(joints=3):
    bounds = JointLimits([-1] * joints, [1] * joints, [.8] * joints, [2] * joints)
    model = SimpleNamespace(joint_names=tuple(f"joint_{i}" for i in range(joints)), limits=bounds)
    driver = Driver(bounds, RobotState(np.zeros(joints), np.zeros(joints)))
    return model, driver


@pytest.mark.parametrize("joints", [1, 3, 7])
def test_injected_robot_and_driver_track_mpc_and_use_one_state_read_per_cycle(backend, joints):
    model, driver = setup(joints)
    session = VelocitySession(model, driver, backend[0](model.limits),
                              planner=backend[1](model.limits))
    session.reset()
    goal_stamps = []

    def goal(state):
        goal_stamps.append(state.stamp)
        return np.full(joints, .3)

    updates = [session.step(goal) for _ in range(100)]
    assert driver.reads == driver.writes == session.steps == 100
    assert driver.stops == 0 and session.fault is None
    assert sum(step.reference.replanned for step in updates) == 20
    assert goal_stamps == [step.cycle.state.stamp for step in updates]
    assert np.max(np.abs(driver._state.q - .3)) < 1e-4
    assert all(step.cycle.cycle_ms >= step.cycle.control_ms for step in updates)


def test_mpc_reference_and_additional_tasks_reach_the_same_controller(backend):
    model, driver = setup(1)
    seen = []

    def tasks(injected, state):
        assert injected is model
        seen.append(state.stamp)
        return [LinearTask([[1]], [.2], 10.)]

    session = VelocitySession(model, driver, backend[0](model.limits),
                              planner=backend[1](model.limits), task_factory=tasks,
                              posture_weight=.1)
    step = session.step([0])
    assert step.reference.success and step.reference.replanned
    assert step.command.velocity[0] > .03  # The additional task affects control.
    assert seen == [0.] and driver.reads == driver.writes == 1


def test_wbc_only_session_accepts_generic_model_and_bound_integrator(backend):
    model, driver = setup(3)
    session = VelocitySession(model, driver, backend[0](model.limits),
                              task_factory=lambda m, s: [LinearTask(np.eye(3), [.1, -.1, 0])])
    for _ in range(20):
        step = session.step()
        assert step.reference is None
    assert driver._state.q[0] > 0 and driver._state.q[1] < 0
    assert session.plan is None and session.last_reference is None


@pytest.mark.parametrize("failure", ["planner", "tasks", "write"])
def test_failed_cycle_discards_mpc_reference_and_requires_reset(backend, failure):
    model, driver = setup()
    planner = backend[1](model.limits)
    session = VelocitySession(model, driver, backend[0](model.limits), planner=planner)
    session.step([.2] * 3)
    assert session.plan is not None
    if failure == "planner":
        original = planner.solve
        planner.solve = lambda *args: PlanResult(QPResult("infeasible", detail="injected"))
    elif failure == "tasks":
        session.task_factory = lambda m, s: [object()]
    else:
        driver.fail_write = True
    before = driver._state
    with pytest.raises(RuntimeError):
        session.step([.3] * 3)
    assert session.plan is None and session.last_reference is None
    assert session.fault and session.failure and driver.stops == 1
    np.testing.assert_array_equal(driver._state.q, before.q)
    counts = driver.reads, driver.writes, driver.stops
    with pytest.raises(RuntimeError, match="faulted"):
        session.step([.2] * 3)
    assert counts == (driver.reads, driver.writes, driver.stops)
    if failure == "planner":
        planner.solve = original
    session.task_factory = None
    driver.fail_write = False
    session.reset()
    assert not session.fault and session.steps == 0
    assert session.step([.2] * 3).reference.replanned


def test_planner_reset_failure_uses_driver_stop_and_keeps_reference_invalid(backend):
    model, driver = setup()
    planner = backend[1](model.limits)
    session = VelocitySession(model, driver, backend[0](model.limits), planner=planner)
    session.step([.2] * 3)
    before = driver._state
    original = planner.reset

    def fail():
        raise RuntimeError("planner reset failed")

    planner.reset = fail
    with pytest.raises(RuntimeError, match="planner reset failed"):
        session.reset()
    assert session.plan is None and session.last_reference is None
    assert session.steps == 1 and session.fault and driver.stops == 1
    assert driver._state.stamp == before.stamp
    planner.reset = original
    session.reset()
    assert session.steps == 0 and not session.fault
    assert session.step([.2] * 3).reference.replanned


def test_cancelled_task_invalidates_new_plan_and_preserves_interrupt(backend):
    model, driver = setup()
    interruption = KeyboardInterrupt("cancelled")

    def tasks(model, state):
        raise interruption

    session = VelocitySession(model, driver, backend[0](model.limits),
                              planner=backend[1](model.limits), task_factory=tasks)
    with pytest.raises(KeyboardInterrupt) as caught:
        session.step([.2] * 3)
    assert caught.value is interruption
    assert session.plan is None and driver.stops == 1 and driver.writes == 0


def test_cycle_budget_includes_planning_and_blocks_command_write(monkeypatch, backend):
    import wholebodyx.hardware as hardware

    model, driver = setup()
    planner = backend[1](model.limits)
    original = planner.solve
    now = [0.]
    monkeypatch.setattr(hardware, "perf_counter", lambda: now[0])

    def delayed(*args):
        now[0] += .01
        return original(*args)

    planner.solve = delayed
    session = VelocitySession(model, driver, backend[0](model.limits), planner=planner,
                              max_cycle_time=.005)
    with pytest.raises(RuntimeError, match="execution deadline"):
        session.step([.2] * 3)
    assert driver.writes == 0 and driver.stops == 1 and session.plan is None


def test_integrator_requires_explicit_or_bound_period_and_rejects_mismatch():
    bounds = JointLimits([-1], [1], [1])
    from wholebodyx import VelocityCommand

    command = VelocityCommand([.1], 0, .02)
    plant = JointIntegrator(bounds, RobotState([0], [0]))
    with pytest.raises(ValueError, match="configure control_dt"):
        plant.write(command)
    bound = JointIntegrator(bounds, RobotState([0], [0]), control_dt=.02)
    with pytest.raises(ValueError, match="differs"):
        bound.write(command, .01)
    bound.write(command)
    assert bound.read().stamp == .02


def test_urdf_pipeline_combines_cartesian_tasks_mpc_and_reordered_backend(backend):
    import importlib.util
    from pathlib import Path

    pytest.importorskip("pinocchio")
    path = Path(__file__).resolve().parents[2] / "examples/python/urdf_tracking.py"
    spec = importlib.util.spec_from_file_location("urdf_pipeline_test", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    controller = "native" if backend[0].__name__ == "NativeWBC" else "python"
    planner = "native" if controller == "python" and importlib.util.find_spec("wholebodyx_native") else "python"
    report = example.run(controller, mpc=True, mpc_backend=planner,
                         backend_joints=("wrist_joint", "shoulder", "elbow"))
    assert report["mode"] == "MPC + WBC" and report["replans"] > 1
    assert report["final_position_error_m"] < 1e-4
    assert report["final_orientation_error_rad"] < 1e-4
    assert report["max_constraint_violation"] <= 1e-6
