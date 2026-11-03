"""External factories must remain independent and obey command contracts."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import BackendRegistry, ControlResult, JointMPC, KinematicWBC, QPResult, VelocityCommand
from wholebodyx.session import TrackingSession


def test_factories_are_lazy_and_role_specific():
    calls = []
    registry = BackendRegistry().register("external", controller=lambda limits: calls.append(limits) or KinematicWBC(limits))
    assert calls == []
    assert "external" in registry.names("controller") and "external" not in registry.names("planner")
    session = TrackingSession("external", mode="WBC", registry=registry)
    assert len(calls) == 1 and session.planner is None and session.reference_manager is None
    session.step()
    assert session.history[-1]["mpc_ms"] is None
    with pytest.raises(ValueError, match="does not provide a planner"):
        registry.create_planner("external", session.model.limits)


def test_sessions_own_distinct_instances_and_registry_snapshots():
    registry = BackendRegistry().register("external", controller=KinematicWBC, planner=JointMPC)
    first = TrackingSession("external", registry=registry)
    second = TrackingSession("external", registry=registry)
    assert first.controller is not second.controller and first.planner is not second.planner
    registry.register("later", controller=KinematicWBC)
    assert "later" not in first.registry.names("controller")
    first.step()
    second.step()
    second.reset()
    assert first.plan is not None and second.plan is None
    assert first.plant.read().stamp == .02 and second.plant.read().stamp == 0


def test_failed_reconfiguration_keeps_working_backend_and_plan():
    def broken(limits):
        raise ImportError("external solver ABI mismatch")
    registry = BackendRegistry().register("broken", planner=broken)
    session = TrackingSession(registry=registry)
    session.step()
    before = session.controller, session.planner, session.plan, session.plant.read()
    with pytest.raises(ImportError, match="ABI mismatch"):
        session.configure("python", "MPC + WBC", "broken")
    assert all(a is b for a, b in zip(before, (session.controller, session.planner, session.plan, session.plant.read())))
    assert session.mpc_backend == "python"
    session.step()


def test_wbc_only_does_not_load_selected_unused_planner():
    def unavailable(limits):
        pytest.fail("unused MPC factory executed")
    registry = BackendRegistry().register("unavailable", planner=unavailable)
    session = TrackingSession(mode="WBC", mpc_backend="unavailable", registry=registry)
    session.step()
    assert session.planner is None and session.plan is None


def test_missing_native_api_is_reported_without_python_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "wholebodyx_native", SimpleNamespace())
    registry = BackendRegistry()
    statuses = {s.name: s for s in registry.status()}
    assert statuses["python"].available and not statuses["native"].available
    assert "missing JointLimits" in statuses["native"].detail
    assert registry.names("controller", available_only=True) == ("python",)
    with pytest.raises(ImportError, match="Incompatible"):
        TrackingSession("native")
    TrackingSession().step()


def test_registration_rejects_duplicates_and_bad_factory_contracts():
    registry = BackendRegistry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register("python", controller=KinematicWBC)
    with pytest.raises(TypeError, match="callable"):
        registry.register("bad", planner=object())
    registry.register("bad", controller=lambda limits: object())
    with pytest.raises(TypeError, match="reset, step"):
        TrackingSession("bad", mode="WBC", registry=registry)
    with pytest.raises(ValueError, match="unknown backend"):
        TrackingSession("typo")


@pytest.mark.parametrize("kind", ["result", "command", "stamp", "violation", "position"])
def test_bad_adapter_output_preserves_state_and_discards_plan(kind):
    session = TrackingSession()
    session.step()
    before = session.plant.read()

    def bad_result(*args):
        if kind == "result":
            return object()
        velocity = np.full(6, 100.) if kind == "position" else np.zeros(6)
        command = VelocityCommand(velocity, before.stamp if kind != "stamp" else 0, before.stamp + .02)
        if kind == "command":
            command = object()
        return ControlResult(QPResult("solved", velocity, violation=np.nan if kind == "violation" else 0), command)

    session.controller = SimpleNamespace(step=bad_result)
    with pytest.raises((ValueError, RuntimeError)):
        session.step()
    assert session.plant.read() is before and session.plan is None and session.failure


@pytest.mark.parametrize("exception_type", [OSError, ZeroDivisionError, KeyboardInterrupt, SystemExit])
def test_adapter_exception_discards_plan_and_preserves_original_error(exception_type):
    session = TrackingSession()
    session.step()
    before = session.plant.read()
    failure = exception_type("injected adapter failure")
    def fail(*args):
        raise failure
    session.controller = SimpleNamespace(step=fail)
    with pytest.raises(exception_type) as caught:
        session.step()
    assert caught.value is failure
    assert session.plant.read() is before and session.plan is None and session.failure


def test_cancelled_scenario_configuration_restores_previous_session(monkeypatch):
    session = TrackingSession()
    session.step()
    before = session.controller, session.planner, session.plant.read(), session.plan, session.goal
    failure = KeyboardInterrupt()
    def interrupt(*args):
        raise failure
    monkeypatch.setattr(session.registry, "create_planner", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        session.set_scenario("mpc_retargeting")
    assert caught.value is failure
    assert session.scenario == "tracking"
    assert all(actual is original for actual, original in zip(
        (session.controller, session.planner, session.plant.read(), session.plan, session.goal), before))


def test_diagnostic_failure_preserves_an_already_committed_simulation_step(monkeypatch):
    session = TrackingSession()
    session.step()
    before = session.plant.read()
    def fail(*args):
        raise OSError("diagnostics unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(session, "reference", fail)
        with pytest.raises(OSError, match="diagnostics unavailable"):
            session.step()
    assert session.plant.read().stamp == before.stamp + session.dt
    assert session.steps == 2 and len(session.history) == 1
    assert session.plan is None and session.failure
    session.step()
    assert session.steps == 3 and session.failure is None


@pytest.mark.parametrize("component", ["controller", "planner"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_failed_session_reset_keeps_simulation_and_discards_reference(monkeypatch, component, exception_type):
    session = TrackingSession()
    session.step()
    before = session.plant.read()
    history, trails = session.history, session.trails
    steps, replans = session.steps, session.replans
    session.prediction(session.frames[0])
    failure = exception_type("reset interrupted")

    def fail():
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(getattr(session, component), "reset", fail)
        with pytest.raises(exception_type) as caught:
            session.reset()
    assert caught.value is failure
    assert session.plant.read() is before and session.steps == steps
    assert session.history is history and session.trails is trails and session.replans == replans
    assert session.plan is None and session.failure == "reset interrupted"
    assert session.prediction(session.frames[0]).shape == (0, 3)
    session.reset()
    assert session.plant.read().stamp == 0 and session.steps == 0
    assert session.plan is None and session.failure is None
    session.step()
    assert session.steps == 1 and session.plan is not None


@pytest.mark.parametrize("component", ["controller", "planner"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_scenario_switch_rolls_back_when_new_component_reset_fails(component, exception_type):
    armed = False
    failure = exception_type("new scenario reset failed")
    base = KinematicWBC if component == "controller" else JointMPC

    class Adapter(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.resets = 0

        def reset(self):
            self.resets += 1
            if armed and self.resets == 2:
                raise failure
            super().reset()

    registry = BackendRegistry().register("fragile", **{component: Adapter})
    session = (TrackingSession("fragile", registry=registry) if component == "controller"
               else TrackingSession(mpc_backend="fragile", registry=registry))
    session.step()
    session.prediction(session.frames[0])
    before = session.__dict__.copy()
    armed = True
    with pytest.raises(exception_type) as caught:
        session.set_scenario("mpc_retargeting")
    assert caught.value is failure
    assert session.__dict__.keys() == before.keys()
    assert all(session.__dict__[key] is value for key, value in before.items())
    session.step()
    assert session.steps == 2 and session.failure is None
    armed = False
    session.set_scenario("mpc_retargeting")
    assert session.scenario == "mpc_retargeting" and session.steps == 0
    session.step()
    assert session.steps == 1
