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
