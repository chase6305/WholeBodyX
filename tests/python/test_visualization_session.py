import numpy as np
import pytest

from wholebodyx import ControlResult, PlanResult, QPResult
from wholebodyx.visualization.session import TrackingSession


def test_live_goal_change_and_mode_switch(backend):
    name = "native" if backend[0].__name__ == "NativeWBC" else "python"
    session = TrackingSession(name)
    for _ in range(7):
        session.step()
    goal = session.goal.copy()
    goal[0] += 0.1
    session.set_goal(goal)
    assert session.plan is None and session.prediction("left_ee").shape == (0, 3)
    stamp = session.plant.read().stamp
    session.step()
    assert session.plan.stamp == stamp  # Replan immediately on goal changes.
    assert session.prediction("left_ee").shape == (16, 3)
    session.configure(name, "WBC")
    assert session.prediction("left_ee").shape == (0, 3)
    session.step()
    assert session.history[-1]["mpc_ms"] is None
    session.reset()
    assert session.plant.read().stamp == 0
    assert not session.history and session.plan is None


def test_visual_geometry_matches_control_model():
    session = TrackingSession()
    q = session.plant.read().q
    for frame in session.frames:
        points = session.links(q, frame)
        np.testing.assert_allclose(points[-1], session.model.pose(q, frame).translation)
        np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), session.model.lengths)
    with pytest.raises(ValueError):
        session.set_goal([9] * 6)


def test_prediction_cache_isolated_and_rebuilt_after_replanning():
    session = TrackingSession()
    session.step()
    for frame in session.frames:
        expected = np.array([session.model.pose(q, frame).translation for q in session.plan.positions])
        prediction = session.prediction(frame)
        np.testing.assert_allclose(prediction, expected, atol=1e-14)
        prediction[:] = 99
        np.testing.assert_allclose(session.prediction(frame), expected, atol=1e-14)
    session.set_goal(session.goal + .1)
    assert not session.prediction("left_ee").size
    session.step()
    expected = np.array([session.model.pose(q, "left_ee").translation for q in session.plan.positions])
    np.testing.assert_allclose(session.prediction("left_ee"), expected, atol=1e-14)


@pytest.mark.parametrize("scenario", ["tracking", "coordination", "cartesian_circle", "disturbance_recovery"])
def test_position_diagnostics_match_full_poses(scenario):
    session = TrackingSession(scenario=scenario)
    for _ in range(8):
        sample = session.step()
        state, target = session.plant.read(), session.reference()
        positions = [session.model.pose(state.q, f).translation for f in session.frames]
        for k, side in enumerate(("left", "right")):
            np.testing.assert_allclose(session.trails[k][-1], positions[k], atol=1e-14)
            assert sample[f"{side}_error"] == pytest.approx(
                np.linalg.norm(positions[k] - target.poses[k].translation), abs=1e-14)
        relative = (positions[1] - positions[0]) - (target.poses[1].translation - target.poses[0].translation)
        assert sample["relative_error"] == pytest.approx(np.linalg.norm(relative), abs=1e-14)


@pytest.mark.parametrize("stamp,offset", [
    (0., [0] * 6), (2. - 2e-9, [0] * 6),
    (2. - 1e-9, [-.2, .2, .1, .2, -.2, -.1]),
    (4. - 2e-9, [-.2, .2, .1, .2, -.2, -.1]),
    (4. - 1e-9, [.15, -.1, .05, -.15, .1, -.05]),
])
def test_joint_reference_schedule_without_pose_evaluation(stamp, offset, monkeypatch):
    session = TrackingSession(scenario="mpc_retargeting")
    session.set_goal([2.75] * 6)
    expected = np.clip(session.goal + offset, session.model.limits.lower, session.model.limits.upper)
    np.testing.assert_allclose(session.reference(stamp).q, expected, atol=1e-14)

    def unexpected_pose(*args):
        raise AssertionError("joint target scheduling should not evaluate poses")

    monkeypatch.setattr(session.model, "pose", unexpected_pose)
    actual = session._reference_goal(stamp)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    actual[:] = 0
    np.testing.assert_allclose(session._reference_goal(stamp), expected, atol=1e-14)


def test_prediction_cache_checks_goal_without_cartesian_reference(monkeypatch):
    session = TrackingSession()
    session.step()
    expected = session.prediction("left_ee")

    def unexpected_reference(*args):
        raise AssertionError("prediction validity only requires the joint target")

    monkeypatch.setattr(session, "reference", unexpected_reference)
    np.testing.assert_array_equal(session.prediction("left_ee"), expected)
    session.set_goal(session.goal + .1)
    assert session.prediction("left_ee").shape == (0, 3)


def test_scenario_switch_resets_clock_and_selects_tasks():
    session = TrackingSession()
    session.step()
    session.set_scenario("cartesian_circle")
    assert session.mode == "WBC" and session.plant.read().stamp == 0
    assert not session.history and session.plan is None
    with pytest.raises(ValueError):
        session.configure("python", "MPC + WBC")
    session.step()
    session.set_scenario("mpc_retargeting")
    assert session.mode == "MPC + WBC"
    session.step()
    assert session.replans == 1 and session.plan.stamp == 0


def test_long_session_bounds_diagnostics_memory():
    session = TrackingSession()
    for _ in range(1510):
        session.step()
    assert session.replans == 302 and len(session.replan_stamps) == 300
    assert len(session.history) == 300 and all(len(t) == 200 for t in session.trails)
    assert session.replan_stamps[0] > 0
    session.reset()
    assert not session.replan_stamps and session.replans == 0


def test_scheduled_goal_change_clears_prediction_at_display_boundary(backend):
    name = "native" if backend[0].__name__ == "NativeWBC" else "python"
    session = TrackingSession(name, scenario="mpc_retargeting")
    for _ in range(99):
        session.step()
    assert session.plan is not None
    session.step()  # State time reaches 2 s, immediately before the next MPC solve.
    assert session.plant.read().stamp == pytest.approx(2)
    assert session.plan is None and not session.prediction("left_ee").size
    sample = session.step()
    assert sample["replan_reason"] == "goal_changed"
    assert session.plan.stamp == pytest.approx(2)


@pytest.mark.parametrize("stage", ["mpc", "wbc"])
def test_control_failure_invalidates_prediction_without_advancing_state(stage, monkeypatch):
    session = TrackingSession()
    for _ in range(5): session.step()
    assert session.prediction("left_ee").size
    before = session.plant.read()
    if stage == "mpc":
        monkeypatch.setattr(session.planner, "solve", lambda *args: PlanResult(QPResult("infeasible")))
    else:
        monkeypatch.setattr(session.controller, "step", lambda *args: ControlResult(QPResult("infeasible")))
    with pytest.raises(RuntimeError, match="infeasible"):
        session.step()
    assert session.plan is None and not session.prediction("left_ee").size
    assert session.plant.read() is before
    assert "infeasible" in session.failure


def test_simulated_configuration_offset_is_validated_atomically():
    session = TrackingSession()
    before = session.plant.read()
    for offset in ([9] * 6, [1], [np.nan] * 6):
        with pytest.raises(ValueError):
            session.plant.inject_configuration_offset(offset)
        assert session.plant.read() is before
    session.plant.inject_configuration_offset([.01] * 6)
    after = session.plant.read()
    assert after.stamp == before.stamp
    np.testing.assert_array_equal(after.v, before.v)
    np.testing.assert_allclose(after.q, before.q + .01)
