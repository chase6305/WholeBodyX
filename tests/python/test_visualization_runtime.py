"""Reset and fault semantics must not execute leftover single-step requests."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx import ControlResult, QPResult
from wholebodyx.visualization.session import TrackingSession
from wholebodyx.visualization.viewer import ControlViewer


def test_link_orientations_preserve_cylinder_axes():
    from wholebodyx.visualization.viewer import link_orientations

    angles = np.linspace(-np.pi, np.pi, 101)
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    points = np.vstack((np.zeros(3), np.cumsum(directions, axis=0)))
    wxyz = link_orientations(points)
    np.testing.assert_allclose(np.linalg.norm(wxyz, axis=1), 1, atol=1e-14)
    rotations = Rotation.from_quat(np.roll(wxyz, -1, axis=1))
    np.testing.assert_allclose(rotations.apply([0, 0, 1]), directions, atol=1e-14)
    np.testing.assert_allclose(rotations.apply([0, 1, 0]),
                               np.tile([0, 0, 1], (len(angles), 1)), atol=1e-14)


def paused_viewer(session, steps, reset=False):
    # Exercise the real control-loop dispatch without opening a GUI socket.
    viewer = object.__new__(ControlViewer)
    viewer.session = session
    viewer._lock = threading.Lock()
    viewer._pending = {"reset": reset, "steps": steps, "configuration": None, "goal": None,
                       "scenario": None, "refresh": False}
    viewer.running = SimpleNamespace(value=False)
    viewer.mpc_backend = SimpleNamespace(disabled=False)
    viewer.status = SimpleNamespace(content="")
    viewer._last_error = None
    viewer.render = lambda: None
    return viewer


def test_reset_discards_queued_steps():
    session = TrackingSession()
    session.step()
    viewer = paused_viewer(session, steps=4, reset=True)
    viewer.tick()
    viewer.tick()
    assert session.plant.read().stamp == 0 and session.steps == 0
    assert not viewer._pending["steps"]


def test_solver_fault_discards_queued_steps_and_keeps_state():
    session = TrackingSession(mode="WBC")
    calls = []

    def fail(*args):
        calls.append(args)
        return ControlResult(QPResult("infeasible"))

    session.controller = SimpleNamespace(step=fail)
    viewer = paused_viewer(session, steps=4)
    viewer.tick()
    viewer.tick()
    assert len(calls) == 1 and not viewer._pending["steps"]
    assert session.plant.read().stamp == 0 and session.steps == 0
    assert "infeasible" in viewer.status.content
    # A late checkbox refresh must not erase the fault diagnostic.
    previous = viewer.status.content
    viewer._pending["refresh"] = True
    viewer.tick()
    assert viewer.status.content == previous


class PredictionHandle:
    visible = False
    writes = 0

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self.writes += 1
        self._points = value


def prediction_viewer(session):
    viewer = object.__new__(ControlViewer)
    viewer.session = session
    viewer._displayed_plans = [None, None]
    return viewer


def test_prediction_geometry_only_written_for_new_plans():
    session = TrackingSession()
    viewer = prediction_viewer(session)
    handles = [PredictionHandle(), PredictionHandle()]
    for _ in range(100):
        session.step()
        for k, frame in enumerate(session.frames):
            viewer._render_prediction(k, frame, handles[k])
    assert session.replans == 20
    assert [h.writes for h in handles] == [20, 20]
    assert all(h.visible for h in handles)


def test_prediction_goal_edit_reset_and_recovery_refresh_display():
    session = TrackingSession()
    viewer = prediction_viewer(session)
    handle = PredictionHandle()
    session.step()
    viewer._render_prediction(0, "left_ee", handle)
    first = handle.points.copy()
    session.set_goal(session.goal + .1)
    viewer._render_prediction(0, "left_ee", handle)
    assert not handle.visible and handle.writes == 1
    session.step()
    viewer._render_prediction(0, "left_ee", handle)
    assert handle.visible and handle.writes == 2
    assert not np.array_equal(first, handle.points)
    session.reset()
    viewer._render_prediction(0, "left_ee", handle)
    assert not handle.visible
    session.step()
    viewer._render_prediction(0, "left_ee", handle)
    assert handle.visible and handle.writes == 3


def test_prediction_expiry_hides_cached_geometry():
    from wholebodyx import RobotState

    session = TrackingSession()
    viewer = prediction_viewer(session)
    handle = PredictionHandle()
    session.step()
    viewer._render_prediction(0, "left_ee", handle)
    state = session.plant.read()
    expired = RobotState(state.q, state.v, session.plan.valid_until)
    session.plant.read = lambda: expired
    viewer._render_prediction(0, "left_ee", handle)
    assert not handle.visible and handle.writes == 1


def test_goal_geometry_reuses_values_and_tracks_edits():
    from wholebodyx.visualization.viewer import segments

    session = TrackingSession(scenario="mpc_retargeting")
    viewer = prediction_viewer(session)
    viewer._displayed_goals = [None, None]
    handle = PredictionHandle()
    for _ in range(100):
        viewer._render_goal(0, "left_ee", session.reference(0).q, handle)
    assert handle.writes == 1 and handle.visible
    # Scheduled retargeting uses the effective reference, not the base goal.
    q = session.reference(2).q
    viewer._render_goal(0, "left_ee", q, handle)
    assert handle.writes == 2
    np.testing.assert_array_equal(handle.points, segments(session.links(q, "left_ee")))
    q[0] += .1
    viewer._render_goal(0, "left_ee", q, handle)
    assert handle.writes == 3
    session.model.lengths[0] += .01
    viewer._render_goal(0, "left_ee", q, handle)
    assert handle.writes == 4
    np.testing.assert_array_equal(handle.points, segments(session.links(q, "left_ee")))


def test_hidden_goal_skips_geometry_and_reappears(monkeypatch):
    session = TrackingSession()
    viewer = prediction_viewer(session)
    viewer._displayed_goals = [None, None]
    handle = PredictionHandle()
    viewer._render_goal(0, "left_ee", session.goal, handle)
    for scenario in ("coordination", "cartesian_circle"):
        session.set_scenario(scenario)
        with monkeypatch.context() as patch:
            def unexpected_links(*args):
                raise AssertionError("hidden goals should not compute link geometry")
            patch.setattr(session, "links", unexpected_links)
            viewer._render_goal(0, "left_ee", session.goal, handle)
        assert not handle.visible and handle.writes == 1
    session.set_scenario("tracking")
    viewer._render_goal(0, "left_ee", session.goal, handle)
    assert handle.visible and handle.writes == 2


def render_viewer(session):
    """Exercise render() with lightweight scene handles, without opening sockets."""
    viewer = prediction_viewer(session)
    viewer._displayed_goals = [None, None]
    viewer._last_plot = -1.
    viewer.server = SimpleNamespace(atomic=nullcontext)
    viewer.error_plot = SimpleNamespace(data=None)
    viewer.timing_plot = SimpleNamespace(data=None)
    viewer.arms = [([SimpleNamespace() for _ in range(3)],
                    [SimpleNamespace() for _ in range(4)],
                    SimpleNamespace(), PredictionHandle(), PredictionHandle(), PredictionHandle())
                   for _ in session.frames]
    return viewer


def test_render_reuses_drawn_endpoints_and_updates_paused_target(monkeypatch):
    session = TrackingSession(mode="WBC")
    session.step()
    viewer = render_viewer(session)
    history = [dict(row) for row in session.history]

    def unexpected_translations(*args):
        raise AssertionError("render telemetry should reuse the drawn endpoints")

    monkeypatch.setattr(session.model, "translations", unexpected_translations)
    viewer.render()
    initial = viewer.error_plot.data
    # A normal redraw at the same timestamp does not rebuild plots.
    viewer.render()
    assert viewer.error_plot.data is initial
    session.set_goal(session.goal + .1)
    viewer._last_plot = -1.  # tick() forces plots after paused target edits.
    viewer.render()
    reference = session.reference()
    for k in range(2):
        endpoint = viewer.arms[k][1][-1].position
        expected = np.linalg.norm(endpoint - reference.poses[k].translation)
        np.testing.assert_allclose(viewer.error_plot.data[k + 1][-1], expected, atol=1e-14)
    assert viewer.error_plot.data[0][-1] == session.plant.read().stamp
    assert list(session.history) == history


def test_render_initial_and_reset_plots_have_valid_time_domain():
    session = TrackingSession(mode="WBC")
    viewer = render_viewer(session)
    viewer.render()
    np.testing.assert_array_equal(viewer.error_plot.data[0], [0.])
    assert all(np.isfinite(a).all() for a in viewer.error_plot.data)
    assert all(np.isnan(a).all() for a in viewer.timing_plot.data[1:])
    session.step()
    session.reset()
    viewer._last_plot = -1.
    viewer.render()
    np.testing.assert_array_equal(viewer.error_plot.data[0], [0.])


def test_plot_columns_preserve_missing_samples_and_own_data():
    from wholebodyx.visualization.viewer import plot_columns

    history = [dict(time=i * .02, left_error=.1, right_error=.2,
                    wbc_ms=0. if i == 0 else None, mpc_ms=.5 if i % 5 == 0 else None)
               for i in range(300)]
    fields = ("time", "left_error", "right_error", "wbc_ms", "mpc_ms")
    columns = plot_columns(history)
    for field, column in zip(fields, columns):
        expected = np.array([np.nan if row[field] is None else row[field] for row in history])
        np.testing.assert_array_equal(column, expected)
        assert column.flags.c_contiguous and column.dtype == np.dtype(float)
    columns[0][0] = 99
    assert history[0]["time"] == 0
    assert columns[3][0] == 0 and np.isnan(columns[3][1])
    assert all(column.shape == (0,) for column in plot_columns([]))
