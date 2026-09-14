"""Acceptance checks for complete example behaviors, including mixed backends."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wholebodyx import RobotState
from wholebodyx.model import PlanarDualArm
from wholebodyx.tasks import FramePoseTask, RelativePositionTask

ROOT = Path(__file__).resolve().parents[2]


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"examples/python/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend_name(backend):
    return "native" if backend[0].__name__ == "NativeWBC" else "python"


@pytest.mark.parametrize("scenario", ["cartesian_circle", "coordination", "mpc_retargeting", "disturbance_recovery"])
def test_complete_scenario(backend, scenario):
    r = example("_common").run(scenario, backend_name(backend))
    assert r["max_constraint_violation"] <= 1e-6
    assert r["final_position_error_m"] < 1e-4
    if scenario == "cartesian_circle":
        assert r["max_position_error_m"] < 1e-4
        assert r["max_orientation_error_rad"] < 1e-4
    elif scenario == "coordination":
        assert r["max_position_error_m"] < 1e-4
        assert r["max_relative_error_m"] < 3e-5
    elif scenario == "mpc_retargeting":
        assert r["scheduled_goal_changes"] == 2
        for stamp in (2.0, 4.0):
            assert any(abs(t - stamp) < 1e-9 for t in r["replan_stamps_s"])
    else:
        assert r["disturbances"] == r["tracking_replans"] == 2
        for stamp in (2.06, 4.14):
            assert any(abs(t - stamp) < 1e-9 for t in r["replan_stamps_s"])


def test_disturbance_csv_explains_position_discontinuities(tmp_path):
    path = tmp_path / "disturbance.csv"
    example("_common").run("disturbance_recovery", csv_path=path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    q = np.array([[float(row[f"q_{k}"]) for k in range(6)] for row in rows])
    v = np.array([[float(row[f"v_{k}"]) for k in range(6)] for row in rows])
    offset = np.array([[float(row[f"offset_{k}_rad"]) for k in range(6)] for row in rows])
    np.testing.assert_allclose(np.diff(q, axis=0), .02 * v[1:] + offset[1:], atol=1e-12)
    events = [r for r in rows if r["replan_reason"] == "tracking_error"]
    np.testing.assert_allclose([float(r["control_stamp_s"]) for r in events], [2.06, 4.14])


def test_constraint_fault_and_recovery_example(backend):
    r = example("joint_limit_handling").run(backend_name(backend))
    assert abs(r["bounded_position_rad"] - 1) < 1e-6
    assert r["fault_status"] == "infeasible" and not r["fault_has_command"]
    assert r["recovered_velocity_rad_s"] < 0


def test_weight_example_analytic_answers(backend):
    rows = example("task_weight_tradeoff").run(backend_name(backend))["results"]
    np.testing.assert_allclose([r["velocity_rad_s"] for r in rows],
                               [r["analytic_velocity_rad_s"] for r in rows], atol=1e-6)
    assert all(b["positive_task_residual"] < a["positive_task_residual"] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize("reordered", [False, True])
def test_supplied_urdf_example(backend, reordered):
    pytest.importorskip("pinocchio")
    names = ["wrist_joint", "shoulder", "elbow"] if reordered else None
    r = example("urdf_tracking").run(backend_name(backend), backend_joints=names)
    assert r["final_position_error_m"] < 1e-4
    assert r["final_orientation_error_rad"] < 1e-4
    assert r["backend_joint_order"] == (tuple(names) if reordered else r["joints"])


@pytest.mark.parametrize("scenario", ["mpc_retargeting", "disturbance_recovery"])
def test_mixed_backend_cli(backend, scenario):
    if backend_name(backend) != "native":
        return  # The native fixture handles optional-extension availability.
    result = subprocess.run([sys.executable, str(ROOT / "examples/python/mixed_backends.py"), "--scenario", scenario],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    results = json.loads(result.stdout)
    assert {(r["wbc_backend"], r["mpc_backend"]) for r in results} == {
        ("python", "python"), ("python", "native"), ("native", "python"), ("native", "native")}
    assert all(r["final_position_error_m"] < 1e-4 for r in results)
    if scenario == "mpc_retargeting":
        assert all(r["scheduled_goal_changes"] == 2 for r in results)
    else:
        assert all(r["disturbances"] == r["tracking_replans"] == 2 for r in results)


def test_csv_contains_every_step_and_measured_motion(tmp_path):
    path = tmp_path / "nested/trace.csv"
    result = example("_common").run("cartesian_circle", steps=20, csv_path=path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert float(rows[-1]["time"]) == pytest.approx(0.4)
    assert rows[0]["mpc_ms"] == ""  # Missing solve samples are not zero-duration solves.
    q = np.array([[float(row[f"q_{k}"]) for k in range(6)] for row in rows])
    v = np.array([[float(row[f"v_{k}"]) for k in range(6)] for row in rows])
    np.testing.assert_allclose(np.diff(q, axis=0), 0.02 * v[1:], atol=1e-12)
    errors = np.array([[float(row[key]) for key in ("left_error", "right_error")] for row in rows])
    assert result["rms_position_error_m"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result["max_position_error_m"] == pytest.approx(errors.max())


@pytest.mark.parametrize("scenario", ["tracking", "cartesian_circle", "mpc_retargeting"])
def test_export_and_summary_paths_have_equal_metrics(tmp_path, scenario):
    from wholebodyx.runner import run

    summary = run(scenario, steps=30)
    exported = run(scenario, steps=30, csv_path=tmp_path / "trace.csv")
    assert summary == exported
    with (tmp_path / "trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    orientations = [float(row[f"orientation_error_{k}_rad"]) for row in rows for k in range(2)]
    assert summary["max_orientation_error_rad"] == max(orientations)
    assert len(rows) == 30


def test_csv_keeps_completed_steps_if_solver_fails(tmp_path, monkeypatch):
    module = example("_common")
    original_step = module.TrackingSession.step

    def fail_on_third_step(session):
        if session.steps == 2:
            raise RuntimeError("injected planner failure")
        return original_step(session)

    monkeypatch.setattr(module.TrackingSession, "step", fail_on_third_step)
    path = tmp_path / "partial.csv"
    with pytest.raises(RuntimeError, match="injected planner failure"):
        module.run("mpc_retargeting", steps=10, csv_path=path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and float(rows[-1]["time"]) == pytest.approx(.04)


def test_relative_task_world_jacobian_and_sign():
    model = PlanarDualArm()
    state = RobotState([0.2, 0.6, -0.4, -0.2, -0.6, 0.4], np.zeros(6))
    relative = lambda q: model.pose(q, "right_ee").translation - model.pose(q, "left_ee").translation
    task = RelativePositionTask("left_ee", "right_ee", relative(state.q) + [0.01, -0.02, 0]).linearize(model, state)
    eps = 1e-6
    numerical = np.column_stack([(relative(state.q + d) - relative(state.q - d)) / (2 * eps)
                                 for d in eps * np.eye(6)])
    np.testing.assert_allclose(task.jacobian, numerical, atol=1e-8)
    np.testing.assert_allclose(task.target_velocity, [0.04, -0.08, 0], atol=1e-12)


def test_coordination_reuses_geometry_and_matches_relative_task(monkeypatch):
    from wholebodyx.experiments import FRAMES, reference, wbc_tasks

    model = PlanarDualArm()
    rng = np.random.default_rng(29)
    for _ in range(10):
        state = RobotState(rng.uniform(-1, 1, 6), np.zeros(6))
        target = reference(model, "coordination", rng.uniform(-1, 1, 6), .7)
        expected = RelativePositionTask(*FRAMES,
            target.poses[1].translation - target.poses[0].translation, weight=20).linearize(model, state)
        calls = {"pose": [], "jacobian": []}
        with monkeypatch.context() as patch:
            for name in calls:
                original = getattr(model, name)

                def counted(q, frame, method=original, key=name):
                    calls[key].append(frame)
                    return method(q, frame)

                patch.setattr(model, name, counted)
            actual = wbc_tasks(model, state, "coordination", target)[1]
        assert calls == {"pose": list(FRAMES), "jacobian": list(FRAMES)}
        np.testing.assert_allclose(actual.jacobian, expected.jacobian, atol=1e-14)
        np.testing.assert_allclose(actual.target_velocity, expected.target_velocity, atol=1e-14)
        np.testing.assert_array_equal(actual.weight, expected.weight)


def test_pose_feedforward_and_dimension_validation():
    model = PlanarDualArm()
    state = RobotState(np.zeros(6), np.zeros(6))
    pose = model.pose(state.q, "left_ee")
    twist = [0.1, 0.2, 0, 0, 0, 0.3]
    task = FramePoseTask("left_ee", pose, feedforward=twist).linearize(model, state)
    np.testing.assert_allclose(task.target_velocity, twist, atol=1e-12)
    with pytest.raises(ValueError):
        FramePoseTask("left_ee", pose, feedforward=[1, 2, 3]).linearize(model, state)


def test_custom_registered_factories_complete_recovery():
    result = example("custom_backends").run()
    assert (result["wbc_backend"], result["mpc_backend"]) == ("smooth_wbc", "long_mpc")
    assert result["disturbances"] == result["tracking_replans"] == 2
    assert result["final_position_error_m"] < 1e-4
    assert result["max_constraint_violation"] <= 1e-6


@pytest.mark.parametrize("mode", ["mpc", "wbc"])
def test_h1_fixed_pelvis_example(backend, mode, tmp_path):
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external HumanoidAssets H1 asset is not present")
    path = tmp_path / "h1.csv"
    result = module.run(mode=mode, backend=backend_name(backend), csv_path=path)
    assert result["joints"] == 19 and result["fixed_pelvis"]
    assert result["final_joint_error_rad"] < 1e-5
    assert result["max_foot_error_m"] < 1e-9
    assert result["max_constraint_violation"] <= 1e-6
    assert result["replans"] == (80 if mode == "mpc" else 0)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    names = module.H1Session().model.joint_names
    q = np.array([[float(row[f"q_{name}"]) for name in names] for row in rows])
    v = np.array([[float(row[f"v_{name}"]) for name in names] for row in rows])
    np.testing.assert_allclose(np.diff(q, axis=0), .02 * v[1:], atol=1e-12)
    assert np.max(np.abs(v)) <= .6 + 1e-6
    assert np.max(np.abs(np.diff(np.vstack((np.zeros(19), v)), axis=0))) <= .04 + 1e-6


@pytest.mark.parametrize("mode", ["mpc", "wbc"])
def test_h1_pickup_and_return(backend, mode, tmp_path):
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 asset is not present")
    result = module.run(mode=mode, motion="pickup", backend=backend_name(backend),
                        csv_path=tmp_path / "pickup.csv")
    assert result["base_model"] == "support_foot_kinematic"
    assert result["object_attached"] and 3 <= result["grasp_time_s"] < 4
    assert .70 < result["max_pelvis_pitch_rad"] < .80
    assert result["object_lift_m"] > .2
    assert result["final_joint_error_rad"] < 1e-5
    assert result["max_constraint_violation"] <= 1e-6
    assert result["max_foot_error_m"] < (.001 if mode == "mpc" else .02)
    with (tmp_path / "pickup.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    first = next(i for i, row in enumerate(rows) if row["object_attached"] == "True")
    assert float(rows[first]["grasp_distance_m"]) <= .015
    positions = np.array([[float(row[f"object_{a}_m"]) for a in "xyz"] for row in rows])
    np.testing.assert_allclose(positions[:first + 1], np.tile(positions[0], (first + 1, 1)), atol=1e-12)
    assert abs(float(rows[-1]["pelvis_pitch_rad"])) < 1e-5


def test_h1_pickup_does_not_attach_distant_object_and_reset_restores():
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 asset is not present")
    session = module.H1Session(motion="pickup")
    original = session.object_position.copy()
    session.object_position += [2., 0, 0]
    for _ in range(205):
        session.step()
    assert not session.object_attached
    session.reset()
    assert session.grasp_time is None and not session.object_attached
    np.testing.assert_array_equal(session.object_position, original)


def test_h1_bow_goal_is_smooth_at_phase_boundaries():
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 asset is not present")
    session = module.H1Session(motion="pickup")
    eps = 1e-5
    for stamp in (.5, 2.5, 4., 6.):
        center = session.goal(stamp)
        for sign in (-1, 1):
            assert np.max(np.abs(session.goal(stamp + sign * eps) - center)) / eps < 1e-4
    np.testing.assert_array_equal(session.goal(0), np.zeros(19))
    np.testing.assert_array_equal(session.goal(8), np.zeros(19))
    assert np.max(np.abs(session.goal(3))) == pytest.approx(1.)
    for stamp in (0., .5, 2.5, 3., 4., 6., 8.):
        np.testing.assert_array_equal(session.goal_velocity(stamp), np.zeros(19))
    for stamp in (1., 2., 4.5, 5.5):
        numerical = (session.goal(stamp + eps) - session.goal(stamp - eps)) / (2 * eps)
        np.testing.assert_allclose(session.goal_velocity(stamp), numerical, atol=1e-9)


@pytest.mark.parametrize("autoplay,loop,expected_steps,expected_resets", [
    (True, True, 3, 1), (True, False, 2, 0), (False, True, 0, 0),
])
def test_h1_viewer_autoplay_and_loop(monkeypatch, tmp_path, autoplay, loop, expected_steps, expected_resets):
    from types import SimpleNamespace
    from contextlib import nullcontext

    module = example("h1_mpc_wbc")
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="test"/>')
    counts = dict(steps=0, resets=0, sleeps=0, stopped=False)
    controls = {}

    def checkbox(name, initial_value):
        controls[name] = SimpleNamespace(value=initial_value)
        return controls[name]

    gui = SimpleNamespace(add_markdown=lambda *args: SimpleNamespace(content=""),
                          add_checkbox=checkbox,
                          add_slider=lambda *args, **kwargs: SimpleNamespace(value=1.),
                          add_button=lambda *args: SimpleNamespace(on_click=lambda callback: None))
    server = SimpleNamespace(gui=gui, scene=SimpleNamespace(set_up_direction=lambda *a: None, add_grid=lambda *a, **k: None),
                             atomic=nullcontext, on_client_connect=lambda callback: None,
                             stop=lambda: counts.update(stopped=True))
    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=lambda **kwargs: server))
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace())
    session = SimpleNamespace(urdf=urdf, dt=.02, steps=398, replans=0,
                              model=SimpleNamespace(pose=lambda *args: SimpleNamespace(translation=np.array([0., 0., -1.]))),
                              plant=SimpleNamespace(read=lambda: SimpleNamespace(q=np.zeros(19))),
                              world_transform=lambda q: (np.eye(3), np.zeros(3)), phase=lambda: "test")

    def reset():
        counts["resets"] += 1
        session.steps = 0

    def step():
        counts["steps"] += 1
        session.steps += 1
        return dict(time=session.steps * .02, joint_error_rad=0., pelvis_pitch_rad=0., wbc_ms=0.)

    def sleep(_):
        counts["sleeps"] += 1
        if counts["sleeps"] >= 3:
            raise KeyboardInterrupt

    session.reset, session.step = reset, step
    monkeypatch.setattr(module, "H1Session", lambda *args: session)
    monkeypatch.setattr(module.time, "sleep", sleep)
    module.view(urdf, "mpc", "python", None, "127.0.0.1", 8081, autoplay=autoplay, loop=loop)
    assert counts["steps"] == expected_steps and counts["resets"] == expected_resets
    assert counts["stopped"]


@pytest.mark.parametrize("scenario", ["coordination", "disturbance_recovery"])
def test_different_wbc_mpc_joint_orders_preserve_closed_loop(backend, scenario):
    result = example("joint_order_adapter").run(backend_name(backend), scenario=scenario)
    assert result["max_position_difference_rad"] < 2e-6
    assert result["max_velocity_difference_rad_s"] < 2e-5
    assert result["max_cartesian_difference_m"] < 1e-6
    assert result["final_position_error_m"] < 1e-4
    if scenario == "disturbance_recovery":
        assert result["tracking_replans"] == result["disturbances"] == 2


@pytest.mark.parametrize("mode", ["mpc", "wbc"])
def test_h1_march_alternates_feet_and_settles(backend, mode, tmp_path):
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 asset is not present")
    path = tmp_path / "march.csv"
    result = module.run(motion="march", mode=mode, backend=backend_name(backend), csv_path=path)
    assert not result["fixed_pelvis"] and result["base_model"] == "floor_aligned_kinematic"
    assert result["final_joint_error_rad"] < 1e-3  # Less than 0.06 degrees at the loop seam.
    assert result["max_constraint_violation"] < 1e-6
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    left = np.array([float(row["left_foot_lift_m"]) for row in rows])
    right = np.array([float(row["right_foot_lift_m"]) for row in rows])
    assert left.max() > .09 and right.max() > .09
    # Both feet alternate repeatedly, without dropping below the display floor.
    for start in (100, 200):
        assert left[start:start + 100].max() > .09
        assert right[start:start + 100].max() > .09
    assert min(left.min(), right.min()) > -1e-6
    assert result["max_fore_aft_foot_separation_m"] > .3
    assert np.ptp([float(row["pelvis_z_m"]) for row in rows]) > .02
    for joint in ("torso", "left_shoulder_pitch", "right_shoulder_pitch", "left_elbow", "right_elbow", "left_hip_pitch", "right_hip_pitch"):
        assert np.ptp([float(row[f"q_{joint}_joint"]) for row in rows]) > .1
    velocities = np.array([[float(v) for k, v in row.items() if k.startswith("v_")] for row in rows])
    assert np.max(np.abs(velocities)) <= 4. + 1e-6
    assert np.max(np.abs(np.diff(np.vstack((np.zeros(19), velocities)), axis=0))) <= .48 + 1e-6


def test_h1_amplitude_and_march_feedforward():
    pytest.importorskip("pinocchio")
    module = example("h1_mpc_wbc")
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 asset is not present")
    for invalid in (0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="amplitude"):
            module.H1Session(amplitude=invalid)
    session = module.H1Session(motion="march")
    for stamp in (0., .3, .6, .7, 2., 3., 4., 4.7, 6., 6.3, 6.6, 7., 8.):
        derivative = (session.goal(stamp + 1e-7) - session.goal(stamp - 1e-7)) / 2e-7
        np.testing.assert_allclose(session.goal_velocity(stamp), derivative, atol=2e-6)
