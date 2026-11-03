import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wholebodyx import RobotState, VelocityCommand
from wholebodyx.model import PinocchioModel, PlanarDualArm
from wholebodyx.simulation import JointIntegrator


def check_jacobian(model, q, frame):
    numerical = np.zeros((6, len(q)))
    eps = 1e-6
    for k in range(len(q)):
        d = np.zeros(len(q))
        d[k] = eps
        plus, minus = model.pose(q + d, frame), model.pose(q - d, frame)
        numerical[:3, k] = (plus.translation - minus.translation) / (2 * eps)
        numerical[3:, k] = Rotation.from_matrix(plus.rotation @ minus.rotation.T).as_rotvec() / (2 * eps)
    np.testing.assert_allclose(model.jacobian(q, frame), numerical, atol=1e-8)


@pytest.mark.parametrize("frame", ["left_ee", "right_ee"])
def test_planar_jacobian(frame):
    check_jacobian(PlanarDualArm(), np.array([0.2, 0.6, -0.4, -0.2, -0.6, 0.4]), frame)


@pytest.mark.parametrize("frame,offset", [("left_ee", 0), ("right_ee", 3)])
def test_planar_pose_matches_general_axis_angle_rotation(frame, offset):
    model = PlanarDualArm()
    rng = np.random.default_rng(92)
    configurations = list(rng.uniform(-2.8, 2.8, (30, 6)))
    for angle in (0., np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 1e-12):
        q = np.zeros(6)
        q[offset] = angle
        configurations.append(q)
    for q in configurations:
        pose = model.pose(q, frame)
        theta = np.cumsum(q[offset:offset + 3])
        expected = Rotation.from_rotvec([0, 0, theta[-1]]).as_matrix()
        np.testing.assert_allclose(pose.rotation, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(pose.translation, model.translations(q[None], frame)[0], atol=1e-14)


def test_pose_rotation_validation_matches_existing_tolerances():
    from wholebodyx.model import Pose

    rng = np.random.default_rng(63)
    rotations = list(Rotation.random(30, random_state=rng).as_matrix())
    rotations += [np.diag([1., 1., -1.]), np.zeros((3, 3))]
    for delta in (1e-10, 4e-9, 5e-9, 6e-9, 1e-8, 1e-6):
        rotations += [np.diag([1 + delta, 1., 1.]), np.eye(3) + delta * np.ones((3, 3))]
    for rotation in rotations:
        accepted = (np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                    and np.isclose(np.linalg.det(rotation), 1., atol=1e-8))
        if accepted:
            Pose(np.zeros(3), rotation)
        else:
            with pytest.raises(ValueError, match="SO\\(3\\)"):
                Pose(np.zeros(3), rotation)
    for rotation in (np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.eye(2)):
        with pytest.raises(ValueError):
            Pose(np.zeros(3), rotation)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        Pose(np.zeros(3), np.full((3, 3), 1e308))


@pytest.mark.parametrize("frame", ["left_ee", "right_ee"])
def test_batch_translations_match_pose(frame):
    model = PlanarDualArm()
    q = np.random.default_rng(18).uniform(-2.8, 2.8, (50, 6))
    # Include non-contiguous inputs, as used by trajectory views.
    q = q[::2]
    expected = np.array([model.pose(row, frame).translation for row in q])
    actual = model.translations(q, frame)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert not np.shares_memory(actual, q)
    assert model.translations(np.empty((0, 6)), frame).shape == (0, 3)


@pytest.mark.parametrize("q,frame", [
    (np.zeros(6), "left_ee"), (np.zeros((2, 5)), "left_ee"),
    (np.full((2, 6), np.nan), "left_ee"), (np.zeros((2, 6)), "missing"),
])
def test_batch_translations_reject_invalid_inputs(q, frame):
    with pytest.raises(ValueError):
        PlanarDualArm().translations(q, frame)


def test_urdf_jacobian(tmp_path):
    pytest.importorskip("pinocchio")
    path = tmp_path / "arm.urdf"
    path.write_text('''<robot name="arm">
      <link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="10" velocity="1"/></joint>
      <joint name="tool" type="fixed"><parent link="arm"/><child link="tip"/>
        <origin xyz="1 0 0"/></joint></robot>''')
    model = PinocchioModel(path)
    assert model.joint_names == ("hinge",)
    check_jacobian(model, np.array([0.4]), "tip")
    # Interleave frame and configuration changes with Jacobian calls. Selective
    # frame updates must never return placements left over from another q.
    expected_data = model._model.createData()
    saved = model.pose([.1], "tip")
    saved_translation = saved.translation.copy()
    for q, frame in (([.2], "tip"), ([-.7], "arm"), ([1.1], "tip"), ([0.], "base")):
        model.jacobian(q, "tip")
        actual = model.pose(q, frame)
        model._pin.forwardKinematics(model._model, expected_data, np.array(q))
        model._pin.updateFramePlacements(model._model, expected_data)
        expected = expected_data.oMf[model._model.getFrameId(frame)]
        np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-14)
        np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-14)
    np.testing.assert_array_equal(saved.translation, saved_translation)
    # Name-cache hits must still validate every configuration, and misses must
    # never silently resolve to a previously queried frame.
    for invalid in ([np.nan], [0., 0.]):
        with pytest.raises(ValueError):
            model.pose(invalid, "tip")
        with pytest.raises(ValueError):
            model.jacobian(invalid, "tip")
    cached_count = len(model._frame_ids)
    for index in range(20):
        with pytest.raises(ValueError, match="unknown frame"):
            model.pose([0.], f"missing_{index}")
    assert len(model._frame_ids) == cached_count


def test_backend_rejects_stale_and_invalid_commands():
    model = PlanarDualArm()
    plant = JointIntegrator(model.limits, RobotState(np.zeros(6), np.zeros(6), 1.0))
    with pytest.raises(ValueError):
        plant.write(VelocityCommand(np.zeros(6), 0.0, 0.1), 0.02)
    with pytest.raises(ValueError):
        plant.write(VelocityCommand(np.full(6, 2), 1.0, 1.1), 0.02)
    np.testing.assert_array_equal(plant.read().q, np.zeros(6))


def test_integrator_keeps_time_progress_with_numpy_period_at_large_clock():
    model = PlanarDualArm()
    state = RobotState(np.zeros(6), np.zeros(6), np.float32(1e8))
    plant = JointIntegrator(model.limits, state)
    dt = np.float32(.005)
    plant.write(VelocityCommand(np.zeros(6), state.stamp, state.stamp + float(dt)), dt)
    assert plant.read().stamp == state.stamp + float(dt)
    assert plant.read().stamp > state.stamp


def test_urdf_jacobian_result_isolated_from_workspace_and_borrowed_views(monkeypatch):
    pytest.importorskip("pinocchio")
    path = Path(__file__).resolve().parents[2] / "examples/models/planar_arm.urdf"
    model = PinocchioModel(path)
    frame = model._model.frames[-1].name
    q = np.zeros(model._model.nq)
    first = model.jacobian(q, frame)
    snapshot = first.copy()
    model.jacobian(q + .2, frame)
    np.testing.assert_array_equal(first, snapshot)
    first[:] = 99
    np.testing.assert_array_equal(model.jacobian(q, frame), snapshot)
    borrowed = np.arange(12 * model._model.nv, dtype=float).reshape(6, -1)[:, ::2]
    original = borrowed.copy()
    monkeypatch.setattr(model._pin, "computeFrameJacobian", lambda *args: borrowed)
    returned = model.jacobian(q, frame)
    assert returned.shape == (6, model._model.nv)
    assert not np.shares_memory(returned, borrowed)
    returned[:] = -1
    np.testing.assert_array_equal(borrowed, original)


@pytest.mark.parametrize("mpc", [False, True])
def test_dual_arm_closed_loop(backend, mpc):
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/python/dual_arm_tracking.py"
    spec = importlib.util.spec_from_file_location("dual_arm_tracking", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    name = "native" if backend[0].__name__ == "NativeWBC" else "python"
    result = example.run(name, mpc)
    assert result["final_position_error_m"] < 1e-4
    assert result["joint_error_rad"] < 1e-3
    assert result["max_constraint_violation"] <= 1e-6
