"""Small reproducible task recipes shared by headless examples and the viewer."""

from dataclasses import dataclass

import numpy as np

from .model import Pose
from .tasks import FramePoseTask, PostureTask
from .types import LinearTask

INITIAL_Q = (0.2, 0.6, -0.4, -0.2, -0.6, 0.4)
DEFAULT_GOAL = (0.5, 0.4, -0.3, -0.5, -0.4, 0.3)
FRAMES = ("left_ee", "right_ee")


@dataclass(frozen=True)
class Scenario:
    description: str
    mode: str | None
    steps: int


SCENARIOS = {
    "tracking": Scenario("双手静态位姿目标；可比较 WBC 与 MPC + WBC。", None, 250),
    "cartesian_circle": Scenario("双手圆轨迹，保持各自末端朝向，叠加速度前馈。", "WBC", 400),
    "coordination": Scenario("移动双手中心，同时保持世界坐标系下的相对位置。", "WBC", 400),
    "mpc_retargeting": Scenario("在 2 s、4 s 更换关节目标，立即重规划并平滑执行。", "MPC + WBC", 400),
    "disturbance_recovery": Scenario("在 2.06 s、4.14 s 注入关节位置偏移，检测预测偏差并重规划。", "MPC + WBC", 400),
}

# Kinematic state offsets, not impulses or external forces.
CONFIGURATION_OFFSETS = (
    (2.06, (0.12, -0.08, 0.04, -0.12, 0.08, -0.04)),
    (4.14, (-0.10, 0.07, -0.03, 0.10, -0.07, 0.03)),
)


@dataclass(frozen=True)
class Reference:
    q: np.ndarray
    poses: tuple[Pose, Pose]
    twists: tuple[np.ndarray, np.ndarray]


def reference_configuration(model, scenario, goal, stamp):
    """Joint target schedule, without computing Cartesian poses or twists."""
    q = np.array(goal, dtype=float, copy=True)
    if scenario == "mpc_retargeting":
        if stamp >= 4.0 - 1e-9:
            q += np.array([0.15, -0.1, 0.05, -0.15, 0.1, -0.05])
        elif stamp >= 2.0 - 1e-9:
            q += np.array([-0.2, 0.2, 0.1, 0.2, -0.2, -0.1])
        # A GUI goal may already be near a bound. Schedule waypoints stay valid.
        q = np.clip(q, model.limits.lower, model.limits.upper)
    return q


def reference(model, scenario, goal, stamp):
    """Eight-second Cartesian circles or a finite, two-event target schedule."""
    q = reference_configuration(model, scenario, goal, stamp)
    poses = tuple(model.pose(q, frame) for frame in FRAMES)
    twists = (np.zeros(6), np.zeros(6))
    if scenario in ("cartesian_circle", "coordination"):
        omega, radius = 2 * np.pi / 8, 0.025
        angle = omega * stamp
        delta = radius * np.array([np.cos(angle) - 1, np.sin(angle), 0])
        velocity = radius * omega * np.array([-np.sin(angle), np.cos(angle), 0])
        signs = (1, -1) if scenario == "cartesian_circle" else (1, 1)
        poses = tuple(Pose(p.translation + delta * [1, sign, 1], p.rotation)
                      for p, sign in zip(poses, signs))
        twists = tuple(np.r_[velocity * [1, sign, 1], np.zeros(3)] for sign in signs)
    return Reference(q, poses, twists)


def wbc_tasks(model, state, scenario, target):
    if scenario == "coordination":
        positions = [model.pose(state.q, frame).translation for frame in FRAMES]
        center = 0.5 * (positions[0] + positions[1])
        center_goal = 0.5 * (target.poses[0].translation + target.poses[1].translation)
        jacobians = [model.jacobian(state.q, frame)[:3] for frame in FRAMES]
        center_jacobian = 0.5 * (jacobians[0] + jacobians[1])
        relative_error = ((target.poses[1].translation - target.poses[0].translation)
                          - (positions[1] - positions[0]))
        return [
            LinearTask(center_jacobian, target.twists[0][:3] + 4 * (center_goal - center)),
            # Both tasks use the same sampled geometry; do not evaluate the
            # model again for the relative-position objective.
            LinearTask(jacobians[1] - jacobians[0], 4 * relative_error, weight=20),
            PostureTask(target.q, weight=1e-5).linearize(model, state),
        ]
    return [FramePoseTask(frame, pose, feedforward=twist).linearize(model, state)
            for frame, pose, twist in zip(FRAMES, target.poses, target.twists)] + [
        PostureTask(target.q, weight=1e-5 if scenario == "cartesian_circle" else 1e-3).linearize(model, state)
    ]
