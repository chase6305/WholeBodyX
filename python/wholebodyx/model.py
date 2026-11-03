"""World-aligned [linear; angular] Jacobians and fixed-base model adapters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import JointLimits, array


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        translation = array(self.translation, 1, "translation")
        rotation = array(self.rotation, 2, "rotation")
        if translation.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("pose requires 3-D translation and 3x3 rotation")
        # Negated <= checks also reject nonfinite products of finite inputs.
        if (
            not np.max(np.abs(rotation.T @ rotation - np.eye(3))) <= 1e-8
            or not abs(np.linalg.det(rotation) - 1.0) <= 1e-8 + 1e-5
        ):
            raise ValueError("rotation must belong to SO(3)")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation", rotation)


class RobotModel(Protocol):
    joint_names: tuple[str, ...]
    limits: JointLimits

    def pose(self, q, frame: str) -> Pose: ...

    def jacobian(self, q, frame: str) -> np.ndarray: ...


class PlanarDualArm:
    """Analytic 2 x 3R test fixture, not a humanoid or a dynamics simulator."""

    joint_names = tuple(f"{side}_{j}" for side in ("left", "right") for j in range(3))

    def __init__(self):
        self.lengths = np.array([0.4, 0.3, 0.2])
        self.limits = JointLimits(np.full(6, -2.8), np.full(6, 2.8), np.ones(6), np.full(6, 4.0))

    def _chain(self, q, frame):
        q = array(q, 1, "q")
        if q.size != 6:
            raise ValueError("planar dual arm needs six joints")
        if frame not in ("left_ee", "right_ee"):
            raise ValueError(f"unknown frame: {frame}")
        offset = 0 if frame == "left_ee" else 3
        return np.cumsum(q[offset : offset + 3]), offset

    def pose(self, q, frame):
        theta, offset = self._chain(q, frame)
        cosine, sine = np.cos(theta), np.sin(theta)
        translation = np.array(
            [self.lengths @ cosine, self.lengths @ sine + (0.4 if offset == 0 else -0.4), 0.0]
        )
        # All joints rotate around world Z. Reuse the terminal sine/cosine
        # already evaluated for translation instead of a general rotvec conversion.
        terminal_cosine, terminal_sine = cosine[-1], sine[-1]
        return Pose(
            translation,
            [[terminal_cosine, -terminal_sine, 0], [terminal_sine, terminal_cosine, 0], [0, 0, 1]],
        )

    def translations(self, configurations, frame):
        """Batch end-effector positions for an (N, 6) trajectory, including N=0.

        Avoid constructing and validating rotations when only positions are
        needed. The returned (N, 3) array owns its storage.
        """
        q = array(configurations, 2, "configurations")
        if q.shape[1] != 6:
            raise ValueError("planar dual arm needs six joints")
        if frame not in ("left_ee", "right_ee"):
            raise ValueError(f"unknown frame: {frame}")
        offset = 0 if frame == "left_ee" else 3
        theta = np.cumsum(q[:, offset : offset + 3], axis=1)
        positions = np.zeros((len(q), 3))
        positions[:, 0] = np.cos(theta) @ self.lengths
        positions[:, 1] = np.sin(theta) @ self.lengths + (0.4 if offset == 0 else -0.4)
        return positions

    def jacobian(self, q, frame):
        theta, offset = self._chain(q, frame)
        jacobian = np.zeros((6, 6))
        jacobian[0, offset : offset + 3] = -np.cumsum((self.lengths * np.sin(theta))[::-1])[::-1]
        jacobian[1, offset : offset + 3] = np.cumsum((self.lengths * np.cos(theta))[::-1])[::-1]
        jacobian[5, offset : offset + 3] = 1
        return jacobian


class PinocchioModel:
    """Optional URDF adapter. Each instance owns its Pinocchio Data workspace."""

    def __init__(self, urdf):
        import pinocchio as pin

        self._pin = pin
        self._model = pin.buildModelFromUrdf(str(Path(urdf).resolve(strict=True)))
        if self._model.nq != self._model.nv or any(
            j.nq != 1 or j.nv != 1 for j in self._model.joints[1:]
        ):
            raise ValueError(
                "initial URDF adapter supports fixed-base scalar Euclidean joints only"
            )
        self._data = self._model.createData()
        # The URDF topology is fixed for this adapter's lifetime. Cache only
        # successful name resolutions; invalid user input must not grow it.
        self._frame_ids = {}
        self.joint_names = tuple(self._model.names[1:])
        self.limits = JointLimits(
            self._model.lowerPositionLimit,
            self._model.upperPositionLimit,
            self._model.velocityLimit,
        )

    def _prepare(self, q, frame):
        q = array(q, 1, "q")
        if q.size != self._model.nq:
            raise ValueError("configuration dimensions differ")
        frame_id = self._frame_ids.get(frame)
        if frame_id is None:
            if not self._model.existFrame(frame):
                raise ValueError(f"unknown frame: {frame}")
            frame_id = self._model.getFrameId(frame)
            self._frame_ids[frame] = frame_id
        return q, frame_id

    def pose(self, q, frame):
        q, frame_id = self._prepare(q, frame)
        self._pin.forwardKinematics(self._model, self._data, q)
        # Only the requested frame is consumed. Joint FK remains current;
        # avoid refreshing every auxiliary frame in a large URDF.
        placement = self._pin.updateFramePlacement(self._model, self._data, frame_id)
        return Pose(placement.translation, placement.rotation)

    def jacobian(self, q, frame):
        q, frame_id = self._prepare(q, frame)
        return np.array(
            self._pin.computeFrameJacobian(
                self._model, self._data, q, frame_id, self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            ),
            copy=True,
        ).reshape(6, self._model.nv)
