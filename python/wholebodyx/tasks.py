"""Task descriptions linearized into backend-neutral least-squares terms."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .types import LinearTask, array, positive


@dataclass
class PostureTask:
    target: np.ndarray
    gain: float = 4.0
    weight: float = 1.0
    feedforward: np.ndarray | None = None

    def linearize(self, model, state):
        target = array(self.target, 1, "posture target")
        if target.shape != state.q.shape:
            raise ValueError("posture target dimensions differ")
        velocity = positive(self.gain, "gain") * (target - state.q)
        if self.feedforward is not None:
            ff = array(self.feedforward, 1, "posture feedforward")
            if ff.shape != target.shape:
                raise ValueError("feedforward dimensions differ")
            velocity += ff
        return LinearTask(np.eye(state.q.size), velocity, self.weight)


@dataclass
class FramePositionTask:
    frame: str
    target: np.ndarray
    gain: float = 4.0
    weight: float = 1.0

    def linearize(self, model, state):
        target = array(self.target, 1, "position target")
        if target.shape != (3,):
            raise ValueError("position target must have three elements")
        error = target - model.pose(state.q, self.frame).translation
        return LinearTask(model.jacobian(state.q, self.frame)[:3], positive(self.gain, "gain") * error, self.weight)


@dataclass
class FramePoseTask:
    frame: str
    target: object  # model.Pose; kept as a value object at the task boundary.
    gain: float = 4.0
    weight: np.ndarray | float = 1.0
    feedforward: np.ndarray | None = None  # World-aligned [linear; angular] twist.

    def linearize(self, model, state):
        current = model.pose(state.q, self.frame)
        angular = Rotation.from_matrix(self.target.rotation @ current.rotation.T).as_rotvec()
        error = np.concatenate((self.target.translation - current.translation, angular))
        velocity = positive(self.gain, "gain") * error
        if self.feedforward is not None:
            ff = array(self.feedforward, 1, "pose feedforward")
            if ff.shape != (6,):
                raise ValueError("pose feedforward must have six elements")
            velocity += ff
        return LinearTask(model.jacobian(state.q, self.frame), velocity, self.weight)


@dataclass
class RelativePositionTask:
    """Track p(second) - p(first), expressed in the world frame.

    This is a soft translational task, not a rigid grasp or collision constraint.
    """

    first: str
    second: str
    target: np.ndarray
    gain: float = 4.0
    weight: float = 1.0

    def linearize(self, model, state):
        target = array(self.target, 1, "relative position target")
        if target.shape != (3,):
            raise ValueError("relative position target must have three elements")
        relative = model.pose(state.q, self.second).translation - model.pose(state.q, self.first).translation
        jacobian = model.jacobian(state.q, self.second)[:3] - model.jacobian(state.q, self.first)[:3]
        return LinearTask(jacobian, positive(self.gain, "gain") * (target - relative), self.weight)
