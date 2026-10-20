"""Fixed-base contracts: SI units, model joint order, one monotonic clock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .mpc import PlanResult
    from .wbc import ControlResult


def array(value, ndim, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    result.setflags(write=False)
    return result


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _braking_speed(distance, acceleration, dt):
    """Largest velocity whose discrete maximum-braking rollout fits distance.

    A velocity command is integrated for dt before the next slew-limited
    command. For k positive intervals, travel is
    dt * (k * velocity - acceleration * dt * k * (k - 1) / 2).
    Continuous sqrt(2*a*distance) bounds omit that first held interval.
    """
    distance = np.maximum(distance, 0.)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        slew = acceleration * dt
        root = np.sqrt(2.) * (np.sqrt(distance) / np.sqrt(acceleration)) / dt
        count = np.maximum(1., np.ceil(np.hypot(.5, root) - .5))
        speed = (distance / count) / dt + .5 * (count - 1.) * slew
        # At enormous interval counts the discrete correction is below float
        # resolution. Keep extreme finite inputs from producing inf/inf NaNs.
        continuous = np.sqrt(2.) * np.sqrt(distance) * np.sqrt(acceleration)
        speed = np.where(np.isfinite(count), speed, continuous)
        speed = np.where(count == 1., distance / dt, speed)
        speed = np.where(slew == 0., 0., speed)
        return np.minimum(speed, distance / dt)


@dataclass(frozen=True)
class RobotState:
    q: np.ndarray
    v: np.ndarray
    stamp: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "q", array(self.q, 1, "q"))
        object.__setattr__(self, "v", array(self.v, 1, "v"))
        if self.q.size == 0 or self.q.shape != self.v.shape:
            raise ValueError("fixed-base q and v must have equal, nonzero dimensions")
        stamp = float(self.stamp)
        if not np.isfinite(stamp) or self.stamp < 0:
            raise ValueError("stamp must be finite and nonnegative")
        object.__setattr__(self, "stamp", stamp)


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray | None = None

    def __post_init__(self):
        for name in ("lower", "upper", "velocity"):
            object.__setattr__(self, name, array(getattr(self, name), 1, name))
        if not self.lower.size or not (self.lower.shape == self.upper.shape == self.velocity.shape):
            raise ValueError("joint limit dimensions must match")
        if np.any(self.lower >= self.upper) or np.any(self.velocity <= 0):
            raise ValueError("require lower < upper and positive velocity limits")
        if self.acceleration is not None:
            a = array(self.acceleration, 1, "acceleration")
            if a.shape != self.lower.shape or np.any(a <= 0):
                raise ValueError("acceleration limits must be positive and match joint dimensions")
            object.__setattr__(self, "acceleration", a)

    def check_state(self, state):
        if state.q.shape != self.lower.shape:
            raise ValueError("state and limits dimensions differ")

    def velocity_bounds(self, state, dt):
        self.check_state(state)
        positive(dt, "dt")
        lo = np.maximum(-self.velocity, (self.lower - state.q) / dt)
        hi = np.minimum(self.velocity, (self.upper - state.q) / dt)
        if self.acceleration is not None:
            slew = self.acceleration * dt
            np.maximum(lo, state.v - slew, out=lo)
            np.minimum(hi, state.v + slew, out=hi)
            # This upper bound on stopping travel avoids square roots for
            # joints comfortably inside their limits. Only nearby joints need
            # the exact discrete envelope.
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                travel = self.velocity * (self.velocity / self.acceleration + dt)
            below = state.q - self.lower
            above = self.upper - state.q
            near_lower = (lo < 0) & (below < travel)
            near_upper = (hi > 0) & (above < travel)
            if np.any(near_lower):
                lo[near_lower] = np.maximum(lo[near_lower], -_braking_speed(
                    below[near_lower], self.acceleration[near_lower], dt))
            if np.any(near_upper):
                hi[near_upper] = np.minimum(hi[near_upper], _braking_speed(
                    above[near_upper], self.acceleration[near_upper], dt))
            # At the braking boundary the interval can collapse to one value.
            # Subtracting joint positions can invert it by a few roundoff ULPs.
            inverted = lo > hi
            if np.any(inverted):
                scale = np.maximum(np.maximum(np.abs(self.lower), np.abs(self.upper)), np.abs(state.q))
                close = inverted & np.isfinite(lo) & np.isfinite(hi) & (lo - hi <= (16 * np.finfo(float).eps * scale) / dt)
                midpoint = .5 * lo[close] + .5 * hi[close]
                lo[close] = hi[close] = midpoint
        return lo, hi


@dataclass(frozen=True)
class LinearTask:
    jacobian: np.ndarray
    target_velocity: np.ndarray
    weight: np.ndarray | float = 1.0

    def __post_init__(self):
        j = array(self.jacobian, 2, "jacobian")
        b = array(self.target_velocity, 1, "target_velocity")
        w = np.asarray(self.weight, dtype=float)
        if w.ndim == 0:
            scalar = w.item()
            if not np.isfinite(scalar):
                raise ValueError("weight must be finite")
            if scalar < 0:
                raise ValueError("task weights must be nonnegative")
            # full() already owns its buffer: do not copy it through array().
            w = np.full(b.size, scalar)
            w.setflags(write=False)
        else:
            w = array(w, 1, "weight")
            if np.any(w < 0):
                raise ValueError("task weights must be nonnegative")
        if j.shape[0] != b.size or w.shape != b.shape or not b.size or not j.shape[1]:
            raise ValueError("task dimensions must match and be nonzero")
        object.__setattr__(self, "jacobian", j)
        object.__setattr__(self, "target_velocity", b)
        object.__setattr__(self, "weight", w)


@dataclass(frozen=True)
class VelocityCommand:
    velocity: np.ndarray
    stamp: float
    valid_until: float

    def __post_init__(self):
        object.__setattr__(self, "stamp", float(self.stamp))
        object.__setattr__(self, "valid_until", float(self.valid_until))
        object.__setattr__(self, "velocity", array(self.velocity, 1, "velocity"))
        if not self.velocity.size or not np.isfinite([self.stamp, self.valid_until]).all():
            raise ValueError("invalid command")
        if self.stamp < 0 or self.valid_until <= self.stamp:
            raise ValueError("invalid command validity interval")

    @property
    def mode(self):
        return "velocity"


class Controller(Protocol):
    def reset(self) -> None: ...

    def step(self, state: RobotState, tasks: list[LinearTask], dt: float) -> ControlResult: ...


class Planner(Protocol):
    def reset(self) -> None: ...

    def solve(self, state: RobotState, goal: np.ndarray) -> PlanResult: ...
