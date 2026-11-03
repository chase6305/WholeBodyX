"""Floating-base numeric contracts; importing them never loads Pinocchio.

Configurations use [translation, quaternion xyzw, scalar joints]. Velocities
use the model tangent order. Contact wrenches are local [force; moment].
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import ulp
from typing import ClassVar, NamedTuple, Protocol

import numpy as np

from .qp import QPResult, QuadraticProgram
from .types import _snapshot


class DynamicsTerms(NamedTuple):
    """One independent model evaluation; nine-item tuple access is retained."""

    mass: np.ndarray
    bias: np.ndarray
    contact_jacobian: np.ndarray
    contact_drift: np.ndarray
    contact_poses: list
    com: np.ndarray
    com_velocity: np.ndarray
    com_drift: np.ndarray
    com_jacobian: np.ndarray


class ContactFormulation(NamedTuple):
    problem: QuadraticProgram
    terms: DynamicsTerms
    contact_rhs: np.ndarray


def _window(stamp, valid_until):
    stamp, valid_until = float(stamp), float(valid_until)
    if (not np.isfinite([stamp, valid_until]).all() or stamp < 0
            or valid_until <= stamp):
        raise ValueError("invalid dynamics validity interval")
    return stamp, valid_until


def _covers(start, valid_until, stamp, dt):
    stamp, dt = float(stamp), float(dt)
    end = stamp + dt
    return (start == stamp and np.isfinite(end) and end > stamp
            and end - valid_until <= min(4 * max(ulp(end), ulp(valid_until)), 1e-6 * dt))


@dataclass(frozen=True)
class DynamicsState:
    q: np.ndarray
    v: np.ndarray
    stamp: float = 0.

    def __post_init__(self):
        q, v = _snapshot(self.q, 1, "dynamics state q"), _snapshot(self.v, 1, "dynamics state v")
        if v.size < 6 or q.size != v.size + 1:
            raise ValueError("floating-base state requires nq=nv+1 and nv>=6")
        if abs(np.linalg.norm(q[3:7]) - 1.) > 1e-8:
            raise ValueError("floating-base state quaternion must be normalized")
        stamp = float(self.stamp)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("dynamics state stamp must be finite and nonnegative")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "stamp", stamp)


@dataclass(frozen=True)
class CoMState:
    """Measured world-frame CoM state, distinct from a control reference."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    stamp: float = 0.

    def __post_init__(self):
        for name in ("position", "velocity", "acceleration"):
            value = _snapshot(getattr(self, name), 1, "CoM state " + name)
            if value.shape != (3,):
                raise ValueError("CoM state requires three-dimensional vectors")
            object.__setattr__(self, name, value)
        stamp = float(self.stamp)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("CoM state stamp must be finite and nonnegative")
        object.__setattr__(self, "stamp", stamp)


@dataclass(frozen=True)
class CoMReference:
    """World-frame position and feedforward, covering one execution interval."""

    position: np.ndarray
    stamp: float
    valid_until: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        for name in ("position", "velocity", "acceleration"):
            value = _snapshot(getattr(self, name), 1, "CoM " + name)
            if value.shape != (3,):
                raise ValueError("CoM reference requires three-dimensional vectors")
            object.__setattr__(self, name, value)
        stamp, end = _window(self.stamp, self.valid_until)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "valid_until", end)

    def covers(self, stamp, dt):
        return _covers(self.stamp, self.valid_until, stamp, dt)


@dataclass(frozen=True)
class TorqueCommand:
    """Actuated-joint torque in SI units and model order, excluding the base."""

    torque: np.ndarray
    stamp: float
    valid_until: float

    def __post_init__(self):
        torque = _snapshot(self.torque, 1, "torque")
        if not torque.size:
            raise ValueError("torque command must contain actuated joints")
        stamp, end = _window(self.stamp, self.valid_until)
        object.__setattr__(self, "torque", torque)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "valid_until", end)

    @property
    def mode(self):
        return "torque"

    def covers(self, stamp, dt):
        return _covers(self.stamp, self.valid_until, stamp, dt)


@dataclass(frozen=True)
class DynamicsControlResult(Mapping):
    """Accepted inverse-dynamics output with legacy dictionary-style reads.

    Arrays own snapshots. Mapping keys match the original ContactWBC.solve()
    dictionary; new callers use attributes and the time-bounded command.
    """

    solver: QPResult
    command: TorqueCommand
    acceleration: np.ndarray
    wrench: np.ndarray
    contact_rhs: np.ndarray
    terms: DynamicsTerms
    dynamics_residual: float
    contact_residual: float

    _KEYS: ClassVar[tuple[str, ...]] = (
        "acceleration", "torque", "wrench", "contact_rhs", "terms", "violation",
        "iterations", "dynamics_residual", "contact_residual",
    )

    def __post_init__(self):
        if not isinstance(self.solver, QPResult) or not self.solver.success:
            raise ValueError("dynamics result requires an accepted solver result")
        decision = _snapshot(self.solver.x, 1, "dynamics solver decision")
        if not decision.size:
            raise ValueError("dynamics solver decision must be nonempty")
        object.__setattr__(self, "solver", replace(self.solver, x=decision))
        if not isinstance(self.command, TorqueCommand):
            raise ValueError("dynamics result requires a TorqueCommand")
        for name in ("acceleration", "wrench", "contact_rhs"):
            object.__setattr__(self, name, _snapshot(getattr(self, name), 1, name))
        nv, nc = self.acceleration.size, self.wrench.size
        if (nv != self.command.torque.size + 6 or not nc or nc % 6
                or self.contact_rhs.shape != (nc,)
                or self.terms.mass.shape != (nv, nv)
                or self.terms.contact_jacobian.shape != (nc, nv)):
            raise ValueError("dynamics result dimensions differ")
        for name in ("dynamics_residual", "contact_residual", "violation"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError("dynamics residuals must be finite and nonnegative")
            if name != "violation":
                object.__setattr__(self, name, value)

    @property
    def torque(self):
        return self.command.torque

    @property
    def violation(self):
        return self.solver.violation

    @property
    def iterations(self):
        return self.solver.iterations

    def __getitem__(self, key):
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._KEYS)

    def __len__(self):
        return len(self._KEYS)


class DynamicsController(Protocol):
    def reset(self) -> None: ...

    def step(self, state: DynamicsState, reference: CoMReference,
             dt: float) -> DynamicsControlResult: ...
