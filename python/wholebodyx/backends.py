"""Explicit, per-application factories for independent WBC and MPC libraries."""

from dataclasses import dataclass
from typing import Callable

from .mpc import JointMPC
from .types import Controller, JointLimits, Planner
from .wbc import KinematicWBC


def _native_controller(limits):
    from .native import NativeWBC
    return NativeWBC(limits)


def _native_planner(limits):
    from .native import NativeMPC
    return NativeMPC(limits)


def _sparse_planner(limits):
    from .sparse_mpc import SparseJointMPC
    return SparseJointMPC(limits)


def _native_sparse_planner(limits):
    from .native import NativeSparseMPC
    return NativeSparseMPC(limits)


def _check_native():
    from .native import _extension
    _extension()


def _check_native_sparse():
    from .native import _sparse_extension
    _sparse_extension()


@dataclass(frozen=True)
class BackendStatus:
    name: str
    controller: bool
    planner: bool
    available: bool
    detail: str = ""


@dataclass(frozen=True)
class _Backend:
    controller: Callable[[JointLimits], Controller] | None
    planner: Callable[[JointLimits], Planner] | None
    check: Callable[[], None] | None


class BackendRegistry:
    """Factories must create a fresh, single-owner instance on each call.

    Registration never imports an optional library or executes a factory.
    Availability checks are explicit; they establish loadability, not solver quality.
    """

    def __init__(self):
        self._backends = {}
        self.register("python", controller=KinematicWBC, planner=JointMPC)
        self.register("native", controller=_native_controller, planner=_native_planner, check=_check_native)
        self.register("sparse", planner=_sparse_planner)
        self.register("native-sparse", planner=_native_sparse_planner, check=_check_native_sparse)

    def register(self, name, *, controller=None, planner=None, check=None):
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("backend name must be a nonempty string without surrounding whitespace")
        if name in self._backends:
            raise ValueError(f"backend already registered: {name}")
        if controller is None and planner is None:
            raise ValueError("register at least one controller or planner factory")
        if any(value is not None and not callable(value) for value in (controller, planner, check)):
            raise TypeError("backend factories and check must be callable")
        self._backends[name] = _Backend(controller, planner, check)
        return self

    def copy(self):
        result = object.__new__(BackendRegistry)
        result._backends = self._backends.copy()
        return result

    def status(self):
        results = []
        for name, backend in self._backends.items():
            detail = ""
            try:
                if backend.check is not None:
                    backend.check()
            except Exception as exc:
                # Diagnostics must retain failures from third-party import checks.
                detail = f"{type(exc).__name__}: {exc}"
            results.append(BackendStatus(name, backend.controller is not None,
                                         backend.planner is not None, not detail, detail))
        return tuple(results)

    def names(self, role, *, available_only=False):
        if role not in ("controller", "planner"):
            raise ValueError("role must be controller or planner")
        if available_only:
            return tuple(s.name for s in self.status() if s.available and getattr(s, role))
        return tuple(name for name, backend in self._backends.items() if getattr(backend, role) is not None)

    def _create(self, name, limits, role):
        if name not in self._backends:
            raise ValueError(f"unknown backend: {name}")
        factory = getattr(self._backends[name], role)
        if factory is None:
            raise ValueError(f"backend {name!r} does not provide a {role}")
        instance = factory(limits)
        methods = ("reset", "step" if role == "controller" else "solve")
        if not all(callable(getattr(instance, method, None)) for method in methods):
            raise TypeError(f"backend {name!r} must implement {role} methods: {', '.join(methods)}")
        return instance

    def create_controller(self, name, limits) -> Controller:
        return self._create(name, limits, "controller")

    def create_planner(self, name, limits) -> Planner:
        return self._create(name, limits, "planner")
