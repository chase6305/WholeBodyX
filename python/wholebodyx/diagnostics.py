"""On-demand dependency checks and tiny control solves; never opens a viewer."""

import importlib
from importlib.metadata import PackageNotFoundError, version
import platform
import sys

import numpy as np

from .backends import BackendRegistry
from .runtime import MPCReferenceManager
from .simulation import JointIntegrator
from .types import JointLimits, LinearTask, RobotState


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _solve_check(registry, name):
    limits = JointLimits([-2], [2], [1], [4])
    state = RobotState([0], [0])
    planner_name = name if name in registry.names("planner") else "python"
    controller_name = name if name in registry.names("controller") else "python"
    planner = registry.create_planner(planner_name, limits)
    reference = MPCReferenceManager(planner, limits).update(state, [.3], .02)
    if not reference.success:
        raise RuntimeError(f"MPC failed: {reference.solver.detail}")
    controller = registry.create_controller(controller_name, limits)
    result = controller.step(state, [LinearTask([[1]], reference.velocity)], .02)
    if not result.solver.success or result.command is None:
        raise RuntimeError(f"WBC failed: {result.solver.detail}")
    plant = JointIntegrator(limits, state)
    plant.write(result.command, .02)
    if not np.isfinite(plant.read().q).all() or plant.read().q[0] <= 0:
        raise RuntimeError("WBC did not move toward the test goal")
    return planner_name, controller_name


def diagnose():
    """Missing optional features are reported; the caller chooses requirements."""
    registry = BackendRegistry()
    checks = {}
    for status in registry.status():
        detail = status.detail
        check = "MPC + WBC solve"
        if status.available:
            try:
                planner, controller = _solve_check(registry, status.name)
                check = f"{planner} MPC + {controller} WBC solve"
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
        checks[status.name] = {"available": not detail, "check": check, "detail": detail}
    for name, module in (("urdf", "pinocchio"), ("viz", "viser.uplot")):
        detail = ""
        try:
            importlib.import_module(module)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
        checks[name] = {"available": not detail, "check": f"import {module}", "detail": detail}
    return {
        "schema_version": 1,
        "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
        "versions": {name: _version(name) for name in
                     ("wholebodyx", "wholebodyx-native", "numpy", "scipy", "osqp", "pin", "viser")},
        "checks": checks,
    }
