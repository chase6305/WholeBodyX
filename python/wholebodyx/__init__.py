"""WholeBodyX Python API. Native extensions are loaded only on explicit request."""

from importlib import import_module as _import_module

from .mpc import JointMPC, JointPlan, PlanResult
from .qp import OSQPSolver, QPResult, QPSolver, QuadraticProgram
from .runtime import MPCReferenceManager, ReferenceUpdate
from .types import Controller, JointLimits, LinearTask, Planner, RobotState, VelocityCommand
from .wbc import ControlResult, KinematicWBC
from .backends import BackendRegistry, BackendStatus
from .joint_order import JointOrder, ReorderedController, ReorderedPlanner
from .hardware import (CommandGate, CycleResult, ExecutionDeadlineError, ModelOrderRobotBackend,
                       VelocityControlLoop, VelocityRobotBackend)
from .dynamics_types import (CoMReference, CoMState, ContactFormulation, DynamicsControlResult,
                             DynamicsController, DynamicsState, DynamicsTerms, TorqueCommand)

# Keep optional formulations out of the basic velocity API's import path.
# Values are cached after first access, preserving normal class identity and
# from-import behavior without loading contact geometry for every application.
_LAZY_EXPORTS = {
    "ProcessMPCWorker": ".process_mpc", "PlanningToken": ".process_mpc", "PlanningReply": ".process_mpc",
    "AsyncMPCReferenceManager": ".async_runtime",
    "VelocitySession": ".execution", "VelocityStep": ".execution",
    "ContactSpec": ".contact",
    "SparseQuadraticProgram": ".sparse_qp", "SparseJointMPC": ".sparse_mpc",
    "CoMReferenceManager": ".centroidal_runtime", "CoMReferenceUpdate": ".centroidal_runtime",
    "CoMPlan": ".centroidal", "CoMPlanner": ".centroidal", "CoMPreviewMPC": ".centroidal",
    "TorqueCommandGate": ".torque", "TorqueControlLoop": ".torque",
    "TorqueCycleResult": ".torque", "TorqueRobotBackend": ".torque",
    "FixedContactSimulator": ".dynamics_simulation",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "ProcessMPCWorker", "PlanningToken", "PlanningReply", "AsyncMPCReferenceManager",
    "BackendRegistry", "BackendStatus",
    "CoMReference", "CoMState", "CoMReferenceManager", "CoMReferenceUpdate",
    "CoMPlan", "CoMPlanner", "CoMPreviewMPC",
    "DynamicsControlResult", "DynamicsController", "DynamicsState", "TorqueCommand",
    "DynamicsTerms", "ContactFormulation",
    "VelocitySession", "VelocityStep",
    "ContactSpec",
    "SparseQuadraticProgram", "SparseJointMPC",
    "TorqueCommandGate", "TorqueControlLoop", "TorqueCycleResult", "TorqueRobotBackend",
    "FixedContactSimulator",
    "JointOrder", "ReorderedController", "ReorderedPlanner",
    "CommandGate", "CycleResult", "ExecutionDeadlineError", "ModelOrderRobotBackend",
    "VelocityControlLoop", "VelocityRobotBackend",
    "ControlResult", "Controller", "JointLimits", "JointMPC", "JointPlan",
    "KinematicWBC", "LinearTask", "OSQPSolver", "PlanResult", "Planner",
    "MPCReferenceManager", "ReferenceUpdate", "QPResult", "QPSolver", "QuadraticProgram", "RobotState", "VelocityCommand",
]
