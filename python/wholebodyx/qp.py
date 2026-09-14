"""Backend-independent convex QP and an OSQP adapter."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import osqp
from scipy import sparse

from .types import array


@dataclass(frozen=True)
class QuadraticProgram:
    hessian: np.ndarray
    gradient: np.ndarray
    constraint: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for name, ndim in (("hessian", 2), ("constraint", 2)):
            object.__setattr__(self, name, array(getattr(self, name), ndim, name))
        self._validate_vectors()
        if not np.allclose(self.hessian, self.hessian.T, rtol=0, atol=1e-10):
            raise ValueError("QP Hessian must be symmetric")
        try:
            # WBC regularization and MPC control cost normally make H positive
            # definite. Keep the eigenvalue fallback for singular PSD inputs
            # and the existing absolute tolerance for roundoff.
            np.linalg.cholesky(self.hessian)
        except np.linalg.LinAlgError:
            if np.linalg.eigvalsh(self.hessian)[0] < -1e-10:
                raise ValueError("QP Hessian must be positive semidefinite")

    def _validate_vectors(self):
        object.__setattr__(self, "gradient", array(self.gradient, 1, "gradient"))
        for name in ("lower", "upper"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or np.isnan(value).any():
                raise ValueError("bounds must be vectors without NaNs")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        n, m = self.gradient.size, self.lower.size
        if not n or self.hessian.shape != (n, n) or self.constraint.shape != (m, n) or self.upper.shape != (m,):
            raise ValueError("QP dimensions do not match")


def _immutable(value):
    """Bytes-backed arrays cannot be made writable via setflags()."""
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _immutable_matrix_key(value):
    """Retain immutable storage and describe its view, not just array identity.

    A read-only flag alone is insufficient: an owning ndarray can be made
    writable again. Only bytes-backed arrays qualify for constant-time reuse.
    """
    base = value
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        return None
    return (base, value.shape, value.strides, value.dtype.str,
            value.__array_interface__["data"][0])


def _same_matrix(snapshot, value, previous_key, key):
    if (key is not None and previous_key is not None
            and key[0] is previous_key[0] and key[1:] == previous_key[1:]):
        return True
    return np.array_equal(snapshot, value)


def _upper_csc(hessian):
    # H is already dense. Avoid CSC -> COO -> CSC in scipy.sparse.triu.
    return sparse.csc_matrix(np.triu(hessian))


class _FixedQP:
    """Validate a fixed H/A once; each instance still validates dynamic vectors."""

    def __init__(self, hessian, constraint):
        p = QuadraticProgram(hessian, np.zeros(len(hessian)), constraint,
                             np.zeros(len(constraint)), np.zeros(len(constraint)))
        self._hessian, self._constraint = _immutable(p.hessian), _immutable(p.constraint)

    def problem(self, gradient, lower, upper):
        p = object.__new__(QuadraticProgram)
        # Share immutable buffers, not ndarray metadata (shape/dtype/strides).
        for name, value in (("hessian", self._hessian.view()), ("constraint", self._constraint.view()),
                            ("gradient", gradient), ("lower", lower), ("upper", upper)):
            object.__setattr__(p, name, value)
        p._validate_vectors()
        return p


@dataclass(frozen=True)
class QPResult:
    status: str
    x: np.ndarray | None = None
    iterations: int = 0
    violation: float = float("inf")
    detail: str = ""

    @property
    def success(self):
        return self.status == "solved" and self.x is not None


class QPSolver(Protocol):
    def reset(self) -> None: ...

    def solve(self, problem: QuadraticProgram, warm_start=None) -> QPResult: ...


class OSQPSolver:
    """Reuse OSQP workspace when the sparsity structure is unchanged.

    Instances own mutable solver state; use one instance per control/planning loop.
    """

    def __init__(self, *, eps_abs=1e-8, eps_rel=1e-8, max_iter=20000, polishing=False):
        if not np.isfinite([eps_abs, eps_rel]).all() or eps_abs <= 0 or eps_rel <= 0:
            raise ValueError("solver tolerances must be finite and positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(polishing, bool):
            raise ValueError("polishing must be boolean")
        self._settings = dict(eps_abs=float(eps_abs), eps_rel=float(eps_rel),
                              max_iter=int(max_iter), polishing=polishing)
        self.reset()

    def reset(self):
        self._solver = None
        self._pattern = None
        self._hessian = self._constraint = None
        self._sparse_constraint = None
        self._sparse_hessian = None
        self._matrix_keys = (None, None)

    def solve(self, problem, warm_start=None):
        p = problem
        if np.any(p.lower > p.upper) or np.isposinf(p.lower).any() or np.isneginf(p.upper).any():
            return QPResult("infeasible", detail="inconsistent constraint bounds")
        if warm_start is not None:
            warm_start = array(warm_start, 1, "warm_start")
            if warm_start.shape != p.gradient.shape:
                raise ValueError("warm start dimensions differ")
        keys = (_immutable_matrix_key(p.hessian), _immutable_matrix_key(p.constraint))
        same_h = (self._solver is not None
                  and _same_matrix(self._hessian, p.hessian, self._matrix_keys[0], keys[0]))
        same_a = (self._solver is not None
                  and _same_matrix(self._constraint, p.constraint, self._matrix_keys[1], keys[1]))
        same_matrices = same_h and same_a
        if not same_matrices:
            h = self._sparse_hessian if same_h else _upper_csc(p.hessian)
            a = self._sparse_constraint if same_a else sparse.csc_matrix(p.constraint)
            pattern = tuple((v.shape, v.indptr.tobytes(), v.indices.tobytes()) for v in (h, a))
        try:
            if same_matrices:
                self._solver.update(q=p.gradient, l=p.lower, u=p.upper)
            elif pattern != self._pattern:
                self._solver = osqp.OSQP()
                self._solver.setup(P=h, q=p.gradient, A=a, l=p.lower, u=p.upper,
                                   verbose=False, warm_starting=True, **self._settings)
                self._pattern = pattern
            else:
                updates = dict(q=p.gradient, l=p.lower, u=p.upper)
                if not same_h:
                    updates["Px"] = h.data
                if not same_a:
                    updates["Ax"] = a.data
                self._solver.update(**updates)
            # Own snapshots: caller mutations must invalidate reuse. Retain
            # each unchanged matrix independently of the other matrix.
            if not same_h:
                self._hessian = p.hessian.copy()
                self._sparse_hessian = h
            if not same_a:
                self._constraint = p.constraint.copy()
                self._sparse_constraint = a
            self._matrix_keys = keys
            if warm_start is not None:
                self._solver.warm_start(x=warm_start)
            result = self._solver.solve(raise_error=False)
        except osqp.OSQPException as exc:
            self.reset()
            return QPResult("error", detail=str(exc))
        status = result.info.status.lower()
        if result.info.status_val != 1 or result.x is None:
            if "infeasible" in status:
                status = "infeasible"
            elif "maximum iterations" in status:
                status = "iteration_limit"
            else:
                status = "error"
            return QPResult(status, iterations=result.info.iter, detail=result.info.status)
        if np.shape(result.x) != p.gradient.shape or not np.isfinite(result.x).all():
            return QPResult("error", iterations=result.info.iter, detail="invalid or nonfinite solution")
        x = array(result.x, 1, "solution")
        # Reuse the exact matrix passed to OSQP; condensed MPC constraints are
        # mostly zero. Keep the independent absolute feasibility check.
        ax = self._sparse_constraint @ x
        if not np.isfinite(ax).all():
            return QPResult("error", iterations=result.info.iter, detail="nonfinite constraint evaluation")
        violation = float(max(np.max(p.lower - ax, initial=0), np.max(ax - p.upper, initial=0)))
        if violation > 1e-6:
            return QPResult("error", iterations=result.info.iter, violation=violation,
                            detail="solution exceeds absolute constraint tolerance")
        return QPResult("solved", x, result.info.iter, violation, result.info.status)
