"""Backend-independent convex QP and an OSQP adapter."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import osqp
from scipy import sparse

from .types import array
from ._solver_execution import OSQP_EXECUTION_LOCK


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
        self._validate_hessian()
        self._remember_matrix_validation()

    def _validate_hessian(self):
        if not np.isfinite(self.hessian).all():
            raise ValueError("QP Hessian must be finite")
        if self.hessian.dtype == np.dtype(float):
            # Finite float64 inputs need only the existing absolute tolerance.
            # Keep allclose for caller-replaced dtypes to avoid integer overflow.
            symmetric = np.max(np.abs(self.hessian - self.hessian.T), initial=0) <= 1e-10
        else:
            symmetric = np.allclose(self.hessian, self.hessian.T, rtol=0, atol=1e-10)
        if not symmetric:
            raise ValueError("QP Hessian must be symmetric")
        try:
            # WBC regularization and MPC control cost normally make H positive
            # definite. Keep the eigenvalue fallback for singular PSD inputs
            # and the existing absolute tolerance for roundoff.
            np.linalg.cholesky(self.hessian)
        except np.linalg.LinAlgError:
            if np.linalg.eigvalsh(self.hessian)[0] < -1e-10:
                raise ValueError("QP Hessian must be positive semidefinite")

    def _validate_constraint(self):
        if not np.isfinite(self.constraint).all():
            raise ValueError("QP constraint must be finite")

    def _remember_matrix_validation(self):
        object.__setattr__(self, "_validated_hessian_key", _immutable_matrix_key(self.hessian))
        object.__setattr__(self, "_validated_constraint_key", _immutable_matrix_key(self.constraint))

    def _validate_vectors(self):
        object.__setattr__(self, "gradient", array(self.gradient, 1, "gradient"))
        for name in ("lower", "upper"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or np.isnan(value).any():
                raise ValueError("bounds must be vectors without NaNs")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        self._check_vectors()

    def _check_vectors(self):
        n, m = self.gradient.size, self.lower.size
        if (self.gradient.ndim != 1 or self.lower.ndim != 1 or not n or
                self.hessian.shape != (n, n) or self.constraint.shape != (m, n) or self.upper.shape != (m,)):
            raise ValueError("QP dimensions do not match")
        if any(np.iscomplexobj(value) for value in
               (self.hessian, self.gradient, self.constraint, self.lower, self.upper)):
            raise ValueError("QP storage must remain real-valued")
        if not np.isfinite(self.gradient).all() or np.isnan(self.lower).any() or np.isnan(self.upper).any():
            raise ValueError("QP vectors must contain a finite gradient and bounds without NaNs")


def _immutable(value):
    """Bytes-backed arrays cannot be made writable via setflags()."""
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _immutable_matrix_key(value):
    """Retain immutable storage and describe its view, not just array identity.

    A read-only flag alone is insufficient: an owning ndarray can be made
    writable again. Only bytes-backed arrays qualify for constant-time reuse.
    """
    if sparse.issparse(value):
        if not sparse.isspmatrix_csc(value):
            return None
        keys = tuple(_immutable_matrix_key(part) for part in (value.data, value.indices, value.indptr))
        return (value.shape, keys) if all(key is not None for key in keys) else None
    base = value
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        return None
    return (base, value.shape, value.strides, value.dtype.str,
            value.__array_interface__["data"][0])


def _same_immutable_key(previous, current):
    if previous is None or current is None:
        return False
    if isinstance(previous[0], bytes):
        return previous[0] is current[0] and previous[1:] == current[1:]
    return (previous[0] == current[0] and
            all(a[0] is b[0] and a[1:] == b[1:] for a, b in zip(previous[1], current[1])))


def _same_matrix(snapshot, value, previous_key, key):
    if sparse.issparse(value):
        if not sparse.issparse(snapshot) or snapshot.shape != value.shape:
            return False
        if _same_immutable_key(previous_key, key):
            return True
        return all(np.array_equal(a, b) for a, b in
                   zip((snapshot.data, snapshot.indices, snapshot.indptr),
                       (value.data, value.indices, value.indptr)))
    if sparse.issparse(snapshot):
        return False
    if _same_immutable_key(previous_key, key):
        return True
    return np.array_equal(snapshot, value)


class _UpperCSC:
    """One bounded, immutable index template for a dense Hessian's upper CSC."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.shape = None
        self.rows = self.columns = self.indptr = None

    def convert(self, hessian):
        triangle = np.triu(hessian)
        if self.shape == hessian.shape:
            values = triangle[self.rows, self.columns]
            # Equal nnz and all old positions still nonzero prove that no
            # position was removed or added, including same-count exchanges.
            if np.count_nonzero(triangle) == values.size and np.all(values != 0):
                return sparse.csc_matrix((values, self.rows.view(), self.indptr.view()),
                                         shape=hessian.shape)
        result = sparse.csc_matrix(triangle)
        columns = np.repeat(np.arange(hessian.shape[1], dtype=result.indices.dtype),
                            np.diff(result.indptr))
        # Own bytes, not mutable arrays shared with the returned CSC object.
        self.rows, self.columns, self.indptr = (
            np.frombuffer(value.tobytes(), dtype=value.dtype)
            for value in (result.indices, columns, result.indptr))
        self.shape = hessian.shape
        return result


def _upper_csc(hessian, cache=None):
    if sparse.issparse(hessian):
        if cache is not None:
            cache.clear()
        return sparse.triu(hessian, format="csc")
    if cache is not None:
        return cache.convert(hessian)
    return sparse.csc_matrix(np.triu(hessian))


def _scaled_csc(matrix, scale, *, rows=False):
    """Own sorted CSC values for A*D or D*H*D without sparse products."""
    result = matrix.astype(float, copy=True)
    result.sum_duplicates()
    # OSQP sorts CSC indices at setup. Keep the update order identical before
    # recording the pattern; sparse diagonal products can reverse row order.
    result.sort_indices()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if rows:
            result.data *= scale[result.indices]
        result.data *= np.repeat(scale, np.diff(result.indptr))
    # Numerical underflow can remove entries even with a positive scale.
    result.eliminate_zeros()
    return result


class _FixedQP:
    """Validate a fixed H/A once; each instance still validates dynamic vectors."""

    _problem_type = QuadraticProgram
    _freeze_matrix = staticmethod(lambda value, name: _immutable(value))
    _view_matrix = staticmethod(np.ndarray.view)

    def __init__(self, hessian, constraint):
        _, problem = self.with_problem(hessian, np.zeros(np.shape(hessian)[0]), constraint,
                                       np.zeros(np.shape(constraint)[0]), np.zeros(np.shape(constraint)[0]))
        self._remember(problem)

    @classmethod
    def with_problem(cls, hessian, gradient, constraint, lower, upper):
        """Validate actual vectors once and retain the certified immutable model."""
        p = object.__new__(cls._problem_type)
        for name, value in (("hessian", cls._freeze_matrix(hessian, "hessian")),
                            ("constraint", cls._freeze_matrix(constraint, "constraint")),
                            ("gradient", gradient), ("lower", lower), ("upper", upper)):
            object.__setattr__(p, name, value)
        p._validate_vectors()
        p._validate_hessian()
        p._validate_constraint()
        p._remember_matrix_validation()
        return cls._from_problem(p)

    def with_hessian(self, hessian, gradient, lower, upper):
        """Keep certified A storage and validate a replacement H and vectors."""
        hessian = self._freeze_matrix(hessian, "hessian")
        if hessian.shape != self._hessian.shape:
            raise ValueError("QP dimensions do not match")
        p = self.problem(gradient, lower, upper)
        object.__setattr__(p, "hessian", hessian)
        p._validate_hessian()
        object.__setattr__(p, "_validated_hessian_key", _immutable_matrix_key(hessian))
        # Retain A's original certificate; changed metadata is checked by the solver.
        return type(self)._from_problem(p)

    @classmethod
    def _from_problem(cls, p):
        template = object.__new__(cls)
        template._remember(p)
        return template, p

    def _remember(self, p):
        # Share certified bytes, never mutable matrix metadata or a new trust token.
        self._hessian, self._constraint = self._view_matrix(p.hessian), self._view_matrix(p.constraint)
        self._hessian_key = p._validated_hessian_key
        self._constraint_key = p._validated_constraint_key

    def problem(self, gradient, lower, upper):
        p = object.__new__(self._problem_type)
        # Share immutable buffers, not ndarray metadata (shape/dtype/strides).
        for name, value in (("hessian", self._view_matrix(self._hessian)),
                            ("constraint", self._view_matrix(self._constraint)),
                            ("gradient", gradient), ("lower", lower), ("upper", upper)):
            object.__setattr__(p, name, value)
        p._validate_vectors()
        object.__setattr__(p, "_validated_hessian_key", self._hessian_key)
        object.__setattr__(p, "_validated_constraint_key", self._constraint_key)
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

    def __init__(self, *, eps_abs=1e-8, eps_rel=1e-8, eps_prim_inf=1e-8, eps_dual_inf=1e-8,
                 max_iter=20000, polishing=False,
                 variable_scaling=None, adaptive_rho_interval=0):
        tolerances = [eps_abs, eps_rel, eps_prim_inf, eps_dual_inf]
        if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
            raise ValueError("solver tolerances must be finite and positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(polishing, bool):
            raise ValueError("polishing must be boolean")
        if (isinstance(adaptive_rho_interval, bool) or
                not isinstance(adaptive_rho_interval, (int, np.integer)) or adaptive_rho_interval < 0):
            raise ValueError("adaptive_rho_interval must be a nonnegative integer")
        self._variable_scaling = None
        if variable_scaling is not None:
            scale = array(variable_scaling, 1, "variable scaling")
            if not scale.size or np.any(scale <= 0):
                raise ValueError("variable scaling must be nonempty and positive")
            self._variable_scaling = _immutable(scale)
        self._settings = dict(eps_abs=float(eps_abs), eps_rel=float(eps_rel),
                              eps_prim_inf=float(eps_prim_inf), eps_dual_inf=float(eps_dual_inf),
                              max_iter=int(max_iter), polishing=polishing,
                              adaptive_rho_interval=int(adaptive_rho_interval))
        self.reset()

    @property
    def variable_scaling(self):
        """Fixed x = D*z scaling; results and warm starts retain physical x."""
        return None if self._variable_scaling is None else self._variable_scaling.view()

    def reset(self):
        self._solver = None
        self._pattern = None
        self._hessian = self._constraint = None
        self._sparse_constraint = None
        self._sparse_hessian = None
        self._physical_constraint = None
        self._matrix_keys = (None, None)
        self._upper_cache = _UpperCSC()

    def solve(self, problem, warm_start=None):
        p = problem
        p._check_vectors()
        scale = self._variable_scaling
        if scale is not None and scale.shape != p.gradient.shape:
            raise ValueError("variable scaling and QP dimensions differ")
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
        # Public read-only flags can be reversed and sparse metadata can be
        # replaced. Revalidate changed inputs before touching the workspace.
        # Only the exact immutable view certified at construction may bypass
        # matrix validation; solver-owned value snapshots protect reuse later.
        if not same_h and not _same_immutable_key(getattr(p, "_validated_hessian_key", None), keys[0]):
            p._validate_hessian()
        if not same_a and not _same_immutable_key(getattr(p, "_validated_constraint_key", None), keys[1]):
            p._validate_constraint()
        if not same_matrices:
            h = self._sparse_hessian if same_h else _upper_csc(p.hessian, self._upper_cache)
            a = self._sparse_constraint if same_a else sparse.csc_matrix(p.constraint)
            if not same_a:
                physical_a = a
            if scale is not None:
                if not same_h:
                    h = _scaled_csc(h, scale, rows=True)
                if not same_a:
                    physical_a = a.copy()
                    a = _scaled_csc(a, scale)
                if not np.isfinite(h.data).all() or not np.isfinite(a.data).all():
                    raise ValueError("nonfinite scaled QP matrices")
            pattern = tuple((v.shape, v.indptr.tobytes(), v.indices.tobytes()) for v in (h, a))
        gradient = p.gradient
        if scale is not None:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                try:
                    gradient = gradient * scale
                    if warm_start is not None:
                        warm_start = warm_start / scale
                except FloatingPointError as exc:
                    raise ValueError("nonfinite scaled QP vectors") from exc
        try:
            if same_matrices:
                self._solver.update(q=gradient, l=p.lower, u=p.upper)
            elif pattern != self._pattern:
                self._solver = osqp.OSQP()
                self._solver.setup(P=h, q=gradient, A=a, l=p.lower, u=p.upper,
                                   verbose=False, warm_starting=True, **self._settings)
                self._pattern = pattern
            else:
                updates = dict(q=gradient, l=p.lower, u=p.upper)
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
                self._physical_constraint = physical_a
            self._matrix_keys = keys
            if warm_start is not None:
                self._solver.warm_start(x=warm_start)
            with OSQP_EXECUTION_LOCK:
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
        with np.errstate(over="ignore", invalid="ignore"):
            solution = result.x if scale is None else result.x * scale
        if not np.isfinite(solution).all():
            return QPResult("error", iterations=result.info.iter, detail="nonfinite physical solution")
        x = array(solution, 1, "solution")
        # Evaluate the original A in physical coordinates. Cached CSC storage
        # avoids scanning dense zeros; numerical scaling never changes the
        # independent absolute feasibility threshold.
        ax = self._physical_constraint @ x
        if not np.isfinite(ax).all():
            return QPResult("error", iterations=result.info.iter, detail="nonfinite constraint evaluation")
        violation = float(max(np.max(p.lower - ax, initial=0), np.max(ax - p.upper, initial=0)))
        if violation > 1e-6:
            return QPResult("error", iterations=result.info.iter, violation=violation,
                            detail="solution exceeds absolute constraint tolerance")
        return QPResult("solved", x, result.info.iter, violation, result.info.status)
