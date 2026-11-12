"""Backend-independent convex QP and an OSQP adapter."""

from copy import copy
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import osqp
from scipy import sparse

from .types import array
from ._solver_execution import OSQP_EXECUTION_LOCK


class _OSQPFailure(Exception):
    pass


# OSQP 1.0 raises ValueError directly; later versions expose OSQPException.
# Translate only errors from backend calls so input validation keeps its
# original ValueError and does not discard a healthy cached workspace.
_OSQP_ERRORS = (ValueError,)
if hasattr(osqp, "OSQPException"):
    _OSQP_ERRORS += (osqp.OSQPException,)


def _osqp_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except _OSQP_ERRORS as exc:
        raise _OSQPFailure(str(exc)) from exc


def _real_qp_array(value, ndim, name):
    """Reject complex QP inputs before a float conversion can discard values."""
    value = np.asarray(value)
    if value.dtype.kind == "c":
        raise ValueError(f"{name} must be real-valued")
    return array(value, ndim, name)


@dataclass(frozen=True)
class QuadraticProgram:
    """Minimize 1/2 x'Hx + g'x subject to lower <= A x <= upper.

    Construction copies caller data. Infinite bounds are allowed; inconsistent
    bound intervals are reported as infeasible by the solver.
    """

    hessian: np.ndarray
    gradient: np.ndarray
    constraint: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for name, ndim in (("hessian", 2), ("constraint", 2)):
            object.__setattr__(self, name, _real_qp_array(getattr(self, name), ndim, name))
        self._validate_vectors()
        self._validate_hessian()
        self._remember_matrix_validation()

    def _validate_hessian(self):
        if not np.isfinite(self.hessian).all():
            raise ValueError("QP Hessian must be finite")
        if self.hessian.dtype == np.dtype(float):
            # Finite float64 inputs need only the existing absolute tolerance.
            # Keep allclose for caller-replaced dtypes to avoid integer overflow.
            symmetric = np.abs(self.hessian - self.hessian.T).max(initial=0) <= 1e-10
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
        object.__setattr__(
            self, "_validated_constraint_key", _immutable_matrix_key(self.constraint)
        )

    def _validate_vectors(self):
        object.__setattr__(self, "gradient", _real_qp_array(self.gradient, 1, "gradient"))
        for name in ("lower", "upper"):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind == "c":
                raise ValueError("bounds must be real-valued")
            value = np.array(value, dtype=float, copy=True)
            if value.ndim != 1 or np.isnan(value).any():
                raise ValueError("bounds must be vectors without NaNs")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        # The new vectors already have real storage, a finite gradient,
        # and NaN-free bounds. Only dimensions remain to be checked here.
        # solve() still checks the public buffers again before every update.
        self._check_dimensions()

    def _check_dimensions(self):
        variable_count = self.gradient.size
        constraint_count = self.lower.size
        if (
            self.gradient.ndim != 1
            or self.lower.ndim != 1
            or not variable_count
            or self.hessian.shape != (variable_count, variable_count)
            or self.constraint.shape != (constraint_count, variable_count)
            or self.upper.shape != (constraint_count,)
        ):
            raise ValueError("QP dimensions do not match")

    def _check_vectors(self):
        self._check_dimensions()
        if any(
            value.dtype.kind == "c"
            for value in (self.hessian, self.gradient, self.constraint, self.lower, self.upper)
        ):
            raise ValueError("QP storage must remain real-valued")
        if (
            not np.isfinite(self.gradient).all()
            or np.isnan(self.lower).any()
            or np.isnan(self.upper).any()
        ):
            raise ValueError("QP vectors must contain a finite gradient and bounds without NaNs")


def _immutable(value):
    """Bytes-backed arrays cannot be made writable via setflags()."""
    matrix = np.asarray(value, dtype=float)
    return np.frombuffer(matrix.tobytes(), dtype=float).reshape(matrix.shape)


class _DenseMatrixKey(NamedTuple):
    storage: bytes
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    address: int

    def matches(self, other):
        """Equal values alone do not certify ownership or view metadata."""
        return (
            self.storage is other.storage
            and self.shape == other.shape
            and self.strides == other.strides
            and self.dtype == other.dtype
            and self.address == other.address
        )


class _SparseMatrixKey(NamedTuple):
    shape: tuple[int, int]
    buffers: tuple[_DenseMatrixKey, ...]


def _immutable_matrix_key(value):
    """Retain immutable storage and describe its view, not just array identity.

    A read-only flag alone is insufficient: an owning ndarray can be made
    writable again. Only bytes-backed arrays qualify for constant-time reuse.
    """
    if sparse.issparse(value):
        if not sparse.isspmatrix_csc(value):
            return None
        buffers = tuple(
            _immutable_array_key(part) for part in (value.data, value.indices, value.indptr)
        )
        return (
            _SparseMatrixKey(value.shape, buffers)
            if all(key is not None for key in buffers)
            else None
        )
    return _immutable_array_key(value)


def _immutable_array_key(value):
    base = value
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        return None
    return _DenseMatrixKey(
        base, value.shape, value.strides, value.dtype.str, value.__array_interface__["data"][0]
    )


def _sparse_view(value):
    """Share immutable CSC buffers with independent matrix and array metadata."""
    # The source already has validated CSC structure. A shallow matrix copy
    # avoids repeating SciPy's constructor checks, while the three new views
    # isolate dtype/shape/strides changes made by a retained public problem.
    result = copy(value)
    result.data = value.data.view()
    result.indices = value.indices.view()
    result.indptr = value.indptr.view()
    return result


def _matrix_snapshot(value, key):
    # A matching certificate covers bytes, not the caller's mutable metadata.
    # Ordinary public arrays still need an owned copy to detect later writes.
    if key is None:
        return value.copy()
    return _sparse_view(value) if sparse.issparse(value) else value.view()


def _same_immutable_key(previous, current):
    if previous is None or type(previous) is not type(current):
        return False
    if isinstance(previous, _DenseMatrixKey):
        return previous.matches(current)
    return previous.shape == current.shape and all(
        old.matches(new) for old, new in zip(previous.buffers, current.buffers)
    )


def _same_matrix(snapshot, value, previous_key, key):
    """Reuse certified storage identity or compare against owned values."""
    if sparse.issparse(value):
        if (
            not sparse.issparse(snapshot)
            or snapshot.shape != value.shape
            or snapshot.format != value.format
        ):
            return False
        if _same_immutable_key(previous_key, key):
            return True
        return all(
            np.array_equal(old, new)
            for old, new in zip(
                (snapshot.data, snapshot.indices, snapshot.indptr),
                (value.data, value.indices, value.indptr),
            )
        )
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
        self._template = None

    def convert(self, hessian):
        triangle = np.triu(hessian)
        if self.shape == hessian.shape:
            values = triangle[self.rows, self.columns]
            # Equal nnz and all old positions still nonzero prove that no
            # position was removed or added, including same-count exchanges.
            if np.count_nonzero(triangle) == values.size and (values != 0).all():
                result = copy(self._template)
                result.data = values
                result.indices = self.rows.view()
                result.indptr = self.indptr.view()
                return result
        result = sparse.csc_matrix(triangle)
        columns = np.repeat(
            np.arange(hessian.shape[1], dtype=result.indices.dtype), np.diff(result.indptr)
        )
        # Own bytes, not mutable arrays shared with the returned CSC object.
        self.rows, self.columns, self.indptr = (
            np.frombuffer(value.tobytes(), dtype=value.dtype)
            for value in (result.indices, columns, result.indptr)
        )
        # Keep only an empty CSC wrapper and the immutable structural indices.
        # Every returned matrix receives fresh values; none are retained here.
        self._template = sparse.csc_matrix(hessian.shape, dtype=hessian.dtype)
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


def _divide_constraint_rows(values, divisor):
    """Reject overflow or lost nonzero coefficients/bounds before updating OSQP."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            result = values / divisor
        except FloatingPointError as exc:
            raise ValueError("unrepresentable normalized QP constraints") from exc
    if ((values != 0) & (result == 0)).any():
        raise ValueError("unrepresentable normalized QP constraints: nonzero underflow")
    return result


def _normalized_constraint(matrix, column_scale=None):
    """Own canonical CSC; cache nonunit row divisors without dense matrices."""
    result = matrix.astype(float, copy=True)
    # Public sparse storage may have changed since construction. Compute norms
    # from actual coefficients, after combining duplicates and removing zeros.
    result.has_sorted_indices = False
    result.has_canonical_format = False
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("nonfinite normalized QP constraints")
    divisor = np.zeros(result.shape[0])
    np.maximum.at(divisor, result.indices, np.abs(result.data))
    divisor[divisor == 0] = 1.
    # Box constraints and already normalized models need no bound transforms.
    if (divisor == 1).all():
        divisor = None
    else:
        result.data = _divide_constraint_rows(result.data, divisor[result.indices])
        divisor = _immutable(divisor)
    if column_scale is not None:
        # The canonical CSC is already owned. Apply D here to avoid a second
        # full copy/canonicalization through _scaled_csc for combined scaling.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            result.data *= np.repeat(column_scale, np.diff(result.indptr))
        result.eliminate_zeros()
    return result, divisor


class _FixedQP:
    """Validate a fixed H/A once; each instance still validates dynamic vectors."""

    _problem_type = QuadraticProgram
    _freeze_matrix = staticmethod(lambda value, name: _immutable(value))
    _view_matrix = staticmethod(np.ndarray.view)

    def __init__(self, hessian, constraint):
        problem = self._build_problem(
            hessian,
            np.zeros(np.shape(hessian)[0]),
            constraint,
            np.zeros(np.shape(constraint)[0]),
            np.zeros(np.shape(constraint)[0]),
        )
        self._remember(problem)

    @classmethod
    def with_problem(cls, hessian, gradient, constraint, lower, upper):
        """Validate actual vectors once and retain the certified immutable model."""
        problem = cls._build_problem(hessian, gradient, constraint, lower, upper)
        return cls._from_problem(problem)

    @classmethod
    def _build_problem(cls, hessian, gradient, constraint, lower, upper):
        """Validate a snapshot without allocating an intermediate template."""
        problem = object.__new__(cls._problem_type)
        for name, value in (
            ("hessian", cls._freeze_matrix(hessian, "hessian")),
            ("constraint", cls._freeze_matrix(constraint, "constraint")),
            ("gradient", gradient),
            ("lower", lower),
            ("upper", upper),
        ):
            object.__setattr__(problem, name, value)
        problem._validate_vectors()
        problem._validate_hessian()
        problem._validate_constraint()
        problem._remember_matrix_validation()
        return problem

    def with_hessian(self, hessian, gradient, lower, upper):
        """Keep certified A storage and validate a replacement H and vectors."""
        hessian = self._freeze_matrix(hessian, "hessian")
        if hessian.shape != self._hessian.shape:
            raise ValueError("QP dimensions do not match")
        problem = self.problem(gradient, lower, upper)
        object.__setattr__(problem, "hessian", hessian)
        problem._validate_hessian()
        object.__setattr__(problem, "_validated_hessian_key", _immutable_matrix_key(hessian))
        # Retain A's original certificate; changed metadata is checked by the solver.
        return type(self)._from_problem(problem)

    @classmethod
    def _from_problem(cls, problem):
        template = object.__new__(cls)
        template._remember(problem)
        return template, problem

    def _remember(self, problem):
        # Share certified bytes, never mutable matrix metadata or a new trust token.
        self._hessian, self._constraint = (
            self._view_matrix(problem.hessian),
            self._view_matrix(problem.constraint),
        )
        self._hessian_key = problem._validated_hessian_key
        self._constraint_key = problem._validated_constraint_key

    def problem(self, gradient, lower, upper):
        problem = object.__new__(self._problem_type)
        # Share immutable buffers, not ndarray metadata (shape/dtype/strides).
        for name, value in (
            ("hessian", self._view_matrix(self._hessian)),
            ("constraint", self._view_matrix(self._constraint)),
            ("gradient", gradient),
            ("lower", lower),
            ("upper", upper),
        ):
            object.__setattr__(problem, name, value)
        problem._validate_vectors()
        object.__setattr__(problem, "_validated_hessian_key", self._hessian_key)
        object.__setattr__(problem, "_validated_constraint_key", self._constraint_key)
        return problem


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
    Optional constraint normalization divides A and bounds by each row's largest
    absolute coefficient. Results and residual acceptance retain physical units.
    """

    def __init__(
        self,
        *,
        eps_abs=1e-8,
        eps_rel=1e-8,
        eps_prim_inf=1e-8,
        eps_dual_inf=1e-8,
        max_iter=20000,
        polishing=False,
        variable_scaling=None,
        normalize_constraints=False,
        adaptive_rho_interval=0,
    ):
        tolerances = [eps_abs, eps_rel, eps_prim_inf, eps_dual_inf]
        if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
            raise ValueError("solver tolerances must be finite and positive")
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(polishing, bool):
            raise ValueError("polishing must be boolean")
        if not isinstance(normalize_constraints, bool):
            raise ValueError("normalize_constraints must be boolean")
        if (
            isinstance(adaptive_rho_interval, bool)
            or not isinstance(adaptive_rho_interval, (int, np.integer))
            or adaptive_rho_interval < 0
        ):
            raise ValueError("adaptive_rho_interval must be a nonnegative integer")
        self._variable_scaling = None
        self._normalize_constraints = normalize_constraints
        if variable_scaling is not None:
            scale = array(variable_scaling, 1, "variable scaling")
            if not scale.size or (scale <= 0).any():
                raise ValueError("variable scaling must be nonempty and positive")
            self._variable_scaling = _immutable(scale)
        self._settings = dict(
            eps_abs=float(eps_abs),
            eps_rel=float(eps_rel),
            eps_prim_inf=float(eps_prim_inf),
            eps_dual_inf=float(eps_dual_inf),
            max_iter=int(max_iter),
            polishing=polishing,
            adaptive_rho_interval=int(adaptive_rho_interval),
        )
        # Initialize base state before a subclass's runtime reset hook is ready.
        OSQPSolver.reset(self)

    @property
    def variable_scaling(self):
        """Fixed x = D*z scaling; results and warm starts retain physical x."""
        return None if self._variable_scaling is None else self._variable_scaling.view()

    @property
    def normalize_constraints(self):
        """Whether positive row normalization is applied before variable scaling."""
        return self._normalize_constraints

    def reset(self):
        self._solver = None
        self._pattern = None
        self._hessian = self._constraint = None
        self._sparse_constraint = None
        self._sparse_hessian = None
        self._physical_constraint = None
        self._constraint_divisor = None
        self._matrix_keys = (None, None)
        self._upper_cache = _UpperCSC()

    def solve(self, problem, warm_start=None):
        problem._check_vectors()
        scale = self._variable_scaling
        if scale is not None and scale.shape != problem.gradient.shape:
            raise ValueError("variable scaling and QP dimensions differ")
        if (
            (problem.lower > problem.upper).any()
            or np.isposinf(problem.lower).any()
            or np.isneginf(problem.upper).any()
        ):
            return QPResult("infeasible", detail="inconsistent constraint bounds")
        if warm_start is not None:
            warm_start = array(warm_start, 1, "warm_start")
            if warm_start.shape != problem.gradient.shape:
                raise ValueError("warm start dimensions differ")
        try:
            solver_warm_start = self._update_workspace(problem, warm_start)
            if solver_warm_start is not None:
                _osqp_call(self._solver.warm_start, x=solver_warm_start)
            with OSQP_EXECUTION_LOCK:
                result = _osqp_call(self._solver.solve, raise_error=False)
        except _OSQPFailure as exc:
            self.reset()
            return QPResult("error", detail=str(exc))
        return self._validate_result(problem, result)

    def _update_workspace(self, problem, warm_start):
        """Update OSQP and return the warm start in solver coordinates.

        Matrix identity, sparsity, and numeric values have separate reuse rules:
        unchanged matrices need only new vectors; the same sparsity permits a
        numeric update; changed sparsity requires a fresh OSQP workspace.
        """
        scale = self._variable_scaling
        divisor = self._constraint_divisor
        keys = (_immutable_matrix_key(problem.hessian), _immutable_matrix_key(problem.constraint))
        same_hessian = self._solver is not None and _same_matrix(
            self._hessian, problem.hessian, self._matrix_keys[0], keys[0]
        )
        same_constraint = self._solver is not None and _same_matrix(
            self._constraint, problem.constraint, self._matrix_keys[1], keys[1]
        )
        same_matrices = same_hessian and same_constraint

        # A reversible read-only flag is not a validation certificate. Only
        # matching immutable storage and view metadata can bypass a new check.
        if not same_hessian and not _same_immutable_key(
            getattr(problem, "_validated_hessian_key", None), keys[0]
        ):
            problem._validate_hessian()
            # Sparse validation may cast index/data buffers into new mutable
            # storage. Snapshot ownership must describe the validated buffers.
            keys = (_immutable_matrix_key(problem.hessian), keys[1])
        if not same_constraint and not _same_immutable_key(
            getattr(problem, "_validated_constraint_key", None), keys[1]
        ):
            problem._validate_constraint()
            keys = (keys[0], _immutable_matrix_key(problem.constraint))

        if not same_matrices:
            solver_hessian = (
                self._sparse_hessian
                if same_hessian
                else _upper_csc(problem.hessian, self._upper_cache)
            )
            solver_constraint = (
                self._sparse_constraint
                if same_constraint
                else sparse.csc_matrix(problem.constraint)
            )
            if not same_constraint:
                physical_constraint = solver_constraint
                if self._normalize_constraints:
                    physical_constraint = solver_constraint.copy()
                    solver_constraint, divisor = _normalized_constraint(solver_constraint, scale)
            if scale is not None:
                if not same_hessian:
                    solver_hessian = _scaled_csc(solver_hessian, scale, rows=True)
                if not same_constraint and not self._normalize_constraints:
                    physical_constraint = solver_constraint.copy()
                    solver_constraint = _scaled_csc(solver_constraint, scale)
                if (
                    not np.isfinite(solver_hessian.data).all()
                    or not np.isfinite(solver_constraint.data).all()
                ):
                    raise ValueError("nonfinite scaled QP matrices")
            pattern = tuple(
                (matrix.shape, matrix.indptr.tobytes(), matrix.indices.tobytes())
                for matrix in (solver_hessian, solver_constraint)
            )

        solver_gradient = problem.gradient
        solver_lower, solver_upper = problem.lower, problem.upper
        if divisor is not None:
            solver_lower = _divide_constraint_rows(solver_lower, divisor)
            solver_upper = _divide_constraint_rows(solver_upper, divisor)
        solver_warm_start = warm_start
        if scale is not None:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                try:
                    solver_gradient = solver_gradient * scale
                    if warm_start is not None:
                        solver_warm_start = warm_start / scale
                except FloatingPointError as exc:
                    raise ValueError("nonfinite scaled QP vectors") from exc

        if same_matrices:
            _osqp_call(self._solver.update, q=solver_gradient, l=solver_lower, u=solver_upper)
        elif pattern != self._pattern:
            self._solver = osqp.OSQP()
            _osqp_call(
                self._solver.setup,
                P=solver_hessian,
                q=solver_gradient,
                A=solver_constraint,
                l=solver_lower,
                u=solver_upper,
                verbose=False,
                warm_starting=True,
                **self._settings,
            )
            self._pattern = pattern
        else:
            updates = dict(q=solver_gradient, l=solver_lower, u=solver_upper)
            if not same_hessian:
                updates["Px"] = solver_hessian.data
            if not same_constraint:
                updates["Ax"] = solver_constraint.data
            _osqp_call(self._solver.update, **updates)

        # Record snapshots after setup/update returns. Preserve the original A
        # independently of solver scaling for physical residual validation.
        if not same_hessian:
            self._hessian = _matrix_snapshot(problem.hessian, keys[0])
            self._sparse_hessian = solver_hessian
        if not same_constraint:
            self._constraint = _matrix_snapshot(problem.constraint, keys[1])
            self._sparse_constraint = solver_constraint
            self._physical_constraint = physical_constraint
            self._constraint_divisor = divisor
        self._matrix_keys = keys
        return solver_warm_start

    def _validate_result(self, problem, result):
        """Map solver coordinates to physical units and enforce acceptance."""
        status = result.info.status.lower()
        if result.info.status_val != 1 or result.x is None:
            if "infeasible" in status:
                status = "infeasible"
            elif "maximum iterations" in status:
                status = "iteration_limit"
            else:
                status = "error"
            return QPResult(status, iterations=result.info.iter, detail=result.info.status)
        if np.shape(result.x) != problem.gradient.shape or not np.isfinite(result.x).all():
            return QPResult(
                "error", iterations=result.info.iter, detail="invalid or nonfinite solution"
            )
        solution = result.x
        if self._variable_scaling is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                solution = result.x * self._variable_scaling
            if not np.isfinite(solution).all():
                return QPResult(
                    "error", iterations=result.info.iter, detail="nonfinite physical solution"
                )
        solution = array(solution, 1, "solution")
        constraint_values = self._physical_constraint @ solution
        if not np.isfinite(constraint_values).all():
            return QPResult(
                "error", iterations=result.info.iter, detail="nonfinite constraint evaluation"
            )
        violation = float(
            max(
                (problem.lower - constraint_values).max(initial=0),
                (constraint_values - problem.upper).max(initial=0),
            )
        )
        if violation > 1e-6:
            return QPResult(
                "error",
                iterations=result.info.iter,
                violation=violation,
                detail="solution exceeds absolute constraint tolerance",
            )
        return QPResult("solved", solution, result.info.iter, violation, result.info.status)
