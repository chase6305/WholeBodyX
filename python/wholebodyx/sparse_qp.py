"""Sparse convex QP snapshots accepted by the existing OSQP adapter."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .qp import QuadraticProgram, _FixedQP, _immutable, _immutable_matrix_key


def _frozen_csc(value, name):
    result = sparse.csc_matrix(value, dtype=float, copy=True)
    result.check_format(full_check=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError(f"{name} must be finite")
    for attribute in ("data", "indices", "indptr"):
        part = getattr(result, attribute)
        setattr(result, attribute, np.frombuffer(part.tobytes(), dtype=part.dtype))
    return result


def _view(value):
    # New matrix metadata; immutable numeric and index buffers may be shared.
    return sparse.csc_matrix((value.data.view(), value.indices.view(), value.indptr.view()),
                             shape=value.shape, copy=False)


@dataclass(frozen=True)
class SparseQuadraticProgram(QuadraticProgram):
    """Full symmetric H and CSC A; vectors follow QuadraticProgram semantics.

    Diagonal dominance certifies common diagonal costs without densification.
    Other Hessians use a smallest-eigenvalue check once at construction.
    """

    hessian: sparse.csc_matrix
    constraint: sparse.csc_matrix

    def __post_init__(self):
        object.__setattr__(self, "hessian", _frozen_csc(self.hessian, "hessian"))
        object.__setattr__(self, "constraint", _frozen_csc(self.constraint, "constraint"))
        self._validate_vectors()
        self._validate_hessian()
        self._remember_matrix_validation()

    def _validate_hessian(self):
        h = self.hessian
        h.check_format(full_check=True)
        # Public sparse metadata can replace the original float buffers. Work
        # in solver precision so integer absolute values/sums cannot overflow
        # and incorrectly certify diagonal dominance.
        h = h.astype(float, copy=False)
        if not np.isfinite(h.data).all():
            raise ValueError("QP Hessian must be finite")
        difference = h - h.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
            raise ValueError("QP Hessian must be symmetric")
        diagonal = h.diagonal()
        off_diagonal = np.asarray(abs(h).sum(axis=1)).ravel() - np.abs(diagonal)
        if np.min(diagonal - off_diagonal) >= -1e-10:
            return
        from scipy.sparse.linalg import ArpackNoConvergence, eigsh
        try:
            if h.shape[0] <= 3:
                minimum = np.linalg.eigvalsh(h.toarray())[0]
            else:
                minimum = eigsh(h, k=1, which="SA", return_eigenvectors=False,
                                 v0=np.random.default_rng(0).normal(size=h.shape[0]), tol=1e-10)[0]
        except ArpackNoConvergence as exc:
            raise ValueError("could not verify sparse QP convexity") from exc
        if not np.isfinite(minimum) or minimum < -1e-10:
            raise ValueError("QP Hessian must be positive semidefinite")

    def _validate_constraint(self):
        self.constraint.check_format(full_check=True)
        if not np.isfinite(self.constraint.data).all():
            raise ValueError("QP constraint must be finite")


class _SparseFixedQP(_FixedQP):
    """Validate constant sparse matrices once and copy dynamic vectors per use."""

    _problem_type = SparseQuadraticProgram
    _freeze_matrix = staticmethod(_frozen_csc)
    _view_matrix = staticmethod(_view)

    @classmethod
    def separable(cls, temporal_hessian, joints, constraint):
        """Build H = B kron I after validating the small temporal cost B.

        A Kronecker product with an identity repeats B's eigenvalues. Keep
        this proof inside the builder; arbitrary sparse QPs still run their
        full convexity check.
        """
        if isinstance(joints, bool) or not isinstance(joints, (int, np.integer)) or joints < 1:
            raise ValueError("joint count must be a positive integer")
        count = np.shape(temporal_hessian)[0]
        block = QuadraticProgram(temporal_hessian, np.zeros(count), np.empty((0, count)), [], [])
        p = object.__new__(SparseQuadraticProgram)
        hessian = sparse.kron(block.hessian, sparse.eye(joints, format="csc"), format="csc")
        object.__setattr__(p, "hessian", _frozen_csc(hessian, "hessian"))
        object.__setattr__(p, "constraint", _frozen_csc(constraint, "constraint"))
        for name, size in (("gradient", hessian.shape[0]), ("lower", p.constraint.shape[0]),
                           ("upper", p.constraint.shape[0])):
            object.__setattr__(p, name, np.zeros(size))
        p._validate_vectors()
        p._remember_matrix_validation()
        return cls._from_problem(p)[0]

    def dense_template(self):
        """Materialize a dense diagnostic template with the same validated H/A."""
        result = object.__new__(_FixedQP)
        result._hessian = _immutable(self._hessian.toarray())
        result._constraint = _immutable(self._constraint.toarray())
        result._hessian_key = _immutable_matrix_key(result._hessian)
        result._constraint_key = _immutable_matrix_key(result._constraint)
        return result

