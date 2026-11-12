"""Changed public QP inputs must be revalidated before an OSQP update."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram


@pytest.mark.parametrize("kind", ["nonconvex", "asymmetric", "nonfinite_hessian", "nonfinite_constraint",
                                  "nonfinite_gradient", "nan_lower", "nan_upper", "gradient_shape",
                                  "complex_hessian", "complex_constraint", "complex_gradient", "complex_lower", "complex_upper"])
@pytest.mark.parametrize('normalize', [False, True])
def test_changed_dense_problem_rejects_invalid_input_and_keeps_workspace_recoverable(kind, normalize):
    problem = QuadraticProgram(np.eye(2), [-1., -1.], np.eye(2), [-2., -2.], [2., 2.])
    solver = OSQPSolver(normalize_constraints=normalize)
    assert solver.solve(problem).success
    workspace = solver._solver
    field = {"nonconvex": "hessian", "asymmetric": "hessian", "nonfinite_hessian": "hessian",
             "nonfinite_constraint": "constraint", "nonfinite_gradient": "gradient",
             "nan_lower": "lower", "nan_upper": "upper", "gradient_shape": "gradient",
             "complex_hessian": "hessian", "complex_constraint": "constraint", "complex_gradient": "gradient",
             "complex_lower": "lower", "complex_upper": "upper"}[kind]
    value = getattr(problem, field)
    original = value.copy()
    value.setflags(write=True)
    if kind.startswith("complex_"):
        value.dtype = np.complex64
    elif kind == "asymmetric":
        value[1, 0] = .5
    elif kind == "gradient_shape":
        value.shape = (1, 2)
    else:
        value.flat[0] = -1. if kind == "nonconvex" else np.nan
    with pytest.raises(ValueError):
        solver.solve(problem)
    assert solver._solver is workspace
    value.dtype = original.dtype
    value.shape = original.shape
    value[:] = original
    recovered = solver.solve(problem)
    assert recovered.success
    np.testing.assert_allclose(recovered.x, [1., 1.], atol=1e-6)


@pytest.mark.parametrize("kind", ["nonconvex", "asymmetric", "nan_hessian", "nan_constraint", "bad_indices",
                                  "complex_hessian", "complex_constraint"])
@pytest.mark.parametrize('normalize', [False, True])
def test_sparse_replacement_buffers_cannot_reuse_a_stale_constructor_certificate(kind, normalize):
    problem = SparseQuadraticProgram(sparse.eye(2), [-1., -1.], sparse.eye(2), [-2., -2.], [2., 2.])
    solver = OSQPSolver(variable_scaling=[.1, 2.], normalize_constraints=normalize)
    assert solver.solve(problem).success
    target = problem.constraint if kind in ("nan_constraint", "bad_indices", "complex_constraint") else problem.hessian
    original = target.copy()
    if kind.startswith("complex_"):
        target.data = target.data.view(np.complex64)
    elif kind == "asymmetric":
        changed = sparse.csc_matrix([[1., 0.], [.5, 1.]])
        target.data, target.indices, target.indptr = changed.data, changed.indices, changed.indptr
    elif kind == "bad_indices":
        target.indices = np.full_like(target.indices, 99)
    else:
        data = target.data.copy()
        data[0] = -1. if kind == "nonconvex" else np.nan
        target.data = data
    with pytest.raises(ValueError):
        solver.solve(problem)
    target.data, target.indices, target.indptr = original.data, original.indices, original.indptr
    assert solver.solve(problem).success


def test_sparse_dtype_metadata_change_is_not_a_trusted_immutable_view():
    problem = SparseQuadraticProgram(sparse.csc_matrix([[1., -.5], [-.5, 1.]]), [0., 0.],
                                     sparse.eye(2), [-1., -1.], [1., 1.])
    problem.hessian.data = problem.hessian.data.view(np.int64)
    with pytest.raises(ValueError, match="positive semidefinite"):
        OSQPSolver().solve(problem)


def test_fixed_templates_skip_duplicate_convexity_validation_but_check_changed_metadata(monkeypatch):
    from wholebodyx.qp import _FixedQP

    template = _FixedQP(np.diag([3., 4., 5.]), np.eye(3))
    problem = template.problem([0., 0., 0.], [-1.] * 3, [1.] * 3)
    original, calls = QuadraticProgram._validate_hessian, []

    def validate(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(QuadraticProgram, "_validate_hessian", validate)
    solver = OSQPSolver()
    assert solver.solve(problem).success
    assert calls == []
    problem.hessian.strides = (8, 8)  # Different PSD matrix, same bytes.
    assert solver.solve(problem).success
    assert calls == [1]
    assert solver.solve(template.problem([0.] * 3, [-1.] * 3, [1.] * 3)).success
    assert calls == [1]


@pytest.mark.parametrize("field,value", [("gradient", np.nan), ("gradient", np.inf),
                                        ("lower", np.nan), ("upper", np.nan)])
def test_sparse_vector_mutation_is_rejected_before_cached_workspace_update(field, value):
    problem = SparseQuadraticProgram(sparse.eye(2), [-1., -1.], sparse.eye(2), [-2., -2.], [2., 2.])
    solver = OSQPSolver()
    assert solver.solve(problem).success
    workspace = solver._solver
    vector = getattr(problem, field)
    original = vector.copy()
    vector.setflags(write=True)
    vector[0] = value
    with pytest.raises(ValueError):
        solver.solve(problem)
    assert solver._solver is workspace
    vector[:] = original
    recovered = solver.solve(problem)
    assert recovered.success
    np.testing.assert_allclose(recovered.x, [1., 1.], atol=1e-6)
