"""Sparse validation can replace immutable buffers with mutable arrays."""

from contextlib import nullcontext

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, SparseQuadraticProgram


def replace_buffer(matrix, part):
    value = getattr(matrix, part)
    dtype = np.dtype(float).newbyteorder() if part == 'data' else np.dtype(float)
    setattr(matrix, part, np.frombuffer(value.astype(dtype).tobytes(), dtype=dtype))
    assert not getattr(matrix, part).flags.writeable
    return nullcontext() if part == 'data' else pytest.warns(UserWarning, match='non-integer dtype')


@pytest.mark.parametrize('part', ['indices', 'indptr', 'data'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_changed_constraint_after_validation_replaced_storage_is_detected(part, normalize, scale):
    p = SparseQuadraticProgram(sparse.eye(2), [-2., -2.],
                               sparse.csc_matrix([[1., 0.], [0., 2.]]), [-1., -1.], [1., .5])
    solver = OSQPSolver(normalize_constraints=normalize, variable_scaling=scale)
    with replace_buffer(p.constraint, part):
        assert solver.solve(p).success
    # SciPy has replaced non-integer index buffers or non-native data with
    # writable arrays. These arrays must not share the solver's snapshot.
    current = getattr(p.constraint, part)
    assert current.flags.writeable
    if part == 'indices':
        current[:] = [1, 0]
        expected = [.5, .5]
    elif part == 'indptr':
        current[:] = [0, 2, 2]
        expected = [.25, 2.]
    else:
        current *= 2.
        expected = [.5, .125]
    result = solver.solve(p)
    assert result.success, result
    np.testing.assert_allclose(result.x, expected, atol=2e-6, rtol=0)
    ax = p.constraint @ result.x
    physical = max((p.lower - ax).max(initial=0), (ax - p.upper).max(initial=0), 0.)
    assert physical <= 1e-6
    assert result.violation == pytest.approx(physical, abs=1e-14)


@pytest.mark.parametrize('part', ['indices', 'indptr', 'data'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_invalid_hessian_after_validation_replaced_storage_keeps_healthy_workspace(part, normalize, scale):
    p = SparseQuadraticProgram(sparse.eye(2), [-.2, -.3], sparse.eye(2), [-1.] * 2, [1.] * 2)
    solver = OSQPSolver(normalize_constraints=normalize, variable_scaling=scale)
    with replace_buffer(p.hessian, part):
        assert solver.solve(p).success
    workspace = solver._solver
    current = getattr(p.hessian, part)
    original = current.copy()
    assert current.flags.writeable
    if part == 'indices':
        current[:] = [1, 0]  # Symmetric indefinite H.
    elif part == 'indptr':
        current[:] = [0, 2, 2]  # Asymmetric H.
    else:
        current *= -1.  # Negative definite H.
    with pytest.raises(ValueError, match='symmetric|positive semidefinite'):
        solver.solve(p)
    assert solver._solver is workspace
    current[:] = original
    recovered = solver.solve(p)
    assert recovered.success
    np.testing.assert_allclose(recovered.x, [.2, .3], atol=1e-7, rtol=0)
