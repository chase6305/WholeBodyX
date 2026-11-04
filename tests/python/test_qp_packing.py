"""Dense CSC packing must preserve structure, scaling and workspace isolation."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram
from wholebodyx.qp import _UpperCSC, _upper_csc


def assert_packing(actual, matrix):
    expected = sparse.csc_matrix(np.triu(matrix))
    for field in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.shape == expected.shape
    assert actual.has_sorted_indices and actual.has_canonical_format


@pytest.mark.parametrize('joints', [1, 6, 19, 30, 100])
def test_cached_packing_matches_fresh_csc_through_support_changes(joints):
    rng, cache = np.random.default_rng(20261022), _UpperCSC()
    for count in range(30):
        h = rng.normal(size=(joints, joints))
        h[rng.random(h.shape) < .6] = 0
        for changed in (h, h * .5, -h, h.T, np.zeros_like(h), np.eye(joints)):
            assert_packing(_upper_csc(changed, cache), changed)


def test_same_count_position_exchange_rebuilds_the_index_template():
    cache = _UpperCSC()
    first = np.array([[2., .1, 0.], [.1, 2., 0.], [0., 0., 2.]])
    second = np.array([[2., 0., .1], [0., 2., 0.], [.1, 0., 2.]])
    _upper_csc(first, cache)
    assert_packing(_upper_csc(second, cache), second)
    assert_packing(_upper_csc(first, cache), first)


def test_cache_owns_immutable_metadata_separate_from_returned_csc():
    cache = _UpperCSC()
    h = np.diag([1., 2., 3.])
    first = _upper_csc(h, cache)
    first.indices[:] = 0
    first.indptr[:] = 0
    first.data[:] = 99
    second = _upper_csc(h * 2, cache)
    assert_packing(second, h * 2)
    with pytest.raises(ValueError):
        second.indices.setflags(write=True)
    second.indices.shape = (1, 3)
    second.indptr.dtype = np.float32
    assert_packing(_upper_csc(h * 3, cache), h * 3)


def test_packing_cache_retains_only_one_shape_and_clears_for_sparse_input():
    cache = _UpperCSC()
    for n in (30, 1, 19, 6):
        assert_packing(_upper_csc(np.eye(n), cache), np.eye(n))
        assert cache.rows.size == cache.columns.size == n
        assert cache.indptr.size == n + 1
    matrix = sparse.csc_matrix([[2., -.1], [-.1, 1.]])
    actual = _upper_csc(matrix, cache)
    assert_packing(actual, matrix.toarray())
    assert cache.shape is None and cache.rows is cache.columns is cache.indptr is None


@pytest.mark.parametrize('scaling', [None, [.2, 2., .5]])
def test_solver_switches_dimensions_support_and_sparse_formats_with_scaling(scaling):
    solver = OSQPSolver(variable_scaling=scaling)
    reference = OSQPSolver(variable_scaling=scaling)
    rng = np.random.default_rng(20261022)
    for k in range(45):
        n = 3 if scaling is not None else (1, 6, 3)[k % 3]
        jac = np.eye(n) if k % 5 == 0 else rng.normal(size=(n, n))
        h = jac.T @ jac + .1 * np.eye(n)
        g = rng.normal(size=n)
        bounds = np.full(n, .3)
        if k % 7 == 0:
            problem = SparseQuadraticProgram(sparse.csc_matrix(h), g, sparse.eye(n), -bounds, bounds)
        else:
            problem = QuadraticProgram(h, g, np.eye(n), -bounds, bounds)
        actual = solver.solve(problem)
        reference.reset()
        expected = reference.solve(QuadraticProgram(h, g, np.eye(n), -bounds, bounds))
        assert actual.success and expected.success
        np.testing.assert_allclose(actual.x, expected.x, atol=1e-6, rtol=0)
        assert actual.violation <= 1e-6
    solver.reset()
    assert solver._upper_cache.shape is None


def test_dense_csc_template_is_unscaled_even_when_scaled_storage_underflows():
    # Only test packing here: tiny solver coordinates are not a useful QP.
    cache = _UpperCSC()
    first = np.array([[1., 1e-200], [1e-200, 1.]])
    h = _upper_csc(first, cache)
    scaled = (sparse.diags([1e-100, 1e-100]) @ h @ sparse.diags([1e-100, 1e-100])).tocsc()
    assert scaled.nnz < h.nnz
    second = np.array([[2., .25], [.25, 3.]])
    assert_packing(_upper_csc(second, cache), second)
