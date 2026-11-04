"""Scaled CSC updates retain coordinate order and the physical optimum."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram
from wholebodyx.qp import _scaled_csc


def problem(compact, h, g, a, lower, upper):
    if compact:
        return SparseQuadraticProgram(sparse.csc_matrix(h), g, sparse.csc_matrix(a), lower, upper)
    return QuadraticProgram(h, g, a, lower, upper)


@pytest.mark.parametrize("compact", [False, True])
def test_changed_scaled_constraints_keep_optimum_with_unequal_bounds(compact):
    solver = OSQPSolver(variable_scaling=[.5, 2.])
    initial = problem(compact, np.eye(2), [-2., 1.], [[1., 1.], [1., 2.]], [-2., -1.], [1., 2.])
    assert solver.solve(initial).success
    workspace = solver._solver
    changed = problem(compact, np.eye(2), [4., 5.], [[3., 3.], [-1., -3.]], [-2., -1.], [1., 2.])
    result = solver.solve(changed)
    assert result.success
    # The optimum is (0, -2/3). A reversed value update returned (0, -1/3),
    # which passes the independent feasibility check but has a worse cost.
    np.testing.assert_allclose(result.x, [0., -2. / 3.], atol=2e-7, rtol=0)
    assert .5 * result.x @ result.x + changed.gradient @ result.x == pytest.approx(-28. / 9., abs=1e-6)
    assert solver._solver is workspace
    assert solver._sparse_constraint.has_sorted_indices


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("changes", ["hessian", "constraint", "both"])
def test_random_scaled_updates_match_cold_unscaled_reference(compact, changes):
    rng = np.random.default_rng(20261024)
    solver = OSQPSolver(variable_scaling=[.2, 1., 3.])
    fixed_a = rng.normal(size=(5, 3))
    for k in range(50):
        b = rng.normal(size=(3, 3))
        h = b.T @ b + np.eye(3) if changes != "constraint" else np.eye(3)
        a = rng.normal(size=(5, 3)) if changes != "hessian" else fixed_a
        # All states are feasible at zero; distinct row bounds expose a row
        # permutation that symmetric identical bounds would conceal.
        lower, upper = -rng.uniform(.1, 2., 5), rng.uniform(.1, 2., 5)
        p = problem(compact, h, rng.normal(size=3) * 4, a, lower, upper)
        actual, expected = solver.solve(p), OSQPSolver().solve(p)
        assert actual.success and expected.success, (k, actual, expected)
        np.testing.assert_allclose(actual.x, expected.x, atol=2e-6, rtol=2e-6)
        assert actual.violation <= 1e-6
        if k == 24:
            solver.reset()


@pytest.mark.parametrize("rows", [False, True])
def test_direct_scaling_matches_sparse_products_without_mutating_storage(rows):
    # Unsorted entries, duplicate coordinates, an empty column and zeros.
    a = sparse.csc_matrix(([2., 3., -1., 0., 5.], [2, 0, 2, 1, 0], [0, 3, 3, 5]), shape=(3, 3))
    before = tuple(v.copy() for v in (a.data, a.indices, a.indptr))
    scale = np.array([.1, 2., 3.])
    diagonal = sparse.diags(scale, format="csc")
    expected = diagonal @ a @ diagonal if rows else a @ diagonal
    actual = _scaled_csc(a, scale, rows=rows)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    assert actual.has_canonical_format
    for old, current in zip(before, (a.data, a.indices, a.indptr)):
        np.testing.assert_array_equal(old, current)


@pytest.mark.parametrize("compact", [False, True])
def test_scaling_underflow_changes_workspace_pattern_and_recovers(compact):
    solver = OSQPSolver(variable_scaling=[1e-200, 1.])
    for a in ([[1e-200, 0.], [0., 1.]], [[1., 0.], [0., 1.]], [[1e-200, 0.], [0., 1.]]):
        p = problem(compact, np.diag([1., 2.]), [0., -1.], a, [-1., -1.], [1., 1.])
        result = solver.solve(p)
        assert result.success
        np.testing.assert_allclose(result.x, [0., .5], atol=1e-6)
        assert solver._sparse_constraint.nnz == (1 if a[0][0] == 1e-200 else 2)


@pytest.mark.parametrize("compact", [False, True])
def test_weak_constraint_updates_do_not_prematurely_report_infeasible(compact):
    from pathlib import Path
    import json

    fixture = json.loads((Path(__file__).parents[1] / 'data/scaled_qp_weak_constraint.json').read_text())
    solver = OSQPSolver(variable_scaling=fixture['scaling'])
    for row in fixture['problems']:
        p = problem(compact, fixture['hessian'], row['g'], fixture['constraint'], row['lower'], row['upper'])
        expected = np.asarray(row['expected'])
        ax = np.asarray(fixture['constraint']) @ expected
        assert np.max(p.lower - ax) <= 1e-12 and np.max(ax - p.upper) <= 1e-12
        result = solver.solve(p)
        assert result.success, (row['round'], result)
        np.testing.assert_allclose(result.x, expected, atol=1e-6, rtol=0)
        assert result.violation <= 1e-6


@pytest.mark.parametrize('name', ['eps_prim_inf', 'eps_dual_inf'])
@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_invalid_infeasibility_certificate_tolerance_is_rejected(name, value):
    with pytest.raises(ValueError, match='finite and positive'):
        OSQPSolver(**{name: value})


def test_explicit_certificate_tolerances_reach_the_workspace_and_preserve_real_infeasibility():
    solver = OSQPSolver(eps_prim_inf=1e-9, eps_dual_inf=1e-10)
    # Both rows are individually consistent; the shared variable cannot satisfy both.
    p = QuadraticProgram([[1.]], [0.], [[1.], [1.]], [1., -2.], [2., -1.])
    result = solver.solve(p)
    assert result.status == 'infeasible' and result.x is None
    assert solver._solver.settings.eps_prim_inf == 1e-9
    assert solver._solver.settings.eps_dual_inf == 1e-10
