"""Row scaling recovers accuracy at a tiny active constraint."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram


def fixture():
    path = Path(__file__).resolve().parents[1] / 'data/scaled_qp_small_row.json'
    data = json.loads(path.read_text())
    return tuple(np.asarray(data[name]) for name in
                 ('hessian', 'constraint', 'gradient', 'lower', 'upper', 'expected', 'dual'))


def test_small_row_kkt_fixture_distinguishes_primal_residual_from_decision_error():
    h, a, g, lower, upper, expected, dual = fixture()
    assert np.linalg.eigvalsh(h).min() > 1.
    np.testing.assert_allclose(h @ expected + g + a.T @ dual, 0., atol=1e-14)
    assert np.all(a @ expected >= lower) and np.all(a @ expected <= upper)
    np.testing.assert_allclose(dual * (upper - a @ expected), 0., atol=1e-14)
    assert np.all(dual >= 0.)
    # A physically small residual in the active 8e-5 row does not imply a
    # similarly small distance to the unique KKT optimum.
    perturbed = expected.copy()
    perturbed[0] += 3.5e-5
    assert np.max(np.abs(perturbed - expected)) > 3e-5
    assert max(np.max(lower - a @ perturbed), np.max(a @ perturbed - upper), 0.) < 1e-8


@pytest.mark.parametrize('compact', [False, True])
def test_row_normalization_recovers_small_row_optimum_in_physical_coordinates(compact):
    h, a, g, lower, upper, expected, _ = fixture()
    row_norm = np.max(np.abs(a), axis=1)
    assert row_norm.min() < 1e-4 and np.all(row_norm > 0.)
    constructor = SparseQuadraticProgram if compact else QuadraticProgram
    problem = constructor(
        sparse.csc_matrix(h) if compact else h, g,
        sparse.csc_matrix(a / row_norm[:, None]) if compact else a / row_norm[:, None],
        lower / row_norm, upper / row_norm,
    )
    result = OSQPSolver(variable_scaling=np.geomspace(.2, 3., 3)).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, expected, atol=1e-7, rtol=0)
    assert max(np.max(lower - a @ result.x), np.max(a @ result.x - upper), 0.) < 1e-8


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, np.sqrt(.6), 3.]])
def test_adapter_normalization_recovers_small_row_optimum_without_rewriting_problem(compact, scale):
    h, a, g, lower, upper, expected, _ = fixture()
    constructor = SparseQuadraticProgram if compact else QuadraticProgram
    problem = constructor(sparse.csc_matrix(h) if compact else h, g,
                          sparse.csc_matrix(a) if compact else a, lower, upper)
    solver = OSQPSolver(variable_scaling=scale, normalize_constraints=True)
    for warm_start in (None, expected, None):
        result = solver.solve(problem, warm_start=warm_start)
        assert result.success
        np.testing.assert_allclose(result.x, expected, atol=1e-7, rtol=0)
        physical = max((lower - a @ result.x).max(), (a @ result.x - upper).max(), 0.)
        assert result.violation == pytest.approx(physical, abs=1e-14)
