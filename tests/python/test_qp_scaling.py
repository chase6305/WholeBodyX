"""Diagonal solver coordinates preserve physical QP inputs and acceptance."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import OSQPSolver, QuadraticProgram, SparseQuadraticProgram


@pytest.mark.parametrize("compact", [False, True])
def test_scaling_preserves_optima_across_value_pattern_and_workspace_changes(compact):
    scale = [.1, 3.]
    solver = OSQPSolver(variable_scaling=scale, adaptive_rho_interval=100)
    reference = OSQPSolver()
    h = np.array([[4., 1.], [1., 2.]])
    a = np.array([[1., 0.], [0., 1.], [1., 1.]])

    def problem(hessian, gradient, constraint):
        count = len(constraint)
        constructor = SparseQuadraticProgram if compact else QuadraticProgram
        return constructor(sparse.csc_matrix(hessian) if compact else hessian, gradient,
                           sparse.csc_matrix(constraint) if compact else constraint,
                           np.full(count, -.5), np.full(count, .7))

    for p in (problem(h, [-8., 3.], a), problem(h, [-3., -2.], a),
              problem(2 * h, [-8., 3.], a), problem(2 * h, [-8., 3.], 2 * a),
              problem(2 * h, [-8., 3.], np.array([[1., 1.]])), problem(h, [-8., 3.], a)):
        actual, expected = solver.solve(p, warm_start=[.1, -.1]), reference.solve(p)
        assert actual.success and expected.success
        np.testing.assert_allclose(actual.x, expected.x, atol=2e-6, rtol=1e-6)
        assert actual.violation <= 1e-6
    solver.reset()
    assert solver._physical_constraint is None
    assert solver.solve(p).success


def test_input_scaling_is_immutable_and_views_have_independent_metadata():
    original = np.array([.1, 3.])
    solver = OSQPSolver(variable_scaling=original)
    original[:] = 99
    first = solver.variable_scaling
    with pytest.raises(ValueError):
        first.setflags(write=True)
    first.shape = (1, 2)
    np.testing.assert_array_equal(solver.variable_scaling, [.1, 3.])
    with pytest.raises(AttributeError):
        solver.variable_scaling = [1, 1]


def test_warm_start_is_scaled_and_result_is_returned_in_physical_units(monkeypatch):
    solver = OSQPSolver(variable_scaling=[.1])
    problem = QuadraticProgram([[2.]], [-4.], [[1.]], [-1.], [.25])
    assert solver.solve(problem).success
    calls, original = [], solver._solver.warm_start

    def warm_start(**kwargs):
        calls.append(kwargs["x"].copy())
        original(**kwargs)

    monkeypatch.setattr(solver._solver, "warm_start", warm_start)
    accepted = solver.solve(problem, warm_start=[.125])
    np.testing.assert_allclose(calls[0], [1.25], atol=0., rtol=0.)
    np.testing.assert_allclose(accepted.x, [.25], atol=1e-7)


def test_independent_residual_uses_physical_constraint_after_mapping_back(monkeypatch):
    solver = OSQPSolver(variable_scaling=[.1])
    problem = QuadraticProgram([[1.]], [0.], [[1.]], [-1.], [1.])
    assert solver.solve(problem).success
    info = SimpleNamespace(status="solved", status_val=1, iter=1)
    monkeypatch.setattr(solver._solver, "solve",
                        lambda **kwargs: SimpleNamespace(info=info, x=np.array([15.])))
    rejected = solver.solve(problem)
    assert not rejected.success and rejected.x is None
    assert rejected.violation == pytest.approx(.5)


@pytest.mark.parametrize("scaling", [[], [0.], [-1.], [np.nan], [np.inf], [[1.]]])
def test_invalid_scaling_is_rejected(scaling):
    with pytest.raises(ValueError):
        OSQPSolver(variable_scaling=scaling)


@pytest.mark.parametrize("interval", [True, -1, .5])
def test_invalid_adaptive_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        OSQPSolver(adaptive_rho_interval=interval)


def test_mismatched_scaling_and_nonfinite_transformation_are_rejected():
    problem = QuadraticProgram([[1.]], [-1.], [[1.]], [-1.], [1.])
    with pytest.raises(ValueError, match="dimensions"):
        OSQPSolver(variable_scaling=[1, 2]).solve(problem)
    with pytest.raises(ValueError, match="scaled QP matrices"):
        OSQPSolver(variable_scaling=[1e308]).solve(problem)
    with pytest.raises(ValueError, match="scaled QP vectors"):
        OSQPSolver(variable_scaling=[1e-3]).solve(problem, warm_start=[1e308])


@pytest.mark.parametrize("scale", [None, [2.]])
@pytest.mark.parametrize("solution", [np.nan, np.inf])
def test_nonfinite_solver_output_is_rejected_with_and_without_scaling(monkeypatch, scale, solution):
    solver = OSQPSolver(variable_scaling=scale)
    problem = QuadraticProgram([[1.]], [0.], [[1.]], [-1.], [1.])
    assert solver.solve(problem).success
    info = SimpleNamespace(status="solved", status_val=1, iter=7)
    monkeypatch.setattr(solver._solver, "solve",
                        lambda **kwargs: SimpleNamespace(info=info, x=np.array([solution])))
    rejected = solver.solve(problem)
    assert rejected.status == "error" and rejected.x is None
    assert rejected.iterations == 7


def test_finite_scaled_solution_overflow_is_rejected(monkeypatch):
    solver = OSQPSolver(variable_scaling=[2.])
    problem = QuadraticProgram([[1.]], [0.], [[1.]], [-1.], [1.])
    assert solver.solve(problem).success
    info = SimpleNamespace(status="solved", status_val=1, iter=7)
    monkeypatch.setattr(solver._solver, "solve", lambda **kwargs: SimpleNamespace(
        info=info, x=np.array([np.finfo(float).max])))
    rejected = solver.solve(problem)
    assert rejected.status == "error" and rejected.x is None
    assert "physical solution" in rejected.detail
