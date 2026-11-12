"""Optional row normalization preserves physical inputs and solver lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import (JointLimits, JointMPC, OSQPSolver, QuadraticProgram, RobotState,
                        SparseJointMPC, SparseQuadraticProgram)
from wholebodyx.qp import _FixedQP, _normalized_constraint
from wholebodyx.sparse_qp import _SparseFixedQP


def problem(compact, hessian, gradient, constraint, lower, upper):
    if compact:
        return SparseQuadraticProgram(sparse.csc_matrix(hessian), gradient,
                                      sparse.csc_matrix(constraint), lower, upper)
    return QuadraticProgram(hessian, gradient, constraint, lower, upper)


@pytest.mark.parametrize('value', [None, 0, 1, 'yes', [], np.bool_(True)])
def test_normalization_requires_a_boolean(value):
    with pytest.raises(ValueError, match='normalize_constraints must be boolean'):
        OSQPSolver(normalize_constraints=value)


def test_normalization_is_opt_in_and_read_only_across_reset():
    assert OSQPSolver().normalize_constraints is False
    solver = OSQPSolver(normalize_constraints=True)
    with pytest.raises(AttributeError):
        solver.normalize_constraints = False
    solver.reset()
    assert solver.normalize_constraints is True


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_kkt_optima_survive_value_pattern_and_dimension_changes(compact, scale):
    rng = np.random.default_rng(20261112)
    solver = OSQPSolver(normalize_constraints=True, variable_scaling=scale)
    previous_workspace = previous_divisor = None
    for k in range(48):
        n = 2 if scale is not None else (1, 2, 4)[k // 8 % 3]
        # Two identical steps, H-only, A-only and simultaneous updates; zero
        # entries enter and leave the CSC pattern, and the row count changes.
        if k % 8 in (0, 4, 6):
            base_a = rng.normal(size=(2 * n + (k // 8 % 2), n))
            if k % 16 == 0:
                base_a[rng.random(base_a.shape) < .3] = 0.
            # The active row must not vanish in the one-variable case.
            base_a[0, 0] = .7
            weights = np.geomspace(1e-5, 10., len(base_a))
            a = weights[:, None] * base_a
        if k % 8 in (0, 2, 6):
            b = rng.normal(size=(n, n))
            h = b.T @ b + np.eye(n)
        expected = rng.uniform(-.3, .3, n)
        ax = a @ expected
        lower, upper = ax - weights, ax + weights
        dual = np.zeros(len(a))
        dual[0] = (1 if k % 2 else -1) / weights[0]
        if dual[0] < 0:
            lower[0] = ax[0]
        else:
            upper[0] = ax[0]
        g = -h @ expected - a.T @ dual
        p = problem(compact, h, g, a, lower, upper)
        result = solver.solve(p, warm_start=np.zeros(n))
        assert result.success, (k, result)
        np.testing.assert_allclose(result.x, expected, atol=2e-6, rtol=0)
        physical_violation = max((lower - a @ result.x).max(initial=0),
                                 (a @ result.x - upper).max(initial=0), 0.)
        assert result.violation == pytest.approx(physical_violation, abs=1e-14)
        # H-only and vector-only updates preserve the A normalization cache.
        if k % 8 in (1, 2, 3, 5, 7):
            assert solver._constraint_divisor is previous_divisor
            assert solver._solver is previous_workspace
        previous_workspace, previous_divisor = solver._solver, solver._constraint_divisor
        if k == 23:
            solver.reset()
            assert solver._constraint_divisor is solver._physical_constraint is None


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_fixed_templates_reuse_normalization_and_keep_original_matrices(compact, scale):
    a = np.array([[1e-4, 0.], [0., -20.]])
    template_type = _SparseFixedQP if compact else _FixedQP
    template = template_type(sparse.eye(2) if compact else np.eye(2),
                             sparse.csc_matrix(a) if compact else a)
    solver = OSQPSolver(normalize_constraints=True, variable_scaling=scale)
    for k, x in enumerate(([.2, -.3], [-.1, .4], [.2, -.3])):
        x = np.asarray(x)
        # Equality bounds exercise both sides, with opposite coefficient signs.
        p = template.problem(-x, a @ x, a @ x)
        result = solver.solve(p, warm_start=x)
        assert result.success
        np.testing.assert_allclose(result.x, x, atol=1e-7, rtol=0)
        if k:
            assert solver._constraint_divisor is divisor
        divisor = solver._constraint_divisor
        physical_a = p.constraint.toarray() if compact else p.constraint
        np.testing.assert_array_equal(physical_a, a)
        np.testing.assert_array_equal(p.lower, a @ x)
        np.testing.assert_array_equal(p.upper, a @ x)
    with pytest.raises(ValueError):
        divisor.setflags(write=True)


@pytest.mark.parametrize('compact', [False, True])
def test_proportional_a_update_still_updates_bound_divisors(compact):
    solver = OSQPSolver(normalize_constraints=True)
    for coefficient, expected in ((1e-4, .2), (2e-4, .4), (1e-4, .2)):
        p = problem(compact, [[1.]], [-1.], [[coefficient]], [-np.inf],
                    [coefficient * expected])
        result = solver.solve(p)
        assert result.success
        np.testing.assert_allclose(result.x, [expected], atol=1e-7, rtol=0)
        if coefficient == 2e-4:
            assert solver._solver is workspace
        workspace = solver._solver


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_empty_and_zero_rows_and_infinite_bounds(compact, scale):
    solver = OSQPSolver(normalize_constraints=True, variable_scaling=scale)
    for a, lower, upper in ((np.empty((0, 2)), [], []),
                             ([[0., 0.]], [-np.inf], [np.inf]),
                             ([[0., 0.], [1e-6, 0.]], [0., -np.inf], [0., 2e-7])):
        p = problem(compact, np.eye(2), [-.1, .2], a, lower, upper)
        result = solver.solve(p)
        assert result.success
        np.testing.assert_allclose(result.x, [.1, -.2], atol=1e-7, rtol=0)
    infeasible = problem(compact, np.eye(2), [0., 0.], [[0., 0.]], [1.], [2.])
    result = solver.solve(infeasible)
    assert result.status == 'infeasible' and result.x is None


@pytest.mark.parametrize('scale', [None, [.1]])
def test_acceptance_uses_original_physical_rows_even_when_normalized_residual_is_small(
    monkeypatch, scale
):
    solver = OSQPSolver(normalize_constraints=True, variable_scaling=scale)
    p = QuadraticProgram([[1.]], [0.], [[1e8]], [-np.inf], [0.])
    assert solver.solve(p).success
    info = SimpleNamespace(status='solved', status_val=1, iter=7)
    # Normalized residual is 1e-8, but the physical residual is one whole unit.
    z = np.array([1e-8 / (1. if scale is None else scale[0])])
    monkeypatch.setattr(solver._solver, 'solve',
                        lambda **kwargs: SimpleNamespace(info=info, x=z))
    result = solver.solve(p)
    assert result.status == 'error' and result.x is None and result.iterations == 7
    assert result.violation == pytest.approx(1.)


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('a,lower,upper', [
    ([[1e-300, 0.]], [-np.inf], [1e100]),   # Bound overflow.
    ([[1e300, 0.]], [-1e-300], [1.]),      # Bound becomes zero.
    ([[1e300, 1e-300]], [-1.], [1.]),      # Coefficient becomes zero.
])
def test_unrepresentable_normalization_preserves_healthy_workspace(compact, a, lower, upper):
    solver = OSQPSolver(normalize_constraints=True)
    valid = problem(compact, np.eye(2), [-.2, .3], np.eye(2), [-1., -1.], [1., 1.])
    assert solver.solve(valid).success
    workspace, divisor = solver._solver, solver._constraint_divisor
    invalid = problem(compact, np.eye(2), [-.2, .3], a, lower, upper)
    with pytest.raises(ValueError, match='unrepresentable normalized QP constraints'):
        solver.solve(invalid)
    assert solver._solver is workspace and solver._constraint_divisor is divisor
    result = solver.solve(valid)
    assert result.success
    np.testing.assert_allclose(result.x, [.2, -.3], atol=1e-7)


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('coefficient,bad_bound', [(1e-300, 1e100), (1e300, 1e-300)])
def test_invalid_bounds_with_cached_a_do_not_update_the_backend(compact, coefficient, bad_bound):
    solver = OSQPSolver(normalize_constraints=True)
    valid = problem(compact, [[1.]], [0.], [[coefficient]], [0.], [0.])
    assert solver.solve(valid).success
    workspace, divisor = solver._solver, solver._constraint_divisor
    invalid = problem(compact, [[1.]], [0.], [[coefficient]], [0.], [bad_bound])
    with pytest.raises(ValueError, match='unrepresentable normalized QP constraints'):
        solver.solve(invalid)
    assert solver._solver is workspace and solver._constraint_divisor is divisor
    assert solver.solve(valid).success


@pytest.mark.parametrize('compact', [False, True])
def test_public_a_mutation_rebuilds_normalization_without_changing_physical_bounds(compact):
    p = problem(compact, [[1.]], [-1.], [[1e-4]], [-np.inf], [2e-5])
    solver = OSQPSolver(normalize_constraints=True)
    assert solver.solve(p).success
    if compact:
        p.constraint.data = p.constraint.data * 2
    else:
        p.constraint.setflags(write=True)
        p.constraint[:] *= 2
    result = solver.solve(p)
    assert result.success
    np.testing.assert_allclose(result.x, [.1], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(p.upper, [2e-5])


def test_finite_tiny_row_scale_is_accepted():
    solver = OSQPSolver(normalize_constraints=True)
    coefficient = 1e-300
    p = QuadraticProgram([[1.]], [-1.], [[coefficient]],
                         [coefficient * .2], [coefficient * .2])
    result = solver.solve(p, warm_start=[.2])
    assert result.success
    np.testing.assert_allclose(result.x, [.2], atol=1e-7, rtol=0)
    assert result.violation <= 1e-6


def test_smallest_subnormal_row_can_be_normalized_without_reciprocal_overflow():
    smallest = np.nextafter(0., 1.)
    p = QuadraticProgram([[1.]], [-1.], [[smallest]], [0.], [0.])
    result = OSQPSolver(normalize_constraints=True).solve(p)
    assert result.success
    np.testing.assert_allclose(result.x, [0.], atol=1e-7, rtol=0)


def test_unit_row_norms_skip_bound_transforms_without_sharing_mutable_csc_storage():
    matrix = sparse.csc_matrix([[1., -.5], [0., -1.], [0., 0.]])
    normalized, divisor = _normalized_constraint(matrix, np.array([.2, 3.]))
    assert divisor is None
    np.testing.assert_array_equal(normalized.toarray(), matrix.toarray() * [.2, 3.])
    normalized.data[:] = 99.
    np.testing.assert_array_equal(matrix.toarray(), [[1., -.5], [0., -1.], [0., 0.]])


def test_normalization_combines_duplicate_entries_and_ignores_stale_sparse_format_flags():
    # Unsorted rows, duplicate coordinates, a zero row and an empty column.
    matrix = sparse.csc_matrix(([2., 3., -1., 0., 5.], [2, 0, 2, 1, 0], [0, 3, 3, 5]),
                               shape=(3, 3))
    expected = matrix.toarray()
    original = tuple(value.copy() for value in (matrix.data, matrix.indices, matrix.indptr))
    # Replaced public buffers can leave these metadata flags stale.
    matrix.has_sorted_indices = matrix.has_canonical_format = True
    normalized, divisor = _normalized_constraint(matrix)
    np.testing.assert_array_equal(divisor, [5., 1., 1.])
    np.testing.assert_array_equal(normalized.toarray(), expected / divisor[:, None])
    assert normalized.has_canonical_format and normalized.nnz == 3
    for before, after in zip(original, (matrix.data, matrix.indices, matrix.indptr)):
        np.testing.assert_array_equal(after, before)
    normalized.data[:] = 99.
    np.testing.assert_array_equal(matrix.toarray(), expected)


def test_solver_can_switch_dense_and_sparse_models_with_nonunit_rows():
    solver = OSQPSolver(normalize_constraints=True, variable_scaling=[.2, 3.])
    for compact in (False, True, True, False, False, True):
        p = problem(compact, np.eye(2), [-1., 1.], [[1e-4, 0.], [0., -20.]],
                    [-np.inf, -np.inf], [2e-5, 6.])
        result = solver.solve(p)
        assert result.success
        np.testing.assert_allclose(result.x, [.2, -.3], atol=1e-7, rtol=0)
        if compact:
            assert solver._solver is workspace
        workspace = solver._solver


@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('scale', [None, [.2, 3.]])
def test_identical_sparse_buffers_in_a_different_format_are_not_the_same_constraint(normalize, scale):
    p = SparseQuadraticProgram(sparse.eye(2), [-4., -2.],
                               sparse.csc_matrix([[1., 2.], [0., 3.]]), [-np.inf] * 2, [1.] * 2)
    solver = OSQPSolver(normalize_constraints=normalize, variable_scaling=scale)
    for matrix_type, expected in ((sparse.csc_matrix, [2.6, -.8]),
                                  (sparse.csr_matrix, [1., -1. / 3.]),
                                  (sparse.csc_matrix, [2.6, -.8])):
        # Changing compressed storage interpretation preserves data, indices,
        # indptr and shape, but transposes this square constraint matrix.
        p.constraint.__class__ = matrix_type
        result = solver.solve(p)
        assert result.success
        np.testing.assert_allclose(result.x, expected, atol=2e-7, rtol=0)
        ax = p.constraint @ result.x
        physical = max((ax - p.upper).max(initial=0), (p.lower - ax).max(initial=0), 0.)
        assert physical <= 1e-6
        assert result.violation == pytest.approx(physical, abs=1e-14)


@pytest.mark.parametrize('planner_type', [JointMPC, SparseJointMPC])
def test_custom_normalizing_solver_preserves_mpc_plans_and_physical_limits(planner_type):
    limits = JointLimits([-1.] * 3, [1.] * 3, [.7] * 3, [3.] * 3)
    solver = OSQPSolver(normalize_constraints=True)
    planner = planner_type(limits, horizon=5, dt=.08, solver=solver)
    reference = planner_type(limits, horizon=5, dt=.08)
    state = RobotState([.2, -.3, .1], [0., 0., 0.])
    for k in range(24):
        goal = np.sin(k / 8. + np.arange(3)) * .5
        actual, expected = planner.solve(state, goal), reference.solve(state, goal)
        assert actual.solver.success and expected.solver.success
        plan = actual.plan
        np.testing.assert_allclose(plan.positions, expected.plan.positions, atol=2e-6, rtol=0)
        np.testing.assert_allclose(plan.velocities, expected.plan.velocities, atol=3e-6, rtol=0)
        assert np.max(limits.lower - plan.positions) <= 1e-6
        assert np.max(plan.positions - limits.upper) <= 1e-6
        assert np.max(np.abs(plan.velocities) - limits.velocity) <= 1e-6
        changes = np.diff(np.vstack((state.v, plan.velocities)), axis=0)
        assert np.max(np.abs(changes) - limits.acceleration * plan.dt) <= 1e-6
        q, v = plan.sample(state.stamp + .04)
        state = RobotState(q, v, state.stamp + .04)
        if k == 11:
            planner.reset()
            reference.reset()


@pytest.mark.parametrize('compact', [False, True])
def test_soak_generator_checks_kkt_optima_across_eleven_decades_of_physical_row_units(compact):
    from soak_scaled_qp import Case

    case = Case(3, compact, 20261112, solver_normalize_rows=True, row_exponents=(-10., 1.))
    for _ in range(80):
        case.step()
    assert case.report['solves'] == case.report['attempts'] == 80
    assert case.report['max_solution_difference'] <= 3e-5
    assert case.report['max_physical_violation'] <= 1e-6

