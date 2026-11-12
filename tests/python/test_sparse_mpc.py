"""Sparse and condensed formulations solve the same constrained trajectory."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import (BackendRegistry, JointLimits, JointMPC, OSQPSolver,
                        RobotState, SparseJointMPC, SparseQuadraticProgram)


@pytest.mark.parametrize("horizon", [1, 5, 20])
@pytest.mark.parametrize("slew", [False, True])
def test_sparse_and_condensed_optima_match_with_random_bounds_and_soft_goals(horizon, slew):
    rng = np.random.default_rng(137)
    n = 3
    limits = JointLimits(-rng.uniform(.5, 2, n), rng.uniform(.5, 2, n),
                          rng.uniform(.3, 1., n), rng.uniform(2, 4, n) if slew else None)
    dense = JointMPC(limits, horizon=horizon, dt=.08, tracking_weight=2.7,
                     control_weight=.4, terminal_weight=9.1)
    compact = SparseJointMPC(limits, horizon=horizon, dt=.08, tracking_weight=2.7,
                             control_weight=.4, terminal_weight=9.1)
    state = RobotState(rng.uniform(-.2, .2, n), rng.uniform(-.1, .1, n))
    for k in range(15):
        goal = rng.uniform(-2, 2, n)  # Deliberately includes unreachable soft targets.
        a, b = dense.solve(state, goal), compact.solve(state, goal)
        assert a.solver.success and b.solver.success
        np.testing.assert_allclose(a.plan.velocities, b.plan.velocities, atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(a.plan.positions, b.plan.positions, atol=2e-6, rtol=1e-5)
        q, v = a.plan.sample(state.stamp + .04)
        state = RobotState(q, v, state.stamp + .04)


def test_sparse_problem_is_linear_in_horizon_storage_and_uses_existing_solver():
    limits = JointLimits([-2] * 19, [2] * 19, [1] * 19, [4] * 19)
    state = RobotState(np.zeros(19), np.zeros(19))
    counts = []
    for horizon in (20, 40):
        planner = SparseJointMPC(limits, horizon=horizon)
        problem = planner.formulate(state, np.full(19, .3))
        assert isinstance(problem, SparseQuadraticProgram)
        assert sparse.isspmatrix_csc(problem.hessian) and sparse.isspmatrix_csc(problem.constraint)
        counts.append(problem.hessian.nnz + problem.constraint.nnz)
        result = planner.solve(state, np.full(19, .3))
        assert result.plan is not None and result.solver.violation <= 1e-6
    assert counts[1] < 2.1 * counts[0]


def test_generic_osqp_solver_can_switch_between_dense_and_sparse_workspaces():
    from wholebodyx import QuadraticProgram

    solver = OSQPSolver()
    dense = QuadraticProgram(np.eye(2), [-2, -3], np.eye(2), [-1, -1], [1, 1])
    compact = SparseQuadraticProgram(sparse.eye(2), [-2, -3], sparse.eye(2), [-1, -1], [1, 1])
    for problem in (dense, compact, compact, dense, dense, compact):
        result = solver.solve(problem)
        assert result.success
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-6)


@pytest.mark.parametrize('field', ['hessian', 'constraint'])
@pytest.mark.parametrize('mutation', ['data', 'dtype', 'shape'])
def test_sparse_buffers_are_immutable_but_returned_matrix_metadata_is_independent(field, mutation):
    limits = JointLimits([-2], [2], [1])
    planner = SparseJointMPC(limits, horizon=3)
    state = RobotState([0], [0])
    first = planner.formulate(state, [.3])
    original = {name: getattr(first, name).toarray() for name in ('hessian', 'constraint')}
    for matrix in (first.hessian, first.constraint):
        for part in (matrix.data, matrix.indices, matrix.indptr):
            with pytest.raises(ValueError):
                part.setflags(write=True)
    matrix = getattr(first, field)
    if mutation == 'data':
        matrix.data = np.full_like(matrix.data, np.nan)
    elif mutation == 'dtype':
        matrix.data.dtype = np.int64
    else:
        matrix.data.shape = (1, matrix.data.size)
    second = planner.formulate(state, [.3])
    for name in original:
        np.testing.assert_array_equal(getattr(second, name).toarray(), original[name])
    assert planner.solve(state, [.3]).solver.success


@pytest.mark.parametrize('planner_type', [JointMPC, SparseJointMPC])
@pytest.mark.parametrize('field', ['hessian', 'constraint'])
@pytest.mark.parametrize('buffer', ['data', 'indices', 'indptr'])
@pytest.mark.parametrize('mutation', ['shape', 'replace'])
def test_retained_sparse_metadata_cannot_change_templates_or_solver_snapshots(
    planner_type, field, buffer, mutation
):
    limits = JointLimits([-2., -2.], [2., 2.], [1., 1.], [4., 4.])
    planner = planner_type(limits, horizon=5)
    state, goal = RobotState([.1, -.2], [.05, -.05]), [.7, -.4]
    expected = planner.solve(state, goal).plan
    template = planner._template
    first = (
        template.problem(*planner._vectors(state, goal))
        if planner_type is JointMPC else planner.formulate(state, goal)
    )
    assert planner.solver.solve(first).success
    matrix = getattr(first, field)
    part = getattr(matrix, buffer)
    if mutation == 'shape':
        part.shape = (1, part.size)
    else:
        setattr(matrix, buffer, np.zeros_like(part))
    # Shape and both cached structural flags belong to each matrix object.
    matrix._shape = (1, 1)
    matrix.has_sorted_indices = False
    matrix.has_canonical_format = False
    actual = planner.solve(state, goal).plan
    assert actual is not None
    np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-7, rtol=0)
    np.testing.assert_allclose(actual.velocities, expected.velocities, atol=1e-6, rtol=0)
    for matrix in (template._hessian, template._constraint):
        assert matrix.has_sorted_indices and matrix.has_canonical_format


def test_sparse_backend_works_in_existing_retargeting_and_failure_recovery():
    from wholebodyx.runner import run

    registry = BackendRegistry()
    assert "sparse" in registry.names("planner") and "sparse" not in registry.names("controller")
    result = run("disturbance_recovery", mpc_backend="sparse")
    assert result["disturbances"] == result["tracking_replans"] == 2
    assert result["final_position_error_m"] < 1e-4
    assert result["max_constraint_violation"] <= 1e-6


@pytest.mark.parametrize("hessian", [np.diag([-1., 1.]), [[1, 3], [0, 1]],
                                    [[np.nan]], np.diag([-1, 1, 1, 1])])
def test_sparse_qp_rejects_nonconvex_asymmetric_or_nonfinite_cost(hessian):
    n = len(hessian)
    with pytest.raises(ValueError):
        SparseQuadraticProgram(sparse.csc_matrix(hessian), np.zeros(n), sparse.eye(n),
                                np.zeros(n), np.ones(n))
