"""Sparse and condensed formulations solve the same constrained trajectory."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import (BackendRegistry, JointLimits, JointMPC, OSQPSolver, QPResult,
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


def test_sparse_plan_and_qp_decision_stay_physical_with_large_joint_offsets():
    center = np.array([1000., -1100., 0.])
    velocity, acceleration = [1., .8, .9], [2., 3., 4.]
    absolute = SparseJointMPC(JointLimits(center - 2., center + 2., velocity, acceleration))
    reference = SparseJointMPC(JointLimits([-2.] * 3, [2.] * 3, velocity, acceleration))
    for index, target in enumerate(([.75, -.55, .1], [-.4, .65, -.2], [.01, .02, .03])):
        state = RobotState(center + [.15, -.2, .1], [.04, -.02, 0.], index * .08)
        goal = center + target
        problem = absolute.formulate(state, goal)
        actual = absolute.solve(state, goal)
        expected = reference.solve(RobotState(state.q - center, state.v, state.stamp), goal - center)
        assert actual.plan is not None and expected.plan is not None
        np.testing.assert_array_equal(problem.lower[:3], state.q)
        np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=1e-7, rtol=0.)
        np.testing.assert_allclose(actual.plan.positions - center, expected.plan.positions,
                                   atol=1e-8, rtol=0.)
        position_size = actual.plan.velocities.size
        np.testing.assert_allclose(actual.solver.x[:position_size].reshape(-1, 3) - center,
                                   expected.solver.x[:position_size].reshape(-1, 3),
                                   atol=1e-8, rtol=0.)
        values = problem.constraint @ actual.solver.x
        violation = max((problem.lower - values).max(), (values - problem.upper).max(), 0.)
        assert actual.solver.violation == violation <= 1e-6


def test_sparse_solver_override_receives_the_absolute_qp(monkeypatch):
    planner = SparseJointMPC(JointLimits([998.], [1002.], [1.]), horizon=5)
    calls = []

    def solve(problem, warm_start=None):
        calls.append(problem)
        return QPResult("infeasible")

    monkeypatch.setattr(planner.solver, "solve", solve)
    outcome = planner.solve(RobotState([1000.], [0.]), [1000.4])
    assert outcome.plan is None and len(calls) == 1
    assert calls[0].lower[0] == 1000.


def test_sparse_solver_override_receives_the_complete_horizon(monkeypatch):
    horizon = 128
    planner = SparseJointMPC(JointLimits([-1.], [1.], [.8], [2.]), horizon=horizon)
    state, goal = RobotState([0.], [0.]), [.001]
    calls = []

    def solve(problem, warm_start=None):
        calls.append(problem)
        return QPResult("infeasible")

    monkeypatch.setattr(planner.solver, "solve", solve)
    assert planner.solve(state, goal).plan is None
    assert len(calls) == 1 and calls[0].gradient.shape == (2 * horizon,)
    assert calls[0].constraint.shape == (4 * horizon, 2 * horizon)


def test_stalled_sparse_prefix_spends_one_iteration_budget_before_full_solve():
    horizon, dt = 128, .05
    limits = JointLimits([-1.], [1.], [.8], [.1])
    state, goal = RobotState([0.], [0.]), [.06]
    full = OSQPSolver(
        variable_scaling=np.r_[np.full(horizon, dt), np.ones(horizon)], adaptive_rho_interval=100,
    )
    expected = SparseJointMPC(limits, horizon, dt, solver=full).solve(state, goal)
    actual = SparseJointMPC(limits, horizon, dt).solve(state, goal)
    assert actual.plan is not None and expected.plan is not None
    assert actual.solver.iterations == expected.solver.iterations + 6000
    np.testing.assert_array_equal(actual.solver.x, expected.solver.x)


@pytest.mark.parametrize("offset", [0., 1000.])
def test_short_step_slew_model_exports_the_original_sparse_decision(offset):
    n, horizon, dt = 6, 60, .004
    phases = np.arange(n) * .37
    center = offset * np.cos(phases)
    limits = JointLimits(center - 1., center + 1., [.8] * n, [2.] * n)
    reference_limits = JointLimits([-1.] * n, [1.] * n, [.8] * n, [2.] * n)
    reference_solver = OSQPSolver(
        variable_scaling=np.r_[np.full(n * horizon, .01), np.ones(n * horizon)],
        adaptive_rho_interval=100,
    )
    reference = SparseJointMPC(reference_limits, horizon=horizon, dt=dt, solver=reference_solver)
    planner = SparseJointMPC(limits, horizon=horizon, dt=dt)
    for k in range(3):
        state = RobotState(center + .01 * np.cos(phases + .1 * k), .03 * np.sin(phases + .4 * k),
                           k * dt)
        goal = center + .55 * np.sin(phases + .025 * k)
        expected = reference.solve(RobotState(state.q - center, state.v, state.stamp), goal - center)
        actual = planner.solve(state, goal)
        assert actual.plan is not None and expected.plan is not None
        assert planner.solver._solver is None
        assert actual.solver.x.shape == (2 * n * horizon,)
        np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=7e-6, rtol=0)
        problem = planner.formulate(state, goal)
        values = problem.constraint @ actual.solver.x
        assert max((problem.lower - values).max(), (values - problem.upper).max(), 0.) <= 1e-6


def test_analytic_sparse_result_defers_the_expanded_matrices():
    planner = SparseJointMPC(JointLimits([-1.], [1.], [.8], [2.]), horizon=128, dt=.02)
    state = RobotState([0.], [0.])
    before = planner.fixed_matrix_bytes
    direct = planner.solve(state, [.001])
    assert direct.plan is not None and direct.solver.iterations == 0
    assert planner.fixed_matrix_bytes == before

    active = planner.solve(state, [1.05])
    assert active.plan is not None and active.solver.iterations > 0
    assert planner.fixed_matrix_bytes > before


def test_short_step_slew_refusal_retries_original_model(monkeypatch):
    planner = SparseJointMPC(JointLimits([-1.], [1.], [.8], [2.]), horizon=40, dt=.004)
    _, solver = planner._slew_problem
    monkeypatch.setattr(solver, "solve", lambda *args, **kwargs: QPResult("iteration_limit", iterations=3))

    result = planner.solve(RobotState([0.], [0.]), [.005])

    assert result.plan is not None and planner.solver._solver is not None
    assert result.solver.iterations > 3
    assert result.solver.detail.startswith("retry after velocity-difference solve:")


def test_short_step_slew_retries_the_warm_workspace_from_cold(monkeypatch):
    planner = SparseJointMPC(JointLimits([-1.], [1.], [.8], [2.]), horizon=40, dt=.004)
    state, goal = RobotState([0.], [0.]), [.05]
    assert planner.solve(state, goal).plan is not None
    _, solver = planner._slew_problem
    workspace = solver._solver
    original = solver.solve
    calls = []

    def solve(problem, warm_start=None):
        calls.append(warm_start is not None)
        if len(calls) == 1:
            return QPResult("iteration_limit", iterations=7, detail="maximum iterations reached")
        return original(problem, warm_start=warm_start)

    monkeypatch.setattr(solver, "solve", solve)
    result = planner.solve(state, goal)

    assert calls == [True, False] and solver._solver is not workspace
    assert result.plan is not None and planner.solver._solver is None
    assert result.solver.iterations > 7 and result.solver.detail.startswith("cold retry after")


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
