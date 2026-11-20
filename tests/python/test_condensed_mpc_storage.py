"""Compressed condensed models preserve the dense extension contract."""

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import JointLimits, JointMPC, OSQPSolver, QuadraticProgram, RobotState, SparseJointMPC
from wholebodyx.sparse_qp import _SparseFixedQP


def test_large_default_solve_never_materializes_dense_diagnostics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default MPC must not expand joint-space matrices")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbidden)
    limits = JointLimits([-2.] * 30, [2.] * 30, [1.] * 30, [4.] * 30)
    planner = JointMPC(limits, horizon=60)
    assert planner.fixed_matrix_bytes < 3_000_000
    result = planner.solve(RobotState(np.zeros(30), np.zeros(30)), np.full(30, .3))
    assert result.plan is not None and result.solver.violation <= 1e-6
    before = planner.fixed_matrix_bytes
    planner.reset()
    assert planner.fixed_matrix_bytes == before
    assert planner.solve(RobotState(np.zeros(30), np.zeros(30)), np.full(30, -.3)).plan is not None


def test_condensed_short_step_uses_sparse_solve_with_velocity_only_result():
    limits = JointLimits([-1., -1.], [1., 1.], [.8, .8], [2., 2.])
    planner = JointMPC(limits, horizon=60, dt=.004)
    reference = SparseJointMPC(limits, horizon=60, dt=.004)
    state = RobotState([0., 0.], [0., 0.])
    assert planner.solve(state, [.0001, -.0001]).solver.iterations == 0
    assert planner._short_step_planner is None
    before = planner.fixed_matrix_bytes
    goal = [.55, -.55]
    outcome = planner.solve(state, goal)
    expected = reference.solve(state, goal)
    assert outcome.plan is not None and expected.plan is not None
    assert planner.solver._solver is None and planner.fixed_matrix_bytes > before
    assert outcome.solver.x.shape == (120,)
    np.testing.assert_allclose(outcome.plan.velocities, expected.plan.velocities, atol=1e-7)
    problem = planner.formulate(state, goal)
    values = problem.constraint @ outcome.solver.x
    assert max((problem.lower - values).max(), (values - problem.upper).max(), 0.) <= 1e-6


@pytest.mark.parametrize("acceleration", [None, [2., 3., 4.]])
def test_custom_solver_receives_dense_velocity_only_problem(acceleration):
    class DenseOnly(OSQPSolver):
        def solve(self, problem, warm_start=None):
            assert type(problem) is QuadraticProgram
            assert isinstance(problem.hessian, np.ndarray)
            assert isinstance(problem.constraint, np.ndarray)
            assert problem.gradient.shape == (21,)
            return super().solve(problem, warm_start)

    limits = JointLimits([-1.] * 3, [1.] * 3, [.8] * 3, acceleration)
    fast, custom = JointMPC(limits, horizon=7), JointMPC(limits, horizon=7, solver=DenseOnly())
    rng = np.random.default_rng(572)
    for k in range(20):
        state = RobotState(rng.uniform(-.4, .4, 3), rng.uniform(-.1, .1, 3), k * .07)
        goal = rng.uniform(-3., 3., 3)
        a, b = fast.solve(state, goal), custom.solve(state, goal)
        assert a.plan is not None and b.plan is not None
        np.testing.assert_allclose(a.plan.positions, b.plan.positions, atol=1e-7, rtol=0)
        np.testing.assert_allclose(a.plan.velocities, b.plan.velocities, atol=1e-6, rtol=0)


@pytest.mark.parametrize("instance_override", [False, True])
def test_overridden_public_formulation_is_used_by_solve(instance_override):
    calls = []

    def formulate(self, state, goal):
        calls.append(1)
        problem = JointMPC.formulate(self, state, goal)
        return QuadraticProgram(problem.hessian, np.zeros_like(problem.gradient),
                                problem.constraint, problem.lower, problem.upper)

    if instance_override:
        from types import MethodType
        planner = JointMPC(JointLimits([-2.], [2.], [1.]))
        planner.formulate = MethodType(formulate, planner)
    else:
        class CustomPlanner(JointMPC):
            pass
        CustomPlanner.formulate = formulate
        planner = CustomPlanner(JointLimits([-2.], [2.], [1.]))
    result = planner.solve(RobotState([0.], [0.]), [1.])
    assert calls == [1] and result.plan is not None
    np.testing.assert_allclose(result.plan.velocities, 0., atol=1e-12)


def test_dense_diagnostic_metadata_cannot_poison_compressed_model():
    planner = JointMPC(JointLimits([-2., -2.], [2., 2.], [1., 1.]), horizon=6)
    state, goal = RobotState([.1, -.2], [0., 0.]), [.7, -.3]
    before = planner.fixed_matrix_bytes
    first = planner.formulate(state, goal)
    assert planner.fixed_matrix_bytes == before + first.hessian.nbytes + first.constraint.nbytes
    expected = planner.solve(state, goal).plan
    first.hessian.dtype = np.int64
    first.constraint.shape = (first.constraint.size,)
    first.lower.setflags(write=True)
    first.lower[:] = 100.
    second = planner.formulate(state, goal)
    assert second.hessian.dtype == np.float64 and second.constraint.ndim == 2
    actual = planner.solve(state, goal).plan
    np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-7)


def test_instance_solver_override_keeps_dense_contract(monkeypatch):
    planner = JointMPC(JointLimits([-2.], [2.], [1.]))
    original, calls = planner.solver.solve, []

    def solve(problem, warm_start=None):
        calls.append(1)
        assert type(problem) is QuadraticProgram
        return original(problem, warm_start)

    monkeypatch.setattr(planner.solver, "solve", solve)
    assert planner.solve(RobotState([0.], [0.]), [.02]).plan is not None
    assert calls == [1]


@pytest.mark.parametrize("horizon, dt", [(1, .2), (7, .03), (60, .1)])
def test_inactive_joint_mpc_matches_the_public_stationarity_equations(horizon, dt):
    limits = JointLimits([-3., -2., -1.], [3., 4., 2.], [.8, 1.2, .9], [6., 8., 10.])
    planner = JointMPC(limits, horizon=horizon, dt=dt, tracking_weight=7.,
                       control_weight=.3, terminal_weight=14.)
    state = RobotState([.15, -.25, .05], [.02, -.01, 0.], stamp=1e8)
    goal = state.q + [.006, -.008, .012]
    problem = planner.formulate(state, goal)

    result = planner.solve(state, goal)

    assert result.plan is not None
    assert result.solver.iterations == 0 and planner.solver._solver is None
    decision = result.solver.x
    constrained_values = problem.constraint @ decision
    assert np.all(problem.lower <= constrained_values)
    assert np.all(constrained_values <= problem.upper)
    np.testing.assert_allclose(problem.hessian @ decision, -problem.gradient, atol=5e-14)
    expected = np.linalg.solve(problem.hessian, -problem.gradient)
    np.testing.assert_allclose(result.plan.velocities.ravel(), expected, rtol=0., atol=5e-13)


@pytest.mark.parametrize("constraint", ["position", "velocity", "initial_slew"])
def test_inactive_joint_mpc_falls_back_for_tiny_positive_constraint_residuals(constraint):
    velocity = .5
    position = .1 * velocity
    acceleration = velocity / .1
    if constraint == "position":
        position -= 2e-9
    elif constraint == "velocity":
        velocity -= 2e-9
    else:
        acceleration -= 2e-8
    limits = JointLimits([-1.], [position if constraint == "position" else 1.],
                         [velocity if constraint == "velocity" else 1.],
                         [acceleration] if constraint == "initial_slew" else None)
    planner = JointMPC(limits, horizon=1)
    state = RobotState([0.], [0.])
    # Horizon one solves (r + dt²*w)*u = dt*w*(goal - q).
    goal = [.5 * (.1 + .1**2 * 30.) / (.1 * 30.)]

    outcome = planner.solve(state, goal)

    assert outcome.plan is not None
    assert outcome.solver.iterations > 0 and planner.solver._solver is not None


def test_inactive_joint_mpc_can_resume_after_an_active_solve_and_reset():
    limits = JointLimits([-1.], [1.], [.8], [4.])
    planner = JointMPC(limits, horizon=10)
    baseline = JointMPC(limits, horizon=10, unconstrained_fast_path=False)
    for index, goal in enumerate([.01, .3, -.02, -.4, 0.]):
        state = RobotState([0.], [0.], index * .1)
        actual, expected = planner.solve(state, [goal]), baseline.solve(state, [goal])
        assert actual.plan is not None and expected.plan is not None
        assert (actual.solver.iterations == 0) == (index % 2 == 0)
        np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities,
                                   rtol=0., atol=1e-6)
        if index == 3:
            planner.reset()
            baseline.reset()


def test_inactive_joint_mpc_does_not_skip_replacement_adapter():
    planner = JointMPC(JointLimits([-1.], [1.], [.8]))
    replacement = OSQPSolver()
    planner.solver = replacement

    result = planner.solve(RobotState([0.], [0.]), [.01])

    assert result.plan is not None
    assert result.solver.iterations > 0 and replacement._solver is not None


@pytest.mark.parametrize(
    "goal, stamp, explicit", [(.3, .04, False), (-.3, .04, True), (.3, .14, True)]
)
def test_return_to_recent_qp_only_reuses_matching_workspace(goal, stamp, explicit, monkeypatch):
    planner = JointMPC(JointLimits([-1.], [1.], [.8], [4.]))
    assert planner.solve(RobotState([0.], [0.]), [.3]).solver.iterations > 0
    assert planner.solve(RobotState([0.], [0.], .02), [.02]).solver.iterations == 0
    workspace = planner.solver._solver
    warm_start, calls = workspace.warm_start, []

    def record(**values):
        calls.append(values)
        return warm_start(**values)

    monkeypatch.setattr(workspace, "warm_start", record)
    outcome = planner.solve(RobotState([0.], [0.], stamp), [goal])
    assert outcome.plan is not None and outcome.solver.iterations > 0
    assert bool(calls) == explicit


@pytest.mark.parametrize("temporal", [[[-1.]], [[1., .5], [0., 1.]], [[np.nan]], [[np.inf]]])
def test_separable_cost_builder_rejects_invalid_temporal_cost(temporal):
    with pytest.raises(ValueError):
        _SparseFixedQP.separable(temporal, 2, sparse.eye(2 * len(temporal)))


def test_shared_adapter_can_switch_condensed_costs_and_dimensions():
    solver = OSQPSolver()
    planners = [JointMPC(JointLimits([-2.] * n, [2.] * n, [1.] * n), horizon=7,
                         tracking_weight=w, solver=solver) for n, w in [(2, 10.), (2, 2.), (3, 10.)]]
    for _ in range(3):
        for planner in planners:
            n = planner.limits.lower.size
            state, goal = RobotState(np.zeros(n), np.zeros(n)), np.full(n, .3)
            actual = planner.solve(state, goal)
            fresh = JointMPC(planner.limits, horizon=7, tracking_weight=planner.tracking_weight).solve(state, goal)
            assert actual.plan is not None and fresh.plan is not None
            np.testing.assert_allclose(actual.plan.velocities, fresh.plan.velocities, atol=1e-6, rtol=0)
