import numpy as np
import pytest

from wholebodyx import JointLimits, JointMPC, JointPlan, KinematicWBC, OSQPSolver, QuadraticProgram, RobotState


def bounds(acceleration=None):
    return JointLimits([-2, -2], [2, 2], [1, 1], acceleration)


def plan_type(backend):
    if backend[0] is KinematicWBC:
        return JointPlan
    import wholebodyx_native
    return wholebodyx_native.JointPlan


@pytest.mark.parametrize("start", [0.0, 1.0, 1e6])
def test_plan_knot_uses_right_hand_velocity(backend, start):
    plan = plan_type(backend)(np.array([[0], [.1], [.3], [.6], [1.0]]),
                              np.array([[1], [2], [3], [4]]), start, 0.1)
    for time in (start + 0.3, start + 3 * 0.1):
        q, v = plan.sample(time)
        np.testing.assert_allclose(q, [.6], atol=1e-8)
        np.testing.assert_array_equal(v, [4])
    np.testing.assert_array_equal(plan.sample(start + .3 - 1e-5)[1], [3])
    np.testing.assert_array_equal(plan.sample(np.nextafter(plan.valid_until, -np.inf))[1], [4])
    for time in (np.nextafter(start, -np.inf), plan.valid_until, np.inf, np.nan):
        with pytest.raises(ValueError):
            plan.sample(time)


@pytest.mark.parametrize("stamp,dt", [(1e16, .1), (1e308, 1e308), (-1, .1), (0, np.inf)])
def test_unrepresentable_plan_time_window(backend, stamp, dt):
    with pytest.raises(ValueError):
        plan_type(backend)(np.zeros((3, 1)), np.zeros((2, 1)), stamp, dt)


def test_numpy_single_precision_plan_clock_uses_double_arithmetic(backend):
    start, dt = np.float32(1e8), np.float32(.1)
    velocities = np.array([[1.], [2.], [3.], [4.]])
    positions = np.vstack(([0.], float(dt) * np.cumsum(velocities, axis=0)))
    plan = plan_type(backend)(positions, velocities, start, dt)
    assert plan.valid_until == float(start) + 4 * float(dt)
    np.testing.assert_array_equal(plan.sample(start)[0], [0.])
    np.testing.assert_array_equal(plan.sample(start)[1], [1.])
    np.testing.assert_array_equal(plan.sample(float(start) + 3 * float(dt))[1], [4.])
    with pytest.raises(ValueError):
        plan.sample(plan.valid_until)


def test_native_and_python_reject_inconsistent_plan_dynamics(backend):
    with pytest.raises(ValueError):
        plan_type(backend)(np.array([[0.], [1.]]), np.zeros((1, 1)), 0., .1)


@pytest.mark.parametrize("horizon", [1, 6])
@pytest.mark.parametrize("acceleration", [None, [2, 3]])
def test_cached_mpc_matches_direct_rollout_cost(backend, horizon, acceleration):
    planner = backend[1](bounds(acceleration), horizon=horizon, dt=.2, tracking_weight=3,
                          control_weight=.4, terminal_weight=7)
    rng = np.random.default_rng(41)
    for _ in range(3):
        state = RobotState(rng.uniform(-1, 1, 2), rng.uniform(-.1, .1, 2))
        goal, u = rng.uniform(-1, 1, 2), rng.normal(size=2 * horizon)
        problem = planner.formulate(state, goal)

        def cost(decision):
            v = decision.reshape(horizon, 2)
            positions = state.q + .2 * np.cumsum(v, axis=0)
            return .5 * (3 * np.sum((positions[:-1] - goal)**2)
                         + 7 * np.sum((positions[-1] - goal)**2) + .4 * np.sum(v**2))

        direction, eps = rng.normal(size=len(u)), 1e-6
        numerical = (cost(u + eps * direction) - cost(u - eps * direction)) / (2 * eps)
        assert direction @ (problem.hessian @ u + problem.gradient) == pytest.approx(numerical, abs=1e-7)
        positions = state.q + .2 * np.cumsum(u.reshape(horizon, 2), axis=0)
        ax = problem.constraint @ u
        np.testing.assert_allclose(ax[2 * horizon:4 * horizon] - problem.upper[2 * horizon:4 * horizon],
                                   (positions - 2).reshape(-1), atol=1e-12)


def test_fixed_matrices_cannot_be_mutated_and_vectors_still_validate():
    original = bounds([2, 3])
    planner = JointMPC(original, horizon=3)
    state = RobotState([0, 0], [0, 0])
    first = planner.formulate(state, [1, 1])
    for value in (first.hessian, first.constraint, planner.limits.velocity):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(AttributeError):
        planner.dt = .2
    original.velocity.setflags(write=True)
    original.velocity[:] = 9
    first.lower.setflags(write=True)
    first.lower[:] = 99
    expected_hessian = first.hessian.copy()
    first.hessian.dtype = np.int64
    first.constraint.shape = (first.constraint.size,)
    second = planner.formulate(state, [0, 0])
    np.testing.assert_array_equal(second.hessian, expected_hessian)
    assert second.constraint.ndim == 2
    assert second.upper[0] == 1 and second.lower[0] == -1
    for goal in ([np.nan, 0], [0]):
        with pytest.raises(ValueError):
            planner.formulate(state, goal)
    with pytest.raises(ValueError):
        planner.formulate(RobotState([1e308, 1e308], [0, 0]), [-1e308, -1e308])


def test_osqp_reuses_matrices_but_detects_value_and_pattern_changes(monkeypatch):
    solver = OSQPSolver()
    qp = lambda h, g, a: QuadraticProgram(h, g, a, [-2, -2], [2, 2])
    assert solver.solve(qp(np.eye(2), [0, 0], np.eye(2))).success
    calls, real_update = [], solver._solver.update

    def update(**kwargs):
        calls.append(set(kwargs))
        return real_update(**kwargs)

    monkeypatch.setattr(solver._solver, "update", update)
    np.testing.assert_allclose(solver.solve(qp(np.eye(2), [-1, 0], np.eye(2))).x, [1, 0], atol=1e-6)
    assert calls[-1] == {"q", "l", "u"}
    np.testing.assert_allclose(solver.solve(qp(2 * np.eye(2), [-1, 0], np.eye(2))).x, [.5, 0], atol=1e-6)
    assert calls[-1] == {"Px", "q", "l", "u"}
    # Same sparsity, changed A must change the constrained optimum.
    np.testing.assert_allclose(solver.solve(qp(2 * np.eye(2), [-8, 0], 2 * np.eye(2))).x, [1, 0], atol=1e-6)
    assert calls[-1] == {"Ax", "q", "l", "u"}
    old_workspace = solver._solver
    assert solver.solve(qp([[2, 1], [1, 2]], [0, 0], np.eye(2))).success
    assert solver._solver is not old_workspace
    solver.reset()
    assert solver.solve(qp(np.eye(2), [-1, 0], np.eye(2))).success


def test_warm_start_after_large_forward_clock_jump(backend):
    planner = backend[1](bounds(), horizon=3)
    assert planner.solve(RobotState([0, 0], [0, 0]), [.3, .3]).plan is not None
    assert planner.solve(RobotState([0, 0], [0, 0], 1e6), [.3, .3]).plan is not None
    with pytest.raises(ValueError):
        planner.solve(RobotState([0, 0], [0, 0], 1e308), [.3, .3])


@pytest.mark.parametrize("planner_name", ["JointMPC", "SparseJointMPC"])
@pytest.mark.parametrize("elapsed,offset", [(0., 0), (.1, 1), (.2, 2), (.3, 3), (1e6, 3)])
def test_shifted_warm_start_matches_plan_and_is_isolated(planner_name, elapsed, offset):
    import wholebodyx
    from wholebodyx import QPResult

    planner = getattr(wholebodyx, planner_name)(bounds(), horizon=3)
    state = RobotState([0, 0], [0, 0])
    first = planner.solve(state, [.3, -.2]).plan
    original = first.velocities.copy()
    expected = np.vstack((original[offset:], np.repeat(original[-1:], offset, axis=0)))
    if planner_name == "SparseJointMPC":
        positions = state.q + planner.dt * np.cumsum(expected, axis=0)
        expected = np.r_[positions.ravel(), expected.ravel()]
    else:
        expected = expected.ravel()

    class InspectSolver:
        def solve(self, problem, warm_start=None):
            np.testing.assert_array_equal(warm_start, expected)
            assert not np.shares_memory(warm_start, first.velocities)
            warm_start[:] = 99  # A custom solver may modify its own initial guess.
            return QPResult("infeasible")

    planner.solver = InspectSolver()
    planner.solve(RobotState([0, 0], [0, 0], elapsed), [.3, -.2])
    np.testing.assert_array_equal(first.velocities, original)


def test_sparse_feasibility_check_uses_updated_constraint(monkeypatch):
    from types import SimpleNamespace

    solver = OSQPSolver()
    first = QuadraticProgram([[1]], [0], [[1]], [-1], [1])
    assert solver.solve(first).success
    # Force an allegedly solved result that violates the updated A, but would
    # satisfy the old A. The independent feasibility check must reject it.
    info = SimpleNamespace(status="solved", status_val=1, iter=1)
    monkeypatch.setattr(solver._solver, "solve",
                        lambda **kwargs: SimpleNamespace(info=info, x=np.array([.75])))
    updated = QuadraticProgram([[1]], [0], [[2]], [-1], [1])
    result = solver.solve(updated)
    assert result.status == "error"
    assert result.violation == pytest.approx(.5)
    solver.reset()
    assert solver.solve(updated).success


def test_fixed_qp_reuse_avoids_value_scan(monkeypatch):
    planner = JointMPC(bounds(), horizon=3)
    state = RobotState([0, 0], [0, 0])
    solver = OSQPSolver()
    assert solver.solve(planner.formulate(state, [.2, .3])).success

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("immutable matrices should not require a value scan")

    with monkeypatch.context() as patch:
        patch.setattr(np, "array_equal", unexpected_scan)
        assert solver.solve(planner.formulate(state, [.3, .4])).success
    solver.reset()
    assert solver.solve(planner.formulate(state, [.3, .4])).success


def test_immutable_matrix_key_detects_view_metadata_changes():
    from wholebodyx.qp import _immutable, _immutable_matrix_key, _same_matrix

    matrix = _immutable([[1., 2.], [3., 4.]])
    snapshot, key = matrix.copy(), _immutable_matrix_key(matrix)
    for changed in (matrix.T, matrix[:, ::-1], matrix.reshape(4), matrix.view(np.int64)):
        assert not _same_matrix(snapshot, changed, key, _immutable_matrix_key(changed))
    assert _same_matrix(snapshot, matrix.view(), key, _immutable_matrix_key(matrix.view()))


def test_readonly_owning_arrays_do_not_qualify_for_immutable_reuse():
    from wholebodyx.qp import _immutable_matrix_key

    problem = QuadraticProgram([[1.]], [-.5], [[1.]], [-1.], [1.])
    assert not problem.hessian.flags.writeable
    assert _immutable_matrix_key(problem.hessian) is None
    solver = OSQPSolver()
    np.testing.assert_allclose(solver.solve(problem).x, [.5], atol=1e-6)
    problem.hessian.setflags(write=True)
    problem.hessian[:] = 2
    np.testing.assert_allclose(solver.solve(problem).x, [.25], atol=1e-6)


@pytest.mark.parametrize("n", [1, 6, 30])
def test_upper_hessian_csc_matches_sparse_extraction(n):
    from scipy import sparse
    from wholebodyx.qp import _upper_csc

    rng = np.random.default_rng(81)
    h = rng.normal(size=(n, n))
    h[rng.random((n, n)) < .6] = 0
    h = h + h.T
    for matrix in (h, h[::-1, ::-1], np.zeros_like(h), np.eye(n)):
        expected = sparse.triu(sparse.csc_matrix(matrix), format="csc")
        actual = _upper_csc(matrix)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)


def test_constraint_pattern_change_rebuilds_with_cached_hessian():
    solver = OSQPSolver()
    first = QuadraticProgram(np.eye(2), [-2, -2], np.eye(2), [-1, -1], [1, 1])
    assert solver.solve(first).success
    workspace, hessian = solver._solver, solver._sparse_hessian
    coupled = QuadraticProgram(np.eye(2), [-2, -2], [[1, 1]], [-1], [1])
    np.testing.assert_allclose(solver.solve(coupled).x, [.5, .5], atol=1e-6)
    assert solver._solver is not workspace
    assert solver._sparse_hessian is hessian
    # Returning to the original A must rebuild again without losing H.
    np.testing.assert_allclose(solver.solve(first).x, [1, 1], atol=1e-6)
    solver.reset()
    assert solver._sparse_hessian is None and solver._sparse_constraint is None
    np.testing.assert_allclose(solver.solve(coupled).x, [.5, .5], atol=1e-6)
