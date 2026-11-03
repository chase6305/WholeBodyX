import numpy as np
import pytest

from wholebodyx import JointLimits, JointPlan, LinearTask, OSQPSolver, QuadraticProgram, RobotState, VelocityCommand


def limits(n=2, acceleration=None):
    return JointLimits(np.full(n, -2.0), np.full(n, 2.0), np.ones(n), acceleration)


def test_command_mixed_precision_clock_is_validated_after_conversion():
    start = np.float32(1e8)
    command = VelocityCommand([0.], start, float(start) + .1)
    assert type(command.stamp) is float and type(command.valid_until) is float
    assert command.valid_until > command.stamp
    with pytest.raises(ValueError):
        VelocityCommand([0.], start, start)


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="requires a wider floating-point input type")
@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_regularization_must_remain_positive_and_finite_after_conversion(backend, value):
    with pytest.raises(ValueError, match="regularization"):
        backend[0](limits(), regularization=np.longdouble(value))


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="requires a wider floating-point input type")
@pytest.mark.parametrize("stamp", ["1e400", "-1e-400"])
def test_robot_state_rejects_unrepresentable_or_negative_extended_clock(stamp):
    with pytest.raises(ValueError, match="stamp"):
        RobotState([0.], [0.], np.longdouble(stamp))


def test_coupled_qp_is_not_post_clipped(backend):
    wbc, _ = backend
    bounds = JointLimits([-2, -2], [2, 2], [0.2, 1])
    controller = wbc(bounds)
    result = controller.step(RobotState([0, 0], [0, 0]), [LinearTask([[1, 1]], [1])], 0.1)
    assert result.solver.success
    np.testing.assert_allclose(result.command.velocity, [0.2, 0.8], atol=1e-5)


def test_position_and_slew_bounds(backend):
    wbc, _ = backend
    controller = wbc(limits(acceleration=[2, 2]))
    state = RobotState([1.99, 0], [0, 0])
    result = controller.step(state, [LinearTask(np.eye(2), [4, -4])], 0.1)
    np.testing.assert_allclose(result.command.velocity, [0.1, -0.2], atol=1e-6)


@pytest.mark.parametrize("acceleration", [None, [2, 3]])
def test_velocity_bounds_match_independent_intersection_and_own_buffers(acceleration):
    bounds = limits(acceleration=acceleration)
    rng = np.random.default_rng(103)
    for _ in range(30):
        state = RobotState(rng.uniform(-3, 3, 2), rng.uniform(-1, 1, 2))
        dt = float(rng.uniform(.001, .2))
        expected_lo, expected_hi = [], []
        for k in range(2):
            lower = [-bounds.velocity[k], (bounds.lower[k] - state.q[k]) / dt]
            upper = [bounds.velocity[k], (bounds.upper[k] - state.q[k]) / dt]
            if acceleration is not None:
                lower.append(state.v[k] - acceleration[k] * dt)
                upper.append(state.v[k] + acceleration[k] * dt)
                # Independent rollout + bisection oracle, without the closed
                # form used by JointLimits.velocity_bounds().
                changes = np.arange(int(np.ceil(bounds.velocity[k] / (acceleration[k] * dt)))) * acceleration[k] * dt
                for distance, target in ((state.q[k] - bounds.lower[k], lower),
                                         (bounds.upper[k] - state.q[k], upper)):
                    left, right = 0., bounds.velocity[k]
                    for _ in range(55):
                        speed = (left + right) / 2
                        travel = dt * np.maximum(speed - changes, 0).sum()
                        if travel <= max(0., distance):
                            left = speed
                        else:
                            right = speed
                    target.append(-left if target is lower else left)
            expected_lo.append(max(lower))
            expected_hi.append(min(upper))
        lo, hi = bounds.velocity_bounds(state, dt)
        np.testing.assert_allclose(lo, expected_lo, atol=1e-12, rtol=0)
        np.testing.assert_allclose(hi, expected_hi, atol=1e-12, rtol=0)
        lo[:] = 99
        hi[:] = -99
        again = bounds.velocity_bounds(state, dt)
        np.testing.assert_allclose(again[0], expected_lo, atol=1e-12, rtol=0)
        np.testing.assert_allclose(again[1], expected_hi, atol=1e-12, rtol=0)


@pytest.mark.parametrize("direction", [-1., 1.])
def test_wbc_brakes_before_joint_limit_without_losing_next_step_feasibility(backend, direction):
    bounds = JointLimits([-1.], [1.], [1.], [1.])
    controller = backend[0](bounds)
    state = RobotState([direction * .96], [direction * .3])
    for _ in range(30):
        result = controller.step(state, [LinearTask([[1.]], [direction * .4])], .1)
        assert result.solver.success
        state = RobotState(state.q + .1 * result.command.velocity, result.command.velocity, state.stamp + .1)
        assert abs(state.q[0]) <= 1 + 1e-7
        lo, hi = bounds.velocity_bounds(state, .1)
        assert lo[0] <= hi[0] + 1e-7
    assert abs(state.q[0]) == pytest.approx(1., abs=1e-6)


def test_braking_bounds_remain_finite_for_extreme_time_and_acceleration_scales():
    for acceleration, dt in ((1e-300, 1e-100), (1e300, 1e300), (1e300, 1.), (1e-300, 1.)):
        bounds = JointLimits([-1.], [1.], [1.], [acceleration])
        with np.errstate(over="ignore", under="ignore"):
            lo, hi = bounds.velocity_bounds(RobotState([0.], [0.]), dt)
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert lo[0] <= 0 <= hi[0]


@pytest.mark.parametrize("direction", [-1., 1.])
def test_solver_feasibility_residual_does_not_accumulate_into_braking_failure(direction):
    from wholebodyx import KinematicWBC, QPResult

    class ResidualSolver(OSQPSolver):
        def solve(self, problem, warm_start=None):
            result = super().solve(problem, warm_start)
            if result.success:
                return QPResult("solved", result.x + direction * 5e-7,
                                result.iterations, 5e-7, "injected feasibility residual")
            return result

    bounds = JointLimits([-1.], [1.], [1.], [1.])
    controller = KinematicWBC(bounds, solver=ResidualSolver())
    state = RobotState([direction * .96], [direction * .3])
    for _ in range(30):
        result = controller.step(state, [LinearTask([[1.]], [direction * .4])], .1)
        assert result.command is not None
        np.testing.assert_array_equal(result.solver.x, result.command.velocity)
        state = RobotState(state.q + .1 * result.command.velocity, result.command.velocity, state.stamp + .1)
        assert abs(state.q[0]) <= 1 + 1e-12


@pytest.mark.parametrize("solution", [[np.nan], [0, 0], [100.]])
def test_wbc_rejects_invalid_injected_solver_solutions(solution):
    from types import SimpleNamespace
    from wholebodyx import KinematicWBC, QPResult

    solver = SimpleNamespace(solve=lambda problem: QPResult("solved", np.array(solution), iterations=7, violation=0))
    result = KinematicWBC(limits(1), solver=solver).step(RobotState([0], [0]), [], .1)
    assert result.solver.status == "error" and result.command is None and result.solver.x is None
    assert result.solver.iterations == 7
    if solution == [100.]:
        assert result.solver.violation == 99.


def test_failure_has_no_command_and_solver_recovers(backend):
    wbc, _ = backend
    controller = wbc(limits())
    task = LinearTask(np.eye(2), [1, 1])
    result = controller.step(RobotState([10, 0], [0, 0]), [task], 0.1)
    assert result.solver.status == "infeasible"
    assert result.command is None and result.solver.x is None
    result = controller.step(RobotState([0, 0], [0, 0]), [task], 0.1)
    assert result.solver.success


@pytest.mark.parametrize("dt", [0, -0.1, np.nan, np.inf])
def test_invalid_dt(backend, dt):
    with pytest.raises(ValueError):
        backend[0](limits()).step(RobotState([0, 0], [0, 0]), [], dt)


def test_numpy_float32_clock_and_dt_keep_double_precision_command_time(backend):
    state = RobotState([0, 0], [0, 0], np.float32(1e8))
    dt = np.float32(.005)
    result = backend[0](limits()).step(state, [], dt)
    assert isinstance(state.stamp, float)
    assert result.command is not None
    assert result.command.valid_until == state.stamp + float(dt)
    assert result.command.valid_until > state.stamp


def test_invalid_task_dimension(backend):
    with pytest.raises(ValueError):
        backend[0](limits()).step(RobotState([0, 0], [0, 0]), [LinearTask([[1]], [1])], 0.1)


def test_one_step_mpc_analytic_optimum(backend):
    _, mpc = backend
    bounds = JointLimits([-10], [10], [10])
    planner = mpc(bounds, horizon=1, dt=0.2, tracking_weight=2, control_weight=0.5, terminal_weight=3)
    result = planner.solve(RobotState([0], [0]), [1])
    expected = 0.2 * 3 / (0.2**2 * 3 + 0.5)
    np.testing.assert_allclose(result.plan.velocities, [[expected]], atol=1e-6)
    np.testing.assert_allclose(result.plan.sample(0.1)[0], [0.1 * expected], atol=1e-6)


def test_mpc_constraints_and_dynamics(backend):
    _, mpc = backend
    bounds = limits(acceleration=[2, 3])
    state = RobotState([1.7, -1.8], [0.2, -0.1], 3.0)
    planner = mpc(bounds, horizon=8)
    result = planner.solve(state, [4, -4])  # An unreachable goal is a soft objective.
    assert result.solver.success
    plan = result.plan
    assert np.max(plan.positions - bounds.upper) <= 1e-6
    assert np.max(bounds.lower - plan.positions) <= 1e-6
    assert np.max(np.abs(plan.velocities) - bounds.velocity) <= 1e-6
    slew = np.diff(np.vstack((state.v, plan.velocities)), axis=0) / plan.dt
    assert np.max(np.abs(slew) - bounds.acceleration) <= 1e-5
    np.testing.assert_allclose(np.diff(plan.positions, axis=0), plan.dt * plan.velocities, atol=1e-9)
    assert planner.solve(RobotState(plan.positions[1], plan.velocities[0], 3.1), [4, -4]).solver.success
    planner.reset()
    assert planner.solve(state, [0, 0]).solver.success


def test_mpc_infeasible(backend):
    _, mpc = backend
    result = mpc(limits()).solve(RobotState([10, 0], [0, 0]), [0, 0])
    assert result.solver.status == "infeasible"
    assert result.plan is None


def test_wbc_weighted_tradeoff(backend):
    wbc, _ = backend
    controller = wbc(limits(1), regularization=0.1)
    result = controller.step(RobotState([0], [0]), [LinearTask([[1]], [1], 3), LinearTask([[1]], [-1], 1)], 0.1)
    np.testing.assert_allclose(result.command.velocity, [2 / 4.1], atol=1e-6)


@pytest.mark.parametrize("weight", [0., 1., np.array(.5), np.array([.5, .5])])
def test_task_weight_ownership_and_scalar_vector_equivalence(weight):
    original = np.asarray(weight).copy()
    task = LinearTask(np.eye(2), [.1, .2], weight)
    np.testing.assert_array_equal(task.weight, np.broadcast_to(original, (2,)))
    assert not task.weight.flags.writeable
    if isinstance(weight, np.ndarray):
        weight[...] = 99
        np.testing.assert_array_equal(task.weight, np.broadcast_to(original, (2,)))
    task.weight.setflags(write=True)
    task.weight[:] = 7
    other = LinearTask(np.eye(2), [.1, .2], original)
    np.testing.assert_array_equal(other.weight, np.broadcast_to(original, (2,)))


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf, -1., [np.nan, 1],
                                   [1, -1], [[1, 1]], [1]])
def test_task_rejects_invalid_scalar_and_vector_weights(weight):
    with pytest.raises(ValueError):
        LinearTask(np.eye(2), [.1, .2], weight)


def test_reference_expiry_and_immutability():
    plan = JointPlan([[0], [0.1], [0.2]], [[1], [1]], 1.0, 0.1)
    for stamp in (0.99, plan.valid_until, np.nan):
        with pytest.raises(ValueError):
            plan.sample(stamp)
    with pytest.raises(ValueError):
        plan.positions[0, 0] = 9
    with pytest.raises(ValueError):
        JointPlan([[0], [4]], [[1]], 0, 0.1)


def test_qp_infeasible_equalities_and_update():
    solver = OSQPSolver()
    p = QuadraticProgram([[1]], [0], [[1], [1]], [0, 1], [0, 1])
    assert solver.solve(p).status == "infeasible"
    p = QuadraticProgram([[1]], [-0.5], [[1], [1]], [-1, -1], [1, 1])
    np.testing.assert_allclose(solver.solve(p).x, [0.5], atol=1e-6)


@pytest.mark.parametrize("eigenvalue", [0.0, -5e-11, -2e-10])
def test_qp_semidefinite_validation_tolerance(eigenvalue):
    h = np.diag([eigenvalue, 1.0])
    if eigenvalue < -1e-10:
        with pytest.raises(ValueError, match="positive semidefinite"):
            QuadraticProgram(h, [0, 0], np.eye(2), [-1, -1], [1, 1])
    else:
        QuadraticProgram(h, [0, 0], np.eye(2), [-1, -1], [1, 1])


def test_wbc_cache_tracks_task_and_setting_changes():
    from wholebodyx import KinematicWBC

    controller = KinematicWBC(limits(), regularization=.1, smoothing=.2)
    state = RobotState([0, 0], [.1, -.1])
    task = LinearTask(np.eye(2), [.4, -.2])
    for weight, scale, regularization, smoothing in (
        (1, 1, .1, .2), (1, 1, .1, .2), (1, 1, .1, .2),
        (2, 1, .1, .2), (2, 1, .1, .2), (2, .5, .1, .2),
        (2, .5, .3, .2), (2, .5, .3, .4),
    ):
        controller.regularization, controller.smoothing = regularization, smoothing
        task = LinearTask(scale * np.eye(2), [.4, -.2], weight)
        result = controller.step(state, [task], .1)
        expected = (smoothing * state.v + weight * scale * task.target_velocity) / (
            regularization + smoothing + weight * scale**2)
        np.testing.assert_allclose(result.command.velocity, expected, atol=1e-6)
    controller.reset()
    assert controller.step(state, [], .1).solver.success


def test_wbc_cached_problem_isolated_from_caller_changes():
    from wholebodyx import KinematicWBC

    controller = KinematicWBC(limits())
    state = RobotState([0, 0], [0, 0])
    task = LinearTask(np.eye(2), [.2, .3])
    first = controller.formulate(state, [task], .1)
    with pytest.raises(ValueError):
        first.hessian.setflags(write=True)
    first.hessian.shape = (4,)
    first.constraint.dtype = np.int64
    cached = controller.formulate(state, [task], .1)
    np.testing.assert_allclose(cached.hessian, np.eye(2) * 1.000001)
    cached.hessian.shape = (4,)
    cached.constraint.dtype = np.int64
    cached.lower.setflags(write=True)
    cached.lower[:] = 99
    result = controller.step(state, [task], .1)
    np.testing.assert_allclose(result.command.velocity, [.2 / 1.000001, .3 / 1.000001], atol=1e-6)


@pytest.mark.parametrize("factory", [
    lambda: RobotState([np.nan], [0]),
    lambda: RobotState([0], [0, 1]),
    lambda: JointLimits([0], [0], [1]),
    lambda: JointLimits([-1], [1], [0]),
    lambda: LinearTask([[1]], [0], -1),
    lambda: QuadraticProgram([[-1]], [0], [[1]], [-1], [1]),
])
def test_invalid_contracts(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("settings", [{"eps_abs": 0}, {"eps_rel": np.nan}, {"max_iter": True},
                                       {"max_iter": 1.5}, {"polishing": "yes"}])
def test_osqp_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        OSQPSolver(**settings)


def test_osqp_iteration_limit_is_preserved_across_reset():
    solver = OSQPSolver(max_iter=1)
    problem = QuadraticProgram([[1]], [-1], [[1]], [-.5], [.5])
    assert solver.solve(problem).status == "iteration_limit"
    solver.reset()
    assert solver.solve(problem).status == "iteration_limit"
    result = OSQPSolver(eps_abs=1e-7, eps_rel=1e-7, polishing=True).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [.5], atol=1e-6)
