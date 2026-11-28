"""Native sparse formulation, independent optima and application integration."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (BackendRegistry, JointLimits, JointMPC, OSQPSolver, RobotState,
                        SparseJointMPC, VelocitySession)


@pytest.fixture
def native_sparse():
    if importlib.util.find_spec("wholebodyx_native") is None:
        if os.environ.get("WBX_REQUIRE_NATIVE") == "1":
            pytest.fail("native sparse extension is required for this test run")
        pytest.skip("optional native extension is not installed")
    import wholebodyx_native
    if not hasattr(wholebodyx_native, "SparseJointMPC"):
        if os.environ.get("WBX_REQUIRE_NATIVE") == "1":
            pytest.fail("native sparse feature is required; rebuild the extension")
        pytest.skip("installed native wheel predates sparse MPC")
    from wholebodyx.native import NativeSparseMPC
    return NativeSparseMPC


@pytest.mark.parametrize("horizon", [1, 5, 20])
@pytest.mark.parametrize("slew", [False, True])
def test_formulations_and_random_constrained_optima_match(native_sparse, horizon, slew):
    rng = np.random.default_rng(523)
    limits = JointLimits(-rng.uniform(.5, 2, 3), rng.uniform(.5, 2, 3),
                          rng.uniform(.3, 1., 3), rng.uniform(2, 4, 3) if slew else None)
    options = dict(horizon=horizon, dt=.08, tracking_weight=2.7, control_weight=.4, terminal_weight=9.1)
    native, sparse, dense = native_sparse(limits, **options), SparseJointMPC(limits, **options), JointMPC(limits, **options)
    state = RobotState(rng.uniform(-.2, .2, 3), rng.uniform(-.1, .1, 3))
    for _ in range(15):
        goal = rng.uniform(-2, 2, 3)
        a, b = native.formulate(state, goal), sparse.formulate(state, goal)
        for field in ("hessian", "constraint"):
            matrix_a, matrix_b = getattr(a, field), getattr(b, field)
            assert matrix_a.format == "csc" and (matrix_a != matrix_b).nnz == 0
        for field in ("gradient", "lower", "upper"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        results = [planner.solve(state, goal) for planner in (native, sparse, dense)]
        assert all(result.plan is not None for result in results)
        for result in results[:2]:
            np.testing.assert_allclose(result.plan.velocities, results[2].plan.velocities, atol=3e-6, rtol=1e-5)
            np.testing.assert_allclose(result.plan.positions, results[2].plan.positions, atol=2e-6, rtol=1e-5)
        q, v = results[2].plan.sample(state.stamp + .04)
        state = RobotState(q, v, state.stamp + .04)


def test_analytic_solution_snapshot_independence_and_reset(native_sparse):
    limits = JointLimits([-10], [10], [10])
    planner = native_sparse(limits, horizon=1, dt=.2, control_weight=.5, terminal_weight=3.)
    state = RobotState([0], [0])
    exact = .2 * 3 / (.2 * .2 * 3 + .5)
    result = planner.solve(state, [1.])
    np.testing.assert_allclose(result.plan.velocities, [[exact]], atol=1e-6)
    np.testing.assert_allclose(result.solver.x, [.2 * exact, exact], atol=1e-6)
    problem = planner.formulate(state, [1.])
    problem.constraint.data = np.zeros_like(problem.constraint.data)
    planner.reset()
    np.testing.assert_allclose(planner.solve(state, [1.]).plan.velocities, [[exact]], atol=1e-6)
    assert planner.limits.lower is not limits.lower
    with pytest.raises(ValueError):
        planner.limits.lower.setflags(write=True)


def test_failure_recovery_and_clock_jumps(native_sparse):
    planner = native_sparse(JointLimits([-1], [1], [1], [1]), horizon=5)
    state = RobotState([0], [0])
    assert planner.solve(state, [.3]).plan is not None
    failed = planner.solve(RobotState([3], [0], .1), [0])
    assert failed.plan is None and not failed.solver.success and failed.solver.x is None
    for stamp in (1e6, 0., .05, 1.):
        result = planner.solve(RobotState([0], [0], stamp), [.3])
        assert result.plan is not None
    with pytest.raises(ValueError, match="time window"):
        planner.solve(RobotState([0], [0], 1e16), [0])
    with pytest.raises(ValueError):
        planner.solve(state, [0, 1])


def test_abrupt_unreachable_targets_do_not_inherit_bad_numerical_conditioning(native_sparse):
    rng = np.random.default_rng(20261021)
    n = 6
    limits = JointLimits(-rng.uniform(.7, 2, n), rng.uniform(.7, 2, n),
                          rng.uniform(.3, 1, n), rng.uniform(2, 4, n))
    planners = (SparseJointMPC(limits), native_sparse(limits), JointMPC(limits))
    for k in range(40):
        state = RobotState(rng.uniform(-.25, .25, n), rng.uniform(-.1, .1, n), k * .07)
        goal = rng.uniform(-3., 3., n)
        results = [planner.solve(state, goal) for planner in planners]
        assert all(result.plan is not None for result in results)
        for result in results[:2]:
            np.testing.assert_allclose(result.plan.velocities, results[-1].plan.velocities, atol=3e-5, rtol=1e-5)
            assert result.solver.violation <= 1e-6


def _full_sparse_planner(limits, horizon, dt, **weights):
    size = limits.lower.size * horizon
    solver = OSQPSolver(variable_scaling=np.r_[np.full(size, dt), np.ones(size)],
                        adaptive_rho_interval=100)
    return SparseJointMPC(limits, horizon=horizon, dt=dt, solver=solver, **weights)


@pytest.mark.parametrize("slew", [False, True])
@pytest.mark.parametrize("backend", ["sparse", "native-sparse"])
def test_unconstrained_suffix_preserves_the_full_sparse_optimum(request, backend, slew):
    limits = JointLimits([-1., -1.], [1., 1.], [.8, .7], [2., 1.8] if slew else None)
    horizon, dt = 128, .05
    weights = dict(tracking_weight=8., control_weight=.12, terminal_weight=18.)
    state = RobotState([-.2, .23], [.03, -.02], 6.)
    goal = [.28, -.25]
    cls = SparseJointMPC if backend == "sparse" else request.getfixturevalue("native_sparse")
    planner = cls(limits, horizon=horizon, dt=dt, **weights)
    expected = _full_sparse_planner(limits, horizon, dt, **weights).solve(state, goal)
    actual = planner.solve(state, goal)
    assert actual.plan is not None and expected.plan is not None
    assert actual.solver.detail == "unconstrained suffix"
    np.testing.assert_allclose(actual.plan.positions, expected.plan.positions, atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=1e-5, rtol=0)
    problem = planner.formulate(state, goal)
    values = problem.constraint @ actual.solver.x
    assert max(0., np.max(problem.lower - values), np.max(values - problem.upper)) <= 1e-6
    planner.reset()
    replayed = planner.solve(state, goal)
    np.testing.assert_array_equal(replayed.solver.x, actual.solver.x)


@pytest.mark.parametrize("backend", ["sparse", "native-sparse"])
def test_longer_prefix_handles_a_large_short_step_retarget(request, backend):
    limits = JointLimits([-1.], [1.], [.8], [2.])
    horizon, dt = 200, .02
    cls = SparseJointMPC if backend == "sparse" else request.getfixturevalue("native_sparse")
    planner = cls(limits, horizon=horizon, dt=dt)
    reference = _full_sparse_planner(limits, horizon, dt)
    for index, target in enumerate((.15, .9, .15)):
        state, goal = RobotState([0.], [0.], index * dt), [target]
        expected = reference.solve(state, goal)
        actual = planner.solve(state, goal)
        assert actual.plan is not None and expected.plan is not None
        assert actual.solver.detail == "unconstrained suffix"
        np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=1e-5, rtol=0)


@pytest.mark.parametrize("backend", ["sparse", "native-sparse"])
def test_slew_across_the_unconstrained_suffix_boundary_can_force_a_full_solve(request, backend):
    horizon, dt = 128, .05
    limits = JointLimits([-1.], [1.], [.8], [.1])
    state, goal = RobotState([0.], [0.]), [.06]
    cls = SparseJointMPC if backend == "sparse" else request.getfixturevalue("native_sparse")
    planner = cls(limits, horizon=horizon, dt=dt)
    expected = _full_sparse_planner(limits, horizon, dt).solve(state, goal)
    actual = planner.solve(state, goal)
    assert actual.plan is not None and expected.plan is not None
    assert actual.solver.detail != "unconstrained suffix"
    np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=2e-5, rtol=0)
    slew = np.diff(np.r_[state.v[0], actual.plan.velocities[:, 0]])
    assert np.abs(slew).max() <= dt * limits.acceleration[0] + 1e-6


@pytest.mark.parametrize("backend", ["sparse", "native-sparse"])
def test_feasible_full_unconstrained_response_avoids_iterations(request, backend):
    cls = SparseJointMPC if backend == "sparse" else request.getfixturevalue("native_sparse")
    limits = JointLimits([-1., -1.], [1., 1.], [.8, .8], [2., 2.])
    state, goal = RobotState([.2, -.15], [0., 0.]), [.201, -.147]
    horizon, dt = 20, .05
    planner = cls(limits, horizon=horizon, dt=dt)
    actual = planner.solve(state, goal)
    expected = _full_sparse_planner(limits, horizon, dt).solve(state, goal)
    assert actual.plan is not None and expected.plan is not None
    assert actual.solver.iterations == 0 and actual.solver.detail == "unconstrained optimum"
    np.testing.assert_allclose(actual.plan.positions, expected.plan.positions, atol=1e-7, rtol=0)
    np.testing.assert_allclose(actual.plan.velocities, expected.plan.velocities, atol=1e-6, rtol=0)


@pytest.mark.parametrize("backend", ["sparse", "native-sparse"])
def test_unconstrained_response_checks_slew_after_the_first_interval(request, backend):
    cls = SparseJointMPC if backend == "sparse" else request.getfixturevalue("native_sparse")
    horizon, dt = 20, .1
    goal = [.05]
    free = _full_sparse_planner(JointLimits([-1.], [1.], [1.]), horizon, dt)
    unconstrained = free.solve(RobotState([0.], [0.]), goal).plan
    assert abs(unconstrained.velocities[1, 0] - unconstrained.velocities[0, 0]) > dt
    state = RobotState([0.], unconstrained.velocities[0])
    limits = JointLimits([-1.], [1.], [1.], [1.])
    result = cls(limits, horizon=horizon, dt=dt).solve(state, goal)
    assert result.plan is not None and result.solver.detail != "unconstrained optimum"
    slew = np.diff(np.r_[state.v[0], result.plan.velocities[:, 0]])
    assert np.abs(slew).max() <= dt + 1e-6


@pytest.mark.parametrize("option", [{"horizon": True}, {"horizon": 0}, {"dt": 0},
                                    {"tracking_weight": np.inf}, {"control_weight": -1}])
def test_settings_reject_invalid_inputs(native_sparse, option):
    with pytest.raises(ValueError):
        native_sparse(JointLimits([-1], [1], [1]), **option)


def test_generic_execution_and_URDF_order_adapters_accept_native_sparse(native_sparse):
    from wholebodyx import JointOrder, ReorderedPlanner
    from wholebodyx.simulation import JointIntegrator
    limits = JointLimits([-1] * 3, [1] * 3, [.8] * 3, [2.] * 3)
    names = ("a", "b", "c")
    order = JointOrder(names, ("c", "a", "b"))
    planner = ReorderedPlanner(native_sparse(order.limits_to_backend(limits)), order)
    model = SimpleNamespace(joint_names=names, limits=limits)
    driver = JointIntegrator(limits, RobotState(np.zeros(3), np.zeros(3)), control_dt=.02)
    session = VelocitySession(model, driver, BackendRegistry().create_controller("python", limits), planner=planner)
    goal = [.3, -.2, .1]
    session.reset()
    replans = sum(session.step(goal).reference.replanned for _ in range(150))
    assert replans == 30 and not session.fault
    np.testing.assert_allclose(driver.read().q, goal, atol=1e-6)


def test_feature_detection_keeps_old_native_wheels_usable(monkeypatch):
    import wholebodyx.native as module
    old = SimpleNamespace(**{name: object for name in
                              ("JointLimits", "RobotState", "LinearTask", "JointPlan", "KinematicWBC", "JointMPC")})
    monkeypatch.setattr(module, "_extension", lambda: old)
    statuses = {status.name: status for status in BackendRegistry().status()}
    assert statuses["native"].available and not statuses["native-sparse"].available
    assert "missing SparseJointMPC" in statuses["native-sparse"].detail
    with pytest.raises(ImportError, match="SparseJointMPC"):
        module.NativeSparseMPC(JointLimits([-1], [1], [1]))


def test_native_sparse_cli_runs_outside_the_repository(native_sparse, tmp_path):
    import json
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(path).resolve()) for path in
                                                environment.get("PYTHONPATH", "").split(os.pathsep) if path)
    result = subprocess.run([sys.executable, "-m", "wholebodyx", "run", "disturbance_recovery",
                             "--backend", "native", "--mpc-backend", "native-sparse"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["tracking_replans"] == report["disturbances"] == 2
    assert report["final_position_error_m"] < 1e-4
