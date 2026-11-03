"""Native sparse formulation, independent optima and application integration."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (BackendRegistry, JointLimits, JointMPC, RobotState,
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
