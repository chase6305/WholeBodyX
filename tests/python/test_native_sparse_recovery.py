"""Replay the warm-workspace counterexample without accepting inaccurate plans."""

import importlib.util
import os

import numpy as np
import pytest

from wholebodyx import JointLimits, JointMPC, RobotState


def native_type():
    if importlib.util.find_spec('wholebodyx_native') is None:
        if os.environ.get('WBX_REQUIRE_NATIVE') == '1':
            pytest.fail('native extension required')
        pytest.skip('optional native extension unavailable')
    from wholebodyx.native import NativeSparseMPC
    return NativeSparseMPC


def test_warm_counterexample_recovers_once_and_matches_independent_condensed_plan():
    rng = np.random.default_rng(20261026)
    n = 30
    limits = JointLimits(-rng.uniform(.7, 2, n), rng.uniform(.7, 2, n),
                         rng.uniform(.3, 1, n), rng.uniform(2, 4, n))
    planner = native_type()(limits, horizon=60)
    recoveries = []
    # The last scheduled reset before failing round 9968 is round 9850.
    # Advancing the generator reproduces exact inputs without replaying 9850 solves.
    for k in range(9969):
        state = RobotState(rng.uniform(-.25, .25, n), rng.uniform(-.1, .1, n),
                           1e6 if k % 499 == 498 else k * .07)
        goal = rng.uniform(-3., 3., n)
        if k < 9850:
            continue
        if k % 197 == 0:
            planner.reset()
        if k % 101 == 0:
            impossible = RobotState(limits.upper + .2 * limits.velocity, np.zeros(n), state.stamp)
            rejected = planner.solve(impossible, np.zeros(n))
            assert rejected.plan is None and not rejected.solver.success
            assert not rejected.solver.detail.startswith('cold retry after ')
        result = planner.solve(state, goal)
        assert result.plan is not None and result.solver.success, (k, result.solver)
        if result.solver.detail.startswith('cold retry after '):
            recoveries.append(k)
            assert result.solver.detail.count('cold retry after ') == 1
            assert 20000 < result.solver.iterations <= 40000
        plan = result.plan
        assert np.isfinite(plan.positions).all() and np.isfinite(plan.velocities).all()
        assert np.max(limits.lower - plan.positions[1:]) <= 1e-6
        assert np.max(plan.positions[1:] - limits.upper) <= 1e-6
        assert np.max(np.abs(plan.velocities) - limits.velocity) <= 1e-6
        change = np.diff(np.vstack((state.v, plan.velocities)), axis=0)
        assert np.max(np.abs(change) - .1 * limits.acceleration) <= 1e-6
    assert 9968 in recoveries
    reference = JointMPC(limits, horizon=60).solve(state, goal)
    assert reference.plan is not None
    np.testing.assert_allclose(result.plan.positions, reference.plan.positions, atol=1e-5, rtol=0)
    np.testing.assert_allclose(result.plan.velocities, reference.plan.velocities, atol=3e-4, rtol=0)
    assert result.solver.violation <= 1e-6


def test_infeasible_and_invalid_inputs_are_not_cold_retried():
    planner = native_type()(JointLimits([-1.], [1.], [1.], [2.]), horizon=5)
    assert planner.solve(RobotState([0.], [0.]), [.3]).plan is not None
    failed = planner.solve(RobotState([3.], [0.], .1), [0.])
    assert failed.plan is None and not failed.solver.success
    assert not failed.solver.detail.startswith('cold retry after ')
    with pytest.raises(ValueError):
        planner.solve(RobotState([0.], [0.], .2), [0., 0.])
    assert planner.solve(RobotState([0.], [0.], .2), [.3]).plan is not None
