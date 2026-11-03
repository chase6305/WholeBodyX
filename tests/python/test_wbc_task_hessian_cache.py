"""Optional task-model reuse must preserve the actual dense QP and isolation."""

import gc
import weakref

import numpy as np
import pytest

from wholebodyx import JointLimits, KinematicWBC, LinearTask, OSQPSolver, QuadraticProgram, RobotState


def limits(n):
    return JointLimits([-2.] * n, [2.] * n, [1.] * n, [3.] * n)


@pytest.mark.parametrize("joints", [1, 6, 30, 120])
def test_reused_task_model_matches_uncached_closed_loop_with_changing_inputs(joints):
    rng = np.random.default_rng(20261027 + joints)
    cached = KinematicWBC(limits(joints), regularization=.01, smoothing=.02, cache_task_hessian=True)
    original = KinematicWBC(limits(joints), regularization=.01, smoothing=.02)
    state = RobotState(np.zeros(joints), np.zeros(joints))
    jac = rng.normal(size=(max(1, joints // 2), joints))
    weights = rng.uniform(.2, 2., jac.shape[0])
    for k in range(120):
        if k % 17 == 0:
            jac = rng.normal(size=jac.shape)
        if k % 23 == 0:
            weights = rng.uniform(.2, 2., jac.shape[0])
        if k == 50:
            cached.smoothing = original.smoothing = .04
            cached.regularization = original.regularization = .03
        tasks = [LinearTask(jac, rng.normal(size=jac.shape[0]), weights)]
        if k % 9 < 3:
            tasks.append(LinearTask(np.eye(joints), rng.normal(size=joints), .1))
        a, b = cached.formulate(state, tasks, .01), original.formulate(state, tasks, .01)
        for field in ("hessian", "gradient", "constraint", "lower", "upper"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        actual, expected = cached.step(state, tasks, .01), original.step(state, tasks, .01)
        assert actual.solver.success and expected.solver.success
        np.testing.assert_array_equal(actual.command.velocity, expected.command.velocity)
        if k == 70:
            cached.reset()
            original.reset()
        state = RobotState(state.q + .01 * actual.command.velocity, actual.command.velocity, state.stamp + .01)


@pytest.mark.parametrize("field", ["jacobian", "weight", "target_velocity"])
def test_reversible_readonly_task_mutation_is_observed(field):
    controller = KinematicWBC(limits(2), cache_task_hessian=True)
    state = RobotState([0., 0.], [0., 0.])
    task = LinearTask([[1., .2], [.3, 1.]], [.2, -.3], [1., 2.])
    controller.formulate(state, [task], .01)
    changed = getattr(task, field)
    changed.setflags(write=True)
    changed[:] *= 2
    p = controller.formulate(state, [task], .01)
    lo, hi = controller.limits.velocity_bounds(state, .01)
    oracle = QuadraticProgram(1e-6 * np.eye(2) + task.jacobian.T @ (task.weight[:, None] * task.jacobian),
                              -task.jacobian.T @ (task.weight * task.target_velocity), np.eye(2), lo, hi)
    np.testing.assert_array_equal(p.hessian, oracle.hessian)
    np.testing.assert_array_equal(p.gradient, oracle.gradient)
    actual, expected = controller.solver.solve(p), OSQPSolver().solve(oracle)
    np.testing.assert_array_equal(actual.x, expected.x)


@pytest.mark.parametrize("field", ["weight", "target_velocity"])
def test_invalid_task_change_does_not_poison_validated_model(field):
    controller = KinematicWBC(limits(2), cache_task_hessian=True)
    state = RobotState([0., 0.], [0., 0.])
    task = LinearTask(np.eye(2), [.2, -.2])
    controller.formulate(state, [task], .01)
    healthy_template, healthy_cache = controller._template, controller._task_hessian_cache
    value = getattr(task, field)
    saved = value.copy()
    value.setflags(write=True)
    value[:] = -1. if field == "weight" else np.nan
    with pytest.raises(ValueError):
        controller.formulate(state, [task], .01)
    assert controller._template is healthy_template and controller._task_hessian_cache is healthy_cache
    value[:] = saved
    assert controller.step(state, [task], .01).solver.success


def test_cache_keeps_only_latest_batch_and_does_not_alias_public_diagnostics():
    controller = KinematicWBC(limits(3), cache_task_hessian=True)
    state = RobotState(np.zeros(3), np.zeros(3))
    first = controller.formulate(state, [LinearTask(np.eye(3), [.2] * 3)], .01)
    old_hessian = weakref.ref(controller._task_hessian_cache[2])
    first.hessian.shape = (9,)
    first.gradient.setflags(write=True)
    first.gradient[:] = np.nan
    again = controller.formulate(state, [LinearTask(np.eye(3), [.3] * 3)], .01)
    assert again.hessian.shape == (3, 3) and controller.solver.solve(again).success
    for n in (6, 1, 3):
        controller.limits = limits(n)
        controller.formulate(RobotState(np.zeros(n), np.zeros(n)), [], .01)
        assert len(controller._task_hessian_cache[1]) == 0
        assert controller._task_hessian_cache[2].shape == (n, n)
    gc.collect()
    assert old_hessian() is None
    with pytest.raises(ValueError):
        controller._task_hessian_cache[2].setflags(write=True)


def test_default_does_not_keep_task_snapshots_and_option_is_boolean():
    controller = KinematicWBC(limits(1))
    controller.formulate(RobotState([0.], [0.]), [LinearTask([[1.]], [.2])], .01)
    assert not controller.cache_task_hessian and controller._task_hessian_cache is None
    for value in ("yes", 1, None):
        with pytest.raises(TypeError, match="boolean"):
            KinematicWBC(limits(1), cache_task_hessian=value)
