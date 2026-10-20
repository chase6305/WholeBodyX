"""Analytic model, continuous support and closed-loop checks for CoM preview."""
import numpy as np
import pytest

from wholebodyx.centroidal import CoMPlan, CoMPreviewMPC

SUPPORT = [[-.10, -.20], [.10, -.20], [.10, .20], [-.10, .20]]


def test_stationary_equilibrium_and_immutable_plan():
    mpc = CoMPreviewMPC(SUPPORT, .95)
    plan = mpc.plan([.01, .02], [0, 0], [0, 0], np.tile([.01, .02], (20, 1)))
    np.testing.assert_allclose(plan.jerks, 0, atol=1e-7)
    np.testing.assert_allclose(plan.zmp, np.tile([.01, .02], (21, 1)), atol=1e-8)
    with pytest.raises(ValueError):
        plan.states[0, 0, 0] = 3
    for stamp in (-.1, 2., float("nan")):
        with pytest.raises(ValueError):
            plan.sample(stamp)


def test_numpy_single_precision_plan_clock_and_sampling():
    stamp, dt = np.float32(1e8), np.float32(.1)
    states = np.zeros((4, 3, 2))
    states[:, 0] = np.arange(4)[:, None] * float(dt)
    states[:, 1] = 1.
    plan = CoMPlan(states, np.zeros((3, 2)), stamp, dt, np.float32(.95), np.float32(9.81))
    for name in ("stamp", "dt", "height", "gravity"):
        assert type(getattr(plan, name)) is float
    np.testing.assert_array_equal(plan.sample(stamp)[0], [0., 0.])
    sample_stamp = float(stamp) + 1.5 * float(dt)
    np.testing.assert_allclose(plan.sample(sample_stamp)[0], sample_stamp - float(stamp), atol=1e-12)
    with pytest.raises(ValueError):
        plan.sample(float(stamp) + 3 * float(dt))
    mpc = CoMPreviewMPC(SUPPORT, .95, horizon=2, dt=dt)
    planned = mpc.plan([0, 0], [0, 0], [0, 0], np.zeros((2, 2)), stamp=stamp)
    np.testing.assert_allclose(planned.sample(stamp)[0], 0., atol=1e-9)


def test_continuous_zmp_support_and_jerk_dynamics():
    mpc = CoMPreviewMPC(SUPPORT, .95)
    plan = mpc.plan([0., 0.], [0., 0.], [0., 0.], np.tile([.07, .15], (20, 1)))
    for stamp in np.linspace(0., 2., 401, endpoint=False):
        p, v, a = plan.sample(stamp)
        zmp = p - .95 / 9.81 * a
        assert np.max(mpc.normals @ zmp - mpc.offsets) < 1e-6
        h = 1e-6
        if stamp > h:
            before, after = plan.sample(stamp - h), plan.sample(stamp + h)
            np.testing.assert_allclose((after[0] - before[0]) / (2 * h), v, atol=1e-7)
            np.testing.assert_allclose((after[1] - before[1]) / (2 * h), a, atol=1e-7)
    assert np.max(np.abs(plan.jerks)) <= mpc.jerk_limit + 1e-6
    capture = plan.states[-1, 0] + np.sqrt(.95 / 9.81) * plan.states[-1, 1]
    assert np.max(mpc.normals @ capture - mpc.offsets) < 1e-6


def test_unreachable_target_remains_within_support():
    mpc = CoMPreviewMPC(SUPPORT, .95)
    plan = mpc.plan([0, 0], [0, 0], [0, 0], np.tile([1., 1.], (20, 1)))
    assert np.max(mpc.normals @ plan.zmp.T - mpc.offsets[:, None]) < 1e-6
    assert np.linalg.norm(plan.states[-1, 0] - [1, 1]) > .5


def test_infeasible_measured_state_clears_plan():
    mpc = CoMPreviewMPC(SUPPORT, .95)
    mpc.plan([0, 0], [0, 0], [0, 0], np.zeros((20, 2)))
    with pytest.raises(RuntimeError, match="CoM MPC failed"):
        mpc.plan([2, 2], [0, 0], [0, 0], np.zeros((20, 2)))
    assert mpc.last_plan is None
    mpc.reset()
    assert mpc.replans == 0
    mpc.plan([0, 0], [0, 0], [0, 0], np.zeros((20, 2)))


def test_invalid_parameters_and_inconsistent_plan():
    for kwargs in ({"height": -1}, {"height": .9, "dt": 0}, {"height": .9, "horizon": True}):
        with pytest.raises(ValueError):
            CoMPreviewMPC(SUPPORT, **kwargs)
    with pytest.raises(ValueError):
        CoMPreviewMPC([[0, 0], [1, 0], [2, 0]], .95)
    with pytest.raises(ValueError, match="jerk integration"):
        CoMPlan(np.ones((3, 3, 2)), np.zeros((2, 2)), 0., .1, .95, 9.81)



def test_cached_model_settings_and_plan_cannot_be_mutated():
    mpc = CoMPreviewMPC(SUPPORT, .95)
    for key in ("height", "dt", "horizon", "velocity_limit", "gravity"):
        with pytest.raises(AttributeError):
            setattr(mpc, key, 2)
    for value in (mpc.normals, mpc.offsets, mpc.support_vertices):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    plan = mpc.plan([0, 0], [0, 0], [0, 0], np.zeros((20, 2)))
    with pytest.raises(ValueError):
        plan.states.setflags(write=True)
    with pytest.raises(ValueError):
        plan.jerks.setflags(write=True)
