"""Immutable dynamic WBC models must retain validation and isolation."""

import numpy as np
import pytest

from wholebodyx import JointLimits, KinematicWBC, LinearTask, OSQPSolver, QuadraticProgram, RobotState
from wholebodyx.qp import _FixedQP


def test_changed_wbc_models_validate_once_before_solver_reuse(monkeypatch):
    original, calls = QuadraticProgram._validate_hessian, []

    def validate(self):
        calls.append(self.hessian.copy())
        original(self)

    monkeypatch.setattr(QuadraticProgram, '_validate_hessian', validate)
    controller = KinematicWBC(JointLimits([-1] * 3, [1] * 3, [1] * 3))
    state = RobotState([0] * 3, [0] * 3)
    for scale, expected_calls in ((1., 1), (1., 1), (2., 2), (.5, 3), (.5, 3)):
        result = controller.step(state, [LinearTask(scale * np.eye(3), [.2] * 3)], .1)
        assert result.solver.success
        assert len(calls) == expected_calls
    controller.reset()
    assert controller.step(state, [LinearTask(.5 * np.eye(3), [.3] * 3)], .1).solver.success
    assert len(calls) == 3


@pytest.mark.parametrize('field,value', [('hessian', [[-1., 0.], [0., 1.]]),
                                       ('hessian', [[1., .5], [0., 1.]]),
                                       ('hessian', [[np.nan, 0.], [0., 1.]]),
                                       ('hessian', [1., 1.]),
                                       ('constraint', [[1., 0.], [0., np.inf]]),
                                       ('constraint', [[1., 0., 0.]]),
                                       ('gradient', [0., np.nan]),
                                       ('gradient', [0.]),
                                       ('lower', [0., np.nan])])
def test_dynamic_template_validates_actual_model_and_vectors(field, value):
    values = dict(hessian=np.eye(2), gradient=[0., 0.], constraint=np.eye(2),
                  lower=[-1., -1.], upper=[1., 1.])
    values[field] = value
    with pytest.raises(ValueError):
        _FixedQP.with_problem(**values)


def test_failed_dynamic_wbc_model_does_not_replace_healthy_template():
    controller = KinematicWBC(JointLimits([-1] * 2, [1] * 2, [1] * 2))
    state, task = RobotState([0, 0], [0, 0]), LinearTask(np.eye(2), [.2, -.3])
    assert controller.step(state, [task], .1).solver.success
    template = controller._template
    task.weight.setflags(write=True)
    task.weight[:] = -1
    with pytest.raises(ValueError, match='positive semidefinite'):
        controller.step(state, [task], .1)
    assert controller._template is template
    task.weight[:] = 1
    assert controller.step(state, [task], .1).solver.success


def test_dynamic_template_metadata_is_separate_from_first_returned_problem():
    h, a = np.diag([2., 3.]), np.eye(2)
    template, first = _FixedQP.with_problem(h, [-.2, -.3], a, [-1., -1.], [1., 1.])
    h[:] = a[:] = 99
    first.hessian.shape = (4,)
    first.constraint.dtype = np.int64
    first.gradient.setflags(write=True)
    first.gradient[:] = np.nan
    second = template.problem([-.2, -.3], [-1., -1.], [1., 1.])
    np.testing.assert_array_equal(second.hessian, np.diag([2., 3.]))
    np.testing.assert_array_equal(second.constraint, np.eye(2))
    np.testing.assert_allclose(OSQPSolver().solve(second).x, [.1, .1], atol=1e-7)
    with pytest.raises(ValueError):
        second.hessian.setflags(write=True)


@pytest.mark.parametrize('joints', [1, 6, 19, 30])
def test_dynamic_wbc_closed_loop_matches_public_dense_qp(joints):
    rng = np.random.default_rng(20261022 + joints)
    bounds = JointLimits([-1.] * joints, [1.] * joints, [.7] * joints, [2.] * joints)
    controller, reference = KinematicWBC(bounds, regularization=.01, smoothing=.02), OSQPSolver()
    state = RobotState(np.full(joints, .8), np.zeros(joints))
    for k in range(250):
        dt = .01
        jac = rng.normal(size=(max(1, joints // 2), joints))
        task = LinearTask(jac, rng.normal(size=jac.shape[0]), rng.uniform(.1, 2., jac.shape[0]))
        lo, hi = bounds.velocity_bounds(state, dt)
        h = .03 * np.eye(joints) + jac.T @ (task.weight[:, None] * jac)
        g = -.02 * state.v - jac.T @ (task.weight * task.target_velocity)
        expected = reference.solve(QuadraticProgram(h, g, np.eye(joints), lo, hi))
        actual = controller.step(state, [task], dt)
        assert expected.success and actual.solver.success
        velocity = actual.command.velocity
        np.testing.assert_allclose(velocity, np.clip(expected.x, lo, hi), atol=1e-7, rtol=0)
        assert np.all(velocity >= lo) and np.all(velocity <= hi)
        state = RobotState(state.q + dt * velocity, velocity, state.stamp + dt)
