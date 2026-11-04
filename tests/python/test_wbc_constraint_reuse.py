"""Changing Hessians must keep immutable constraints and all validation guards."""

import numpy as np
import pytest

from wholebodyx import JointLimits, KinematicWBC, LinearTask, OSQPSolver, QuadraticProgram, RobotState
from wholebodyx.qp import _FixedQP


def test_dynamic_wbc_validates_each_hessian_and_reuses_one_constraint(monkeypatch):
    original_h, original_a = QuadraticProgram._validate_hessian, QuadraticProgram._validate_constraint
    calls = dict(h=0, a=0)

    def check_h(self):
        calls['h'] += 1
        original_h(self)

    def check_a(self):
        calls['a'] += 1
        original_a(self)

    monkeypatch.setattr(QuadraticProgram, '_validate_hessian', check_h)
    monkeypatch.setattr(QuadraticProgram, '_validate_constraint', check_a)
    controller = KinematicWBC(JointLimits([-1.] * 3, [1.] * 3, [1.] * 3))
    state, problems = RobotState([0.] * 3, [0.] * 3), []
    for scale in (1., 2., .5, 3.):
        p = controller.formulate(state, [LinearTask(scale * np.eye(3), [.2] * 3)], .1)
        assert controller.solver.solve(p).success
        problems.append(p)
    assert calls == dict(h=4, a=1)
    assert all(np.shares_memory(p.constraint, problems[0].constraint) for p in problems)
    assert all(p.constraint is not problems[0].constraint for p in problems[1:])
    controller.reset()
    assert controller.step(state, [LinearTask(4 * np.eye(3), [.2] * 3)], .1).solver.success
    assert calls == dict(h=5, a=1)


@pytest.mark.parametrize('field,value', [('hessian', [[-1., 0.], [0., 1.]]),
                                       ('hessian', [[1., .5], [0., 1.]]),
                                       ('hessian', [[np.nan, 0.], [0., 1.]]),
                                       ('hessian', [1., 1.]),
                                       ('gradient', [0., np.nan]),
                                       ('gradient', [0.]),
                                       ('lower', [np.nan, -1.]),
                                       ('upper', [1.])])
def test_replacement_validates_actual_hessian_and_vectors_without_poisoning_template(field, value):
    template = _FixedQP(np.eye(2), np.eye(2))
    values = dict(hessian=2 * np.eye(2), gradient=[-.2, -.3], lower=[-1., -1.], upper=[1., 1.])
    values[field] = value
    with pytest.raises(ValueError):
        template.with_hessian(**values)
    p = template.problem([-.2, -.3], [-1., -1.], [1., 1.])
    np.testing.assert_allclose(OSQPSolver().solve(p).x, [.2, .3], atol=1e-7)


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'strides', 'lower_nan'])
def test_retained_public_diagnostics_cannot_mutate_future_constraint_metadata(kind):
    a = np.array([[1., 2., 0.], [0., 1., -1.]])
    template, first = _FixedQP.with_problem(np.eye(3), [0.] * 3, a, [-1.] * 2, [1.] * 2)
    if kind == 'shape':
        first.constraint.shape = (6,)
    elif kind == 'dtype':
        first.constraint.dtype = np.int64
    elif kind == 'strides':
        first.constraint.strides = (8, 8)
    else:
        first.lower.setflags(write=True)
        first.lower[:] = np.nan
    next_template, p = template.with_hessian(2 * np.eye(3), [0.] * 3, [-1.] * 2, [1.] * 2)
    np.testing.assert_array_equal(p.constraint, a)
    assert p.constraint.dtype == np.float64 and OSQPSolver().solve(p).success
    p.constraint.shape = (6,)
    np.testing.assert_array_equal(next_template.problem([0.] * 3, [-1.] * 2, [1.] * 2).constraint, a)
    with pytest.raises(ValueError):
        next_template._constraint.setflags(write=True)


def test_replacement_does_not_certify_unvalidated_constraint_storage():
    template = _FixedQP(np.eye(2), np.eye(2))
    healthy = template.problem([0., 0.], [-1., -1.], [1., 1.])
    solver = OSQPSolver()
    assert solver.solve(healthy).success
    workspace = solver._solver
    # Even a replaced private buffer must not gain a fresh validation certificate.
    original = template._constraint
    template._constraint = np.full((2, 2), np.nan)
    _, invalid = template.with_hessian(2 * np.eye(2), [0., 0.], [-1., -1.], [1., 1.])
    with pytest.raises(ValueError, match='constraint must be finite'):
        solver.solve(invalid)
    assert solver._solver is workspace
    template._constraint = original
    _, recovered = template.with_hessian(3 * np.eye(2), [0., 0.], [-1., -1.], [1., 1.])
    assert solver.solve(recovered).success


def test_controller_dimension_change_builds_new_identity_constraints():
    controller = KinematicWBC(JointLimits([-1.] * 3, [1.] * 3, [1.] * 3))
    retained = []
    for n in (3, 1, 6, 3):
        controller.limits = JointLimits([-1.] * n, [1.] * n, [1.] * n)
        state = RobotState(np.zeros(n), np.zeros(n))
        p = controller.formulate(state, [LinearTask(2 * np.eye(n), np.full(n, .2))], .1)
        assert controller.solver.solve(p).success
        np.testing.assert_array_equal(p.constraint, np.eye(n))
        retained.append(p)
    assert [p.constraint.shape for p in retained] == [(3, 3), (1, 1), (6, 6), (3, 3)]


def test_reused_constraint_preserves_scaled_solver_physical_coordinates():
    a = np.array([[1., 2.], [0., 1.]])
    template = _FixedQP(np.eye(2), a)
    solver = OSQPSolver(variable_scaling=[.1, 2.])
    for k in range(20):
        h = (1 + .1 * k) * np.array([[2., .2], [.2, 1.]])
        g, lo, hi = [-.3, .2], [-.5, -.5], [.5, .5]
        template, p = template.with_hessian(h, g, lo, hi)
        actual = solver.solve(p)
        expected = OSQPSolver(variable_scaling=[.1, 2.]).solve(QuadraticProgram(h, g, a, lo, hi))
        assert actual.success and expected.success
        np.testing.assert_allclose(actual.x, expected.x, atol=1e-7, rtol=0)
        assert np.max(a @ actual.x - hi) <= 1e-6 and np.max(lo - a @ actual.x) <= 1e-6
