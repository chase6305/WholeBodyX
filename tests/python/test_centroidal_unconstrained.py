"""A feasible free optimum may bypass OSQP; active constraints and hooks may not."""

import numpy as np
import pytest

from wholebodyx import OSQPSolver, QPResult
from wholebodyx.centroidal import CoMPreviewMPC
from wholebodyx.qp import _FixedQP


SUPPORT = np.array([[-.12, -.09], [.12, -.09], [.12, .09], [-.12, .09]])


def rollout(initial, jerks, dt):
    states = np.empty((len(jerks) + 1, 3, 2))
    states[0] = initial
    for k, jerk in enumerate(jerks):
        p, v, a = states[k]
        states[k + 1] = [p + dt*v + .5*dt**2*a + dt**3/6*jerk,
                         v + dt*a + .5*dt**2*jerk, a + dt*jerk]
    return states


def known_optimum(planner, *, active=False):
    """Construct targets whose physical jerk-cost KKT optimum is known."""
    n = planner.horizon
    initial = np.array([[0., 0.], [.001, -.002], [.002, .003]])
    jerks = np.column_stack((.08*np.cos(np.arange(n)), .06*np.sin(np.arange(n))))
    multiplier = np.zeros(2*n)
    if active:
        jerks[0, 0] = planner.jerk_limit
        multiplier[0] = .001
    states = rollout(initial, jerks, planner.dt)
    # Target velocities exactly match the intended rollout. A tracking offset
    # balances the jerk regularizer and, for an active upper bound, its dual.
    correction = np.linalg.solve(planner.position_map.T, 1e-5*jerks.ravel() + multiplier)
    targets = states[1:, 0] + correction.reshape(n, 2)
    return initial, targets, states[1:, 1], jerks, states


@pytest.mark.parametrize('horizon', [2, 5])
@pytest.mark.parametrize('angle', [0., .3])
def test_feasible_free_kkt_optimum_does_not_create_an_osqp_workspace(horizon, angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    planner = CoMPreviewMPC(SUPPORT @ rotation.T, .8, horizon=horizon)
    initial, targets, velocities, expected, states = known_optimum(planner)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is None
    np.testing.assert_allclose(plan.jerks, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(plan.states, states, atol=2e-12, rtol=0)
    assert planner.validate_plan(plan) == 0
    assert planner.last_plan is plan and planner.replans == 1
    with pytest.raises(ValueError):
        plan.jerks.setflags(write=True)


@pytest.mark.parametrize('horizon', [2, 5])
def test_active_jerk_constraint_uses_osqp_and_keeps_the_kkt_optimum(horizon):
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=horizon)
    initial, targets, velocities, expected, _ = known_optimum(planner, active=True)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is not None
    np.testing.assert_allclose(plan.jerks, expected, atol=3e-5, rtol=0)
    assert planner.validate_plan(plan) <= 1e-6


def test_free_active_free_transitions_and_reset_preserve_plans():
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    saved = None
    for k, active in enumerate((False, True, False, False, True, False)):
        initial, targets, velocities, expected, _ = known_optimum(planner, active=active)
        plan = planner.plan(*initial, targets, stamp=k*.01, target_velocities=velocities)
        np.testing.assert_allclose(plan.jerks, expected, atol=3e-5 if active else 2e-10, rtol=0)
        assert planner.validate_plan(plan) <= 1e-6
        if k == 1:
            workspace = planner.solver._solver
        if k == 2:
            assert planner.solver._solver is workspace
        if saved is not None:
            np.testing.assert_array_equal(saved.jerks, saved_values)
        saved, saved_values = plan, plan.jerks.copy()
        if k == 2:
            planner.reset()
            assert planner.last_plan is None and planner.replans == 0
            assert planner.solver._solver is None


@pytest.mark.parametrize('value', [None, 0, 1, 'yes', np.bool_(True)])
def test_fast_path_option_requires_a_boolean(value):
    with pytest.raises(ValueError, match='unconstrained_fast_path must be boolean'):
        CoMPreviewMPC(SUPPORT, .8, unconstrained_fast_path=value)


def test_explicit_disable_retains_the_backend_and_option_is_read_only():
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3, unconstrained_fast_path=False)
    assert planner.unconstrained_fast_path is False
    with pytest.raises(AttributeError):
        planner.unconstrained_fast_path = True
    initial, targets, velocities, expected, _ = known_optimum(planner)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is not None
    np.testing.assert_allclose(plan.jerks, expected, atol=3e-5, rtol=0)


@pytest.mark.parametrize('kind', ['replacement', 'subclass', 'instance_method', 'class_method'])
def test_custom_solver_hooks_are_always_called(monkeypatch, kind):
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    calls = []
    original = OSQPSolver.solve
    if kind == 'replacement':
        planner.solver = OSQPSolver()
    elif kind == 'subclass':
        class Plugin(OSQPSolver):
            pass
        planner.solver = Plugin()
    elif kind == 'class_method':
        def recording(self, problem, **kwargs):
            calls.append(problem)
            return original(self, problem, **kwargs)
        monkeypatch.setattr(OSQPSolver, 'solve', recording)
    if kind != 'class_method':
        implementation = planner.solver.solve
        def recording(problem, **kwargs):
            calls.append(problem)
            return implementation(problem, **kwargs)
        monkeypatch.setattr(planner.solver, 'solve', recording)
    initial, targets, velocities, _, _ = known_optimum(planner)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert len(calls) == 1 and planner.solver._solver is not None
    assert planner.validate_plan(plan) <= 1e-6


def test_replacement_model_cannot_reuse_an_old_cost_factor():
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    initial, targets, velocities, original_optimum, _ = known_optimum(planner)
    planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is None
    template = planner.problem
    planner.problem = _FixedQP(2*template._hessian, template._constraint)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is not None
    np.testing.assert_allclose(plan.jerks, .5*original_optimum, atol=3e-5, rtol=0)


def test_solver_failure_after_a_free_plan_clears_the_old_plan(monkeypatch):
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    initial, targets, velocities, _, _ = known_optimum(planner)
    planner.plan(*initial, targets, target_velocities=velocities)
    monkeypatch.setattr(planner.solver, 'solve', lambda problem: QPResult('error', detail='injected'))
    with pytest.raises(RuntimeError, match='injected'):
        planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.last_plan is None and planner.replans == 1


@pytest.mark.parametrize('field', ['gradient', 'lower', 'upper'])
def test_changed_nonfinite_vectors_are_checked_before_lapack(monkeypatch, field):
    import wholebodyx.centroidal as module

    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    initial, targets, velocities, _, _ = known_optimum(planner)
    original = planner.problem.problem
    def malformed(*args):
        problem = original(*args)
        vector = getattr(problem, field)
        vector.setflags(write=True)
        vector[0] = np.nan
        return problem
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid vectors must not reach LAPACK')
    monkeypatch.setattr(planner.problem, 'problem', malformed)
    monkeypatch.setattr(module, 'cho_solve', forbidden)
    with pytest.raises(ValueError, match='QP vectors'):
        planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is None


def test_nonfinite_factor_disables_direct_solves(monkeypatch):
    import wholebodyx.centroidal as module

    monkeypatch.setattr(module, 'cholesky', lambda matrix, **kwargs: np.full_like(matrix, np.nan))
    planner = CoMPreviewMPC(SUPPORT, .8, horizon=3)
    initial, targets, velocities, _, _ = known_optimum(planner)
    plan = planner.plan(*initial, targets, target_velocities=velocities)
    assert planner.solver._solver is not None
    assert planner.validate_plan(plan) <= 1e-6
