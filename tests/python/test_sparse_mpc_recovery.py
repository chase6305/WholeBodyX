"""Default sparse planner can cold-retry a refused warm solve once."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import JointLimits, OSQPSolver, RobotState, SparseJointMPC


def planner(solver=None):
    return SparseJointMPC(JointLimits([-1.], [1.], [1.], [2.]), horizon=5, solver=solver)


@pytest.mark.parametrize('status,value', [('maximum iterations reached', 7), ('solved inaccurate', 2)])
def test_default_warm_refusal_gets_one_cold_retry_and_accumulated_iterations(monkeypatch, status, value):
    p = planner()
    state = RobotState([0.], [0.])
    assert p.solve(state, [.3]).plan is not None
    old_workspace = p.solver._solver
    monkeypatch.setattr(old_workspace, 'solve', lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status=status, status_val=value, iter=20000), x=None))
    result = p.solve(state, [.3])
    assert result.plan is not None and result.solver.success
    assert p.solver._solver is not old_workspace
    assert result.solver.iterations > 20000
    assert result.solver.detail.startswith('cold retry after ' + status + '; solved')
    assert result.solver.violation <= 1e-6


def test_custom_solver_retains_single_attempt_policy(monkeypatch):
    solver = OSQPSolver()
    p = planner(solver)
    state = RobotState([0.], [0.])
    assert p.solve(state, [.3]).plan is not None
    workspace = solver._solver
    monkeypatch.setattr(workspace, 'solve', lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status='maximum iterations reached', status_val=7, iter=20000), x=None))
    result = p.solve(state, [.3])
    assert result.plan is None and result.solver.status == 'iteration_limit'
    assert solver._solver is workspace
    assert not result.solver.detail.startswith('cold retry after ')


@pytest.mark.parametrize('status,value', [('primal infeasible', 3), ('interrupted', 10)])
def test_infeasible_and_other_errors_do_not_trigger_retry(monkeypatch, status, value):
    p = planner()
    state = RobotState([0.], [0.])
    assert p.solve(state, [.3]).plan is not None
    workspace = p.solver._solver
    monkeypatch.setattr(workspace, 'solve', lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status=status, status_val=value, iter=25), x=None))
    result = p.solve(state, [.3])
    assert result.plan is None and p.solver._solver is workspace
    assert not result.solver.detail.startswith('cold retry after ')


def test_failed_cold_retry_stops_and_clears_previous_plan(monkeypatch):
    p = planner()
    state = RobotState([0.], [0.])
    assert p.solve(state, [.3]).plan is not None
    old = p.solver._solver
    monkeypatch.setattr(old, 'solve', lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status='maximum iterations reached', status_val=7, iter=20000), x=None))
    # A fresh workspace inherits this deliberate one-iteration budget and also refuses.
    p.solver._settings['max_iter'] = 1
    result = p.solve(state, [.3])
    assert result.plan is None and not result.solver.success
    assert result.solver.iterations == 20001
    assert result.solver.detail.count('cold retry after ') == 1
    assert p._last_plan is None
