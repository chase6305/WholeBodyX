from types import SimpleNamespace

import numpy as np

from wholebodyx import dynamics
from wholebodyx._contact_qp import ScaledContactQP, solve_contact_working_set


def test_working_set_repairs_incorrect_bound_sides():
    # x[0] must reach its upper bound, x[1] its lower, and x[2] is fixed.
    problem = ScaledContactQP(
        np.eye(3), np.array([-3., 3., -5.]), np.eye(3),
        np.array([-1., -1., 0.5]), np.array([1., 1., 0.5]),
    )
    result = solve_contact_working_set(problem, np.array([-1., 1., 0.]))
    assert result is not None
    primal, dual = result
    np.testing.assert_array_equal(primal, [1., -1., 0.5])
    np.testing.assert_array_equal(dual, [2., -2., 4.5])


def test_working_set_rejects_singular_kkt():
    problem = ScaledContactQP(
        np.eye(1), np.zeros(1), np.ones((2, 1)), np.zeros(2), np.zeros(2),
    )
    assert solve_contact_working_set(problem, np.zeros(2)) is None


def test_working_set_enforces_a_weak_inactive_bound():
    problem = ScaledContactQP(
        np.eye(1), np.zeros(1), np.array([[1e-4]]), np.array([2.5e-9]), np.array([np.inf]),
    )
    primal, dual = solve_contact_working_set(problem, np.zeros(1))
    np.testing.assert_allclose(primal, [2.5e-5], rtol=0, atol=1e-14)
    np.testing.assert_allclose(dual, [-.25], rtol=0, atol=1e-14)


def test_controller_reuses_checked_working_set_without_osqp(floating_robot, monkeypatch):
    robot, state = floating_robot
    controller = dynamics.ContactWBC(robot, state)
    target = robot.terms(state).com
    before = controller.solve(state, target)

    def unexpected_solve(**_):
        raise AssertionError("the verified working set already solves the QP")

    monkeypatch.setattr(controller._solver, "solve", unexpected_solve)
    after = controller.solve(state, target)
    assert after.iterations == 0 and after.solver.detail == "working set"
    assert after.violation <= 1e-8
    np.testing.assert_allclose(before.solver.x, after.solver.x, rtol=0, atol=1e-8)
    controller.reset()
    assert controller._last_dual is None


def test_controller_finishes_osqp_when_working_set_fails(floating_robot, monkeypatch):
    robot, state = floating_robot
    controller = dynamics.ContactWBC(robot, state)
    target = robot.terms(state).com
    controller.solve(state, target)
    backend = controller._solver
    solve = backend.solve
    budgets = []

    def staged(**kwargs):
        budgets.append(backend.settings.max_iter)
        if len(budgets) == 1:
            return SimpleNamespace(
                info=SimpleNamespace(status_val=7, iter=600), y=controller._last_dual
            )
        return solve(**kwargs)

    monkeypatch.setattr(dynamics, "solve_contact_working_set", lambda *_: None)
    monkeypatch.setattr(backend, "solve", staged)
    result = controller.solve(state, target)
    assert result.violation <= 1e-5 and result.iterations > 600
    assert budgets == [600, 29400]
    assert backend.settings.max_iter == 30000
    assert controller.solver_recoveries == 0
