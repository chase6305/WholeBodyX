"""A custom QP solver cannot turn an unsafe vector into a valid joint plan."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import JointLimits, JointMPC, QPResult, RobotState, SparseJointMPC


@pytest.mark.parametrize("planner_type", [JointMPC, SparseJointMPC])
@pytest.mark.parametrize("kind", ["short", "long", "nan", "inf", "velocity", "position", "slew"])
def test_bad_solved_result_is_rejected_without_overwriting_warm_history(planner_type, kind):
    limits = JointLimits([-1], [1], [1], [1] if kind == "slew" else None)
    planner = planner_type(limits, horizon=3, dt=.1)
    state = RobotState([.99] if kind == "position" else [0], [0])
    previous = planner.solve(state, [0]).plan
    assert previous is not None
    count = 6 if planner_type is SparseJointMPC else 3
    decision = np.zeros(count)
    if kind in ("short", "long"):
        decision = np.zeros(count + (1 if kind == "long" else -1))
    elif kind in ("nan", "inf"):
        decision[-1] = getattr(np, kind)
    else:
        decision[-3:] = 1.5 if kind == "velocity" else .2
    planner.solver = SimpleNamespace(solve=lambda *args, **kwargs:
                                     QPResult("solved", decision, iterations=7, violation=0.))
    rejected = planner.solve(state, [0])
    assert rejected.plan is None and not rejected.solver.success
    assert rejected.solver.x is None and rejected.solver.iterations == 7
    assert planner._last_plan is previous


@pytest.mark.parametrize("planner_type", [JointMPC, SparseJointMPC])
def test_tiny_constraint_roundoff_remains_within_the_public_absolute_tolerance(planner_type):
    planner = planner_type(JointLimits([-100], [100], [1]), horizon=3, dt=.1)
    decision = np.full(6 if planner_type is SparseJointMPC else 3, 1. + 5e-7)
    planner.solver = SimpleNamespace(solve=lambda *args, **kwargs:
                                     QPResult("solved", decision, violation=5e-7))
    accepted = planner.solve(RobotState([0], [0]), [0])
    assert accepted.plan is not None
    np.testing.assert_allclose(accepted.plan.velocities, 1. + 5e-7, atol=0., rtol=0.)


@pytest.mark.parametrize("planner_type", [JointMPC, SparseJointMPC])
@pytest.mark.parametrize("kind", ["integer_extreme", "complex", "integer_zero"])
def test_custom_solution_uses_real_physical_arithmetic_before_plan_acceptance(planner_type, kind):
    planner = planner_type(JointLimits([-1e20], [1e20], [.8]), horizon=1, dt=.1)
    state = RobotState([0.], [0.])
    previous = planner.solve(state, [0.]).plan
    size = 2 if planner_type is SparseJointMPC else 1
    decision = np.zeros(size, dtype=complex if kind == "complex" else np.int64)
    decision[-1] = .1j if kind == "complex" else -(2**63) if kind == "integer_extreme" else 0
    planner.solver = SimpleNamespace(solve=lambda *args, **kwargs: QPResult("solved", decision, violation=0.))
    result = planner.solve(state, [0.])
    if kind == "integer_zero":
        assert result.plan is not None and result.solver.success
        np.testing.assert_array_equal(result.plan.velocities, [[0.]])
    else:
        assert result.plan is None and not result.solver.success and result.solver.x is None
        assert planner._last_plan is previous
