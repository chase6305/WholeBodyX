"""Non-involutive permutations and asymmetric limits catch direction mistakes."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (ControlResult, JointLimits, JointOrder, JointPlan, LinearTask, MPCReferenceManager,
                       PlanResult, QPResult, ReorderedController, ReorderedPlanner, RobotState, VelocityCommand)


def order():
    return JointOrder(["shoulder", "elbow", "wrist"], ["wrist", "shoulder", "elbow"])


def limits(acceleration=True):
    return JointLimits([-1.2, -.7, -2.1], [1., 1.4, .8], [.15, .6, .9], [2, 3, 4] if acceleration else None)


@pytest.mark.parametrize("model,backend", [([], []), (["a", "a"], ["a", "b"]),
                                          (["a", "b"], ["a", "a"]), (["a", "b"], ["a", "c"]),
                                          (["a", "b"], ["a"]), (["", "a"], ["a", ""]),
                                          (["a "], ["a "]), ("abc", "cab")])
def test_reject_ambiguous_joint_names(model, backend):
    with pytest.raises(ValueError):
        JointOrder(model, backend)


def test_vectors_columns_and_task_residuals_have_the_same_physical_meaning():
    mapping = order()
    vector = np.array([.1, .3, -.7])
    np.testing.assert_array_equal(mapping.to_backend(vector), [-.7, .1, .3])
    np.testing.assert_array_equal(mapping.to_model(mapping.to_backend(vector)), vector)
    jacobian = np.array([[1, 3, -2], [.2, -.1, .7]])
    task = LinearTask(jacobian, [.2, -.3], [2, 7])
    mapped = mapping.task_to_backend(task)
    np.testing.assert_allclose(mapped.jacobian @ mapping.to_backend(vector), jacobian @ vector)
    np.testing.assert_array_equal(mapped.target_velocity, task.target_velocity)
    np.testing.assert_array_equal(mapped.weight, task.weight)
    assert not np.shares_memory(mapped.jacobian, task.jacobian)
    assert not mapped.jacobian.flags.writeable
    for value in (np.zeros(2), np.zeros((3, 2)), np.zeros((1, 1, 3)), [0, 0, np.nan]):
        with pytest.raises(ValueError):
            mapping.to_backend(value)


def test_names_and_converted_values_are_owned_snapshots():
    model, backend = ["a", "b", "c"], ["c", "a", "b"]
    mapping = JointOrder(model, backend)
    model.reverse()
    backend.reverse()
    assert mapping.model_names == ("a", "b", "c")
    values = np.array([[1., 2., 3.]])
    result = mapping.to_backend(values)
    values[:] = 9
    np.testing.assert_array_equal(result, [[3, 1, 2]])


@pytest.mark.parametrize("acceleration", [False, True])
def test_limits_and_state_share_one_permutation(acceleration):
    mapping = order()
    source = limits(acceleration)
    target = mapping.limits_to_backend(source)
    state = RobotState([.2, .25, -.35], [.05, -.12, .08], 2.7)
    converted = mapping.state_to_backend(state)
    assert converted.stamp == state.stamp
    lo, hi = source.velocity_bounds(state, .02)
    mapped_lo, mapped_hi = target.velocity_bounds(converted, .02)
    np.testing.assert_allclose(mapped_lo, mapping.to_backend(lo))
    np.testing.assert_allclose(mapped_hi, mapping.to_backend(hi))
    assert (target.acceleration is None) == (source.acceleration is None)


def test_plan_and_command_keep_timestamps_and_sampling():
    mapping = order()
    q = np.array([.1, .3, -.2])
    velocities = np.array([[.2, -.1, .4], [-.1, .5, .2]])
    model_plan = JointPlan(np.vstack((q, q + .1 * np.cumsum(velocities, axis=0))), velocities, 1.7, .1)
    backend_plan = JointPlan(mapping.to_backend(model_plan.positions), mapping.to_backend(velocities), 1.7, .1)
    result = mapping.plan_to_model(backend_plan)
    for stamp in (1.7, 1.74, 1.8, 1.89):
        actual, expected = result.sample(stamp), model_plan.sample(stamp)
        np.testing.assert_allclose(actual, expected)
    assert result.stamp == backend_plan.stamp and result.valid_until == backend_plan.valid_until
    command = mapping.command_to_model(VelocityCommand(mapping.to_backend(velocities[0]), 1.7, 1.72))
    np.testing.assert_array_equal(command.velocity, velocities[0])
    assert command.stamp == 1.7 and command.valid_until == 1.72


@pytest.mark.parametrize("acceleration", [False, True])
def test_wbc_coupled_tasks_with_asymmetric_hard_bounds(backend, acceleration):
    controller_type, _ = backend
    mapping, bounds = order(), limits(acceleration)
    state = RobotState([.2, .25, -.35], [.05, -.12, .08], 2.7)
    tasks = [LinearTask([[1, 2, -.5], [-.2, .4, 1.3]], [1.1, -.7], [2, 5]),
             LinearTask([[.7, -.5, .2]], [.3], 3)]
    baseline = controller_type(bounds, regularization=.01, smoothing=.03)
    wrapped = ReorderedController(controller_type(mapping.limits_to_backend(bounds), regularization=.01, smoothing=.03), mapping)
    original, result = baseline.step(state, tasks, .02), wrapped.step(state, tasks, .02)
    assert result.solver.success and original.solver.success
    np.testing.assert_allclose(result.command.velocity, original.command.velocity, atol=2e-7)
    lo, hi = bounds.velocity_bounds(state, .02)
    assert np.all(result.command.velocity >= lo - 1e-6) and np.all(result.command.velocity <= hi + 1e-6)
    if not acceleration:
        assert result.command.velocity[0] == pytest.approx(bounds.velocity[0], abs=1e-6)


def test_mpc_prediction_and_reference_validation_use_model_order(backend):
    _, planner_type = backend
    mapping, bounds = order(), limits()
    state, goal = RobotState([.2, .25, -.35], [.05, -.12, .08], 2.7), np.array([.8, -.4, .6])
    baseline = planner_type(bounds, horizon=8)
    wrapped = ReorderedPlanner(planner_type(mapping.limits_to_backend(bounds), horizon=8), mapping)
    original, result = baseline.solve(state, goal), wrapped.solve(state, goal)
    assert original.solver.success and result.solver.success
    np.testing.assert_allclose(result.plan.positions, original.plan.positions, atol=2e-7)
    np.testing.assert_allclose(result.plan.velocities, original.plan.velocities, atol=2e-7)
    # QP vectors may include slacks or auxiliary variables; they are not commands.
    np.testing.assert_allclose(result.solver.x.reshape(8, 3), mapping.to_backend(result.plan.velocities), atol=1e-12)
    assert MPCReferenceManager(wrapped, bounds).update(state, goal, .02).success
    wrapped.reset()
    np.testing.assert_allclose(wrapped.solve(state, goal).plan.positions, original.plan.positions, atol=2e-7)


@pytest.mark.parametrize("kind", ["controller", "planner"])
def test_solver_diagnostics_are_copied_without_guessing_variable_layout(kind):
    x = np.array([7., 8., 9., 10.])  # Three joints plus an auxiliary variable.
    solver = QPResult("solved", x, 12, 0., "external library")
    command = VelocityCommand([.1, .2, .3], 0, .02)
    plan = JointPlan(np.zeros((3, 3)), np.zeros((2, 3)), 0, .1)
    state = RobotState(np.zeros(3), np.zeros(3))
    if kind == "controller":
        wrapped = ReorderedController(SimpleNamespace(reset=lambda: None, step=lambda *args: ControlResult(solver, command)), order())
        result = wrapped.step(state, [], .02)
    else:
        wrapped = ReorderedPlanner(SimpleNamespace(reset=lambda: None, solve=lambda *args: PlanResult(solver, plan)), order())
        result = wrapped.solve(state, np.ones(3))
    x[:] = -99
    np.testing.assert_array_equal(result.solver.x, [7, 8, 9, 10])
    assert result.solver.detail == "external library" and result.solver.iterations == 12


@pytest.mark.parametrize("kind", ["controller", "planner"])
def test_failure_and_reset_forwarding(kind):
    resets = []
    solver = QPResult("infeasible", detail="external infeasible bounds")
    state = RobotState(np.zeros(3), np.zeros(3))
    if kind == "controller":
        wrapped = ReorderedController(SimpleNamespace(reset=lambda: resets.append(1), step=lambda *args: ControlResult(solver)), order())
        result = wrapped.step(state, [], .02)
        assert result.command is None
    else:
        wrapped = ReorderedPlanner(SimpleNamespace(reset=lambda: resets.append(1), solve=lambda *args: PlanResult(solver)), order())
        result = wrapped.solve(state, np.ones(3))
        assert result.plan is None
    assert result.solver.status == "infeasible" and result.solver.detail == solver.detail
    wrapped.reset()
    assert resets == [1]


@pytest.mark.parametrize("kind", ["controller", "planner"])
@pytest.mark.parametrize("failure", ["malformed", "missing", "contradictory", "dimension"])
def test_adapter_rejects_invalid_wrapped_results(kind, failure):
    solver = QPResult("infeasible") if failure == "contradictory" else QPResult("solved", np.zeros(3))
    n = 2 if failure == "dimension" else 3
    state = RobotState(np.zeros(3), np.zeros(3))
    if kind == "controller":
        result = ControlResult(solver, None if failure == "missing" else VelocityCommand(np.zeros(n), 0, .02))
        wrapped = ReorderedController(SimpleNamespace(reset=lambda: None, step=lambda *args: object() if failure == "malformed" else result), order())
        call = lambda: wrapped.step(state, [], .02)
    else:
        result = PlanResult(solver, None if failure == "missing" else JointPlan(np.zeros((3, n)), np.zeros((2, n)), 0, .1))
        wrapped = ReorderedPlanner(SimpleNamespace(reset=lambda: None, solve=lambda *args: object() if failure == "malformed" else result), order())
        call = lambda: wrapped.solve(state, np.ones(3))
    with pytest.raises(ValueError):
        call()
