"""Floating-base contracts exercised without external robot assets."""

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from wholebodyx.dynamics import ContactWBC, DynamicsState, DynamicsTerms, FloatingBaseDynamics


@pytest.fixture
def robot_state(tmp_path):
    path = tmp_path / "standing.urdf"
    path.write_text('''<robot name="standing">
      <link name="base"><inertial><origin xyz="0 0 0.3"/>
        <mass value="10"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
      </inertial></link>
      <link name="arm"><inertial><origin xyz="0 0 0.1"/>
        <mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
      </inertial></link>
      <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
        <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="2" effort="10"/>
      </joint></robot>''', encoding="utf-8")
    robot = FloatingBaseDynamics(path, feet=("base",))
    q = robot.pin.neutral(robot.model)
    q[2] = .07
    return robot, DynamicsState(q, np.zeros(robot.nv))


def test_named_terms_are_independent_snapshots_with_tuple_compatibility(robot_state):
    robot, state = robot_state
    terms = robot.terms(state)
    assert isinstance(terms, DynamicsTerms) and len(terms) == 9
    assert terms.mass is terms[0] and terms.com is terms[5]
    original = terms.mass.copy()
    moved = DynamicsState(robot.pin.integrate(robot.model, state.q, np.full(robot.nv, .001)), state.v)
    robot.terms(moved)
    np.testing.assert_array_equal(terms.mass, original)


def test_formulation_does_not_advance_solver_and_owns_its_qp(robot_state):
    robot, state = robot_state
    controller = ContactWBC(robot, state)
    target = robot.terms(state).com
    first = controller.formulate(state, target)
    original = first.problem.constraint.copy()
    assert controller.previous is None and controller.solver_setups == 0
    first.problem.constraint.setflags(write=True)
    first.problem.constraint[:] = 99
    second = controller.formulate(state, target)
    np.testing.assert_array_equal(second.problem.constraint, original)
    result = controller.solve(state, target)
    assert result["violation"] <= 1e-5
    previous, workspace = controller.previous.copy(), controller._solver
    controller.formulate(state, target + [0, .001, 0])
    np.testing.assert_array_equal(controller.previous, previous)
    assert controller._solver is workspace and controller.solver_setups == 1
    acceleration, wrench = robot.forward(state, result["torque"], result["contact_rhs"], result["terms"])
    np.testing.assert_allclose(acceleration, result["acceleration"], atol=1e-6)
    assert controller.contact_violation(wrench) <= 1e-5


def test_contact_reserve_refreshes_template_and_reset_keeps_configuration(robot_state):
    robot, state = robot_state
    controller = ContactWBC(robot, state)
    target = robot.terms(state).com
    first = controller.formulate(state, target)
    controller.contact_reserve = .5
    second = controller.formulate(state, target)
    assert not np.array_equal(first.problem.constraint, second.problem.constraint)
    expected = second.problem.constraint.copy()
    controller.solve(state, target)
    nominal = controller.nominal.copy()
    anchors = [pose.homogeneous.copy() for pose in controller.anchors]
    controller.reset()
    assert controller.previous is None and controller._solver is None
    assert controller.solver_setups == controller.solver_recoveries == 0
    assert controller.contact_reserve == .5
    np.testing.assert_array_equal(controller.nominal, nominal)
    for actual, original in zip(controller.anchors, anchors):
        np.testing.assert_array_equal(actual.homogeneous, original)
    np.testing.assert_array_equal(controller.formulate(state, target).problem.constraint, expected)
    assert controller.solve(state, target)["violation"] <= 1e-5


def test_construction_does_not_dispatch_a_plugin_reset_override(robot_state):
    class Plugin(ContactWBC):
        reset_called = False
        def reset(self):
            self.reset_called = True
            super().reset()

    robot, state = robot_state
    plugin = Plugin(robot, state)
    assert not plugin.reset_called
    plugin.reset()
    assert plugin.reset_called


@pytest.mark.parametrize("stamp", [-1., np.nan, np.inf])
def test_dynamics_rejects_invalid_clock(robot_state, stamp):
    robot, initial = robot_state
    with pytest.raises(ValueError, match="state"):
        robot.terms(DynamicsState(initial.q, initial.v, stamp))


def test_dynamics_rejects_unrepresentable_integration_interval(robot_state):
    robot, initial = robot_state
    state = DynamicsState(initial.q, initial.v, 1e20)
    with pytest.raises(ValueError, match="interval"):
        robot.integrate(state, np.zeros(robot.nv), .005)


@pytest.mark.parametrize("single_stamp,single_dt", [(True, False), (False, True), (True, True)])
def test_dynamics_integration_uses_double_precision_clock(robot_state, single_stamp, single_dt):
    robot, initial = robot_state
    stamp = np.float32(1e8) if single_stamp else 1e8
    dt = np.float32(.005) if single_dt else .005
    velocity = np.linspace(-.1, .1, robot.nv)
    acceleration = np.linspace(.01, .02, robot.nv)
    state = DynamicsState(initial.q, velocity, stamp)
    result = robot.integrate(state, acceleration, dt)
    assert type(result.stamp) is float
    assert result.stamp == float(stamp) + float(dt)
    np.testing.assert_allclose(result.v, velocity + float(dt) * acceleration, rtol=0, atol=1e-15)
    expected_q = robot.pin.integrate(robot.model, state.q, float(dt) * (velocity + .5 * float(dt) * acceleration))
    np.testing.assert_allclose(result.q, expected_q, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(state.v, velocity)


@pytest.mark.parametrize("failure", ["velocity_overflow", "nonfinite_pose"])
def test_dynamics_rejects_nonfinite_integrated_state(robot_state, failure):
    robot, initial = robot_state
    velocity, acceleration = np.zeros(robot.nv), np.zeros(robot.nv)
    if failure == "velocity_overflow":
        velocity[0] = acceleration[0] = np.finfo(float).max
    else:
        velocity[3] = 1e308
    state = DynamicsState(initial.q, velocity)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="integration"):
            robot.integrate(state, acceleration, .02)
    np.testing.assert_array_equal(state.q, initial.q)
    assert np.isfinite(state.v).all()


def test_large_finite_translation_is_not_arbitrarily_clipped(robot_state):
    robot, initial = robot_state
    velocity = np.zeros(robot.nv)
    velocity[0] = 1e308
    result = robot.integrate(DynamicsState(initial.q, velocity), np.zeros(robot.nv), .005)
    assert result.q[0] == pytest.approx(5e305)
    robot.validate_state(result)


def test_forward_dynamics_rejects_overflowed_solution(robot_state):
    robot, state = robot_state
    terms = robot.terms(state)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="forward dynamics"):
            robot.forward(state, np.full(robot.na, np.finfo(float).max),
                          np.zeros(6 * len(robot.contacts)), terms)
    assert np.isfinite(state.q).all() and np.isfinite(state.v).all() and state.stamp == 0
