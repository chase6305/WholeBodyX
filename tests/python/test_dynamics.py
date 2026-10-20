"""Physics checks for the experimental Pinocchio torque-control example."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
from wholebodyx.dynamics import DynamicsState


@pytest.fixture
def demo_module():
    path = Path(__file__).resolve().parents[2] / "examples/python/h1_dynamics_wbc.py"
    spec = importlib.util.spec_from_file_location("h1_dynamics_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.DEFAULT_URDF.is_file():
        pytest.skip("external H1 assets are absent")
    return module


def test_mass_bias_and_contact_jacobian(demo_module):
    demo = demo_module.DynamicsDemo()
    r, state = demo.robot, demo.state
    terms = r.terms(state)
    mass, bias, jac = terms[:3]
    assert (r.model.nq, r.model.nv) == (26, 25)
    assert np.linalg.eigvalsh(mass).min() > 0
    acceleration = np.linspace(-.2, .3, r.nv)
    np.testing.assert_allclose(r.pin.rnea(r.model, r.data, state.q, state.v, acceleration), mass @ acceleration + bias, atol=1e-9)
    velocity = np.linspace(-.1, .1, r.nv)
    shifted = DynamicsState(r.pin.integrate(r.model, state.q, 1e-7 * velocity), velocity)
    moved = r.terms(shifted)[4]
    finite_difference = np.concatenate([r.pin.log6(a.inverse() * b).vector / 1e-7 for a, b in zip(terms[4], moved)])
    np.testing.assert_allclose(jac @ velocity, finite_difference, atol=1e-7)
    # With no actuator or contact forces the base falls under gravity.
    freefall = np.linalg.solve(mass, -bias)
    assert freefall[2] == pytest.approx(-9.81, abs=1e-8)


def test_static_balance_and_torque_driven_plant(demo_module):
    demo = demo_module.DynamicsDemo(motion="stand")
    r = demo.robot
    command = demo.controller.solve(demo.state, demo.initial_com)
    acc, wrench = r.forward(demo.state, command["torque"], command["contact_rhs"], command["terms"])
    np.testing.assert_allclose(acc, command["acceleration"], atol=1e-6)
    weight = r.pin.computeTotalMass(r.model) * 9.81
    assert wrench[2] + wrench[8] == pytest.approx(weight, abs=.1)
    assert command["dynamics_residual"] < 1e-5
    assert command["contact_residual"] < 1e-6
    assert np.all(np.abs(command["torque"]) <= r.model.effortLimit[6:] + 1e-6)
    for force in command["wrench"].reshape(-1, 6):
        assert force[2] > 0
        assert abs(force[0]) + abs(force[1]) <= .6 * force[2] + 1e-6
        assert abs(force[3]) <= .035 * force[2] + 1e-6
        assert abs(force[4]) <= .10 * force[2] + 1e-6
    perturbed = command["torque"].copy()
    perturbed[-1] += 1.
    changed, _ = r.forward(demo.state, perturbed, command["contact_rhs"], command["terms"])
    assert np.linalg.norm(changed - acc) > .1


def test_double_support_shift_and_reset(demo_module):
    demo = demo_module.DynamicsDemo()
    initial = demo.state.q.copy()
    rows = [demo.step() for _ in range(3200)]
    assert demo.state.stamp == pytest.approx(16.)
    assert np.sqrt(np.mean([r["com_error_m"] ** 2 for r in rows])) < .005
    assert max(r["forward_contact_violation"] for r in rows) < 1e-5
    assert np.max(np.abs(np.diff([r["com_y_m"] for r in rows]))) < .001
    assert np.ptp([r["com_y_m"] for r in rows]) > .05
    assert min(min(r["left_fz_n"], r["right_fz_n"]) for r in rows) > 0
    assert np.ptp([r["left_fz_n"] for r in rows]) > 20
    assert demo.max_contact_error < 1e-4
    assert demo.max_dynamics_residual < 1e-5
    assert demo.max_violation < 1e-5
    assert max(r["forward_acceleration_error"] for r in rows) < 1e-5
    assert np.linalg.norm(demo.state.q[3:7]) == pytest.approx(1., abs=1e-10)
    demo.reset()
    np.testing.assert_allclose(demo.state.q, initial)
    assert demo.state.stamp == 0
    with pytest.raises(ValueError):
        demo.controller.solve(demo.state, [np.nan, 0, 0])


def test_com_feedforward_derivatives(demo_module):
    demo = demo_module.DynamicsDemo()
    for stamp in (0., .7, 2., 4., 7.9, 8., 8.1):
        velocity, acceleration = demo.target_derivatives(stamp)
        h = 1e-4
        np.testing.assert_allclose(velocity, (demo.target(stamp + h) - demo.target(stamp - h)) / (2 * h), atol=1e-8)
        np.testing.assert_allclose(acceleration, (demo.target(stamp + h) - 2 * demo.target(stamp) + demo.target(stamp - h)) / h**2, atol=1e-7)
    with pytest.raises(ValueError, match="feedforward"):
        demo.controller.solve(demo.state, demo.initial_com, target_velocity=[0, np.nan, 0])


@pytest.mark.parametrize("component,value", [(2, -1.), (0, 100.), (3, 10.), (4, 20.), (5, 5.)])
def test_invalid_forward_contact_rejected_before_integration(demo_module, monkeypatch, component, value):
    demo = demo_module.DynamicsDemo()
    before = demo.state.q.copy()
    force = np.array([0., 0., 100., 0., 0., 0.] * 2)
    force[component] = value
    monkeypatch.setattr(demo.robot, "forward", lambda *args, **kwargs: (np.zeros(25), force))
    with pytest.raises(RuntimeError, match="forward contact wrench"):
        demo.step()
    np.testing.assert_array_equal(before, demo.state.q)
    assert demo.state.stamp == 0


@pytest.mark.parametrize("autoplay,loop,steps", [(True, True, 12), (True, False, 4), (False, True, 0)])
def test_viewer_continues_across_cycle_without_reset(demo_module, monkeypatch, tmp_path, autoplay, loop, steps):
    from types import SimpleNamespace as NS
    from contextlib import nullcontext
    import sys
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="mock"/>')
    counts = dict(steps=0, sleeps=0, stops=0)
    demo = NS(state=NS(stamp=7.98), duration=8., target=lambda t: np.zeros(3), controller=NS(anchors=[]),
              robot=NS(terms=lambda state: [None] * 5 + [np.zeros(3)]))

    def step():
        counts["steps"] += 1
        demo.state.stamp += .005
        return dict(time=demo.state.stamp, left_fz_n=290., right_fz_n=290., max_torque_nm=20.,
                    contact_position_error_m=0., com_error_m=0., wbc_ms=1., applied_push_y_n=0., solver_recoveries=0.)

    def reset():
        pytest.fail("loop boundary must not reset the dynamic state")

    def sleep(seconds):
        counts["sleeps"] += 1
        if counts["sleeps"] == 3:
            raise KeyboardInterrupt

    def stop():
        counts["stops"] += 1

    demo.step, demo.reset = step, reset
    gui = NS(add_markdown=lambda *a: NS(content=""),
             add_checkbox=lambda *a, **k: NS(value=k["initial_value"]),
             add_slider=lambda *a, **k: NS(value=k["initial_value"]),
             add_button=lambda *a: NS(on_click=lambda cb: None))
    scene = NS(set_up_direction=lambda *a: None, add_grid=lambda *a, **k: None,
               add_icosphere=lambda *a, **k: NS(position=None))
    server = NS(gui=gui, scene=scene, atomic=nullcontext, on_client_connect=lambda cb: None, stop=stop)
    monkeypatch.setitem(sys.modules, "viser", NS(ViserServer=lambda **k: server))
    monkeypatch.setitem(sys.modules, "trimesh", NS())
    monkeypatch.setattr(demo_module, "DynamicsDemo", lambda *a: demo)
    monkeypatch.setattr(demo_module.time, "sleep", sleep)
    demo_module.view(urdf, "shift", .035, "127.0.0.1", 8081, autoplay, loop)
    assert counts["steps"] == steps
    assert counts["stops"] == 1


def test_reused_solver_matches_fresh_workspace(demo_module):
    from wholebodyx.dynamics import ContactWBC
    demo = demo_module.DynamicsDemo()
    fresh = ContactWBC(demo.robot, demo.state)
    for _ in range(80):
        target = demo.target(demo.state.stamp)
        velocity, acceleration = demo.target_derivatives(demo.state.stamp)
        kwargs = dict(target_velocity=velocity, target_acceleration=acceleration)
        cached = demo.controller.solve(demo.state, target, **kwargs)
        fresh._solver = None
        recreated = fresh.solve(demo.state, target, **kwargs)
        np.testing.assert_allclose(cached["torque"], recreated["torque"], atol=1e-6)
        np.testing.assert_allclose(cached["wrench"], recreated["wrench"], atol=1e-6)
        demo.state = demo.robot.integrate(demo.state, cached["acceleration"], demo.dt)
    assert demo.controller.solver_setups == 1
    assert fresh.solver_setups == 80


def test_solver_failure_discards_workspace_and_recovers(demo_module, monkeypatch):
    from types import SimpleNamespace
    demo = demo_module.DynamicsDemo()
    demo.controller.solve(demo.state, demo.initial_com)
    monkeypatch.setattr(demo.controller._solver, "solve", lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status_val=3, status="primal infeasible"), x=None))
    with pytest.raises(RuntimeError, match="primal infeasible"):
        demo.controller.solve(demo.state, demo.initial_com)
    assert demo.controller.previous is None and demo.controller._solver is None
    result = demo.controller.solve(demo.state, demo.initial_com)
    assert result["violation"] < 1e-5
    assert demo.controller.solver_setups == 2


@pytest.mark.parametrize("bad", [np.zeros(24), np.zeros((25, 1)), np.full(25, np.nan)])
def test_integrator_rejects_invalid_acceleration(demo_module, bad):
    demo = demo_module.DynamicsDemo()
    with pytest.raises(ValueError, match="acceleration"):
        demo.robot.integrate(demo.state, bad, demo.dt)


def test_state_quaternion_and_contact_input_validation(demo_module):
    demo = demo_module.DynamicsDemo()
    state = DynamicsState(demo.state.q.copy(), demo.state.v.copy())
    state.q[3:7] *= 2
    with pytest.raises(ValueError, match="quaternion"):
        demo.robot.terms(state)
    with pytest.raises(ValueError, match="contact acceleration"):
        demo.robot.forward(demo.state, np.zeros(19), np.zeros((12, 1)), demo.robot.terms(demo.state))


@pytest.mark.parametrize("mode", ["wbc", "mpc"])
def test_side_push_is_applied_to_plant_and_recovers(demo_module, mode):
    demo = demo_module.DynamicsDemo(motion="stand", controller=mode, push_force=5.)
    rows = [demo.step() for _ in range(1600)]
    assert sum(row["applied_push_y_n"] * demo.dt for row in rows) == pytest.approx(1.)
    disturbed = [row for row in rows if row["applied_push_y_n"]]
    assert max(row["forward_acceleration_error"] for row in disturbed) > .1
    assert max(row["forward_dynamics_residual"] for row in rows) < 1e-5
    assert max(row["forward_contact_violation"] for row in rows) < 1e-5
    assert rows[-1]["com_error_m"] < .001
    assert np.linalg.norm(demo.state.v) < .002
    assert demo.max_contact_error < 1e-4
    assert demo.controller.contact_reserve == .5
    demo.trigger_push()
    assert demo.step()["applied_push_y_n"] == 3.
    demo.reset()
    assert demo.step()["applied_push_y_n"] == 0.


def test_generalized_external_force_validation(demo_module):
    demo = demo_module.DynamicsDemo()
    terms = demo.robot.terms(demo.state)
    with pytest.raises(ValueError, match="external force"):
        demo.robot.forward(demo.state, np.zeros(19), np.zeros(12), terms, external_force=[0.])
    for value in (-1, .9, np.nan):
        with pytest.raises(ValueError, match="reserve"):
            demo.controller.contact_reserve = value


@pytest.mark.parametrize("mode", ["wbc", "mpc"])
def test_dynamic_cli_summary_and_csv(demo_module, mode, tmp_path):
    import csv
    path = tmp_path / "trace.csv"
    report = demo_module.run(steps=25, csv_path=path, controller=mode)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["steps"] == 25
    assert report["dynamic_mpc"] == (mode == "mpc")
    assert report["mpc_replans"] == (2 if mode == "mpc" else 0)
    assert report["applied_push_impulse_ns"] == 0
    assert report["max_forward_dynamics_residual"] < 1e-5
    assert float(rows[-1]["time"]) == pytest.approx(.125)



def test_numerical_failure_rebuilds_once_and_validates(demo_module, monkeypatch):
    from types import SimpleNamespace
    demo = demo_module.DynamicsDemo()
    demo.controller.solve(demo.state, demo.initial_com)
    monkeypatch.setattr(demo.controller._solver, "solve", lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status_val=7, status="maximum iterations reached"), x=None))
    result = demo.controller.solve(demo.state, demo.initial_com)
    assert demo.controller.solver_recoveries == 1
    assert demo.controller.solver_setups == 2
    assert result["violation"] < 1e-5
    assert result["dynamics_residual"] < 1e-5


def test_numerical_retry_is_bounded(demo_module, monkeypatch):
    from types import SimpleNamespace
    import wholebodyx.dynamics as dynamics
    attempts = []
    def failed_solve(**kwargs):
        attempts.append(True)
        return SimpleNamespace(info=SimpleNamespace(status_val=7, status="maximum iterations reached"), x=None)
    monkeypatch.setattr(dynamics.osqp, "OSQP", lambda: SimpleNamespace(
        setup=lambda **kw: None, update=lambda **kw: None, update_settings=lambda **kw: None,
        warm_start=lambda **kw: None, solve=failed_solve))
    demo = demo_module.DynamicsDemo()
    with pytest.raises(RuntimeError, match="maximum iterations"):
        demo.step()
    assert len(attempts) == 2
    assert demo.state.stamp == 0
    assert demo.controller._solver is None


def test_mpc_replan_preserves_reference_acceleration_and_measured_feedback(demo_module):
    demo = demo_module.DynamicsDemo(controller="mpc", amplitude=.05)
    for _ in range(20):
        demo.step()
    previous = demo.planner.last_plan
    expected_acceleration = previous.sample(demo.state.stamp)[2]
    terms = demo.robot.terms(demo.state)
    measured_position, measured_velocity = terms[5][:2].copy(), terms[6][:2].copy()
    row = demo.step()
    initial = demo.planner.last_plan.states[0]
    np.testing.assert_allclose(initial[0], measured_position, atol=1e-12)
    np.testing.assert_allclose(initial[1], measured_velocity, atol=1e-12)
    np.testing.assert_allclose(initial[2], expected_acceleration, atol=1e-12)
    assert row["mpc_continuity_fallbacks"] == 0


@pytest.mark.parametrize("measured_feasible", [True, False])
def test_mpc_continuity_fallback_is_feasibility_checked_and_bounded(demo_module, measured_feasible):
    from types import SimpleNamespace
    demo = demo_module.DynamicsDemo(controller="mpc")
    demo.planner.last_plan = SimpleNamespace(sample=lambda stamp: (None, None, np.full(2, 100.)))
    if not measured_feasible:
        demo.com_acceleration[:] = 100.
    terms = demo.robot.terms(demo.state)
    targets = np.tile(demo.initial_com[:2], (demo.planner.horizon, 1))
    if measured_feasible:
        plan = demo._replan_com(terms, targets, np.zeros_like(targets))
        np.testing.assert_allclose(plan.states[0, 2], 0.)
        assert demo.planner.replans == 1
    else:
        with pytest.raises(RuntimeError, match="CoM MPC failed"):
            demo._replan_com(terms, targets, np.zeros_like(targets))
        assert demo.planner.last_plan is None
        assert demo.planner.replans == 0
    assert demo.state.stamp == 0.
    assert demo.mpc_continuity_fallbacks == 1
    demo.reset()
    assert demo.mpc_continuity_fallbacks == 0


def test_mpc_shift_push_tracking_and_support_force_smoothness(demo_module, monkeypatch):
    demo = demo_module.DynamicsDemo(controller="mpc", amplitude=.05, push_force=3.)
    iterations = []
    solve = demo.controller.solve
    def measured_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        iterations.append(result["iterations"])
        return result
    monkeypatch.setattr(demo.controller, "solve", measured_solve)
    rows = [demo.step() for _ in range(3200)]
    errors = np.array([r["com_error_m"] for r in rows])
    forces = np.array([[r["left_fz_n"], r["right_fz_n"]] for r in rows])
    # Original measured-acceleration reseeding: 11.62 mm RMSE and 17.24 N
    # largest force step. Bounds protect the observed continuity improvement.
    assert np.sqrt(np.mean(errors**2)) < .0105
    assert np.max(np.abs(np.diff(forces, axis=0))) < 14.
    assert max(r["forward_contact_violation"] for r in rows) <= 1e-5
    assert max(r["forward_dynamics_residual"] for r in rows) <= 1e-5
    assert demo.mpc_continuity_fallbacks == 0
    # Fixed H1 trajectory: protect iteration tails without a wall-clock limit.
    assert max(iterations) <= 8000
    assert demo.controller.solver_setups == 1
    assert demo.controller.solver_recoveries == 0


def test_physical_constraint_error_refines_once(demo_module, monkeypatch):
    from types import SimpleNamespace
    demo = demo_module.DynamicsDemo()
    demo.step()
    solver = demo.controller._solver
    monkeypatch.setattr(solver, 'solve', lambda **kwargs: SimpleNamespace(
        info=SimpleNamespace(status_val=1, status='solved', iter=1), x=np.zeros(56)))
    row = demo.step()
    assert demo.controller.solver_recoveries == 1
    assert demo.controller.solver_setups == 2
    assert demo.controller._solver.settings.eps_abs == 1e-9
    assert row['constraint_violation'] <= 1e-5
    assert row['forward_contact_violation'] <= 1e-5


def test_physical_constraint_error_never_relaxes_acceptance(demo_module, monkeypatch):
    from types import SimpleNamespace
    import wholebodyx.dynamics as dynamics
    tolerances = []
    def setup(**kwargs):
        tolerances.append((kwargs['eps_abs'], kwargs['eps_rel']))
    monkeypatch.setattr(dynamics.osqp, 'OSQP', lambda: SimpleNamespace(
        setup=setup, update=lambda **kw: None, update_settings=lambda **kw: None,
        warm_start=lambda **kw: None, solve=lambda **kw: SimpleNamespace(
            info=SimpleNamespace(status_val=1, status='solved', iter=1), x=np.zeros(56))))
    demo = demo_module.DynamicsDemo()
    with pytest.raises(RuntimeError, match='constraint violation'):
        demo.step()
    assert tolerances == [(1e-8, 1e-8), (1e-9, 1e-9)]
    assert demo.controller.solver_recoveries == 1
    assert demo.controller._solver is None
    assert demo.controller.previous is None
    assert demo.state.stamp == 0.
