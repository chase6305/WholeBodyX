import numpy as np

from wholebodyx import JointLimits, JointMPC, KinematicWBC, LinearTask, RobotState


def test_native_python_formulations_and_solutions(backend):
    if backend[0] is KinematicWBC:
        return
    from wholebodyx.native import NativeMPC, NativeWBC
    rng = np.random.default_rng(17)
    bounds = JointLimits(np.full(3, -2), np.full(3, 2), np.ones(3), np.full(3, 3))
    for _ in range(8):
        state = RobotState(rng.uniform(-1, 1, 3), rng.uniform(-0.2, 0.2, 3))
        task = LinearTask(rng.normal(size=(4, 3)), rng.normal(size=4), [1, 2, 3, 4])
        cpp, python = NativeWBC(bounds, smoothing=0.2), KinematicWBC(bounds, smoothing=0.2)
        a, b = cpp.formulate(state, [task], 0.02), python.formulate(state, [task], 0.02)
        for field in ("hessian", "gradient", "constraint", "lower", "upper"):
            np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-12)
        np.testing.assert_allclose(cpp.step(state, [task], 0.02).command.velocity,
                                   python.step(state, [task], 0.02).command.velocity, atol=2e-6)
        goal = rng.uniform(-1, 1, 3)
        cpp_m, py_m = NativeMPC(bounds, horizon=5), JointMPC(bounds, horizon=5)
        a, b = cpp_m.formulate(state, goal), py_m.formulate(state, goal)
        for field in ("hessian", "gradient", "constraint", "lower", "upper"):
            np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-12)
        np.testing.assert_allclose(cpp_m.solve(state, goal).plan.positions,
                                   py_m.solve(state, goal).plan.positions, atol=2e-6)


def test_braking_boundary_formulations_and_commands_match(backend):
    if backend[0] is KinematicWBC:
        return
    from wholebodyx.native import NativeWBC

    bounds = JointLimits([-1] * 3, [1] * 3, [1, 1, .4], [1, 1, 2])
    state = RobotState([-.96, .96, .99], [-.3, .3, 0])
    tasks = [LinearTask([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [0, .6, -.2])]
    cpp, python = NativeWBC(bounds), KinematicWBC(bounds)
    for _ in range(30):
        a, b = cpp.formulate(state, tasks, .1), python.formulate(state, tasks, .1)
        np.testing.assert_allclose(a.lower, b.lower, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a.upper, b.upper, atol=1e-12, rtol=0)
        actual, expected = cpp.step(state, tasks, .1), python.step(state, tasks, .1)
        assert actual.command is not None and expected.command is not None
        np.testing.assert_allclose(actual.command.velocity, expected.command.velocity, atol=2e-6, rtol=0)
        state = RobotState(state.q + .1 * expected.command.velocity, expected.command.velocity, state.stamp + .1)
