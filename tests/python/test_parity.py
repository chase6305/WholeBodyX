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
