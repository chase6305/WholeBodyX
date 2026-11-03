"""Native entry points release the GIL and preserve the process SIGINT handler."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT restoration regression")
@pytest.mark.parametrize("mode", ["native", "mixed", "dynamics"])
def test_independent_native_planners_restore_python_interrupts(tmp_path, mode, request):
    if importlib.util.find_spec("wholebodyx_native") is None:
        if os.environ.get("WBX_REQUIRE_NATIVE") == "1":
            pytest.fail("native extension is required for this test run")
        pytest.skip("optional native extension is not installed")
    import wholebodyx_native
    if not hasattr(wholebodyx_native, "SparseJointMPC"):
        if os.environ.get("WBX_REQUIRE_NATIVE") == "1":
            pytest.fail("native sparse feature is required; rebuild the extension")
        pytest.skip("installed native wheel predates sparse MPC")
    if mode == "dynamics":
        request.getfixturevalue("floating_robot")
    code = '''
import concurrent.futures, os, signal
import numpy as np
from wholebodyx import JointLimits, RobotState, JointMPC, SparseJointMPC
from wholebodyx.native import NativeMPC, NativeSparseMPC
count = [0]
def handler(*args): count[0] += 1
signal.signal(signal.SIGINT, handler)
limits = JointLimits([-1] * 3, [1] * 3, [1] * 3, [4] * 3)
state = RobotState(np.zeros(3), np.zeros(3))
def job(index):
    if index == 4:
        from wholebodyx.dynamics import ContactWBC, DynamicsState, FloatingBaseDynamics
        model = FloatingBaseDynamics(URDF, feet=("base",))
        q = model.pin.neutral(model.model)
        q[2] = .07
        measured = DynamicsState(q, np.zeros(model.nv))
        controller = ContactWBC(model, measured)
        target = model.terms(measured).com
        for k in range(100):
            assert controller.solve(measured, target, .005).solver.success
        return
    choices = (JointMPC, NativeMPC, SparseJointMPC, NativeSparseMPC) if MODE != "native" else (NativeMPC, NativeSparseMPC) * 2
    planner = choices[index](limits)
    for k in range(100):
        result = planner.solve(state, np.full(3, .3 + .01 * np.sin(k)))
        assert result.plan is not None
for batch in range(4):
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        list(pool.map(job, range(5 if MODE == "dynamics" else 4)))
    os.kill(os.getpid(), signal.SIGINT)
    assert count[0] == batch + 1, "native solvers lost Python's SIGINT handler"
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(path).resolve()) for path in
                                                environment.get("PYTHONPATH", "").split(os.pathsep) if path)
    prefix = f"MODE = {mode!r}\nURDF = {str(tmp_path / 'standing.urdf')!r}\n"
    result = subprocess.run([sys.executable, "-c", prefix + code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
