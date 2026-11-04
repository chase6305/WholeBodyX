import importlib.util
import os
from pathlib import Path
import sys

import pytest

from wholebodyx import JointMPC, KinematicWBC

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


@pytest.fixture
def floating_robot(tmp_path):
    """Small inertial URDF for contact/torque contracts, without external assets."""
    pytest.importorskip("pinocchio")
    import numpy as np
    from wholebodyx.dynamics import DynamicsState, FloatingBaseDynamics

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


@pytest.fixture(params=["python", "native"])
def backend(request):
    if request.param == "python":
        return KinematicWBC, JointMPC
    if importlib.util.find_spec("wholebodyx_native") is None:
        if os.environ.get("WBX_REQUIRE_NATIVE") == "1":
            pytest.fail("native extension is required for this test run")
        pytest.skip("optional native extension is not installed")
    from wholebodyx.native import NativeMPC, NativeWBC
    return NativeWBC, NativeMPC
