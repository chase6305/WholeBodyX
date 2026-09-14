import importlib.util
import os

import pytest

from wholebodyx import JointMPC, KinematicWBC


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
