"""The basic API keeps optional geometry and native loaders out of startup."""

import os
import subprocess
import sys


def test_optional_root_exports_load_on_demand_and_keep_class_identity():
    code = '''
import sys
import osqp
linear_algebra_loaded = "scipy.sparse.linalg" in sys.modules
import wholebodyx as wbx
assert "scipy.spatial" not in sys.modules
assert "wholebodyx.centroidal" not in sys.modules
assert "wholebodyx.sparse_mpc" not in sys.modules
assert "wholebodyx.process_mpc" not in sys.modules
assert "wholebodyx.async_runtime" not in sys.modules
assert "wholebodyx_native" not in sys.modules
assert "pinocchio" not in sys.modules
assert ("scipy.sparse.linalg" in sys.modules) == linear_algebra_loaded
assert set(wbx.__all__) <= set(dir(wbx))
registry = wbx.BackendRegistry()
assert "sparse" in registry.names("planner")
assert "wholebodyx.sparse_mpc" not in sys.modules
planner = wbx.JointMPC(wbx.JointLimits([-1.], [1.], [1.]), horizon=3)
assert ("scipy.sparse.linalg" in sys.modules) == linear_algebra_loaded
from wholebodyx import ContactSpec, SparseJointMPC, CoMReferenceManager
from wholebodyx.contact import ContactSpec as DirectContact
from wholebodyx.sparse_mpc import SparseJointMPC as DirectMPC
assert ContactSpec is DirectContact is wbx.ContactSpec
assert SparseJointMPC is DirectMPC is wbx.SparseJointMPC
assert CoMReferenceManager is wbx.CoMReferenceManager
assert "pinocchio" not in sys.modules
assert "wholebodyx_native" not in sys.modules
try:
    wbx.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown exports must raise AttributeError")
namespace = {}
exec("from wholebodyx import *", namespace)
assert set(wbx.__all__) <= set(namespace)
'''
    environment = os.environ.copy()
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
