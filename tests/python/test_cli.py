"""Installed package commands do not depend on source-tree examples or a GUI."""

import csv
import json
import subprocess
import sys

import pytest

from wholebodyx import cli, diagnostics


def invoke(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "wholebodyx", *args], cwd=tmp_path,
                          capture_output=True, text=True)


def test_list_and_headless_run_outside_repository(tmp_path):
    listing = invoke(tmp_path, "list")
    assert listing.returncode == 0, listing.stderr
    assert json.loads(listing.stdout)["disturbance_recovery"]["steps"] == 400
    path = tmp_path / "trace.csv"
    result = invoke(tmp_path, "run", "tracking", "--mode", "WBC", "--steps", "5", "--csv", str(path))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["steps"] == 5 and summary["replans"] == 0 and summary["mode"] == "WBC"
    with path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 5


@pytest.mark.parametrize("args", [("run", "tracking", "--steps", "0"),
                                 ("run", "coordination", "--mode", "MPC + WBC"),
                                 ("run", "tracking", "--backend", "typo")])
def test_cli_rejects_invalid_runs_without_success_json(tmp_path, args):
    result = invoke(tmp_path, *args)
    assert result.returncode != 0 and not result.stdout.strip()
    assert result.stderr and "Traceback" not in result.stderr


def test_doctor_optional_requirements_control_exit_status(monkeypatch, capsys):
    checks = {key: {"available": key == "python"} for key in ("python", "native", "urdf", "viz")}
    monkeypatch.setattr(diagnostics, "diagnose", lambda: {"checks": checks})
    assert cli.main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["checks"]["native"]["available"] is False
    assert cli.main(["doctor", "--require", "native"]) == 1
    capsys.readouterr()
    checks["python"]["available"] = False
    assert cli.main(["doctor"]) == 1


def test_core_run_does_not_import_optional_packages(tmp_path):
    code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"wholebodyx_native", "pinocchio", "viser", "trimesh"}:
            raise ImportError("blocked optional module: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from wholebodyx.cli import main
raise SystemExit(main(["run", "cartesian_circle", "--steps", "2"]))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["steps"] == 2


def test_doctor_retains_binary_load_error(monkeypatch):
    original = diagnostics.importlib.import_module

    def broken(name):
        if name == "pinocchio":
            raise ImportError("liburdfdom_model.so.4: cannot open shared object")
        return original(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", broken)
    report = diagnostics.diagnose()
    assert report["checks"]["python"]["available"]
    assert not report["checks"]["urdf"]["available"]
    assert "liburdfdom_model.so.4" in report["checks"]["urdf"]["detail"]


def test_planner_only_backend_diagnostics_use_a_real_controller():
    report = diagnostics.diagnose()
    assert report["checks"]["sparse"]["available"]
    assert report["checks"]["sparse"]["check"] == "sparse MPC + python WBC solve"
