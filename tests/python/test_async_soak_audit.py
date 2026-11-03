"""Reject corrupt soak evidence even when its checksum has been recomputed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_async_runtime import _diagnostic_runtime, state


def auditor():
    path = Path(__file__).resolve().parents[2] / "scripts/check_async_soak.py"
    spec = importlib.util.spec_from_file_location("wholebodyx_async_soak_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path):
    manager, worker = _diagnostic_runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.2), [.3], .02)
    manager.update(state(.22, .04), [.3], .02)
    sample = dict(attempt=1, category="tracking_error", detail="measured state diverged from the active plan",
                  injected=None, context=manager.diagnostic_snapshot(), worker={})
    event = dict(backend="sparse", wall_s=.1, **sample)
    case = dict(attempts=2, accepted=1, late_replies=0, uninjected_tracking_faults=1,
                failures={"tracking_error": 1}, failure_samples=[sample],
                failure_samples_by_cause={"tracking_error/uninjected": [sample]})
    report = dict(status="passed", source_unchanged=True, all_workers_reaped=True,
                  wall_s=1., requested_wall_s=1., max_adoption_delay=None,
                  events_written=1, cases={"sparse": case})
    return tmp_path / "soak.json", report, [event]


def write_evidence(evidence):
    path, report, events = evidence
    payload = "".join(json.dumps(event, allow_nan=False) + "\n" for event in events).encode()
    report["events_sha256"] = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".events.jsonl").write_bytes(payload)
    path.write_text(json.dumps(report))
    return path


def test_audit_recomputes_tracking_and_adoption_from_captured_plans(evidence):
    result = auditor().audit(write_evidence(evidence))
    assert result["status"] == "passed" and result["events"] == 1
    assert result["cases"]["sparse"]["uninjected_tracking"] == 1


@pytest.mark.parametrize("corruption,message", [
    ("backend", "unknown event backend"), ("attempt", "attempt is out of order"),
    ("clock", "wall clock"), ("error", "tracking error differs"),
    ("adoption", "adoption reference differs"), ("count", "event count differs"),
    ("retention", "recent samples differ"), ("cause", "per-cause samples differ"),
    ("accounting", "attempt accounting differs"), ("early", "ended early"),
    ("cleanup", "worker cleanup failed"),
])
def test_audit_rejects_inconsistent_evidence_with_valid_checksum(evidence, corruption, message):
    _, report, events = evidence
    event = events[0]
    if corruption == "backend": event["backend"] = "other"
    elif corruption == "attempt": event["attempt"] = 2
    elif corruption == "clock": event["wall_s"] = 2.
    elif corruption == "error": event["context"]["reference"]["tracking_error"] += .01
    elif corruption == "adoption": event["context"]["last_adoption"]["reference_v"] = [.2]
    elif corruption == "count": report["events_written"] = 2
    elif corruption == "retention": report["cases"]["sparse"]["failure_samples"] = []
    elif corruption == "cause": report["cases"]["sparse"]["failure_samples_by_cause"] = {}
    elif corruption == "accounting": report["cases"]["sparse"]["accepted"] = 0
    elif corruption == "early": report["requested_wall_s"] = 2.
    elif corruption == "cleanup": report["all_workers_reaped"] = False
    with pytest.raises(ValueError, match=message):
        auditor().audit(write_evidence(evidence))


def test_audit_rejects_truncated_event_stream(evidence):
    path = write_evidence(evidence)
    path.with_suffix(".events.jsonl").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        auditor().audit(path)


def test_audit_binds_result_to_exact_report_and_does_not_modify_inputs(evidence):
    path = write_evidence(evidence)
    sidecar = path.with_suffix(".events.jsonl")
    before = path.read_bytes(), sidecar.read_bytes()
    result = auditor().audit(path)
    assert result["report_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result["events_sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert (path.read_bytes(), sidecar.read_bytes()) == before


def test_cli_rejects_output_that_overwrites_evidence(evidence):
    import subprocess
    import sys
    path = write_evidence(evidence)
    before = path.read_bytes()
    command = Path(__file__).resolve().parents[2] / "scripts/check_async_soak.py"
    outcome = subprocess.run([sys.executable, str(command), str(path), "--output", str(path)],
                             capture_output=True, text=True)
    assert outcome.returncode != 0 and "must not overwrite" in outcome.stderr
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", ['{"value": NaN}', '{"value": 1e999}', '{"value": 1, "value": 2}'])
def test_audit_rejects_ambiguous_json(payload):
    with pytest.raises(ValueError):
        auditor()._json(payload)


def test_audit_verifies_late_reply_without_mislabeling_it_as_a_fault(evidence):
    _, report, events = evidence
    manager, worker = _diagnostic_runtime(max_adoption_delay=0.)
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.22), [.3], .02)
    event = events[0]
    event.update(category="late_reply", detail="candidate exceeded adoption deadline",
                 context=manager.diagnostic_snapshot())
    sample = {k: v for k, v in event.items() if k not in ("backend", "wall_s")}
    report["max_adoption_delay"] = 0.
    report["cases"]["sparse"].update(accepted=2, late_replies=1, uninjected_tracking_faults=0,
                                      failures={}, failure_samples=[sample],
                                      failure_samples_by_cause={"late_reply/uninjected": [sample]})
    result = auditor().audit(write_evidence(evidence))
    assert result["cases"]["sparse"]["late_replies"] == 1
    assert result["cases"]["sparse"]["failures"] == {}
    report["max_adoption_delay"] = .1
    with pytest.raises(ValueError, match="within its adoption budget"):
        auditor().audit(write_evidence(evidence))

    # The same floating-point grace as the runtime must not be audited as late.
    from math import ulp
    report["max_adoption_delay"] = 0.
    candidate = event["context"]["received"]
    candidate["nominal_start"] = event["context"]["stamp"] - ulp(event["context"]["stamp"])
    with pytest.raises(ValueError, match="within its adoption budget"):
        auditor().audit(write_evidence(evidence))


@pytest.mark.parametrize("corruption", [None, "missing", "stage", "write", "budget", "elapsed", "reference"])
def test_schema3_deadline_evidence_checks_boundary_and_fault_budget(evidence, corruption):
    _, report, events = evidence
    event = events[0]
    manager, worker = _diagnostic_runtime()
    manager.update(state(), [.3], .02)
    worker.finish()
    manager.update(state(.2), [.3], .02)
    event["context"] = manager.diagnostic_snapshot()
    event.update(category="soft_deadline", detail="execution deadline",
                 execution_deadline=dict(mode="velocity", stage="before_write", elapsed_s=.006,
                                         elapsed_finite=True, budget_s=.005, write_completed=False))
    event["context"]["reference_attempted"] = True
    sample = {k: v for k, v in event.items() if k not in ("backend", "wall_s")}
    report.update(diagnostic_schema=3, max_cycle_time=.005)
    report["cases"]["sparse"].update(uninjected_tracking_faults=0, failures={"soft_deadline": 1},
                                      failure_samples=[sample],
                                      failure_samples_by_cause={"soft_deadline/uninjected": [sample]})
    if corruption == "missing": event["execution_deadline"] = None
    elif corruption == "stage": event["execution_deadline"]["stage"] = "unknown"
    elif corruption == "write": event["execution_deadline"]["write_completed"] = True
    elif corruption == "budget": event["execution_deadline"]["budget_s"] = .01
    elif corruption == "elapsed": event["execution_deadline"]["elapsed_s"] = .001
    elif corruption == "reference": event["context"]["reference_attempted"] = False
    if corruption is None:
        assert auditor().audit(write_evidence(evidence))["status"] == "passed"
    else:
        with pytest.raises(ValueError):
            auditor().audit(write_evidence(evidence))
