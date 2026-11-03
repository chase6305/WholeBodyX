"""Audit a completed async MPC soak and its sidecar event stream.

Reads events one at a time. The audit checks recorded evidence, not real-time
scheduling or historical causality. It never modifies the input evidence.
"""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wholebodyx import JointPlan


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "non-finite JSON number")
        return number
    return json.loads(text, parse_constant=_constant, parse_float=finite_float, object_pairs_hook=_object)


def _plan(context, key):
    require(key is not None, "missing captured plan")
    data = context["plans"][key]
    return JointPlan(data["positions"], data["velocities"], data["stamp"], data["dt"])


def _adoption(context):
    adoption = context.get("last_adoption")
    if adoption is None:
        return
    stamp = adoption["stamp"]
    require(stamp <= context["stamp"], "adoption is newer than the event")
    require(np.isfinite(adoption["solve_ms"]) and adoption["solve_ms"] >= 0,
            "invalid adoption solve duration")
    require(abs(adoption["lateness_s"] - max(0., stamp - adoption["nominal_start"])) < 1e-10,
            "adoption lateness differs from timestamps")
    # Earlier archives can contain adoption metadata without a captured plan.
    if "plan" in adoption:
        plan = _plan(context, adoption["plan"])
        require(plan.stamp == adoption["plan_stamp"], "adoption plan timestamp differs")
        q, v = plan.sample(stamp)
        require(np.allclose(q, adoption["reference_q"], rtol=0, atol=1e-10) and
                np.allclose(v, adoption["reference_v"], rtol=0, atol=1e-10),
                "adoption reference differs from captured plan")
        error = float(np.max(np.abs(np.asarray(adoption["measured_q"]) - q)))
        require(abs(error - adoption["tracking_error"]) < 1e-10,
                "adoption tracking error differs from measured entry")


def _deadline(event, report):
    value = event.get("execution_deadline")
    required = report.get("diagnostic_schema", 0) >= 3 and event["category"] == "soft_deadline"
    require(not required or isinstance(value, dict), "missing execution deadline metadata")
    if value is None:
        return
    require(event["category"] == "soft_deadline" and value["mode"] == "velocity",
            "deadline metadata belongs to a different fault or mode")
    require(value["stage"] in ("after_read", "after_tasks", "before_write", "after_write"),
            "unknown deadline detection boundary")
    require(isinstance(value["write_completed"], bool) and
            value["write_completed"] == (value["stage"] == "after_write"),
            "deadline write completion differs from detection boundary")
    require(np.isfinite(value["budget_s"]) and 0 < value["budget_s"] <= report["max_cycle_time"],
            "deadline budget is invalid")
    require(isinstance(value["elapsed_finite"], bool), "invalid elapsed finiteness flag")
    if value["elapsed_finite"]:
        require(value["elapsed_s"] is not None and np.isfinite(value["elapsed_s"]) and
                (value["elapsed_s"] < 0 or value["elapsed_s"] >= value["budget_s"]),
                "deadline elapsed time does not reject the budget")
    else:
        require(value["elapsed_s"] is None, "non-finite elapsed time must be null")
    if required:
        context = event["context"]
        attempted = value["stage"] != "after_read"
        require(context["reference_attempted"] is attempted,
                "reference evaluation differs from deadline boundary")
        require((isinstance(context["reference"], dict) and context["reference"]["success"])
                if attempted else context["reference"] is None,
                "deadline boundary has an inconsistent reference result")


def audit(report_path, events_path=None):
    report_path = Path(report_path)
    report_bytes = report_path.read_bytes()
    report = _json(report_bytes)
    require(report["status"] == "passed", "soak did not pass")
    require(report["source_unchanged"] and report["all_workers_reaped"],
            "source integrity or worker cleanup failed")
    require(report["wall_s"] >= report["requested_wall_s"], "soak ended early")
    events_path = Path(events_path) if events_path is not None else report_path.with_suffix(".events.jsonl")
    cases = report["cases"]
    require(bool(cases), "soak has no cases")
    counts = {name: Counter() for name in cases}
    recent = {name: deque(maxlen=10) for name in cases}
    causes = {name: defaultdict(lambda: deque(maxlen=4)) for name in cases}
    uninjected = {name: deque(maxlen=4) for name in cases}
    tracking = Counter()
    last_attempt = dict.fromkeys(cases, -1)
    digest = hashlib.sha256()
    last_wall, event_count = -1., 0
    with events_path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                event = _json(line)
                name, category = event["backend"], event["category"]
                require(name in cases, "unknown event backend")
                require(np.isfinite(event["wall_s"]) and 0 <= event["wall_s"] and
                        last_wall <= event["wall_s"] <= report["wall_s"],
                        "event wall clock is out of order or outside the run")
                require(isinstance(event["attempt"], int) and not isinstance(event["attempt"], bool) and
                        last_attempt[name] < event["attempt"] < cases[name]["attempts"],
                        "event attempt is out of order or outside the run")
                last_wall, last_attempt[name] = event["wall_s"], event["attempt"]
                context = event["context"]
                _deadline(event, report)
                _adoption(context)
                if category == "late_reply":
                    require(context["before"]["active"] == context["after"]["active"] and
                            context["reference"]["success"], "late reply changed the active plan")
                    candidate = context["received"] or context["before"]["future"]
                    limit = report["max_adoption_delay"]
                    rounding = min(8 * math.ulp(context["stamp"]), context["control_dt"] * .25)
                    require(candidate is not None and limit is not None and
                            context["stamp"] - candidate["nominal_start"] > limit + rounding,
                            "late reply is within its adoption budget")
                elif category == "tracking_error":
                    failure = context["failure"]
                    key = failure["future"]["plan"] if "future plan" in event["detail"] else failure["active"]
                    q, _ = _plan(context, key).sample(context["stamp"])
                    error = float(np.max(np.abs(np.asarray(context["q"]) - q)))
                    require(abs(error - context["reference"]["tracking_error"]) < 1e-10,
                            "tracking error differs from captured plan")
                    if event["injected"] is None:
                        tracking[name] += 1
                        uninjected[name].append(dict(attempt=event["attempt"], stamp=context["stamp"], error=error))
                counts[name][category] += 1
                sample = {k: v for k, v in event.items() if k not in ("backend", "wall_s")}
                recent[name].append(sample)
                cause = category + ("/injected" if event["injected"] is not None else "/uninjected")
                causes[name][cause].append(sample)
                event_count += 1
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"event line {number}: {exc}") from exc
    require(digest.hexdigest() == report["events_sha256"], "event checksum differs")
    require(event_count == report["events_written"], "event count differs")
    summaries = {}
    for name, case in cases.items():
        late = counts[name].pop("late_reply", 0)
        require(late == case["late_replies"], f"{name}: late reply count differs")
        require(dict(counts[name]) == case["failures"], f"{name}: failure counts differ")
        require(case["attempts"] == case["accepted"] + sum(counts[name].values()),
                f"{name}: attempt accounting differs")
        require(tracking[name] == case["uninjected_tracking_faults"], f"{name}: tracking count differs")
        require(list(recent[name]) == case["failure_samples"], f"{name}: recent samples differ")
        require({k: list(v) for k, v in causes[name].items()} == case["failure_samples_by_cause"],
                f"{name}: per-cause samples differ")
        summaries[name] = dict(failures=dict(counts[name]), late_replies=late,
                               uninjected_tracking=tracking[name],
                               recent_uninjected_tracking=list(uninjected[name]))
    return dict(status="passed", events=event_count, events_sha256=digest.hexdigest(), cases=summaries,
                report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="Recorded event consistency and captured-plan checks; no scheduling or causality guarantee")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--events", type=Path, help="sidecar path; defaults to the report's .events.jsonl sibling")
    parser.add_argument("--output", type=Path, help="write audit JSON separately from the input evidence")
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() in (
            args.report.resolve(), (args.events or args.report.with_suffix(".events.jsonl")).resolve()):
        parser.error("audit output must not overwrite input evidence")
    try:
        result = audit(args.report, args.events)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"audit failed: {exc}\n")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
