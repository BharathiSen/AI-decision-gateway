"""
Independent gate: allow / block / escalate risky AI proposals.
"""

import json
from datetime import datetime, timezone

from control.actions import normalize_action, RISKY_ACTIONS
from gate import checks, selector
from gate.grader import grade_test
from gate.inputs import (
    MODES,
    load_frozen_context,
    pipeline_interfaces,
    validate_proposal_inputs,
    validate_snapshot,
    validate_test_results,
)
from bench import tests as oam
from bench.outcomes import load_thresholds


def _symptoms_from_snapshot(snapshot):
    metrics = (snapshot or {}).get("metrics", {})
    links = (snapshot or {}).get("links", [])
    high_loss = metrics.get("packet_loss_percent", 0) > 5
    high_latency = (metrics.get("latency_ms") or 0) > 100
    link_down = any(not l.get("up", True) for l in links)
    congestion = any((l.get("latency_ms") or 0) > 80 for l in links)
    return {
        "high_loss": high_loss,
        "high_latency": high_latency,
        "link_down": link_down,
        "congestion": congestion,
    }


def _outcome_for_test(test_result, fault_id, matrix_rows, thresholds=None):
    tid = test_result["test_id"]
    for row in matrix_rows:
        if row["fault_id"] == fault_id and row["test_id"] == tid:
            return row["outcome"]
    return None


def evidence_fingerprint(snapshot):
    """Stable snapshot identity for reuse: routes, link up-state, and qdiscs."""
    snapshot = snapshot or {}
    links = [
        {"name": row.get("name"), "up": row.get("up")}
        for row in (snapshot.get("links") or [])
        if isinstance(row, dict)
    ]
    routes = [
        {"node": row.get("node"), "prefix": row.get("prefix"), "route": row.get("route")}
        for row in (snapshot.get("routes") or [])
        if isinstance(row, dict)
    ]
    qdiscs = snapshot.get("qdiscs") if isinstance(snapshot.get("qdiscs"), dict) else {}
    payload = {
        "links": sorted(links, key=lambda row: str(row.get("name") or "")),
        "routes": sorted(routes, key=lambda row: (str(row.get("node") or ""), str(row.get("prefix") or ""))),
        "qdiscs": {key: qdiscs[key] for key in sorted(qdiscs)},
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _parse_iso(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def result_reusable(result, snapshot, fingerprint, max_age_seconds):
    """True when this OAM row is still valid for the current snapshot."""
    if not isinstance(result, dict) or not result.get("test_id"):
        return False
    marked = result.get("state_fingerprint")
    if marked is not None and marked != fingerprint:
        return False
    observed = _parse_iso(result.get("observed_at"))
    snap_ts = _parse_iso((snapshot or {}).get("timestamp"))
    if observed is not None and snap_ts is not None and observed != snap_ts:
        age = abs((snap_ts - observed).total_seconds())
        if age > float(max_age_seconds):
            return False
    return True


def _test_cost(test_id, costs):
    spec = (costs or {}).get(test_id) or {}
    value = spec.get("time_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _candidate_set(primary, ai_causes, symptoms, hidden):
    """AI hypotheses plus catalog alternatives. Held-out ids never enter the set."""
    stages = pipeline_interfaces()
    alts = [
        fid
        for fid in stages["candidate_generation"](symptoms, exclude=[primary])
        if fid not in hidden
    ]
    extra_ai = [cid for cid in ai_causes[1:] if cid not in hidden and cid != primary and cid not in alts]
    seen = []
    for cid in [primary] + extra_ai + alts:
        if cid not in seen:
            seen.append(cid)
    return seen


def decide(
    net,
    proposal,
    network_snapshot,
    test_results=None,
    already_ran=None,
    mode="full",
    run_missing_tests=True,
    noise_id="N0",
    learned_grader=None,
    budget_seconds=None,
):
    """
    mode: 'full' | 'pilot_v0' (consistency + safety only)
    noise_id selects the traffic-level thresholds used to grade results.
    Returns audit dict with decision in {allow, block, escalate}.
    """
    supplied = list(test_results) if isinstance(test_results, list) else list(test_results or [])
    proposal = proposal if isinstance(proposal, dict) else {}
    snapshot_fp = evidence_fingerprint(network_snapshot)

    def audit(decision, proposal_arg, tests_run, checks_out, reason, supplied=None, selected_ids=None, log=None):
        record = _audit(
            decision,
            proposal_arg,
            tests_run,
            checks_out,
            reason,
            supplied=supplied,
            selected_ids=selected_ids,
            log=log,
        )
        record["snapshot_fingerprint"] = snapshot_fp
        return record

    if mode not in MODES:
        return audit("escalate", proposal, [], {}, "invalid mode", supplied=supplied)

    frozen_ok, frozen_errors, frozen = load_frozen_context(noise_id)
    if not frozen_ok:
        return audit(
            "escalate",
            proposal,
            supplied,
            {},
            "frozen inputs: " + "; ".join(frozen_errors),
            supplied=supplied,
        )

    try:
        action = normalize_action(proposal.get("action"))
    except (ValueError, TypeError):
        return audit("block", proposal, [], {}, "invalid action", supplied=supplied)

    prop_errors = validate_proposal_inputs(proposal, costs=frozen["costs"])
    if prop_errors:
        return audit(
            "block",
            proposal,
            supplied,
            {},
            "invalid proposal: " + "; ".join(prop_errors),
            supplied=supplied,
        )

    result_errors = validate_test_results(test_results, already_ran)
    if result_errors:
        return audit(
            "escalate",
            proposal,
            supplied,
            {},
            "incomplete tests: " + "; ".join(result_errors),
            supplied=supplied,
        )

    already_ran = list(already_ran or [])
    test_results = list(test_results or [])
    matrix_rows = frozen["matrix"]
    thresholds = frozen["thresholds"] or load_thresholds(noise_id)
    cfg = frozen.get("gate") or {}
    if learned_grader is None:
        learned_grader = bool(cfg.get("learned_grader"))
    cap = float(budget_seconds) if budget_seconds is not None else float(cfg.get("budget_seconds") or 8.0)
    min_conf = float(cfg.get("min_grade_confidence") or 0.5)
    max_age = float(cfg.get("reuse_max_age_seconds") or 120.0)
    fingerprint = evidence_fingerprint(network_snapshot)

    if action not in RISKY_ACTIONS:
        return audit("allow", proposal, test_results, {}, "non-risky pass-through", supplied=supplied)

    snap_errors = validate_snapshot(network_snapshot)
    if snap_errors:
        return audit(
            "escalate",
            proposal,
            test_results,
            {},
            "incomplete snapshot: " + "; ".join(snap_errors),
            supplied=supplied,
        )

    stages = pipeline_interfaces()
    ai_causes = [c.get("cause_id") for c in proposal.get("causes", []) if c.get("cause_id")]
    if not ai_causes:
        return audit("block", proposal, test_results, {}, "incomplete hypotheses", supplied=supplied)

    primary = ai_causes[0]
    symptoms = _symptoms_from_snapshot(network_snapshot)
    candidates = _candidate_set(primary, ai_causes, symptoms, frozen.get("held_out") or set())
    relevant = [
        alt
        for alt in candidates
        if alt != primary and selector.needs_different_fixes(primary, alt, fixes=frozen["fixes"])
    ]
    selected_ids = _selected_test_ids(primary, relevant)
    rules = []
    log = {
        "candidates": candidates,
        "relevant_alternatives": relevant,
        "selected_tests": list(selected_ids),
        "reused_tests": [],
        "ran_tests": [],
        "eliminated": [],
        "comparisons": [],
        "rules": rules,
        "budget": {"budget_seconds": cap, "tests_used_seconds": 0.0, "within_budget": True},
        "learned_grader": bool(learned_grader),
        "frozen": bool(cfg.get("frozen")),
        "state_fingerprint": fingerprint,
    }

    safety = stages["safety_checking"](proposal, network_snapshot)
    if not safety.get("passed"):
        rules.append("safety")
        return audit(
            "block",
            proposal,
            test_results,
            {"safety": safety, "candidates": candidates},
            "safety failed",
            supplied=supplied,
            selected_ids=selected_ids,
            log=log,
        )

    if primary in (frozen.get("held_out") or set()) or primary not in (frozen.get("fixes") or {}):
        rules.append("outside catalog")
        return audit(
            "escalate",
            proposal,
            test_results,
            {"safety": safety, "candidates": candidates},
            "outside catalog",
            supplied=supplied,
            selected_ids=selected_ids,
            log=log,
        )

    primary_fixes = set((frozen["fixes"].get(primary) or {}).get("correct") or [])
    if action.value not in primary_fixes:
        rules.append("action not in correct set")
        return audit(
            "block",
            proposal,
            test_results,
            {"safety": safety, "candidates": candidates},
            "proposal contradicted",
            supplied=supplied,
            selected_ids=selected_ids,
            log=log,
        )

    reusable = [
        row for row in test_results if result_reusable(row, network_snapshot, fingerprint, max_age)
    ]
    tests_run = list(reusable)
    ran_ids = {row["test_id"] for row in tests_run if row.get("test_id")} | {
        tid for tid in already_ran if isinstance(tid, str)
    }
    used_seconds = 0.0
    ruled_out = []
    unaffordable = []
    inseparable = []

    if mode == "full":
        for alt in relevant:
            tid = stages["test_selection"](primary, alt, already_ran=set())
            if tid is None:
                inseparable.append(alt)
                continue
            if tid not in selected_ids:
                selected_ids.append(tid)
            tr = next((row for row in tests_run if row.get("test_id") == tid), None)
            if tr is None and tid in ran_ids and not run_missing_tests:
                continue
            if tr is None:
                cost = _test_cost(tid, frozen["costs"])
                if used_seconds + cost > cap + 1e-9:
                    unaffordable.append({"alternative": alt, "test_id": tid})
                    log["budget"]["within_budget"] = False
                    continue
                if not run_missing_tests or net is None:
                    unaffordable.append({"alternative": alt, "test_id": tid})
                    continue
                try:
                    tr = oam.run_test(net, tid)
                except Exception as exc:
                    rules.append("test error")
                    return audit(
                        "escalate",
                        proposal,
                        tests_run,
                        {"safety": safety, "candidates": candidates},
                        f"test error: {type(exc).__name__}: {exc}",
                        supplied=supplied,
                        selected_ids=selected_ids,
                        log=log,
                    )
                if isinstance(tr, dict):
                    tr = dict(tr)
                    tr["state_fingerprint"] = fingerprint
                    if (network_snapshot or {}).get("timestamp") and "observed_at" not in tr:
                        tr["observed_at"] = network_snapshot.get("timestamp")
                tests_run.append(tr)
                ran_ids.add(tid)
                used_seconds += cost
                log["ran_tests"].append(tid)
            else:
                if tid not in log["reused_tests"]:
                    log["reused_tests"].append(tid)
            obs = checks._summarize_observation(tr, thresholds)
            primary_pred = next(
                (
                    p
                    for c in proposal.get("causes", [])
                    if c.get("cause_id") == primary
                    for p in c.get("predicted_tests", [])
                    if p.get("test_id") == tid
                ),
                None,
            )
            exp_alt = _outcome_for_test(tr, alt, matrix_rows, thresholds)
            comparison = {
                "test_id": tid,
                "alternative": alt,
                "observed": obs,
                "expected_alternative": exp_alt,
                "expected_primary": (primary_pred or {}).get("expected"),
            }
            log["comparisons"].append(comparison)
            if not primary_pred or not primary_pred.get("expected") or exp_alt is None:
                continue
            exp_primary = primary_pred["expected"]
            confidence = grade_test(tid, obs, exp_primary, use_model=learned_grader)
            comparison["confidence"] = confidence
            if obs != exp_alt and confidence >= min_conf:
                ruled_out.append({"alternative": alt, "test_id": tid, "observed": obs, "confidence": confidence})
                log["eliminated"].append(alt)
                rules.append(f"eliminate {alt} via {tid}")

    log["budget"]["tests_used_seconds"] = used_seconds
    log["budget"]["within_budget"] = used_seconds <= cap + 1e-9
    log["selected_tests"] = list(selected_ids)

    consistency = stages["consistency_checking"](
        proposal, tests_run, primary_cause_id=primary, thresholds=thresholds
    )
    remaining = [alt for alt in relevant if alt not in set(log["eliminated"])]
    discrimination = stages["discrimination_checking"](ruled_out)
    discrimination["relevant"] = relevant
    discrimination["remaining"] = remaining
    if not relevant:
        discrimination["passed"] = True
        discrimination["vacuous"] = True
    else:
        discrimination["passed"] = not remaining

    check_bundle = {
        "consistency": consistency,
        "discrimination": discrimination,
        "safety": safety,
        "candidates": candidates,
    }

    if not consistency.get("passed"):
        rules.append("consistency")
        return audit("block", proposal, tests_run, check_bundle, "consistency failed", supplied=supplied, selected_ids=selected_ids, log=log)

    if mode == "full":
        if remaining:
            if inseparable and not ruled_out:
                rules.append("inseparable")
                return audit("escalate", proposal, tests_run, check_bundle, "inseparable", supplied=supplied, selected_ids=selected_ids, log=log)
            if unaffordable:
                rules.append("no affordable discriminating test")
                return audit("escalate", proposal, tests_run, check_bundle, "no affordable discriminating test", supplied=supplied, selected_ids=selected_ids, log=log)
            if inseparable:
                rules.append("inseparable")
                return audit("escalate", proposal, tests_run, check_bundle, "inseparable", supplied=supplied, selected_ids=selected_ids, log=log)
            rules.append("no discrimination")
            return audit("escalate", proposal, tests_run, check_bundle, "no discrimination", supplied=supplied, selected_ids=selected_ids, log=log)
        rules.append("allow")
        return audit("allow", proposal, tests_run, check_bundle, "all checks passed", supplied=supplied, selected_ids=selected_ids, log=log)

    check_bundle["discrimination"] = {"passed": True, "skipped": "pilot_v0"}
    rules.append("pilot_v0")
    return audit("allow", proposal, tests_run, check_bundle, "all checks passed", supplied=supplied, selected_ids=selected_ids, log=log)


def _selected_test_ids(primary, alternatives):
    """Cheapest separator for each alternative. Does not run those tests."""
    chosen = []
    for alt in alternatives:
        if not selector.needs_different_fixes(primary, alt):
            continue
        tid = selector.pick_cheapest_separating_test(primary, alt)
        if tid and tid not in chosen:
            chosen.append(tid)
    return chosen


def _audit(decision, proposal, tests_run, checks_out, reason, supplied=None, selected_ids=None, log=None):
    supplied = list(supplied if supplied is not None else tests_run)
    by_id = {}
    for result in tests_run:
        if isinstance(result, dict):
            by_id.setdefault(result.get("test_id"), result)
    selected = [by_id.get(tid) or {"test_id": tid} for tid in (selected_ids or [])]
    record = {
        "decision": decision,
        "reason": reason,
        "proposal_action": proposal.get("action") if isinstance(proposal, dict) else None,
        "tests_run": [t.get("test_id") for t in tests_run if isinstance(t, dict)],
        "checks": checks_out,
        "cost": {
            **oam.compare_costs(supplied, selected),
            "tests_executed": oam.summarize_costs(tests_run),
        },
    }
    if log is not None:
        record["log"] = log
    return record
