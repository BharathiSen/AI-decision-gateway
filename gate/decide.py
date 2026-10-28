"""
Independent gate: allow / block / escalate risky AI proposals.
"""

from control.actions import normalize_action, RISKY_ACTIONS
from gate import catalog, checks, selector
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


def decide(
    net,
    proposal,
    network_snapshot,
    test_results=None,
    already_ran=None,
    mode="full",
    run_missing_tests=True,
    noise_id="N0",
    learned_grader=True,
):
    """
    mode: 'full' | 'pilot_v0' (consistency + safety only)
    noise_id selects the traffic-level thresholds used to grade results.
    Returns audit dict with decision in {allow, block, escalate}.
    """
    supplied = list(test_results) if isinstance(test_results, list) else list(test_results or [])
    proposal = proposal if isinstance(proposal, dict) else {}

    if mode not in MODES:
        return _audit("escalate", proposal, [], {}, "invalid mode", supplied=supplied)

    frozen_ok, frozen_errors, frozen = load_frozen_context(noise_id)
    if not frozen_ok:
        return _audit(
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
        return _audit("block", proposal, [], {}, "invalid action", supplied=supplied)

    prop_errors = validate_proposal_inputs(proposal, costs=frozen["costs"])
    if prop_errors:
        return _audit(
            "block",
            proposal,
            supplied,
            {},
            "invalid proposal: " + "; ".join(prop_errors),
            supplied=supplied,
        )

    result_errors = validate_test_results(test_results, already_ran)
    if result_errors:
        return _audit(
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

    if action not in RISKY_ACTIONS:
        return _audit("allow", proposal, test_results, {}, "non-risky pass-through", supplied=supplied)

    snap_errors = validate_snapshot(network_snapshot)
    if snap_errors:
        return _audit(
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
        return _audit("block", proposal, test_results, {}, "incomplete hypotheses", supplied=supplied)

    primary = ai_causes[0]
    symptoms = _symptoms_from_snapshot(network_snapshot)
    alts = stages["candidate_generation"](symptoms, exclude=[primary])
    candidates = [primary] + [a for a in alts if a not in ai_causes]

    ruled_out = []
    tests_run = list(test_results)
    ran_ids = {r["test_id"] for r in tests_run if r.get("test_id")} | set(already_ran)
    selected_ids = _selected_test_ids(primary, alts)

    if mode == "full":
        for alt in alts:
            if not selector.needs_different_fixes(primary, alt, fixes=frozen["fixes"]):
                continue
            tid = stages["test_selection"](primary, alt, already_ran=ran_ids)
            if tid is None:
                continue
            if run_missing_tests and tid not in ran_ids:
                tests_run.append(oam.run_test(net, tid))
                ran_ids.add(tid)
            tr = next((r for r in tests_run if r.get("test_id") == tid), None)
            if tr is None:
                continue
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
            if not primary_pred or not primary_pred.get("expected"):
                continue
            exp_alt = _outcome_for_test(tr, alt, matrix_rows, thresholds)
            if exp_alt is None:
                continue
            exp_primary = primary_pred["expected"]
            if obs != exp_alt and grade_test(tid, obs, exp_primary, use_model=learned_grader) >= 0.5:
                ruled_out.append({"alternative": alt, "test_id": tid, "observed": obs})

    consistency = stages["consistency_checking"](
        proposal, tests_run, primary_cause_id=primary, thresholds=thresholds
    )
    discrimination = stages["discrimination_checking"](ruled_out)
    safety = stages["safety_checking"](proposal, network_snapshot)

    check_bundle = {
        "consistency": consistency,
        "discrimination": discrimination,
        "safety": safety,
        "candidates": candidates,
    }

    if not safety.get("passed"):
        return _audit("block", proposal, tests_run, check_bundle, "safety failed", supplied=supplied, selected_ids=selected_ids)
    if not consistency.get("passed"):
        return _audit("block", proposal, tests_run, check_bundle, "consistency failed", supplied=supplied, selected_ids=selected_ids)

    if mode == "full":
        if not discrimination.get("passed"):
            if not any(selector.separating_tests(primary, a, matrix=matrix_rows) for a in alts):
                return _audit("escalate", proposal, tests_run, check_bundle, "inseparable", supplied=supplied, selected_ids=selected_ids)
            return _audit("escalate", proposal, tests_run, check_bundle, "no discrimination", supplied=supplied, selected_ids=selected_ids)
    else:
        check_bundle["discrimination"] = {"passed": True, "skipped": "pilot_v0"}

    return _audit("allow", proposal, tests_run, check_bundle, "all checks passed", supplied=supplied, selected_ids=selected_ids)


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


def _audit(decision, proposal, tests_run, checks_out, reason, supplied=None, selected_ids=None):
    supplied = list(supplied if supplied is not None else tests_run)
    by_id = {}
    for result in tests_run:
        if isinstance(result, dict):
            by_id.setdefault(result.get("test_id"), result)
    selected = [by_id.get(tid) or {"test_id": tid} for tid in (selected_ids or [])]
    return {
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
