"""
Independent gate: allow / block / escalate risky AI proposals.
"""

from control.actions import normalize_action, RISKY_ACTIONS, Action
from gate import catalog, checks, selector
from gate.grader import grade_test
from bench import tests as oam


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


def _outcome_for_test(test_result, fault_id, matrix_rows):
    tid = test_result["test_id"]
    for row in matrix_rows:
        if row["fault_id"] == fault_id and row["test_id"] == tid:
            return row["outcome"]
    return checks._summarize_observation(test_result)


def decide(
    net,
    proposal,
    network_snapshot,
    test_results=None,
    already_ran=None,
    mode="full",
    run_missing_tests=True,
):
    """
    mode: 'full' | 'pilot_v0' (consistency + safety only)
    Returns audit dict with decision in {allow, block, escalate}.
    """
    already_ran = list(already_ran or [])
    test_results = list(test_results or [])
    matrix_rows = selector._load_matrix()

    try:
        action = normalize_action(proposal.get("action"))
    except ValueError:
        return _audit("block", proposal, [], {}, "invalid action")

    if action not in RISKY_ACTIONS:
        return _audit("allow", proposal, test_results, {}, "non-risky pass-through")

    ai_causes = [c.get("cause_id") for c in proposal.get("causes", []) if c.get("cause_id")]
    primary = ai_causes[0] if ai_causes else "AI_PRIMARY"
    symptoms = _symptoms_from_snapshot(network_snapshot)
    alts = catalog.plausible_alternatives(symptoms, exclude=[primary])
    candidates = [primary] + [a for a in alts if a not in ai_causes]

    ruled_out = []
    tests_run = list(test_results)
    ran_ids = {r["test_id"] for r in tests_run} | set(already_ran)

    if mode == "full":
        for alt in alts:
            if not selector.needs_different_fixes(primary, alt):
                continue
            tid = selector.pick_cheapest_separating_test(primary, alt, already_ran=ran_ids)
            if tid is None:
                continue
            if run_missing_tests and tid not in ran_ids:
                tests_run.append(oam.run_test(net, tid))
                ran_ids.add(tid)
            tr = next(r for r in tests_run if r["test_id"] == tid)
            obs = checks._summarize_observation(tr)
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
            exp_alt = _outcome_for_test(tr, alt, matrix_rows)
            exp_primary = primary_pred["expected"] if primary_pred else obs
            if obs != exp_alt and grade_test(tid, obs, exp_primary) >= 0.5:
                ruled_out.append({"alternative": alt, "test_id": tid, "observed": obs})

    consistency = checks.check_consistency(proposal, tests_run, primary_cause_id=primary)
    discrimination = checks.check_discrimination(ruled_out)
    safety = checks.check_safety(proposal, network_snapshot)

    check_bundle = {
        "consistency": consistency,
        "discrimination": discrimination,
        "safety": safety,
        "candidates": candidates,
    }

    if not safety.get("passed"):
        return _audit("block", proposal, tests_run, check_bundle, "safety failed")
    if not consistency.get("passed"):
        return _audit("block", proposal, tests_run, check_bundle, "consistency failed")

    if mode == "full":
        if not discrimination.get("passed"):
            if not any(selector.separating_tests(primary, a) for a in alts):
                return _audit("escalate", proposal, tests_run, check_bundle, "inseparable")
            return _audit("escalate", proposal, tests_run, check_bundle, "no discrimination")
    else:
        check_bundle["discrimination"] = {"passed": True, "skipped": "pilot_v0"}

    return _audit("allow", proposal, tests_run, check_bundle, "all checks passed")


def _audit(decision, proposal, tests_run, checks_out, reason):
    return {
        "decision": decision,
        "reason": reason,
        "proposal_action": proposal.get("action"),
        "tests_run": [t.get("test_id") for t in tests_run],
        "checks": checks_out,
    }
