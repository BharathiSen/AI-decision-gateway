"""B4: catalog-filled checker with VeriFix tests, catalog, and stopping rules.

This is not an LLM. It fills expected outcomes from the development
matrix, not from the agent's predicted_tests, then calls gate.decide.
A live LLM client is not used.

Parity with VeriFix: same candidate set, covering separators, thresholds,
budget, evidence filtering, and decision semantics. Only the source of
expected outcomes differs (catalog matrix vs AI predictions).
"""

from agent.validate import MAX_EXPECTED_LEN
from bench.contracts import parse_t2_prediction
from gate.decide import _candidate_set, _symptoms_from_snapshot, decide as gate_decide
from gate.inputs import load_frozen_context
from gate.selector import covering_separating_tests, held_out_ids


def _compact_expected(test_id, expected):
    """Fit catalog outcomes into the proposal expected-length contract."""
    if not isinstance(expected, str) or len(expected) <= MAX_EXPECTED_LEN:
        return expected
    if test_id == "T2_hop_ping":
        parsed = parse_t2_prediction(expected)
        if not parsed.get("ok"):
            return expected[:MAX_EXPECTED_LEN]
        # Partial prediction: only non-ok hops (still match observed grades).
        by_path = {"primary": [], "backup": []}
        for (path, hop), outcome in (parsed.get("pairs") or {}).items():
            if outcome != "ok" and path in by_path:
                by_path[path].append(f"{hop}={outcome}")
        parts = [f"{path}:" + ",".join(hops) for path, hops in by_path.items() if hops]
        compact = ";".join(parts) if parts else expected[:MAX_EXPECTED_LEN]
        return compact if len(compact) <= MAX_EXPECTED_LEN else compact[:MAX_EXPECTED_LEN]
    return expected[:MAX_EXPECTED_LEN]


def _catalog_predictions(cause_id, test_ids, matrix):
    hidden = held_out_ids()
    if cause_id in hidden:
        return []
    preds = []
    for tid in test_ids:
        expected = next(
            (row["outcome"] for row in matrix if row["fault_id"] == cause_id and row["test_id"] == tid),
            None,
        )
        if expected is not None:
            preds.append({"test_id": tid, "expected": _compact_expected(tid, expected)})
    return preds


def decide(proposal, snapshot, test_results, net=None, run_missing_tests=None, noise_id="N0", budget_seconds=None):
    proposal = proposal or {}
    causes = [c for c in (proposal.get("causes") or []) if c.get("cause_id")]
    if not causes:
        return {
            "method": "B4",
            "decision": "escalate",
            "action": proposal.get("action"),
            "tests_run": [row.get("test_id") for row in (test_results or []) if row.get("test_id")],
            "reason": "no cause",
        }

    primary = causes[0]["cause_id"]
    ok, errs, frozen = load_frozen_context(noise_id or "N0")
    if not ok:
        return {
            "method": "B4",
            "decision": "escalate",
            "action": proposal.get("action"),
            "tests_run": [row.get("test_id") for row in (test_results or []) if row.get("test_id")],
            "reason": "frozen inputs: " + "; ".join(errs),
        }

    symptoms = _symptoms_from_snapshot(snapshot)
    hidden = frozen.get("held_out") or set()
    ai_causes = [c["cause_id"] for c in causes]
    candidates = _candidate_set(primary, ai_causes, symptoms, hidden)
    selected = covering_separating_tests(
        candidates, fixes=frozen.get("fixes"), matrix=frozen.get("matrix") or []
    )
    # Keep secondary causes; only replace primary predictions with catalog outcomes.
    catalog_preds = _catalog_predictions(primary, selected, frozen.get("matrix") or [])
    filled = dict(proposal)
    filled["causes"] = []
    for cause in causes:
        row = dict(cause)
        if row.get("cause_id") == primary:
            row["predicted_tests"] = catalog_preds
        filled["causes"].append(row)

    run_missing = net is not None if run_missing_tests is None else run_missing_tests
    try:
        audit = gate_decide(
            net,
            filled,
            snapshot,
            test_results,
            mode="full",
            run_missing_tests=run_missing,
            noise_id=noise_id or "N0",
            budget_seconds=budget_seconds,
        )
    except Exception as exc:
        return {
            "method": "B4",
            "decision": "escalate",
            "action": proposal.get("action"),
            "tests_run": [row.get("test_id") for row in (test_results or []) if row.get("test_id")],
            "reason": f"checker error: {type(exc).__name__}: {exc}",
        }
    return {
        "method": "B4",
        "implementation": "catalog_filled_gate",
        "decision": audit.get("decision"),
        "action": proposal.get("action"),
        "tests_run": list(audit.get("tests_run") or []),
        "reason": audit.get("reason"),
        "checks": audit.get("checks"),
        "cost": audit.get("cost"),
        "log": audit.get("log"),
        "selected_tests": list((audit.get("log") or {}).get("selected_tests") or []),
    }
