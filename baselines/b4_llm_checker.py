"""B4: catalog-filled checker with VeriFix tests, catalog, and stopping rules.

This is not an LLM. It fills expected outcomes from the development
matrix, not from the agent's predicted_tests, then calls gate.decide.
A live LLM client is not used.
"""

from gate.decide import _candidate_set, _symptoms_from_snapshot, decide as gate_decide
from gate.selector import covering_separating_tests, _load_matrix, held_out_ids


def _catalog_predictions(cause_id, test_ids):
    hidden = held_out_ids()
    if cause_id in hidden:
        return []
    matrix = _load_matrix()
    preds = []
    for tid in test_ids:
        expected = next(
            (row["outcome"] for row in matrix if row["fault_id"] == cause_id and row["test_id"] == tid),
            None,
        )
        if expected is not None:
            preds.append({"test_id": tid, "expected": expected})
    return preds


def decide(proposal, snapshot, test_results, net=None, run_missing_tests=None):
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
    symptoms = _symptoms_from_snapshot(snapshot)
    hidden = held_out_ids()
    candidates = _candidate_set(primary, [c["cause_id"] for c in causes], symptoms, hidden)
    # Same model-independent covering set the gate freezes before predictions.
    selected = covering_separating_tests(candidates, matrix=_load_matrix())
    filled = dict(proposal)
    filled["causes"] = [{
        "cause_id": primary,
        "predicted_tests": _catalog_predictions(primary, selected),
    }]
    run_missing = net is not None if run_missing_tests is None else run_missing_tests
    try:
        audit = gate_decide(
            net,
            filled,
            snapshot,
            test_results,
            mode="full",
            run_missing_tests=run_missing,
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
    }
