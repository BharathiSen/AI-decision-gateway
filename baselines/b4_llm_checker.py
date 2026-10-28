"""B4: checker with VeriFix's permitted tests, catalog, and stopping rules.

The checker stand-in fills expected outcomes from the development matrix
(the catalog), not from the agent's predicted_tests. A live LLM client is
not required for offline or VM runs.
"""

from gate.catalog import plausible_alternatives
from gate.decide import _selected_test_ids, _symptoms_from_snapshot, decide as gate_decide
from gate.selector import _load_matrix, held_out_ids


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
    alts = plausible_alternatives(symptoms, exclude=[primary])
    selected = _selected_test_ids(primary, alts)
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
        "decision": audit.get("decision"),
        "action": proposal.get("action"),
        "tests_run": list(audit.get("tests_run") or []),
        "reason": audit.get("reason"),
        "checks": audit.get("checks"),
        "cost": audit.get("cost"),
    }
