"""B2: LLM checker with test access (stub: rule-based checker)."""

from gate.checks import check_consistency, check_safety
from control.actions import normalize_action, RISKY_ACTIONS


def decide(proposal, snapshot, test_results):
    action = normalize_action(proposal.get("action"))
    if action not in RISKY_ACTIONS:
        return {"method": "B2", "decision": "allow", "reason": "pass-through"}
    consistency = check_consistency(proposal, test_results)
    safety = check_safety(proposal, snapshot)
    if consistency["passed"] and safety["passed"]:
        return {"method": "B2", "decision": "allow", "checks": [consistency, safety]}
    return {"method": "B2", "decision": "block", "checks": [consistency, safety]}
