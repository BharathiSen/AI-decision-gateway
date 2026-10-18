"""B2: LLM checker with test access (stub: rule-based checker)."""

from gate.checks import check_consistency, check_safety
from control.actions import normalize_action, RISKY_ACTIONS


def decide(proposal, snapshot, test_results):
    action = normalize_action(proposal.get("action"))
    if action not in RISKY_ACTIONS:
        return {"method": "B2", "decision": "allow", "action": action.value, "reason": "pass-through"}
    consistency = check_consistency(proposal, test_results)
    safety = check_safety(proposal, snapshot)
    decision = "allow" if consistency["passed"] and safety["passed"] else "block"
    return {"method": "B2", "decision": decision, "action": action.value, "checks": [consistency, safety]}
