"""
Validate agent JSON proposals (Phase 8 schema).
"""

ALLOWED_ACTIONS = {"REROUTE", "REBALANCE", "ESCALATE", "DO_NOTHING", "reroute_traffic", "no_action"}
ALLOWED_TESTS = {
    "T1_e2e_ping",
    "T2_hop_ping",
    "T4_mtu",
    "T5_if_errors",
    "T6_queue_stats",
    "T7_route_check",
    "T8_tcp_vs_ping",
    "T9_oneway_loss",
    "T10_speed",
}


def validate_proposal(data, allowed_actions=None, max_causes=5):
    """
    Returns (ok: bool, errors: list[str]).
    """
    errors = []
    if not isinstance(data, dict):
        return False, ["root must be an object"]

    for key in ("action", "target", "reason"):
        if key not in data:
            errors.append(f"missing {key}")

    allowed = set(allowed_actions or ALLOWED_ACTIONS)
    action = data.get("action")
    if action not in allowed:
        errors.append(f"action {action!r} not in allowed set")

    causes = data.get("causes", [])
    if not isinstance(causes, list):
        errors.append("causes must be an array")
    elif len(causes) > max_causes:
        errors.append(f"at most {max_causes} causes allowed")
    else:
        for i, c in enumerate(causes):
            if not isinstance(c, dict):
                errors.append(f"causes[{i}] must be object")
                continue
            if "cause_id" not in c or "predicted_tests" not in c:
                errors.append(f"causes[{i}] needs cause_id and predicted_tests")
                continue
            preds = c.get("predicted_tests")
            if not isinstance(preds, list):
                errors.append(f"causes[{i}].predicted_tests must be array")
                continue
            for j, p in enumerate(preds):
                if not isinstance(p, dict):
                    errors.append(f"causes[{i}].predicted_tests[{j}] must be object")
                    continue
                tid = p.get("test_id")
                if tid not in ALLOWED_TESTS:
                    errors.append(f"unknown test_id {tid!r}")
                if "expected" not in p:
                    errors.append(f"causes[{i}].predicted_tests[{j}] missing expected")

    return len(errors) == 0, errors


def validate_batch(proposals):
    """Return validation rate for a list of proposals."""
    if not proposals:
        return 0.0, 0, 0
    ok_count = sum(1 for p in proposals if validate_proposal(p)[0])
    return ok_count / len(proposals), ok_count, len(proposals)
