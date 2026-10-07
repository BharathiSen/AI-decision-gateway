"""Gate checks: consistency, discrimination, safety."""

import yaml
from pathlib import Path

from control.actions import normalize_action, RISKY_ACTIONS

_POLICY = Path(__file__).resolve().parent.parent / "policy" / "policy.yaml"


def _load_policy():
    with _POLICY.open() as f:
        return yaml.safe_load(f)


def check_consistency(proposal, test_results, primary_cause_id=None):
    """
    Compare observed tests to predictions on the AI's primary cause
    (first cause if not specified).
    """
    causes = proposal.get("causes") or []
    if not causes:
        return {"passed": False, "reason": "no causes to check"}
    cid = primary_cause_id or causes[0].get("cause_id")
    cause = next((c for c in causes if c.get("cause_id") == cid), causes[0])
    preds = {p["test_id"]: p["expected"] for p in cause.get("predicted_tests", [])}
    by_id = {r["test_id"]: r for r in test_results}

    mismatches = []
    for tid, expected in preds.items():
        obs = by_id.get(tid)
        if obs is None:
            continue
        obs_str = _summarize_observation(obs)
        if not _prediction_matches(expected, obs_str):
            mismatches.append({"test_id": tid, "expected": expected, "observed": obs_str})

    return {
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "checked_cause": cause.get("cause_id"),
    }


def _summarize_observation(obs):
    if obs.get("test_id") == "T1_e2e_ping":
        return f"loss={obs.get('packet_loss_percent')},lat={obs.get('latency_ms')}"
    if obs.get("test_id") == "T7_route_check":
        return str(obs.get("active_path"))
    if obs.get("test_id") == "T10_speed":
        return str(obs.get("mbit_per_sec"))
    return "ok"


def _prediction_matches(expected, observed):
    exp = (expected or "").lower()
    obs = (observed or "").lower()
    if exp in obs or obs in exp:
        return True
    # numeric loose match
    for token in exp.replace(",", " ").split():
        if token and token in obs:
            return True
    return exp == "ok" and obs == "ok"


def check_discrimination(ruled_out_alternatives):
    return {
        "passed": len(ruled_out_alternatives) > 0,
        "ruled_out": ruled_out_alternatives,
    }


def check_safety(proposal, network_snapshot):
    policy = _load_policy()
    try:
        action = normalize_action(proposal.get("action"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    if action not in RISKY_ACTIONS:
        return {"passed": True, "reason": "non-risky action"}

    path_nodes = proposal.get("proposed_path") or []
    allowed_primary = policy.get("primary_path", [])
    allowed_backup = policy.get("backup_path", [])
    if path_nodes == allowed_primary or path_nodes == allowed_backup:
        return {"passed": True, "path": path_nodes}

    if "r2" in path_nodes or "primary" in str(proposal.get("reason", "")).lower():
        return {"passed": True, "path": "primary"}
    if "r3" in path_nodes:
        return {"passed": True, "path": "backup"}

    metrics = (network_snapshot or {}).get("metrics", {})
    if action.name == "REROUTE" and not metrics.get("connectivity"):
        return {"passed": True, "reason": "reroute while degraded"}

    return {"passed": False, "reason": "path not in policy-approved routes"}
