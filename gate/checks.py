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
    from bench.outcomes import grade_result

    return grade_result(obs)


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


PRIMARY_LINKS = ("r1-r2", "r2-r4")
BACKUP_LINKS = ("r1-r3", "r3-r4")


def reroute_choice(proposal):
    """Return primary or backup only for an exact policy path or label."""
    proposal = proposal or {}
    label = proposal.get("path") or proposal.get("proposed_path_label")
    if label in ("primary", "backup"):
        return label
    nodes = proposal.get("proposed_path") or []
    policy = _load_policy()
    if nodes == (policy.get("primary_path") or []):
        return "primary"
    if nodes == (policy.get("backup_path") or []):
        return "backup"
    return None


def path_links_up(snapshot, choice):
    needed = PRIMARY_LINKS if choice == "primary" else BACKUP_LINKS
    by_name = {
        link.get("name"): link.get("up")
        for link in (snapshot or {}).get("links") or []
    }
    down = [name for name in needed if by_name.get(name) is not True]
    return not down, down


def check_safety(proposal, network_snapshot):
    try:
        action = normalize_action(proposal.get("action"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    if action not in RISKY_ACTIONS:
        return {"passed": True, "reason": "non-risky action"}

    choice = reroute_choice(proposal)
    if choice is None:
        return {"passed": False, "reason": "path not in policy-approved routes"}
    up, down = path_links_up(network_snapshot, choice)
    if not up:
        return {"passed": False, "reason": f"{choice} path is down", "down": down}
    return {"passed": True, "path": choice}
