"""Gate checks: consistency, discrimination, safety."""

import yaml
from pathlib import Path

from control.actions import normalize_action, RISKY_ACTIONS

_POLICY = Path(__file__).resolve().parent.parent / "policy" / "policy.yaml"


def _load_policy():
    with _POLICY.open() as f:
        return yaml.safe_load(f)


def check_consistency(proposal, test_results, primary_cause_id=None, thresholds=None):
    """
    Compare observed tests to predictions on the AI's primary cause
    (first cause if not specified). thresholds are the episode's noise
    level limits; None grades at N0.

    passed is False when zero predictions were compared with observations
    (empty predicted_tests, or only tests that were not run).
    """
    from bench.contracts import parse_t2_prediction, parse_t6_prediction, prediction_matches

    causes = proposal.get("causes") or []
    if not causes:
        return {
            "passed": False,
            "reason": "no causes to check",
            "mismatches": [],
            "checked_cause": None,
            "predictions_total": 0,
            "predictions_checked": 0,
            "predictions_unchecked": 0,
            "predictions_matched": 0,
            "predictions_mismatched": 0,
            "predictions": [],
        }
    cid = primary_cause_id or causes[0].get("cause_id")
    cause = next((c for c in causes if c.get("cause_id") == cid), causes[0])
    raw_preds = [p for p in (cause.get("predicted_tests") or []) if isinstance(p, dict)]
    by_id = {r["test_id"]: r for r in test_results if isinstance(r, dict) and r.get("test_id")}

    seen_ids = []
    rows = []
    mismatches = []
    checked = matched = mismatched = unchecked = 0
    for pred in raw_preds:
        tid = pred.get("test_id")
        expected = pred.get("expected")
        record = {
            "test_id": tid,
            "original": expected,
            "normalized": expected,
            "delimiter_normalized": False,
            "status": "unchecked",
            "observed": None,
            "reason": None,
        }
        if tid == "T2_hop_ping":
            parsed = parse_t2_prediction(expected)
            record["normalized"] = parsed.get("normalized")
            record["delimiter_normalized"] = bool(parsed.get("delimiter_normalized"))
        elif tid == "T6_queue_stats":
            parsed = parse_t6_prediction(expected)
            record["normalized"] = parsed.get("normalized")
        if tid in seen_ids:
            record["status"] = "mismatched"
            record["reason"] = f"duplicate test_id {tid!r}"
            mismatched += 1
            checked += 1
            mismatches.append(record)
            rows.append(record)
            continue
        seen_ids.append(tid)
        obs = by_id.get(tid)
        if obs is None:
            record["reason"] = "test was not executed"
            unchecked += 1
            rows.append(record)
            continue
        # Errored rows are unavailable measurement, not negative evidence.
        if obs.get("error"):
            record["reason"] = f"test unavailable: {obs.get('error')}"
            unchecked += 1
            rows.append(record)
            continue
        obs_str = _summarize_observation(obs, thresholds)
        record["observed"] = obs_str
        checked += 1
        if prediction_matches(tid, expected, obs_str):
            record["status"] = "matched"
            matched += 1
        else:
            record["status"] = "mismatched"
            mismatched += 1
            if tid == "T2_hop_ping":
                pred_parse = parse_t2_prediction(expected)
                obs_parse = parse_t2_prediction(obs_str)
                if "backup" in pred_parse.get("paths", ()) and "backup" not in obs_parse.get("paths", ()):
                    record["reason"] = "observed T2 has no backup section"
                else:
                    record["reason"] = "hop/outcome mismatch"
            else:
                record["reason"] = "prediction mismatch"
            mismatches.append(record)
        rows.append(record)

    total = len(raw_preds)
    if checked == 0:
        reason = "no checked predictions"
        passed = False
    else:
        passed = mismatched == 0
        reason = None if passed else "consistency failed"
    return {
        "passed": passed,
        "reason": reason,
        "mismatches": mismatches,
        "checked_cause": cause.get("cause_id"),
        "predictions_total": total,
        "predictions_checked": checked,
        "predictions_unchecked": unchecked,
        "predictions_matched": matched,
        "predictions_mismatched": mismatched,
        "predictions": rows,
    }


def _summarize_observation(obs, thresholds=None):
    from bench.outcomes import grade_result

    return grade_result(obs, thresholds)


def _prediction_matches(expected, observed, test_id=None):
    """Delegate to the shared contracts matcher. Prefer passing test_id."""
    from bench.contracts import prediction_matches, scalar_prediction_matches

    if test_id:
        return prediction_matches(test_id, expected, observed)
    return scalar_prediction_matches(expected, observed)


def check_discrimination(ruled_out_alternatives):
    return {
        "passed": len(ruled_out_alternatives) > 0,
        "ruled_out": ruled_out_alternatives,
    }


PRIMARY_LINKS = ("r1-r2", "r2-r4")
BACKUP_LINKS = ("r1-r3", "r3-r4")


def reroute_choice(proposal):
    """Return primary or backup only for an exact policy path or label.

    A label and a node list that name different paths are ambiguous: the
    controller and the safety check could act on different paths.
    """
    proposal = proposal or {}
    label = proposal.get("path") or proposal.get("proposed_path_label")
    nodes = proposal.get("proposed_path") or []
    policy = _load_policy()
    from_nodes = None
    if nodes == (policy.get("primary_path") or []):
        from_nodes = "primary"
    elif nodes == (policy.get("backup_path") or []):
        from_nodes = "backup"
    if label is not None:
        if label not in ("primary", "backup"):
            return None
        if nodes and from_nodes != label:
            return None
        return label
    return from_nodes


def path_links_up(snapshot, choice):
    needed = PRIMARY_LINKS if choice == "primary" else BACKUP_LINKS
    by_name = {
        link.get("name"): link.get("up")
        for link in (snapshot or {}).get("links") or []
    }
    down = [name for name in needed if by_name.get(name) is not True]
    return not down, down


# Traffic tests that must all be present and healthy to justify a no-op.
# Incomplete batteries and degraded T10 (performance) refuse DO_NOTHING.
NOOP_EVIDENCE_TESTS = ("T1_e2e_ping", "T4_mtu", "T8_tcp_vs_ping", "T10_speed")


def noop_justified(snapshot, test_results, thresholds=None):
    """True when leaving the network unchanged is supported by complete evidence."""
    from bench.outcomes import grade_result, is_degraded, load_thresholds

    thresholds = thresholds or load_thresholds("N0")
    metrics = (snapshot or {}).get("metrics") or {}
    max_loss = (thresholds.get("T1_e2e_ping") or {}).get("max_loss_percent", 5.0)
    loss = metrics.get("packet_loss_percent")
    loss_known = isinstance(loss, (int, float)) and not isinstance(loss, bool)
    # Contradiction needs an explicit broken reading; a missing number is not one.
    snapshot_broken = metrics.get("connectivity") is False or (loss_known and loss > max_loss)
    by_id = {}
    for row in test_results or []:
        if not isinstance(row, dict):
            continue
        tid = row.get("test_id")
        if tid in NOOP_EVIDENCE_TESTS:
            by_id[tid] = row
    # Completeness: every required test must be present and measured.
    for tid in NOOP_EVIDENCE_TESTS:
        row = by_id.get(tid)
        if row is None or row.get("error"):
            return False
        if is_degraded(tid, grade_result(row, thresholds), thresholds):
            return False
    # Healthy OAM next to a broken snapshot is contradictory evidence.
    return not snapshot_broken


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
    needed = PRIMARY_LINKS if choice == "primary" else BACKUP_LINKS
    dead = [
        link.get("name")
        for link in (network_snapshot or {}).get("links") or []
        if link.get("name") in needed
        and isinstance(link.get("packet_loss_percent"), (int, float))
        and link["packet_loss_percent"] >= 100.0
    ]
    if dead:
        return {"passed": False, "reason": f"{choice} path is unreachable", "down": dead}
    return {"passed": True, "path": choice}
