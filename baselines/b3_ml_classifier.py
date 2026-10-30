"""B3: matrix nearest-neighbor on a capped OAM information budget.

This is not a trained ML model. The lookup uses the development
observability matrix only. Held-out faults never enter the signature
table. When the matrix is missing, T1 loss/latency thresholds are the
fallback.
"""

from bench import tests as oam
from bench.tests import DEFAULT_ORDER
from control.actions import Action, normalize_action
from gate.checks import _summarize_observation
from gate.selector import _load_fixes, _load_matrix, held_out_ids

from baselines.common import DEFAULT_BUDGET_SECONDS, PERMITTED_ACTIONS, cheapest_within_budget


def _fallback_action(test_results, snapshot):
    loss, lat = 100.0, 0.0
    for row in test_results or []:
        if row.get("test_id") == "T1_e2e_ping":
            loss = row.get("packet_loss_percent", 100)
            lat = row.get("latency_ms", 0) or 0
            break
    else:
        metrics = (snapshot or {}).get("metrics") or {}
        if "packet_loss_percent" in metrics:
            loss = metrics.get("packet_loss_percent") or 0
            lat = metrics.get("latency_ms") or 0
    if (loss or 0) > 20:
        return Action.REROUTE
    if lat and lat > 150:
        return Action.REROUTE
    return Action.DO_NOTHING


def _select_results(test_results, net, budget_seconds):
    """Use cheapest supplied (or live) tests that fit the budget."""
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    by_id = {row.get("test_id"): row for row in (test_results or []) if row.get("test_id")}
    available = [tid for tid in DEFAULT_ORDER if tid in by_id]
    if net is not None:
        for tid in DEFAULT_ORDER:
            if tid not in available:
                available.append(tid)
    chosen, _used = cheapest_within_budget(available, cap)
    selected = []
    for tid in chosen:
        if tid in by_id:
            selected.append(by_id[tid])
        elif net is not None:
            selected.append(oam.run_test(net, tid))
    return selected


def _match_fault(selected):
    matrix = _load_matrix()
    hidden = held_out_ids()
    if not matrix or not selected:
        return None, None
    obs = {row["test_id"]: _summarize_observation(row) for row in selected if row.get("test_id")}
    best_id = None
    best = (-1.0, -1, 0)
    faults = {row["fault_id"] for row in matrix if row.get("fault_id") not in hidden}
    for fid in faults:
        score = 0
        compared = 0
        for tid, observed in obs.items():
            expected = next(
                (row["outcome"] for row in matrix if row["fault_id"] == fid and row["test_id"] == tid),
                None,
            )
            if expected is None:
                continue
            compared += 1
            exp = (expected or "").lower()
            got = (observed or "").lower()
            if exp == got or exp in got or got in exp:
                score += 1
        if not compared:
            continue
        rank = (score / compared, score, compared)
        if rank > best:
            best = rank
            best_id = fid
    return best_id, best[0] if best_id else None


def _action_for_fault(fault_id):
    if not fault_id:
        return None
    correct = (_load_fixes().get(fault_id) or {}).get("correct") or []
    for name in correct:
        if name in PERMITTED_ACTIONS:
            return name
    return "ESCALATE"


def predict_action(test_results, snapshot=None, net=None, budget_seconds=None):
    selected = _select_results(test_results, net, budget_seconds)
    fault_id, score = _match_fault(selected)
    predicted = _action_for_fault(fault_id)
    if predicted is None:
        predicted = _fallback_action(selected, snapshot).value
    return predicted, selected, fault_id, score


def decide(proposal, snapshot, test_results, net=None, budget_seconds=None):
    pred, selected, fault_id, score = predict_action(
        test_results, snapshot=snapshot, net=net, budget_seconds=budget_seconds
    )
    proposed = (proposal or {}).get("action")
    try:
        proposed_canon = normalize_action(proposed).value
    except (ValueError, TypeError):
        proposed_canon = proposed
    agree = proposed_canon == pred
    return {
        "method": "B3",
        "classifier": "nearest_neighbor",
        "decision": "allow" if agree else "block",
        "action": pred,
        "predicted": pred,
        "matched_fault": fault_id,
        "match_score": score,
        "tests_run": [row.get("test_id") for row in selected if row.get("test_id")],
        "reason": "matrix nn" if fault_id else "t1 fallback",
    }
