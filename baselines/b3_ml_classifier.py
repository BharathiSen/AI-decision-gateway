"""B3: catalog nearest-neighbor comparator on a capped OAM information budget.

Role (H5): not a trained ML model. It matches graded OAM signatures against
the development observability matrix and maps the nearest fault to the
shared action set. Held-out faults never enter the table. Episode noise
thresholds grade observations the same way VeriFix/B4/B5 do. The documented
information budget is DEFAULT_BUDGET_SECONDS (cheapest-first fill).
"""

from bench import tests as oam
from bench.outcomes import load_thresholds
from bench.tests import DEFAULT_ORDER
from control.actions import Action, normalize_action
from gate.checks import _summarize_observation
from gate.selector import _load_fixes, _load_matrix, held_out_ids

from baselines.common import DEFAULT_BUDGET_SECONDS, PERMITTED_ACTIONS, cheapest_within_budget


def _fallback_action(test_results, snapshot, thresholds=None):
    thresholds = thresholds or {}
    max_loss = (thresholds.get("T1_e2e_ping") or {}).get("max_loss_percent", 20.0)
    max_lat = (thresholds.get("T1_e2e_ping") or {}).get("max_latency_ms", 150.0)
    loss, lat = 100.0, 0.0
    for row in test_results or []:
        if row.get("test_id") == "T1_e2e_ping":
            if row.get("error"):
                return Action.ESCALATE
            loss = row.get("packet_loss_percent", 100)
            lat = row.get("latency_ms", 0) or 0
            break
    else:
        metrics = (snapshot or {}).get("metrics") or {}
        if "packet_loss_percent" in metrics:
            loss = metrics.get("packet_loss_percent") or 0
            lat = metrics.get("latency_ms") or 0
    if (loss or 0) > max_loss:
        return Action.REROUTE
    if lat and lat > max_lat:
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
            row = by_id[tid]
            # Errored rows are missing evidence, not signature tokens.
            if isinstance(row, dict) and row.get("error"):
                continue
            selected.append(row)
        elif net is not None:
            fresh = oam.run_test(net, tid)
            if isinstance(fresh, dict) and fresh.get("error"):
                continue
            selected.append(fresh)
    return selected


def _match_fault(selected, thresholds=None):
    matrix = _load_matrix()
    hidden = held_out_ids()
    if not matrix or not selected:
        return None, None
    obs = {
        row["test_id"]: _summarize_observation(row, thresholds)
        for row in selected
        if row.get("test_id")
    }
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
            if expected is None or observed is None:
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


def predict_action(test_results, snapshot=None, net=None, budget_seconds=None, noise_id="N0"):
    if noise_id is None:
        raise ValueError("B3 requires noise_id (episode traffic level)")
    thresholds = load_thresholds(noise_id)
    selected = _select_results(test_results, net, budget_seconds)
    fault_id, score = _match_fault(selected, thresholds=thresholds)
    predicted = _action_for_fault(fault_id)
    if predicted is None:
        predicted = _fallback_action(selected, snapshot, thresholds=thresholds).value
    return predicted, selected, fault_id, score


def decide(proposal, snapshot, test_results, net=None, budget_seconds=None, noise_id=None):
    pred, selected, fault_id, score = predict_action(
        test_results,
        snapshot=snapshot,
        net=net,
        budget_seconds=budget_seconds,
        noise_id=noise_id,
    )
    proposed = (proposal or {}).get("action")
    try:
        proposed_canon = normalize_action(proposed).value
    except (ValueError, TypeError):
        proposed_canon = proposed
    agree = proposed_canon == pred
    return {
        "method": "B3",
        "classifier": "catalog_nearest_neighbor",
        "decision": "allow" if agree else "block",
        "action": pred,
        "predicted": pred,
        "matched_fault": fault_id,
        "match_score": score,
        "noise_id": noise_id,
        "tests_run": [row.get("test_id") for row in selected if row.get("test_id")],
        "reason": "matrix nn" if fault_id else "t1 fallback",
    }
