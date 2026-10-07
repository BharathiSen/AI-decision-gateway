"""B3: ML classifier on test features (stub: threshold on T1 loss)."""

import yaml
from pathlib import Path

from control.actions import Action

_FIXES = Path(__file__).resolve().parent.parent / "config" / "correct_fixes.yaml"


def _features(test_results):
    for r in test_results:
        if r.get("test_id") == "T1_e2e_ping":
            return r.get("packet_loss_percent", 100), r.get("latency_ms", 0)
    return 100, 0


def predict_action(fault_id_hint, test_results):
    loss, lat = _features(test_results)
    if loss > 20:
        return Action.REROUTE
    if lat and lat > 150:
        return Action.REROUTE
    return Action.DO_NOTHING


def decide(proposal, test_results, fault_id_hint="F3"):
    pred = predict_action(fault_id_hint, test_results)
    agree = pred.value == proposal.get("action") or proposal.get("action") in (pred.value, "reroute_traffic")
    return {
        "method": "B3",
        "decision": "allow" if agree else "block",
        "predicted": pred.value,
    }
