"""
B4/B5: adaptive testing without AI predictions.

Picks cheapest test separating true fault from closest alternative fix class.
"""

import yaml
from pathlib import Path

from gate import selector
from gate.checks import check_safety
from control.actions import normalize_action, RISKY_ACTIONS
from bench import tests as oam

_FIXES_PATH = Path(__file__).resolve().parent.parent / "config" / "correct_fixes.yaml"


def _load_fixes():
    with _FIXES_PATH.open() as f:
        doc = yaml.safe_load(f)
    out = {}
    out.update(doc.get("faults", {}))
    out.update(doc.get("compounds", {}))
    return out


def _closest_alt(fault_id, fixes):
    for other in fixes:
        if other == fault_id or other == "H0":
            continue
        if selector.needs_different_fixes(fault_id, other, fixes):
            return other
    return None


def decide_b5(net, proposal, snapshot, test_results, fault_id, run_tests=True):
    fixes = _load_fixes()
    action = normalize_action(proposal.get("action"))
    tests_run = list(test_results)
    ran = {t["test_id"] for t in tests_run}

    if action not in RISKY_ACTIONS:
        return {"method": "B5", "decision": "allow", "tests_run": list(ran), "reason": "pass-through"}

    alt = _closest_alt(fault_id, fixes)
    if alt is None:
        safety = check_safety(proposal, snapshot)
        return {
            "method": "B5",
            "decision": "allow" if safety["passed"] else "block",
            "tests_run": list(ran),
            "reason": "no alt",
        }

    tid = selector.pick_cheapest_separating_test(fault_id, alt, already_ran=ran)
    if tid and run_tests and tid not in ran:
        tests_run.append(oam.run_test(net, tid))
        ran.add(tid)

    if tid is None:
        return {"method": "B5", "decision": "escalate", "tests_run": list(ran), "reason": "inseparable"}

    tr = next(t for t in tests_run if t["test_id"] == tid)
    obs = tr  # observed live
    matrix = selector._load_matrix()
    exp_fault = next((r["outcome"] for r in matrix if r["fault_id"] == fault_id and r["test_id"] == tid), None)
    exp_alt = next((r["outcome"] for r in matrix if r["fault_id"] == alt and r["test_id"] == tid), None)

    from gate.checks import _summarize_observation

    obs_s = _summarize_observation(tr)
    separated = exp_fault is not None and exp_alt is not None and exp_fault != exp_alt and obs_s == exp_fault

    safety = check_safety(proposal, snapshot)
    if not safety["passed"]:
        return {"method": "B5", "decision": "block", "tests_run": list(ran), "reason": "safety"}

    if separated:
        return {"method": "B5", "decision": "allow", "tests_run": list(ran), "separated": alt}
    return {"method": "B5", "decision": "escalate", "tests_run": list(ran), "reason": "test inconclusive"}
