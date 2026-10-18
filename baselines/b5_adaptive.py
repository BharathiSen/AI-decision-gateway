"""
B5: adaptive testing without AI predictions and without the true fault id.

The candidate is the proposal's first cause. The alternative is a catalog
fault that the snapshot symptoms make plausible and that needs a different fix.
"""

from gate import selector
from gate.catalog import plausible_alternatives
from gate.checks import check_safety
from gate.decide import _symptoms_from_snapshot
from control.actions import normalize_action, RISKY_ACTIONS
from bench import tests as oam


def decide_b5(net, proposal, snapshot, test_results, run_tests=True):
    action = normalize_action((proposal or {}).get("action"))
    tests_run = list(test_results or [])
    ran = {t["test_id"] for t in tests_run}
    record = {"method": "B5", "action": action.value, "tests_run": list(ran)}

    if action not in RISKY_ACTIONS:
        return {**record, "decision": "allow", "reason": "pass-through"}

    causes = [c.get("cause_id") for c in (proposal or {}).get("causes") or [] if c.get("cause_id")]
    if not causes:
        return {**record, "decision": "escalate", "reason": "no cause"}

    primary = causes[0]
    symptoms = _symptoms_from_snapshot(snapshot)
    alts = [
        alt
        for alt in plausible_alternatives(symptoms, exclude=[primary])
        if selector.needs_different_fixes(primary, alt)
    ]
    if not alts:
        return {**record, "decision": "escalate", "reason": "no alt"}

    alt = alts[0]
    tid = selector.pick_cheapest_separating_test(primary, alt, already_ran=ran)
    if tid is None:
        return {**record, "decision": "escalate", "reason": "inseparable"}

    if run_tests and net is not None and tid not in ran:
        tests_run.append(oam.run_test(net, tid))
        ran.add(tid)
        record["tests_run"] = list(ran)

    tr = next((t for t in tests_run if t["test_id"] == tid), None)
    if tr is None:
        return {**record, "decision": "escalate", "reason": "inseparable"}

    from gate.checks import _summarize_observation

    matrix = selector._load_matrix()
    exp_cause = next((r["outcome"] for r in matrix if r["fault_id"] == primary and r["test_id"] == tid), None)
    exp_alt = next((r["outcome"] for r in matrix if r["fault_id"] == alt and r["test_id"] == tid), None)
    obs_s = _summarize_observation(tr)
    separated = exp_cause is not None and exp_alt is not None and exp_cause != exp_alt and obs_s == exp_cause

    safety = check_safety(proposal, snapshot)
    if not safety["passed"]:
        return {**record, "decision": "block", "reason": "safety"}
    if separated:
        return {**record, "decision": "allow", "separated": alt}
    return {**record, "decision": "escalate", "reason": "test inconclusive"}
