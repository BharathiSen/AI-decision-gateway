"""
B5: adaptive testing without AI predictions and without the true fault id.

The candidate is the proposal's first cause. The alternative is a catalog
fault that the snapshot symptoms make plausible and that needs a different fix.

Evidence rules match the gate: a test row with `error` is missing evidence
(never graded as an observation), and results are graded with the same
traffic-level thresholds the gate uses for the episode.
"""

from gate import selector
from gate.catalog import plausible_alternatives
from gate.checks import check_safety, noop_justified
from gate.decide import _symptoms_from_snapshot
from control.actions import Action, normalize_action, RISKY_ACTIONS
from bench import tests as oam


def decide_b5(net, proposal, snapshot, test_results, run_tests=True, thresholds=None):
    measured = list(test_results or [])
    tests_run = [t for t in measured if not t.get("error")]
    errored = sorted({t.get("test_id") for t in measured if t.get("error")})
    ran = {t["test_id"] for t in measured if t.get("test_id")}
    try:
        action = normalize_action((proposal or {}).get("action"))
    except ValueError as exc:
        return {
            "method": "B5",
            "action": (proposal or {}).get("action"),
            "tests_run": list(ran),
            "decision": "block",
            "reason": str(exc),
        }
    record = {"method": "B5", "action": action.value, "tests_run": list(ran)}
    if errored:
        record["errored_tests"] = errored

    if action == Action.DO_NOTHING:
        if noop_justified(snapshot, measured, thresholds):
            return {**record, "decision": "allow", "reason": "healthy no-op"}
        return {**record, "decision": "escalate", "reason": "no-op not justified"}
    if action not in RISKY_ACTIONS:
        return {**record, "decision": "escalate", "reason": "non-risky not auto-allowed"}

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
    tid = selector.pick_cheapest_separating_test(primary, alt, already_ran={t["test_id"] for t in tests_run})
    if tid is None:
        return {**record, "decision": "escalate", "reason": "inseparable"}

    if run_tests and net is not None and not any(t["test_id"] == tid for t in tests_run):
        fresh = oam.run_test(net, tid)
        ran.add(tid)
        record["tests_run"] = list(ran)
        if isinstance(fresh, dict) and fresh.get("error"):
            record.setdefault("errored_tests", []).append(tid)
        else:
            tests_run.append(fresh)

    tr = next((t for t in tests_run if t["test_id"] == tid), None)
    if tr is None:
        if tid in (record.get("errored_tests") or []):
            return {**record, "decision": "escalate", "reason": f"test unavailable: {tid}"}
        return {**record, "decision": "escalate", "reason": "inseparable"}

    from gate.checks import _summarize_observation

    matrix = selector._load_matrix()
    exp_cause = next((r["outcome"] for r in matrix if r["fault_id"] == primary and r["test_id"] == tid), None)
    exp_alt = next((r["outcome"] for r in matrix if r["fault_id"] == alt and r["test_id"] == tid), None)
    obs_s = _summarize_observation(tr, thresholds)
    separated = exp_cause is not None and exp_alt is not None and exp_cause != exp_alt and obs_s == exp_cause

    safety = check_safety(proposal, snapshot)
    if not safety["passed"]:
        return {**record, "decision": "block", "reason": "safety"}
    if separated:
        return {**record, "decision": "allow", "separated": alt}
    return {**record, "decision": "escalate", "reason": "test inconclusive"}
