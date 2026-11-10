"""
B5: adaptive testing without AI predictions and without the true fault id.

Parity notes (H5): same episode noise thresholds as VeriFix/B4, same
NOOP_EVIDENCE_TESTS / cause-permit DO_NOTHING rule, same covering separators
when available, and the same positive-support elimination rule (observation
must match the primary catalog outcome and contradict the alternative).
Budget is the shared DEFAULT_BUDGET_SECONDS cap on newly run tests.
"""

from gate import selector
from gate.catalog import plausible_alternatives
from gate.checks import check_safety, noop_justified
from gate.decide import _candidate_set, _symptoms_from_snapshot
from gate.inputs import load_frozen_context
from control.actions import Action, normalize_action, RISKY_ACTIONS
from bench import tests as oam
from bench.contracts import prediction_matches
from baselines.common import DEFAULT_BUDGET_SECONDS, test_cost


def decide_b5(
    net,
    proposal,
    snapshot,
    test_results,
    run_tests=True,
    thresholds=None,
    noise_id="N0",
    budget_seconds=None,
):
    if noise_id is None:
        raise ValueError("B5 requires noise_id (episode traffic level)")
    measured = list(test_results or [])
    tests_run = [t for t in measured if not t.get("error")]
    errored = sorted({t.get("test_id") for t in measured if t.get("error") and t.get("test_id")})
    ran = {t["test_id"] for t in measured if t.get("test_id")}
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    used_seconds = 0.0
    try:
        action = normalize_action((proposal or {}).get("action"))
    except ValueError as exc:
        return {
            "method": "B5",
            "action": (proposal or {}).get("action"),
            "tests_run": list(ran),
            "decision": "block",
            "reason": str(exc),
            "noise_id": noise_id,
        }
    record = {
        "method": "B5",
        "action": action.value,
        "tests_run": list(ran),
        "noise_id": noise_id,
    }
    if errored:
        record["errored_tests"] = errored

    causes = [c.get("cause_id") for c in (proposal or {}).get("causes") or [] if c.get("cause_id")]
    primary = causes[0] if causes else None
    ok, _errs, frozen = load_frozen_context(noise_id)
    fixes = (frozen or {}).get("fixes") or {}
    hidden = ((frozen or {}).get("held_out") or set()) if ok else set()

    if action == Action.DO_NOTHING:
        # Match VeriFix: cause must permit DO_NOTHING and evidence must be complete.
        if not primary:
            return {**record, "decision": "escalate", "reason": "no-op not justified: missing cause"}
        if primary in hidden or primary not in fixes:
            return {**record, "decision": "escalate", "reason": "no-op not justified: outside catalog"}
        permitted = set((fixes.get(primary) or {}).get("correct") or [])
        if "DO_NOTHING" not in permitted:
            return {
                **record,
                "decision": "escalate",
                "reason": "no-op not justified: DO_NOTHING not permitted for cause",
            }
        if noop_justified(snapshot, measured, thresholds, noise_id=noise_id):
            return {**record, "decision": "allow", "reason": "healthy no-op"}
        return {**record, "decision": "escalate", "reason": "no-op not justified"}
    if action not in RISKY_ACTIONS:
        return {**record, "decision": "escalate", "reason": "non-risky not auto-allowed"}

    if not causes:
        return {**record, "decision": "escalate", "reason": "no cause"}

    symptoms = _symptoms_from_snapshot(snapshot)
    candidates = _candidate_set(primary, causes, symptoms, hidden) if ok else [primary]
    selected = selector.covering_separating_tests(
        candidates, fixes=fixes, matrix=(frozen or {}).get("matrix") or []
    ) if ok else []
    record["selected_tests"] = list(selected)

    alts = [
        alt
        for alt in plausible_alternatives(symptoms, exclude=[primary])
        if selector.needs_different_fixes(primary, alt)
    ]
    if not alts:
        return {**record, "decision": "escalate", "reason": "no alt"}

    # Prefer a covering separator shared with VeriFix/B4 when one applies.
    alt = alts[0]
    tid = None
    for candidate_tid in selected:
        if candidate_tid in set(selector.separating_tests(primary, alt)):
            tid = candidate_tid
            break
    if tid is None:
        tid = selector.pick_cheapest_separating_test(
            primary, alt, already_ran={t["test_id"] for t in tests_run}
        )
    if tid is None:
        return {**record, "decision": "escalate", "reason": "inseparable"}

    if run_tests and net is not None and not any(t["test_id"] == tid for t in tests_run):
        cost = test_cost(tid)
        if used_seconds + cost > cap + 1e-9:
            return {**record, "decision": "escalate", "reason": f"budget exhausted before {tid}"}
        fresh = oam.run_test(net, tid)
        used_seconds += cost
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
    # H6-aligned: positive support for primary and contradiction of alternative.
    supports_primary = (
        exp_cause is not None and obs_s is not None and prediction_matches(tid, exp_cause, obs_s)
    )
    contradicts_alt = (
        exp_alt is not None and obs_s is not None and not prediction_matches(tid, exp_alt, obs_s)
    )
    separated = supports_primary and contradicts_alt

    safety = check_safety(proposal, snapshot)
    if not safety["passed"]:
        return {**record, "decision": "block", "reason": "safety"}
    if separated:
        return {**record, "decision": "allow", "separated": alt, "selected_tests": list(selected)}
    return {**record, "decision": "escalate", "reason": "test inconclusive"}
