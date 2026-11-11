"""
B5: adaptive testing without AI predictions and without the true fault id.

Parity notes (H5): same episode noise thresholds as VeriFix/B4, same
NOOP_EVIDENCE_TESTS / cause-permit DO_NOTHING rule, same candidate set and
covering separators, and the same positive-support elimination rule
(observation must match the primary catalog outcome and contradict each
relevant alternative). ALLOW requires every relevant alternative eliminated
(same full-mode bar as VeriFix/B4). Budget is the shared DEFAULT_BUDGET_SECONDS
cap on newly run tests.
"""

from gate import selector
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
    noise_id=None,
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
    matrix = (frozen or {}).get("matrix") or []

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

    if not ok:
        return {**record, "decision": "escalate", "reason": "frozen inputs unavailable"}

    symptoms = _symptoms_from_snapshot(snapshot)
    # Same candidate / relevant sets as VeriFix and B4 (not a one-alt slice).
    candidates = _candidate_set(primary, causes, symptoms, hidden)
    relevant = [
        alt
        for alt in candidates
        if alt != primary and selector.needs_different_fixes(primary, alt, fixes=fixes)
    ]
    selected = list(selector.covering_separating_tests(candidates, fixes=fixes, matrix=matrix))
    record["selected_tests"] = list(selected)
    record["relevant_alternatives"] = list(relevant)
    if not relevant:
        return {**record, "decision": "escalate", "reason": "no alt"}

    from gate.checks import _summarize_observation

    eliminated = []
    unavailable = []
    inseparable = []
    unaffordable = []

    for alt in relevant:
        separators = set(selector.separating_tests(primary, alt, matrix=matrix))
        tid = next((t for t in selected if t in separators), None)
        if tid is None:
            # Same as VeriFix: pick any cheapest separator, then reuse measured rows.
            tid = selector.pick_cheapest_separating_test(primary, alt, matrix=matrix)
        if tid is None:
            inseparable.append(alt)
            continue

        tr = next((t for t in tests_run if t.get("test_id") == tid), None)
        if tr is None and tid in errored:
            unavailable.append({"alternative": alt, "test_id": tid})
            continue
        if tr is None:
            cost = test_cost(tid)
            if used_seconds + cost > cap + 1e-9:
                unaffordable.append({"alternative": alt, "test_id": tid})
                continue
            if not run_tests or net is None:
                # Replay / offline: missing separator evidence cannot eliminate.
                unaffordable.append({"alternative": alt, "test_id": tid})
                continue
            fresh = oam.run_test(net, tid)
            used_seconds += cost
            ran.add(tid)
            record["tests_run"] = list(ran)
            if isinstance(fresh, dict) and fresh.get("error"):
                errored = sorted(set(errored) | {tid})
                record["errored_tests"] = errored
                unavailable.append({"alternative": alt, "test_id": tid})
                continue
            if isinstance(fresh, dict):
                tests_run.append(fresh)
                tr = fresh
            else:
                unavailable.append({"alternative": alt, "test_id": tid})
                continue

        if isinstance(tr, dict) and tr.get("error"):
            unavailable.append({"alternative": alt, "test_id": tid})
            continue
        try:
            obs_s = _summarize_observation(tr, thresholds)
        except ValueError:
            unavailable.append({"alternative": alt, "test_id": tid})
            continue
        if obs_s is None:
            unavailable.append({"alternative": alt, "test_id": tid})
            continue

        exp_cause = next(
            (r["outcome"] for r in matrix if r["fault_id"] == primary and r["test_id"] == tid),
            None,
        )
        exp_alt = next(
            (r["outcome"] for r in matrix if r["fault_id"] == alt and r["test_id"] == tid),
            None,
        )
        # H6: positive support for primary and contradiction of this alternative.
        supports_primary = (
            exp_cause is not None and prediction_matches(tid, exp_cause, obs_s)
        )
        contradicts_alt = (
            exp_alt is not None and not prediction_matches(tid, exp_alt, obs_s)
        )
        if supports_primary and contradicts_alt:
            eliminated.append(alt)
        # else: leave alt in remaining (inconclusive for this separator)

    remaining = [alt for alt in relevant if alt not in set(eliminated)]
    record["eliminated"] = list(eliminated)
    record["remaining_alternatives"] = list(remaining)
    record["budget"] = {
        "budget_seconds": cap,
        "tests_used_seconds": used_seconds,
        "within_budget": used_seconds <= cap + 1e-9,
    }

    safety = check_safety(proposal, snapshot)
    if not safety["passed"]:
        return {**record, "decision": "block", "reason": "safety"}
    if not remaining:
        return {
            **record,
            "decision": "allow",
            "separated": list(eliminated),
            "selected_tests": list(selected),
        }
    if inseparable and not eliminated:
        return {**record, "decision": "escalate", "reason": "inseparable"}
    if unavailable:
        tids = sorted({row["test_id"] for row in unavailable})
        return {**record, "decision": "escalate", "reason": "test unavailable: " + ", ".join(tids)}
    if unaffordable:
        return {**record, "decision": "escalate", "reason": "no affordable discriminating test"}
    if inseparable:
        return {**record, "decision": "escalate", "reason": "inseparable"}
    return {**record, "decision": "escalate", "reason": "no discrimination"}
