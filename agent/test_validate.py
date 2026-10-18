"""Offline validation tests for agent/validate.py."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.validate import MAX_EXPECTED_LEN, MAX_PROPOSAL_BYTES, validate_batch, validate_proposal


def good_proposal():
    return {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "proposed_path": ["r1", "r3", "r4"],
        "reason": "loss on primary",
        "causes": [
            {
                "cause_id": "F3",
                "predicted_tests": [
                    {"test_id": "T1_e2e_ping", "expected": "loss"},
                    {"test_id": "T7_route_check", "expected": "primary"},
                ],
            }
        ],
    }


def check(label, ok, detail=""):
    if not ok:
        raise AssertionError(f"{label}: {detail}")
    print(f"  PASS  {label}")


def main():
    good = good_proposal()
    ok, errs = validate_proposal(good)
    check("well-formed proposal", ok, errs)

    unknown_cause = copy.deepcopy(good)
    unknown_cause["causes"][0]["cause_id"] = "not-in-catalog"
    ok, errs = validate_proposal(unknown_cause)
    check("unknown cause id rejected", not ok, errs)

    held_out = copy.deepcopy(good)
    held_out["causes"][0]["cause_id"] = "U1"
    ok, errs = validate_proposal(held_out)
    check("held-out catalog id accepted", ok, errs)

    bad = copy.deepcopy(good)
    bad["causes"][0]["predicted_tests"][0]["test_id"] = "T99"
    ok, errs = validate_proposal(bad)
    check("unknown test id rejected", not ok, errs)

    missing = copy.deepcopy(good)
    del missing["reason"]
    ok, errs = validate_proposal(missing)
    check("missing reason rejected", not ok, errs)

    empty_target = copy.deepcopy(good)
    empty_target["target"] = "  "
    ok, errs = validate_proposal(empty_target)
    check("blank target rejected", not ok, errs)

    for action in ("REBALANCE", "reroute_traffic", "no_action", "reboot"):
        unsupported = copy.deepcopy(good)
        unsupported["action"] = action
        ok, errs = validate_proposal(unsupported)
        check(f"unsupported action {action} rejected", not ok, errs)

    wider = copy.deepcopy(good)
    wider["action"] = "REBALANCE"
    ok, errs = validate_proposal(wider, allowed_actions=["REROUTE", "DO_NOTHING", "ESCALATE", "REBALANCE"])
    check("caller-allowed REBALANCE accepted", ok, errs)

    not_object = validate_proposal(["REROUTE"])
    check("non-object rejected", not not_object[0], not_object[1])

    bad_causes = copy.deepcopy(good)
    bad_causes["causes"] = "F3"
    ok, errs = validate_proposal(bad_causes)
    check("causes must be a list", not ok, errs)

    bad_cause = copy.deepcopy(good)
    bad_cause["causes"] = ["F3"]
    ok, errs = validate_proposal(bad_cause)
    check("cause entries must be objects", not ok, errs)

    too_many = copy.deepcopy(good)
    too_many["causes"] = copy.deepcopy(good["causes"]) * 6
    ok, errs = validate_proposal(too_many)
    check("more than five causes rejected", not ok, errs)

    long_expected = copy.deepcopy(good)
    long_expected["causes"][0]["predicted_tests"][0]["expected"] = "x" * (MAX_EXPECTED_LEN + 1)
    ok, errs = validate_proposal(long_expected)
    check("overlong expected rejected", not ok, errs)

    huge = copy.deepcopy(good)
    huge["reason"] = "y" * (MAX_PROPOSAL_BYTES + 1)
    ok, errs = validate_proposal(huge)
    check("oversized proposal rejected", not ok, errs)

    no_causes = {
        "action": "DO_NOTHING",
        "target": "HostA_to_HostB",
        "reason": "network looks healthy",
    }
    ok, errs = validate_proposal(no_causes)
    check("proposal without causes accepted", ok, errs)

    rate, ok_n, total = validate_batch([copy.deepcopy(good) for _ in range(95)] + [bad] * 5)
    check("batch rate at least 95%", rate >= 0.95, f"{ok_n}/{total} = {rate:.2%}")
    print(f"validate_batch: {ok_n}/{total} = {rate:.2%}")
    print("PASS")


if __name__ == "__main__":
    main()
