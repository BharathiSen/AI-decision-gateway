"""Offline validation tests for agent/validate.py."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.validate import validate_proposal, validate_batch


def main():
    good = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "proposed_path": ["r1", "r3", "r4"],
        "reason": "loss on primary",
        "causes": [
            {
                "cause_id": "F3",
                "predicted_tests": [
                    {"test_id": "T1_e2e_ping", "expected": "loss=15"},
                    {"test_id": "T7_route_check", "expected": "primary"},
                ],
            }
        ],
    }
    ok, errs = validate_proposal(good)
    assert ok, errs

    bad = copy.deepcopy(good)
    bad["causes"][0]["predicted_tests"][0]["test_id"] = "T99"
    ok2, _ = validate_proposal(bad)
    assert not ok2

    rate, ok_n, total = validate_batch([copy.deepcopy(good) for _ in range(95)] + [bad] * 5)
    assert rate >= 0.95, rate
    print(f"validate_batch: {ok_n}/{total} = {rate:.2%} (target >=95%)")
    print("PASS")


if __name__ == "__main__":
    main()
