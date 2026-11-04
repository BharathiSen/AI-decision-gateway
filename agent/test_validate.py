"""Offline validation tests for agent/validate.py."""

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.validate import (
    MAX_EXPECTED_LEN,
    MAX_PROPOSAL_BYTES,
    SUPPORTED_TARGETS,
    catalog_cause_ids,
    proposal_freshness_errors,
    validate_batch,
    validate_proposal,
)
from agent.decision_agent import _canonical_target
from bench.contracts import (
    TEST_IDS,
    expected_grade_ok,
    llm_proposal_schema,
    parse_t2_prediction,
    parse_t6_prediction,
)


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

    other_flow = copy.deepcopy(good)
    other_flow["target"] = "HostC_to_HostD"
    ok, errs = validate_proposal(other_flow)
    check("unsupported target rejected", not ok, errs)

    schema = llm_proposal_schema(
        ["REROUTE", "DO_NOTHING", "ESCALATE"],
        predictions=True,
        cause_ids=catalog_cause_ids(),
        targets=sorted(SUPPORTED_TARGETS),
    )
    check("llm schema pins target", schema["properties"]["target"].get("enum") == ["HostA_to_HostB"])
    cause_enum = schema["properties"]["causes"]["items"]["properties"]["cause_id"].get("enum") or []
    check("llm schema includes catalog F1", "F1" in cause_enum)
    check("llm schema includes held-out U1", "U1" in cause_enum)
    check("llm schema excludes compounds", "C1" not in cause_enum)
    check(
        "llm schema allows a full OAM catalog",
        schema["properties"]["causes"]["items"]["properties"]["predicted_tests"].get("maxItems")
        == len(TEST_IDS),
    )
    check("target alias hostA to hostB", _canonical_target("hostA to hostB") == "HostA_to_HostB")
    check("target alias hostA", _canonical_target("hostA") == "HostA_to_HostB")
    check("unsupported target alias left alone", _canonical_target("HostC_to_HostD") == "HostC_to_HostD")

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

    for tid, token in (
        ("T1_e2e_ping", "ok"),
        ("T1_e2e_ping", "down"),
        ("T1_e2e_ping", "loss"),
        ("T1_e2e_ping", "latency"),
        ("T1_e2e_ping", "loss+latency"),
        ("T4_mtu", "df_ok"),
        ("T4_mtu", "mtu_fail"),
        ("T7_route_check", "primary"),
        ("T7_route_check", "blackhole"),
        ("T8_tcp_vs_ping", "both_ok"),
        ("T8_tcp_vs_ping", "ping_ok_tcp_fail"),
        ("T9_oneway_loss", "symmetric"),
        ("T9_oneway_loss", "a2b_loss"),
        ("T10_speed", "ok"),
        ("T10_speed", "slow"),
        ("T5_if_errors", "clean"),
        ("T5_if_errors", "errors:r4/r4-eth0"),
        ("T6_queue_stats", "clear"),
        ("T6_queue_stats", "drops:r2/r2-eth1"),
        ("T2_hop_ping", "r2-r4=down"),
        ("T2_hop_ping", "primary:hostA-r1=ok,r2-r4=down;backup:r3-r4=ok"),
    ):
        token_ok = copy.deepcopy(good)
        token_ok["causes"][0]["predicted_tests"] = [{"test_id": tid, "expected": token}]
        ok, errs = validate_proposal(token_ok)
        check(f"grade token {tid} {token} accepted", ok, errs)
        check(f"expected_grade_ok {tid} {token}", expected_grade_ok(tid, token))

    for english in ("success", "timeout / no reply", "no errors", "all hops reachable", "packet loss 100%"):
        prose = copy.deepcopy(good)
        prose["causes"][0]["predicted_tests"][0]["expected"] = english
        ok, errs = validate_proposal(prose)
        check(f"english expected {english!r} rejected", not ok, errs)

    t2_prose = copy.deepcopy(good)
    t2_prose["causes"][0]["predicted_tests"] = [{"test_id": "T2_hop_ping", "expected": "fail at r2-r4"}]
    ok, errs = validate_proposal(t2_prose)
    check("T2 english hop narrative rejected", not ok, errs)
    check("T2 atomic down rejected", not expected_grade_ok("T2_hop_ping", "down"))
    hyphen = "r2\u2011r4=down"
    check("T2 unicode hyphen normalized", expected_grade_ok("T2_hop_ping", hyphen))
    check("T2 whitespace around hop and outcome accepted", expected_grade_ok("T2_hop_ping", "  r2-r4 = down  "))
    check(
        "T2 whitespace around path delimiters accepted",
        expected_grade_ok("T2_hop_ping", " primary : r2-r4 = down ; backup : r3-r4 = ok "),
    )
    check("T2 interior hop whitespace rejected", not expected_grade_ok("T2_hop_ping", "r2 - r4=down"))
    check("T2 placeholder hop= rejected", not expected_grade_ok("T2_hop_ping", "hop=ok"))
    check("T2 IP address hop rejected", not expected_grade_ok("T2_hop_ping", "10.0.24.2=down"))
    check("T2 unknown hop rejected", not expected_grade_ok("T2_hop_ping", "r9-r9=down"))
    check("T2 shared hop without prefix rejected", not expected_grade_ok("T2_hop_ping", "hostA-r1=ok"))
    check("T2 mixed-case topology name accepted", expected_grade_ok("T2_hop_ping", "primary:HOSTA-R1=ok"))
    comma = parse_t2_prediction("primary:r2-r4=down,backup:r3-r4=ok")
    check("T2 comma path transition normalized", comma["ok"] and comma["delimiter_normalized"])
    hop_comma = parse_t2_prediction("primary:hostA-r1=ok,r2-r4=down")
    check("T2 hop comma is not a path rewrite", hop_comma["ok"] and not hop_comma["delimiter_normalized"])
    check("T6 whitespace around iface accepted", expected_grade_ok("T6_queue_stats", " drops : r2/r2-eth1 "))
    check("T6 empty drops list rejected", not expected_grade_ok("T6_queue_stats", "drops:"))
    check("T6 empty backlog list rejected", not expected_grade_ok("T6_queue_stats", "backlog:"))
    check("T6 duplicate iface rejected", not expected_grade_ok("T6_queue_stats", "drops:r2/r2-eth1|r2/r2-eth1"))
    check("T6 r4 iface rejected", not expected_grade_ok("T6_queue_stats", "drops:r4/r4-eth2"))
    check("T6 bare iface rejected", not expected_grade_ok("T6_queue_stats", "drops:r2-eth1"))
    check("T6 mixed-case iface accepted", expected_grade_ok("T6_queue_stats", "drops:R2/R2-ETH1"))
    ordered = parse_t6_prediction("drops:r2/r2-eth1|r1/r1-eth1")
    reversed_ifaces = parse_t6_prediction("drops:r1/r1-eth1|r2/r2-eth1")
    check(
        "T6 iface order does not change the parsed set",
        ordered["ok"] and ordered["ifaces"] == reversed_ifaces["ifaces"],
    )

    dup_ids = copy.deepcopy(good)
    dup_ids["causes"][0]["predicted_tests"] = [
        {"test_id": "T1_e2e_ping", "expected": "loss"},
        {"test_id": "T1_e2e_ping", "expected": "down"},
    ]
    ok, errs = validate_proposal(dup_ids)
    check("duplicate test_id rejected", not ok and any("duplicate test_id" in err for err in errs), errs)
    nbsp = copy.deepcopy(good)
    nbsp["causes"][0]["predicted_tests"][0]["expected"] = "ok"
    ok, errs = validate_proposal(nbsp)
    check("plain ok still accepted", ok, errs)
    wrong_token = copy.deepcopy(good)
    wrong_token["causes"][0]["predicted_tests"] = [{"test_id": "T1_e2e_ping", "expected": "primary"}]
    ok, errs = validate_proposal(wrong_token)
    check("T7 token rejected on T1", not ok, errs)

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

    ts = datetime.now(timezone.utc).isoformat()
    fresh = dict(no_causes, observed_at=ts)
    check("fresh proposal bound to telemetry", not proposal_freshness_errors(fresh, {"timestamp": ts}))
    stale = dict(no_causes, observed_at="2020-01-01T00:00:00+00:00")
    check(
        "stale observed_at rejected",
        any("does not match" in err for err in proposal_freshness_errors(stale, {"timestamp": ts})),
    )
    old = "2020-01-01T00:00:00+00:00"
    aged = proposal_freshness_errors(dict(no_causes, observed_at=old), {"timestamp": old})
    check("old proposal rejected even when timestamps match", any("age" in err for err in aged), aged)
    check("missing observed_at rejected", proposal_freshness_errors(dict(no_causes), {"timestamp": ts}) == ["missing observed_at"])
    print("PASS")


if __name__ == "__main__":
    main()
