"""B2: full OAM battery, then fixed rules on graded outcomes.

Module name is historical. The LLM checker is B4.
"""

from bench import tests as oam
from bench.outcomes import grade_result
from gate.checks import path_links_up

from baselines.common import BACKUP_PATH, PERMITTED_ACTIONS


def _grades(test_results):
    out = {}
    for row in test_results or []:
        tid = row.get("test_id")
        if tid:
            out[tid] = grade_result(row)
    return out


def _backup_unreachable(snapshot, grades):
    up, _down = path_links_up(snapshot, "backup")
    if not up:
        return True
    t2 = grades.get("T2_hop_ping") or ""
    if "backup:" in t2:
        return "down" in t2.split("backup:", 1)[-1]
    return False


def _rule_action(grades, snapshot):
    t1 = grades.get("T1_e2e_ping")
    t4 = grades.get("T4_mtu")
    t5 = grades.get("T5_if_errors")
    t6 = grades.get("T6_queue_stats")
    t7 = grades.get("T7_route_check")
    t8 = grades.get("T8_tcp_vs_ping")
    t10 = grades.get("T10_speed")
    backup_down = _backup_unreachable(snapshot, grades)

    def or_escalate(action):
        if action == "REROUTE" and backup_down:
            return "ESCALATE"
        return action

    if t8 == "ping_ok_tcp_fail":
        return "ESCALATE"
    if t7 in ("blackhole", "return_mismatch"):
        return or_escalate("REROUTE")
    if t1 == "down" or (t1 and "loss" in t1):
        return or_escalate("REROUTE")
    if t4 == "mtu_fail":
        return or_escalate("REROUTE")
    if t6 and t6.startswith("drops"):
        return or_escalate("REROUTE")
    if t10 == "slow" and (t1 is None or t1 == "ok"):
        return or_escalate("REROUTE")
    if t5 and t5.startswith("errors"):
        return or_escalate("REROUTE")
    if t1 and "latency" in t1:
        return or_escalate("REROUTE")
    return "DO_NOTHING"


def decide(proposal, snapshot, test_results, net=None, run_battery=False):
    results = list(test_results or [])
    if run_battery and net is not None:
        bundle = oam.run_all(net)
        results = list(bundle.get("results") or results)
    grades = _grades(results)
    action = _rule_action(grades, snapshot)
    if action not in PERMITTED_ACTIONS:
        action = "ESCALATE"
    path = list(BACKUP_PATH) if action == "REROUTE" else []
    return {
        "method": "B2",
        "decision": "allow",
        "action": action,
        "path": path,
        "tests_run": [row.get("test_id") for row in results if row.get("test_id")],
        "reason": "full battery then rules",
    }
