"""Offline checks for OAM grading and expected-signature rules."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.outcomes import grade_result, is_degraded, parse_qdisc_stats, signature
from bench.signature_checks import load_expected, separation_errors
from bench.tests import evaluate_against_thresholds

THRESHOLDS = {
    "T1_e2e_ping": {"max_loss_percent": 5.0, "max_latency_ms": 50.0},
    "T2_hop_ping": {"max_loss_percent": 10.0, "max_latency_ms": 80.0},
    "T8_tcp_vs_ping": {"tcp_must_succeed": True},
    "T10_speed": {"min_mbit": 1.0},
}


def _hop(dst, loss=0.0, latency=10.0):
    return {"dst_ip": dst, "packet_loss_percent": loss, "latency_ms": latency}


def _t2(primary, backup):
    return {
        "test_id": "T2_hop_ping",
        "path": "primary",
        "hops": primary,
        "hops_primary": primary,
        "hops_backup": backup,
    }


def _healthy_primary():
    return [
        _hop("10.0.1.1"),
        _hop("10.0.12.2"),
        _hop("10.0.24.2"),
        _hop("10.0.4.2"),
    ]


def _healthy_backup():
    return [
        _hop("10.0.1.1"),
        _hop("10.0.13.2"),
        _hop("10.0.34.2"),
        _hop("10.0.4.2"),
    ]


def check(label, ok, detail=""):
    if not ok:
        raise AssertionError(f"{label}: {detail}")
    print(f"  PASS  {label}")


def test_grades():
    check("T1 ok", grade_result({"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 20}, THRESHOLDS) == "ok")
    check("T1 loss", grade_result({"test_id": "T1_e2e_ping", "packet_loss_percent": 15, "latency_ms": 20}, THRESHOLDS) == "loss")
    check("T1 latency", grade_result({"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 120}, THRESHOLDS) == "latency")
    check("T1 down", grade_result({"test_id": "T1_e2e_ping", "packet_loss_percent": 100, "latency_ms": None}, THRESHOLDS) == "down")

    primary = _healthy_primary()
    primary[2] = _hop("10.0.24.2", loss=20)
    last_mile = _healthy_primary()
    last_mile[3] = _hop("10.0.4.2", loss=20)
    backup = _healthy_backup()
    f3 = grade_result(_t2(primary, backup), THRESHOLDS)
    f11 = grade_result(_t2(last_mile, backup), THRESHOLDS)
    check("T2 separates r2-r4 loss from last mile", f3 != f11 and "r2-r4=loss" in f3 and "r4-hostB=loss" in f11, f"{f3} vs {f11}")

    down = _healthy_primary()
    down[2] = _hop("10.0.24.2", loss=100, latency=None)
    backup_down = _healthy_backup()
    backup_down[2] = _hop("10.0.34.2", loss=100, latency=None)
    f1 = grade_result(_t2(down, _healthy_backup()), THRESHOLDS)
    f12 = grade_result(_t2(_healthy_primary(), backup_down), THRESHOLDS)
    check("T2 link down", "r2-r4=down" in f1, f1)
    check("T2 backup down", "r3-r4=down" in f12 and "primary:" in f12, f12)

    check("T4 df_ok", grade_result({"test_id": "T4_mtu", "df_ping_ok": True}, THRESHOLDS) == "df_ok")
    check("T4 mtu_fail", grade_result({"test_id": "T4_mtu", "df_ping_ok": False}, THRESHOLDS) == "mtu_fail")

    check(
        "T5 clean",
        grade_result({"test_id": "T5_if_errors", "interfaces": {"r4/r4-eth0": {"rx_errors": 0, "tx_errors": 0}}}, THRESHOLDS) == "clean",
    )
    check(
        "T5 errors",
        grade_result({"test_id": "T5_if_errors", "interfaces": {"r4/r4-eth0": {"rx_errors": 3, "tx_errors": 0}}}, THRESHOLDS) == "errors:r4/r4-eth0",
    )

    parsed = parse_qdisc_stats("Sent 10 bytes 1 pkt (dropped 4, overlimits 0 requeues 0)\n backlog 120b 2p requeues 0")
    check("qdisc parse", parsed["drops"] == 4 and parsed["backlog_bytes"] == 120 and parsed["backlog_packets"] == 2, str(parsed))
    check(
        "T6 drops",
        grade_result({"test_id": "T6_queue_stats", "queues": {"r2/r2-eth1": "dropped 4\n backlog 0b 0p"}}, THRESHOLDS) == "drops:r2/r2-eth1",
    )
    check(
        "T6 clear",
        grade_result({"test_id": "T6_queue_stats", "queues": {"r2/r2-eth1": {"drops": 0, "backlog_bytes": 0, "backlog_packets": 0}}}, THRESHOLDS) == "clear",
    )

    check(
        "T7 blackhole",
        grade_result({"test_id": "T7_route_check", "r1_to_dest": "via 10.0.12.2", "r2_to_dest": "blackhole 10.0.4.0/24", "r4_to_src": "via 10.0.24.1", "active_path": "primary"}, THRESHOLDS) == "blackhole",
    )
    check(
        "T7 return mismatch",
        grade_result({"test_id": "T7_route_check", "r1_to_dest": "via 10.0.12.2", "r2_to_dest": "via 10.0.24.2", "r4_to_src": "via 10.0.34.1", "active_path": "primary"}, THRESHOLDS) == "return_mismatch",
    )
    check(
        "T7 primary",
        grade_result({"test_id": "T7_route_check", "r1_to_dest": "via 10.0.12.2", "r4_to_src": "via 10.0.24.1", "active_path": "primary"}, THRESHOLDS) == "primary",
    )

    check("T8 tcp fail", grade_result({"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": False}, THRESHOLDS) == "ping_ok_tcp_fail")
    check("T8 both ok", grade_result({"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True}, THRESHOLDS) == "both_ok")
    check("T9 a2b", grade_result({"test_id": "T9_oneway_loss", "hostA_to_hostB": {"packet_loss_percent": 20}, "hostB_to_hostA": {"packet_loss_percent": 0}}, THRESHOLDS) == "a2b_loss")
    check("T9 b2a", grade_result({"test_id": "T9_oneway_loss", "hostA_to_hostB": {"packet_loss_percent": 0}, "hostB_to_hostA": {"packet_loss_percent": 20}}, THRESHOLDS) == "b2a_loss")
    check("T10 slow", grade_result({"test_id": "T10_speed", "mbit_per_sec": 0.4}, THRESHOLDS) == "slow")
    check("T10 ok", grade_result({"test_id": "T10_speed", "mbit_per_sec": 8}, THRESHOLDS) == "ok")

    alarms = evaluate_against_thresholds([
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 20},
        {"test_id": "T10_speed", "mbit_per_sec": 8},
    ])
    check("healthy battery has no alarms", alarms == [], str(alarms))
    alarms = evaluate_against_thresholds([
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 40, "latency_ms": 20},
        {"test_id": "T5_if_errors", "interfaces": {"r4/r4-eth0": {"rx_errors": 2, "tx_errors": 0}}},
    ])
    check("loss and errors alarm", "T1_loss_high" in alarms and "T5_if_errors_degraded" in alarms, str(alarms))
    check("blackhole is degraded", is_degraded("T7_route_check", "blackhole"))
    check("primary route is not degraded", not is_degraded("T7_route_check", "primary"))


def _row(fault, test, outcome):
    return {"fault_id": fault, "test_id": test, "outcome": outcome}


def test_expected_catalog_covers_faults():
    root = Path(__file__).resolve().parent.parent
    with (root / "config" / "faults.yaml").open(encoding="utf-8") as f:
        catalog = [item["id"] for item in yaml.safe_load(f)["faults"]]
    spec = load_expected()
    missing = [fid for fid in catalog if fid not in spec["faults"]]
    check("expected signatures cover the catalog", not missing, str(missing))
    pair_keys = {(p["fault_a"], p["fault_b"]) for p in spec["pairs"]}
    required_pairs = {
        ("F3", "F11"),
        ("F1", "F9"),
        ("H0", "F10"),
        ("F12", "F1"),
        ("F5", "F3"),
        ("F4", "F3"),
        ("F6", "F3"),
        ("U2", "F3"),
        ("U1", "H0"),
    }
    check("required pairs are listed", required_pairs <= pair_keys, str(required_pairs - pair_keys))


def _synthetic_matrix():
    """Graded outcomes that satisfy the required separators."""
    tests = [
        "T1_e2e_ping",
        "T2_hop_ping",
        "T4_mtu",
        "T5_if_errors",
        "T6_queue_stats",
        "T7_route_check",
        "T8_tcp_vs_ping",
        "T9_oneway_loss",
        "T10_speed",
    ]
    healthy = {
        "T1_e2e_ping": "ok",
        "T2_hop_ping": "primary:hostA-r1=ok,r1-r2=ok,r2-r4=ok,r4-hostB=ok;backup:hostA-r1=ok,r1-r3=ok,r3-r4=ok,r4-hostB=ok",
        "T4_mtu": "df_ok",
        "T5_if_errors": "clean",
        "T6_queue_stats": "clear",
        "T7_route_check": "primary",
        "T8_tcp_vs_ping": "both_ok",
        "T9_oneway_loss": "symmetric",
        "T10_speed": "ok",
    }
    overrides = {
        "F1": {"T1_e2e_ping": "down", "T2_hop_ping": healthy["T2_hop_ping"].replace("r2-r4=ok", "r2-r4=down", 1)},
        "F2": {"T1_e2e_ping": "down", "T2_hop_ping": healthy["T2_hop_ping"].replace("r1-r2=ok", "r1-r2=down", 1)},
        "F3": {"T1_e2e_ping": "loss", "T2_hop_ping": healthy["T2_hop_ping"].replace("r2-r4=ok", "r2-r4=loss", 1), "T9_oneway_loss": "a2b_loss"},
        "F4": {"T5_if_errors": "errors:r4/r4-eth0"},
        "F5": {"T1_e2e_ping": "latency", "T2_hop_ping": healthy["T2_hop_ping"].replace("r2-r4=ok", "r2-r4=delay", 1)},
        "F6": {"T6_queue_stats": "drops:r2/r2-eth1", "T10_speed": "slow"},
        "F7": {"T6_queue_stats": "drops:r1/r1-eth1", "T10_speed": "slow"},
        "F9": {"T7_route_check": "blackhole", "T1_e2e_ping": "down"},
        "F10": {"T8_tcp_vs_ping": "ping_ok_tcp_fail"},
        "F11": {"T1_e2e_ping": "loss", "T2_hop_ping": healthy["T2_hop_ping"].replace("r4-hostB=ok", "r4-hostB=loss", 1)},
        "F12": {"T2_hop_ping": healthy["T2_hop_ping"].replace("r3-r4=ok", "r3-r4=down", 1)},
        "U1": {"T4_mtu": "mtu_fail"},
        "U2": {"T9_oneway_loss": "b2a_loss", "T1_e2e_ping": "loss"},
        "U5": {"T7_route_check": "return_mismatch"},
    }
    spec = load_expected()
    rows = []
    for fid in spec["faults"]:
        outcomes = dict(healthy)
        outcomes.update(overrides.get(fid, {}))
        for tid in tests:
            rows.append(_row(fid, tid, outcomes[tid]))
    return rows


def test_signature_rules():
    spec = load_expected()
    rows = _synthetic_matrix()
    errors = separation_errors(rows, spec)
    check("synthetic matrix satisfies signature rules", not errors, "\n".join(errors))

    collapsed = [dict(r) for r in rows]
    for row in collapsed:
        if row["fault_id"] in ("F3", "F11") and row["test_id"] == "T2_hop_ping":
            row["outcome"] = "primary:r2-r4=ok"
    errors = separation_errors(collapsed, spec)
    check("identical T2 for F3 and F11 is rejected", any("F3 vs F11" in err for err in errors), str(errors))

    graded = signature([
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 12},
        {"test_id": "T7_route_check", "r1_to_dest": "via 10.0.12.2", "r4_to_src": "via 10.0.24.1", "active_path": "primary"},
    ], THRESHOLDS)
    check("signature map", graded["T1_e2e_ping"] == "ok" and graded["T7_route_check"] == "primary", str(graded))


def main():
    test_grades()
    test_expected_catalog_covers_faults()
    test_signature_rules()
    print("PASS")


if __name__ == "__main__":
    main()
