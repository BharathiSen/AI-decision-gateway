"""Grade OAM results into compact outcomes that can separate faults.

Outcomes are strings the observability matrix and the consistency check
both use. Thresholds come from config/thresholds.yaml (per traffic level),
or from results/calibration_report.json proposed limits when that file
exists (see calibration.grading_from_report in thresholds.yaml).
"""

import json
import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "config"
_REPORT = _ROOT / "results" / "calibration_report.json"

HOP_NAMES = {
    "10.0.1.1": "hostA-r1",
    "10.0.12.2": "r1-r2",
    "10.0.24.2": "r2-r4",
    "10.0.4.2": "r4-hostB",
    "10.0.13.2": "r1-r3",
    "10.0.34.2": "r3-r4",
}

_HEALTHY = {
    "T1_e2e_ping": "ok",
    "T4_mtu": "df_ok",
    "T5_if_errors": "clean",
    "T6_queue_stats": "clear",
    "T7_route_check": "primary",
    "T8_tcp_vs_ping": "both_ok",
    "T9_oneway_loss": "symmetric",
    "T10_speed": "ok",
}


def _thresholds_from_report(noise_id):
    if not _REPORT.is_file():
        return None
    try:
        doc = json.loads(_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    level = (doc.get("levels") or {}).get(noise_id) or {}
    proposed = level.get("proposed_thresholds")
    if isinstance(proposed, dict) and proposed:
        return proposed
    return None


def load_thresholds(noise_id="N0"):
    with (_CONFIG / "thresholds.yaml").open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    calibration = doc.get("calibration") or {}
    if calibration.get("grading_from_report", True):
        from_report = _thresholds_from_report(noise_id)
        if from_report is not None:
            return from_report
    return doc["by_noise_level"].get(noise_id, doc["by_noise_level"]["N0"])


def parse_qdisc_stats(text):
    """Pull drop and backlog counters out of `tc -s qdisc` text."""
    drops = 0
    for match in re.finditer(r"dropped (\d+)", text or ""):
        drops += int(match.group(1))
    backlog_bytes = 0
    backlog_packets = 0
    backlog = re.search(r"backlog (\d+)b(?:\s+(\d+)p)?", text or "")
    if backlog:
        backlog_bytes = int(backlog.group(1))
        if backlog.group(2):
            backlog_packets = int(backlog.group(2))
    return {
        "drops": drops,
        "backlog_bytes": backlog_bytes,
        "backlog_packets": backlog_packets,
        # netem "loss N%" counts those discards as dropped. That is not a queue.
        "loss_rule": bool(re.search(r"\bloss\b", text or "")),
        "rate_rule": bool(re.search(r"\brate\b|\btbf\b", text or "")),
    }


def _hop_outcome(hop, thresholds):
    limits = thresholds.get("T2_hop_ping", {})
    max_loss = limits.get("max_loss_percent", 10.0)
    max_lat = limits.get("max_latency_ms", 80.0)
    loss = hop.get("packet_loss_percent")
    if loss is None:
        loss = 100.0
    if loss >= 100.0:
        return "down"
    if loss > max_loss:
        return "loss"
    latency = hop.get("latency_ms")
    if latency is not None and latency > max_lat:
        return "delay"
    return "ok"


def _grade_hops(hops, thresholds):
    parts = []
    for hop in hops or []:
        label = HOP_NAMES.get(hop.get("dst_ip"), hop.get("dst_ip") or "?")
        parts.append(f"{label}={_hop_outcome(hop, thresholds)}")
    return ",".join(parts)


def _grade_t1(result, thresholds):
    limits = thresholds.get("T1_e2e_ping", {})
    max_loss = limits.get("max_loss_percent", 5.0)
    max_lat = limits.get("max_latency_ms", 50.0)
    loss = result.get("packet_loss_percent")
    if loss is None:
        loss = 100.0
    latency = result.get("latency_ms")
    if loss >= 100.0:
        return "down"
    flags = []
    if loss > max_loss:
        flags.append("loss")
    if latency is not None and latency > max_lat:
        flags.append("latency")
    return "+".join(flags) if flags else "ok"


def _grade_t2(result, thresholds):
    primary = result.get("hops_primary")
    backup = result.get("hops_backup")
    if primary is None and result.get("path", "primary") == "primary":
        primary = result.get("hops")
    if backup is None and result.get("path") == "backup":
        backup = result.get("hops")
    chunks = []
    if primary is not None:
        chunks.append("primary:" + _grade_hops(primary, thresholds))
    if backup is not None:
        chunks.append("backup:" + _grade_hops(backup, thresholds))
    if chunks:
        return ";".join(chunks)
    path = result.get("path") or "primary"
    return f"{path}:{_grade_hops(result.get('hops') or [], thresholds)}"


def _grade_t5(result):
    bad = []
    for name, stats in sorted((result.get("interfaces") or {}).items()):
        if not isinstance(stats, dict):
            continue
        if stats.get("rx_errors", 0) or stats.get("tx_errors", 0):
            bad.append(name)
    if not bad:
        return "clean"
    return "errors:" + "|".join(bad)


def _queue_maps(result):
    """Normalize T6 queues to {name: {drops, backlog_bytes, backlog_packets}}."""
    queues = result.get("queues") or {}
    parsed = {}
    for name, value in queues.items():
        if isinstance(value, dict) and "drops" in value:
            parsed[name] = value
        else:
            parsed[name] = parse_qdisc_stats(value if isinstance(value, str) else "")
    return parsed


def _grade_t6(result):
    drops = []
    backlog = []
    for name, stats in sorted(_queue_maps(result).items()):
        # Packet-loss netem reports drops too. Count them only when the
        # qdisc is a rate limit or has no loss rule.
        intentional_loss = stats.get("loss_rule") and not stats.get("rate_rule")
        if stats.get("drops", 0) and not intentional_loss:
            drops.append(name)
        elif stats.get("backlog_bytes", 0) or stats.get("backlog_packets", 0):
            backlog.append(name)
    if drops:
        return "drops:" + "|".join(drops)
    if backlog:
        return "backlog:" + "|".join(backlog)
    return "clear"


def _grade_t7(result):
    r1 = result.get("r1_to_dest") or ""
    r2 = result.get("r2_to_dest") or ""
    r4 = result.get("r4_to_src") or ""
    blob = f"{r1} {r2} {r4}"
    if "blackhole" in blob:
        return "blackhole"
    forward_primary = "10.0.12.2" in r1
    forward_backup = "10.0.13.2" in r1
    return_primary = "10.0.24.1" in r4
    return_backup = "10.0.34.1" in r4
    if (forward_primary and return_backup) or (forward_backup and return_primary):
        return "return_mismatch"
    if forward_backup or result.get("active_path") == "backup":
        return "backup"
    if forward_primary or result.get("active_path") == "primary":
        return "primary"
    return str(result.get("active_path") or "none")


def _grade_t8(result):
    ping_ok = bool(result.get("ping_ok"))
    tcp_ok = bool(result.get("tcp_ok"))
    if ping_ok and tcp_ok:
        return "both_ok"
    if ping_ok and not tcp_ok:
        return "ping_ok_tcp_fail"
    if tcp_ok:
        return "ping_fail_tcp_ok"
    return "both_fail"


def _grade_t9(result, thresholds):
    max_loss = thresholds.get("T1_e2e_ping", {}).get("max_loss_percent", 5.0)
    a2b = (result.get("hostA_to_hostB") or {}).get("packet_loss_percent", 0.0)
    b2a = (result.get("hostB_to_hostA") or {}).get("packet_loss_percent", 0.0)
    margin = 5.0
    a_bad = a2b > max_loss
    b_bad = b2a > max_loss
    if a_bad and a2b > b2a + margin:
        return "a2b_loss"
    if b_bad and b2a > a2b + margin:
        return "b2a_loss"
    return "symmetric"


def _grade_t10(result, thresholds):
    minimum = thresholds.get("T10_speed", {}).get("min_mbit", 1.0)
    mbit = result.get("mbit_per_sec")
    if mbit is None or mbit < minimum:
        return "slow"
    return "ok"


def grade_result(result, thresholds=None, noise_id="N0"):
    """Return one compact outcome string for a single OAM result dict."""
    if thresholds is None:
        thresholds = load_thresholds(noise_id)
    test_id = result.get("test_id")
    if test_id == "T1_e2e_ping":
        return _grade_t1(result, thresholds)
    if test_id == "T2_hop_ping":
        return _grade_t2(result, thresholds)
    if test_id == "T4_mtu":
        return "df_ok" if result.get("df_ping_ok") else "mtu_fail"
    if test_id == "T5_if_errors":
        return _grade_t5(result)
    if test_id == "T6_queue_stats":
        return _grade_t6(result)
    if test_id == "T7_route_check":
        return _grade_t7(result)
    if test_id == "T8_tcp_vs_ping":
        return _grade_t8(result)
    if test_id == "T9_oneway_loss":
        return _grade_t9(result, thresholds)
    if test_id == "T10_speed":
        return _grade_t10(result, thresholds)
    return "unknown"


def signature(test_results, thresholds=None, noise_id="N0"):
    """Map test_id -> outcome for a battery of OAM results."""
    if thresholds is None:
        thresholds = load_thresholds(noise_id)
    return {r["test_id"]: grade_result(r, thresholds) for r in test_results}


def healthy_outcome(test_id):
    return _HEALTHY.get(test_id, "ok")


def is_degraded(test_id, outcome, thresholds=None):
    """True when an outcome is not the healthy reading for that test.

    T2 is degraded when any hop is loss, down, or delay. T7 backup is a
    path choice, not a fault; blackhole and return_mismatch are faults.
    T8 may skip the TCP alarm when the traffic level says TCP need not succeed.
    """
    if test_id == "T2_hop_ping":
        return any(token in (outcome or "") for token in ("=loss", "=down", "=delay"))
    if test_id == "T7_route_check":
        return outcome in ("blackhole", "return_mismatch")
    if test_id == "T8_tcp_vs_ping":
        if thresholds and not thresholds.get("T8_tcp_vs_ping", {}).get("tcp_must_succeed", True):
            return outcome in ("both_fail", "ping_fail_tcp_ok")
        return outcome != "both_ok"
    if test_id == "T6_queue_stats":
        return outcome != "clear"
    return outcome != healthy_outcome(test_id)
