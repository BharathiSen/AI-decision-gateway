"""
OAM test suite T1–T10 for the diamond topology.

Each test returns a dict with raw output and parsed fields. run_all()
executes the full battery (T10 last).
"""

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

HOPS = [
    ("hostA", "r1", "10.0.1.1"),
    ("r1", "r2", "10.0.12.2"),
    ("r2", "r4", "10.0.24.2"),
    ("r4", "hostB", "10.0.4.2"),
]
HOPS_BACKUP = [
    ("hostA", "r1", "10.0.1.1"),
    ("r1", "r3", "10.0.13.2"),
    ("r3", "r4", "10.0.34.2"),
    ("r4", "hostB", "10.0.4.2"),
]

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def _parse_ping(output):
    loss_match = re.search(r"(\d+)% packet loss", output)
    rtt_match = re.search(r"= [\d.]+/([\d.]+)/", output)
    return {
        "packet_loss_percent": float(loss_match.group(1)) if loss_match else 100.0,
        "latency_ms": float(rtt_match.group(1)) if rtt_match else None,
        "raw": output,
    }


def T1_e2e_ping(net, count=10, interval=0.2):
    t0 = time.monotonic()
    out = net.get("hostA").cmd(
        f"ping -c {count} -i {interval} -W 1 {net.get('hostB').IP()}"
    )
    parsed = _parse_ping(out)
    parsed["connectivity"] = parsed["packet_loss_percent"] < 100.0
    parsed["duration_seconds"] = time.monotonic() - t0
    return {"test_id": "T1_e2e_ping", **parsed}


def _hop_ping(net, src, dst_ip, count=10, interval=0.2):
    out = net.get(src).cmd(f"ping -c {count} -i {interval} -W 1 {dst_ip}")
    p = _parse_ping(out)
    p["src"] = src
    p["dst_ip"] = dst_ip
    return p


def T2_hop_ping(net, path="primary"):
    t0 = time.monotonic()
    hops = HOPS if path == "primary" else HOPS_BACKUP
    # Sequential: Mininet node.cmd() deadlocks if called from several threads.
    results = [_hop_ping(net, src, dst_ip) for src, _node, dst_ip in hops]
    return {
        "test_id": "T2_hop_ping",
        "path": path,
        "hops": results,
        "duration_seconds": time.monotonic() - t0,
    }


def T4_mtu(net):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    out = net.get("hostA").cmd(
        f"ping -c 3 -M do -s 1400 -W 2 {hostB}"
    )
    ok = "0% packet loss" in out or "1 received" in out
    return {
        "test_id": "T4_mtu",
        "df_ping_ok": ok,
        "raw": out,
        "duration_seconds": time.monotonic() - t0,
    }


def T5_if_errors(net):
    t0 = time.monotonic()
    stats = {}
    for name in ("r1", "r2", "r3", "r4"):
        node = net.get(name)
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            rx = node.cmd(f"cat /sys/class/net/{intf.name}/statistics/rx_errors").strip()
            tx = node.cmd(f"cat /sys/class/net/{intf.name}/statistics/tx_errors").strip()
            stats[f"{name}/{intf.name}"] = {"rx_errors": int(rx or 0), "tx_errors": int(tx or 0)}
    return {"test_id": "T5_if_errors", "interfaces": stats, "duration_seconds": time.monotonic() - t0}


def T6_queue_stats(net):
    t0 = time.monotonic()
    queues = {}
    for name in ("r2", "r3"):
        node = net.get(name)
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            qdisc = node.cmd(f"tc -s qdisc show dev {intf.name}")
            queues[f"{name}/{intf.name}"] = qdisc.strip()
    return {"test_id": "T6_queue_stats", "queues": queues, "duration_seconds": time.monotonic() - t0}


def T7_route_check(net):
    t0 = time.monotonic()
    r1 = net.get("r1").cmd("ip route show 10.0.4.0/24").strip()
    r4 = net.get("r4").cmd("ip route show 10.0.1.0/24").strip()
    active = ctl.get_active_path(net)
    return {
        "test_id": "T7_route_check",
        "r1_to_dest": r1,
        "r4_to_src": r4,
        "active_path": active,
        "duration_seconds": time.monotonic() - t0,
    }


def T8_tcp_vs_ping(net):
    t0 = time.monotonic()
    ping = T1_e2e_ping(net, count=5, interval=0.2)
    hostB = net.get("hostB").IP()
    tcp_out = net.get("hostA").cmd(
        f"timeout 5 bash -c 'echo | nc -w 3 {hostB} 5201' 2>&1; echo exit:$?"
    )
    tcp_ok = "exit:0" in tcp_out or "succeeded" in tcp_out.lower()
    return {
        "test_id": "T8_tcp_vs_ping",
        "ping_ok": ping["connectivity"],
        "tcp_ok": tcp_ok,
        "tcp_raw": tcp_out,
        "duration_seconds": time.monotonic() - t0,
    }


def T9_oneway_loss(net):
    t0 = time.monotonic()
    a2b = _hop_ping(net, "hostA", net.get("hostB").IP(), count=10)
    b2a = _hop_ping(net, "hostB", net.get("hostA").IP(), count=10)
    return {
        "test_id": "T9_oneway_loss",
        "hostA_to_hostB": a2b,
        "hostB_to_hostA": b2a,
        "duration_seconds": time.monotonic() - t0,
    }


def T10_speed(net, duration=3):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    out = net.get("hostA").cmd(
        f"timeout 20 iperf3 -c {hostB} -t {duration} --connect-timeout 2000"
    )
    mbit = None
    m = re.search(r"([\d.]+)\s+Mbits/sec", out)
    if m:
        mbit = float(m.group(1))
    return {
        "test_id": "T10_speed",
        "mbit_per_sec": mbit,
        "raw": out[-2000:] if len(out) > 2000 else out,
        "duration_seconds": time.monotonic() - t0,
    }


TEST_FUNCS = {
    "T1_e2e_ping": T1_e2e_ping,
    "T2_hop_ping": T2_hop_ping,
    "T4_mtu": T4_mtu,
    "T5_if_errors": T5_if_errors,
    "T6_queue_stats": T6_queue_stats,
    "T7_route_check": T7_route_check,
    "T8_tcp_vs_ping": T8_tcp_vs_ping,
    "T9_oneway_loss": T9_oneway_loss,
    "T10_speed": T10_speed,
}

DEFAULT_ORDER = [
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


def run_test(net, test_id, **kwargs):
    if test_id not in TEST_FUNCS:
        raise ValueError(f"Unknown test {test_id!r}")
    return TEST_FUNCS[test_id](net, **kwargs)


def run_all(net, test_ids=None):
    """Run full OAM battery; T10 always last."""
    ids = list(test_ids or DEFAULT_ORDER)
    if "T10_speed" in ids:
        ids = [i for i in ids if i != "T10_speed"] + ["T10_speed"]
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for tid in ids:
        results.append(run_test(net, tid))
    return {"started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(), "results": results}


def load_thresholds(noise_id="N0"):
    with (_CONFIG / "thresholds.yaml").open() as f:
        doc = yaml.safe_load(f)
    return doc["by_noise_level"].get(noise_id, doc["by_noise_level"]["N0"])


def evaluate_against_thresholds(test_results, noise_id="N0"):
    """Return list of alarms (empty if all within healthy thresholds)."""
    th = load_thresholds(noise_id)
    alarms = []
    by_id = {r["test_id"]: r for r in test_results}
    t1 = by_id.get("T1_e2e_ping", {})
    if t1:
        lim = th.get("T1_e2e_ping", {})
        if t1.get("packet_loss_percent", 100) > lim.get("max_loss_percent", 100):
            alarms.append("T1_loss_high")
        lat = t1.get("latency_ms")
        if lat is not None and lat > lim.get("max_latency_ms", 1e9):
            alarms.append("T1_latency_high")
    t10 = by_id.get("T10_speed", {})
    if t10 and t10.get("mbit_per_sec") is not None:
        if t10["mbit_per_sec"] < th.get("T10_speed", {}).get("min_mbit", 0):
            alarms.append("T10_speed_low")
    return alarms
