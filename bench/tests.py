"""
OAM test suite T1–T10 for the diamond topology.

Each test returns a dict with raw output and parsed fields. run_all()
executes the full battery (T10 last).
"""

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

try:
    from bench.mininet_util import run_in_node
except ImportError:
    from mininet_util import run_in_node

try:
    from bench.outcomes import grade_result, is_degraded, load_thresholds as _load_thresholds
    from bench.outcomes import parse_qdisc_stats
except ImportError:
    from outcomes import grade_result, is_degraded, load_thresholds as _load_thresholds
    from outcomes import parse_qdisc_stats

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

def _parse_ping(output):
    loss_match = re.search(r"(\d+)% packet loss", output)
    rtt_match = re.search(r"= [\d.]+/([\d.]+)/", output)
    return {
        "packet_loss_percent": float(loss_match.group(1)) if loss_match else 100.0,
        "latency_ms": float(rtt_match.group(1)) if rtt_match else None,
        "raw": output,
    }


def _counter(text):
    nums = re.findall(r"\d+", text or "")
    return int(nums[-1]) if nums else 0


# Linux `ping` with no -s sends 56 bytes of ICMP data. The 8-byte ICMP
# header is part of that echo request. T4 passes a different -s size.
_ICMP_HEADER = 8
_PING_DATA_BYTES = 56


def _icmp_probe(count, data_bytes=_PING_DATA_BYTES):
    """Echo requests this test transmits. Replies are not added again."""
    count = int(count)
    return {
        "kind": "icmp_echo",
        "packets": count,
        "bytes": count * (_ICMP_HEADER + int(data_bytes)),
    }


def T1_e2e_ping(net, count=10, interval=0.2):
    t0 = time.monotonic()
    out = run_in_node(
        net.get("hostA"),
        f"ping -c {count} -i {interval} -W 1 {net.get('hostB').IP()}",
    )
    parsed = _parse_ping(out)
    parsed["connectivity"] = parsed["packet_loss_percent"] < 100.0
    parsed["duration_seconds"] = time.monotonic() - t0
    parsed["probes"] = [_icmp_probe(count)]
    return {"test_id": "T1_e2e_ping", **parsed}


def _hop_ping(net, src, dst_ip, count=10, interval=0.2):
    out = run_in_node(net.get(src), f"ping -c {count} -i {interval} -W 1 {dst_ip}")
    p = _parse_ping(out)
    p["src"] = src
    p["dst_ip"] = dst_ip
    return p


def _ping_hops(net, hops, count=10, interval=0.2):
    # Sequential: Mininet node.cmd() deadlocks if called from several threads.
    return [_hop_ping(net, src, dst_ip, count=count, interval=interval) for src, _node, dst_ip in hops]


def T2_hop_ping(net, path="primary", count=10, interval=0.2):
    """Hop pings on the requested path, plus both paths for fault separation.

    Backup hops are what separate a dead backup (F12) from a healthy
    primary. `hops` stays the requested path so existing callers still
    read a single path.
    """
    t0 = time.monotonic()
    primary = _ping_hops(net, HOPS, count=count, interval=interval)
    backup = _ping_hops(net, HOPS_BACKUP, count=count, interval=interval)
    selected = primary if path != "backup" else backup
    return {
        "test_id": "T2_hop_ping",
        "path": path,
        "hops": selected,
        "hops_primary": primary,
        "hops_backup": backup,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe((len(HOPS) + len(HOPS_BACKUP)) * count)],
    }


def T4_mtu(net):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    payload = 1400
    count = 5
    out = run_in_node(
        net.get("hostA"),
        f"ping -c {count} -i 0.2 -M do -s {payload} -W 2 {hostB}",
    )
    # Match the loss number. A substring check treats "100% packet loss"
    # as success because it contains "0% packet loss".
    loss = _parse_ping(out)["packet_loss_percent"]
    return {
        "test_id": "T4_mtu",
        "df_ping_ok": loss == 0.0,
        "raw": out,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe(count, payload)],
    }


def _snmp_input_errors(node):
    """IP header errors plus ICMP checksum errors. netem corrupt bumps these; loss does not."""
    text = run_in_node(node, "cat /proc/net/snmp")
    total = 0
    pending = None
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in ("Ip:", "Icmp:"):
            continue
        if pending == parts[0]:
            header = pending_header
            for name, value in zip(header, parts[1:]):
                if name in ("InHdrErrors", "InCsumErrors"):
                    try:
                        total += int(value)
                    except ValueError:
                        pass
            pending = None
        else:
            pending = parts[0]
            pending_header = parts[1:]
    return total


def T5_if_errors(net, probe_count=0):
    t0 = time.monotonic()
    stats = {}
    for name in ("r1", "r2", "r3", "r4"):
        node = net.get(name)
        intfs = [intf.name for intf in node.intfList() if intf.name != "lo"]
        if not intfs:
            continue
        script = "; ".join(
            f'echo "{intf} $(cat /sys/class/net/{intf}/statistics/rx_errors) $(cat /sys/class/net/{intf}/statistics/tx_errors)"'
            for intf in intfs
        )
        for line in run_in_node(node, script).splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            stats[f"{name}/{parts[0]}"] = {"rx_errors": _counter(parts[1]), "tx_errors": _counter(parts[2])}
    if probe_count:
        hostB = net.get("hostB")
        before = _snmp_input_errors(hostB)
        _hop_ping(net, "hostA", hostB.IP(), count=probe_count, interval=0.05)
        delta = max(0, _snmp_input_errors(hostB) - before)
        stats["hostB/icmp"] = {"rx_errors": delta, "tx_errors": 0}
    probes = [_icmp_probe(probe_count)] if probe_count else []
    return {
        "test_id": "T5_if_errors",
        "interfaces": stats,
        "duration_seconds": time.monotonic() - t0,
        "probes": probes,
    }


def T6_queue_stats(net):
    t0 = time.monotonic()
    queues = {}
    for name in ("r1", "r2", "r3"):
        node = net.get(name)
        intfs = [intf.name for intf in node.intfList() if intf.name != "lo"]
        if not intfs:
            continue
        script = "; ".join(f'echo "@@{intf}"; tc -s qdisc show dev {intf}' for intf in intfs)
        current = None
        chunk = []

        def store(intf_name, lines, _name=name):
            if not intf_name:
                return
            raw = "\n".join(lines).strip()
            parsed = parse_qdisc_stats(raw)
            parsed["raw"] = raw
            queues[f"{_name}/{intf_name}"] = parsed

        for line in run_in_node(node, script).splitlines():
            if line.startswith("@@"):
                store(current, chunk)
                current = line[2:].strip()
                chunk = []
            else:
                chunk.append(line)
        store(current, chunk)
    return {
        "test_id": "T6_queue_stats",
        "queues": queues,
        "duration_seconds": time.monotonic() - t0,
        "probes": [],
    }


def T7_route_check(net):
    t0 = time.monotonic()
    r1 = run_in_node(net.get("r1"), "ip route show 10.0.4.0/24").strip()
    r2 = run_in_node(net.get("r2"), "ip route show 10.0.4.0/24").strip()
    r4 = run_in_node(net.get("r4"), "ip route show 10.0.1.0/24").strip()
    if "blackhole" in r1 or "blackhole" in r2:
        active = "blackhole"
    elif "10.0.12.2" in r1:
        active = "primary"
    elif "10.0.13.2" in r1:
        active = "backup"
    else:
        active = ctl.get_active_path(net)
    return {
        "test_id": "T7_route_check",
        "r1_to_dest": r1,
        "r2_to_dest": r2,
        "r4_to_src": r4,
        "active_path": active,
        "duration_seconds": time.monotonic() - t0,
        "probes": [],
    }


def T8_tcp_vs_ping(net):
    t0 = time.monotonic()
    ping = T1_e2e_ping(net, count=5, interval=0.2)
    hostB = net.get("hostB").IP()
    # Write-only connect. nc would print iperf3's binary handshake, and
    # Mininet's UTF-8 shell decoder crashes on the leading 0xff byte.
    tcp_out = run_in_node(
        net.get("hostA"),
        f"timeout 5 bash -c 'echo >/dev/tcp/{hostB}/5201' >/dev/null 2>&1; echo exit:$?",
    )
    tcp_ok = "exit:0" in tcp_out or "succeeded" in tcp_out.lower()
    return {
        "test_id": "T8_tcp_vs_ping",
        "ping_ok": ping["connectivity"],
        "tcp_ok": tcp_ok,
        "tcp_raw": tcp_out,
        "duration_seconds": time.monotonic() - t0,
        "probes": list(ping.get("probes") or []) + [{"kind": "tcp_connect", "packets": 1}],
    }


def T9_oneway_loss(net, count=10, interval=0.2):
    t0 = time.monotonic()
    a2b = _hop_ping(net, "hostA", net.get("hostB").IP(), count=count, interval=interval)
    b2a = _hop_ping(net, "hostB", net.get("hostA").IP(), count=count, interval=interval)
    return {
        "test_id": "T9_oneway_loss",
        "hostA_to_hostB": a2b,
        "hostB_to_hostA": b2a,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe(count * 2)],
    }


_SUMMARY_LINE = re.compile(
    r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+[\d.]+\s+[KMG]?Bytes\s+([\d.]+)\s*([KMG])bits/sec"
)


def _link_mbit():
    """Configured diamond link rate. A faster reading is not a completed transfer."""
    return float(ctl.BASELINE_LINK_PARAMS["bw"])


def _summary_mbit(output, seconds):
    """Bitrate from the iperf3 summary that covers this transfer.

    The last bits/sec token is not used. A short interval can say Gbits/sec
    and was parsed as 2930 Mbit/s. Anything above the configured link rate
    is rejected.
    """
    if not output:
        return None
    if "unable to send" in output or "error -" in output or "interrupt -" in output:
        return None
    scale = {"K": 1e-3, "M": 1.0, "G": 1e3}
    chosen = None
    for match in _SUMMARY_LINE.finditer(output):
        start, end, value, unit = match.groups()
        if abs((float(end) - float(start)) - float(seconds)) > 0.25:
            continue
        chosen = float(value) * scale[unit]
    if chosen is None or chosen <= 0 or chosen > _link_mbit():
        return None
    return chosen


_SUMMARY_BYTES = re.compile(
    r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+([\d.]+)\s+([KMG]?)Bytes"
)


def _summary_bytes(output, seconds):
    """Bytes on the iperf3 summary that covers this transfer. No estimate."""
    if not output or "unable to send" in output or "error -" in output or "interrupt -" in output:
        return None
    scale = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    chosen = None
    for match in _SUMMARY_BYTES.finditer(output):
        start, end, value, unit = match.groups()
        if abs((float(end) - float(start)) - float(seconds)) > 0.25:
            continue
        chosen = float(value) * scale[unit]
    if chosen is None or chosen <= 0:
        return None
    return int(chosen)


def _parse_mbit(output):
    """One-second transfer. Same rule as measure_speed_once."""
    return _summary_mbit(output, 1.0)


def accepted_speed(result):
    """A positive bitrate at or under the configured link rate."""
    if not isinstance(result, dict):
        return None
    try:
        mbit = float(result.get("mbit_per_sec"))
    except (TypeError, ValueError):
        return None
    if mbit != mbit or mbit <= 0 or mbit > _link_mbit():
        return None
    return mbit


def _parse_mbit_intervals(output):
    """One-second iperf3 lines. The summary line spans more than one second."""
    found = []
    seen = set()
    scale = {"K": 1e-3, "M": 1.0, "G": 1e3}
    pattern = re.compile(
        r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+[\d.]+\s+[KMG]?Bytes\s+([\d.]+)\s*([KMG])bits/sec"
    )
    for match in pattern.finditer(output or ""):
        start, end, value, unit = match.groups()
        if abs((float(end) - float(start)) - 1.0) > 0.25 or start in seen:
            continue
        seen.add(start)
        found.append(float(value) * scale[unit])
    return found


def measure_speed_once(net, port=5201):
    """One one-second iperf3, the same shape as a working N0 T10 sample.

    A multi-second interval report makes this iperf3 close its control
    socket with "Bad file descriptor". One second, with no interval flag, does not.
    """
    dest = net.get("hostB").IP()
    port_arg = "" if int(port) == 5201 else f"-p {int(port)} "
    out = run_in_node(
        net.get("hostA"),
        f"timeout 8 iperf3 -c {dest} {port_arg}-t 1 --connect-timeout 2000",
    )
    return _parse_mbit(out), out


def T10_speed(net, duration=3, interval=None):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    if interval:
        mbit, out = measure_speed_once(net, port=5201)
        intervals = [mbit] if mbit is not None else []
    else:
        out = run_in_node(
            net.get("hostA"),
            f"timeout 20 iperf3 -c {hostB} -t {duration} --connect-timeout 2000",
        )
        mbit = _summary_mbit(out, duration)
        intervals = []
    return {
        "test_id": "T10_speed",
        "mbit_per_sec": mbit,
        "intervals": intervals,
        "raw": out[-2000:] if len(out) > 2000 else out,
        "duration_seconds": time.monotonic() - t0,
        "probes": [{"kind": "iperf3", "bytes": _summary_bytes(out, 1 if interval else duration)}],
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


def _probe_total(probes, field):
    """Sum probes that reported this field. None if a reported value is unknown.

    A probe may omit a field it does not measure, such as a TCP connect
    with no byte size. An empty probe list is a local test: it sent nothing.
    """
    if not probes:
        return 0
    total = 0
    reported = False
    for probe in probes:
        if field not in probe:
            continue
        value = probe[field]
        if value is None or isinstance(value, bool) or not isinstance(value, int):
            return None
        reported = True
        total += value
    if not reported:
        return 0
    return total


def summarize_costs(results):
    """Sum elapsed time and reported probe traffic. Missing fields stay unknown."""
    rows = []
    total_duration = 0.0
    total_packets = 0
    total_bytes = 0
    duration_known = True
    packets_known = True
    bytes_known = True
    for result in results or []:
        result = result or {}
        duration = result.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            duration_known = False
            duration = None
        else:
            duration = float(duration)
            total_duration += duration
        probes = result.get("probes")
        if not isinstance(probes, list):
            packets = None
            byte_count = None
        else:
            packets = _probe_total(probes, "packets")
            byte_count = _probe_total(probes, "bytes")
        if packets is None:
            packets_known = False
        else:
            total_packets += packets
        if byte_count is None:
            bytes_known = False
        else:
            total_bytes += byte_count
        rows.append({
            "test_id": result.get("test_id"),
            "duration_seconds": duration,
            "additional_packets": packets,
            "additional_bytes": byte_count,
        })
    return {
        "tests": rows,
        "test_count": len(rows),
        "total_duration_seconds": total_duration if duration_known else None,
        "total_additional_packets": total_packets if packets_known else None,
        "total_additional_bytes": total_bytes if bytes_known else None,
    }


def compare_costs(full_results, selected_results):
    """Full battery versus a subset of the same result objects. No second run."""
    return {
        "full_battery": summarize_costs(full_results),
        "selected_tests": summarize_costs(selected_results),
    }


def run_all(net, test_ids=None, ping_count=None, error_probe=None, ping_interval=None, speed_seconds=None):
    """Run full OAM battery; T10 always last.

    ping_count and error_probe are optional. The fault-matrix run uses
    them so a 20% loss and a few corrupted packets are not a coin flip
    on 10 pings. The default battery is unchanged. ping_interval overrides
    the short matrix spacing; healthy calibration passes 0.2.
    """
    ids = list(test_ids or DEFAULT_ORDER)
    if "T10_speed" in ids:
        ids = [i for i in ids if i != "T10_speed"] + ["T10_speed"]
    started = datetime.now(timezone.utc).isoformat()
    results = []
    if "T10_speed" in ids:
        from bench.mininet_util import ensure_iperf3_server

        ensure_iperf3_server(net)
    for tid in ids:
        kwargs = {}
        if ping_count and tid in ("T1_e2e_ping", "T2_hop_ping", "T9_oneway_loss"):
            kwargs["count"] = ping_count
            kwargs["interval"] = 0.05 if ping_interval is None else ping_interval
        if error_probe and tid == "T5_if_errors":
            kwargs["probe_count"] = error_probe
        if speed_seconds and tid == "T10_speed":
            kwargs["duration"] = speed_seconds
        results.append(run_test(net, tid, **kwargs))
    return {
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "cost": summarize_costs(results),
    }


def load_thresholds(noise_id="N0"):
    return _load_thresholds(noise_id)


def evaluate_against_thresholds(test_results, noise_id="N0"):
    """Return list of alarms (empty if graded outcomes are healthy)."""
    th = load_thresholds(noise_id)
    alarms = []
    seen = set()
    for result in test_results:
        tid = result.get("test_id")
        outcome = grade_result(result, th)
        if tid == "T1_e2e_ping":
            if outcome == "down" or "loss" in outcome:
                alarms.append("T1_loss_high")
            if "latency" in outcome:
                alarms.append("T1_latency_high")
            continue
        if tid == "T10_speed" and outcome == "slow":
            alarms.append("T10_speed_low")
            continue
        if is_degraded(tid, outcome, th) and tid not in seen:
            alarms.append(f"{tid}_degraded")
            seen.add(tid)
    return alarms


def _check_costs():
    """Accounting only. Does not start Mininet."""
    ping = {
        "test_id": "T1_e2e_ping",
        "duration_seconds": 1.5,
        "probes": [_icmp_probe(10)],
    }
    route = {
        "test_id": "T7_route_check",
        "duration_seconds": 0.2,
        "probes": [],
    }
    speed = {
        "test_id": "T10_speed",
        "duration_seconds": 1.0,
        "probes": [{"kind": "iperf3", "bytes": None}],
    }
    known = summarize_costs([ping, route])
    assert known["total_additional_packets"] == 10
    assert known["total_additional_bytes"] == 10 * (_ICMP_HEADER + _PING_DATA_BYTES)
    assert known["total_duration_seconds"] == 1.7
    missing = summarize_costs([ping, speed])
    assert missing["total_additional_bytes"] is None
    assert missing["total_additional_packets"] == 10
    assert missing["tests"][0]["additional_packets"] == 10
    assert missing["tests"][0]["additional_bytes"] == 640
    measured = summarize_costs([ping, {
        "test_id": "T10_speed",
        "duration_seconds": 1.0,
        "probes": [{"kind": "iperf3", "bytes": 1024 ** 2}],
    }])
    assert measured["total_additional_bytes"] == 640 + 1024 ** 2
    assert measured["total_additional_packets"] == 10
    compared = compare_costs([ping, route, speed], [ping])
    assert compared["selected_tests"]["total_duration_seconds"] == 1.5
    assert compared["selected_tests"]["total_duration_seconds"] < compared["full_battery"]["total_duration_seconds"]
    assert compared["full_battery"]["total_additional_bytes"] is None
    unmeasured = compare_costs([ping, route], [ping, {"test_id": "T2_hop_ping"}])
    assert unmeasured["selected_tests"]["total_additional_bytes"] is None
    assert unmeasured["selected_tests"]["tests"][0]["additional_bytes"] == 640
    text = "[  5]   0.00-1.00   sec  1.00 MBytes  8.39 Mbits/sec\n"
    assert _summary_bytes(text, 1) == 1024 ** 2
    assert _summary_bytes("unable to send control message", 1) is None
    longer = (
        "[  5]   0.00-1.00   sec  1.00 MBytes  8.39 Mbits/sec\n"
        "[  5]   0.00-3.00   sec  3.00 MBytes  8.39 Mbits/sec\n"
    )
    assert _summary_bytes(longer, 3) == 3 * 1024 ** 2
    assert _icmp_probe(5, 1400)["bytes"] == 5 * (_ICMP_HEADER + 1400)
    tcp = summarize_costs([{
        "test_id": "T8_tcp_vs_ping",
        "duration_seconds": 0.4,
        "probes": [_icmp_probe(5), {"kind": "tcp_connect", "packets": 1}],
    }])
    assert tcp["total_additional_packets"] == 6
    assert tcp["total_additional_bytes"] == 5 * (_ICMP_HEADER + _PING_DATA_BYTES)


if __name__ == "__main__":
    _check_costs()
    print("test cost: PASS")
