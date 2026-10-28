"""
OAM test suite T1–T10 for the diamond topology.

Each test returns a dict with raw output and parsed fields. run_all()
executes the full battery (T10 last).
"""

import re
import statistics
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

# Default episode battery (also used for new healthy calibration runs).
EPISODE_PING_COUNT = 10
EPISODE_PING_INTERVAL = 0.2
EPISODE_SPEED_SECONDS = 3
EPISODE_BATTERY = {
    "ping_count": EPISODE_PING_COUNT,
    "ping_interval": EPISODE_PING_INTERVAL,
    "speed_seconds": EPISODE_SPEED_SECONDS,
}
LEGACY_BATTERY = {"ping_count": 3, "ping_interval": 0.2, "speed_seconds": 1}

_REPLY_RTT = re.compile(r"time[=<]([\d.]+) ms")


def _parse_ping(output):
    """latency_ms is the median reply RTT. One VM scheduler stall moves the
    average of 10 pings by tens of ms; a delay fault moves every reply.
    The ping summary average is the fallback when no reply lines parse.
    """
    output = output or ""
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    rtt_match = re.search(r"= [\d.]+/([\d.]+)/", output)
    rtts = [float(value) for value in _REPLY_RTT.findall(output)]
    average = float(rtt_match.group(1)) if rtt_match else None
    return {
        "packet_loss_percent": float(loss_match.group(1)) if loss_match else 100.0,
        "latency_ms": statistics.median(rtts) if rtts else average,
        "latency_avg_ms": average,
        "rtt_ms": rtts,
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
    return {"test_id": "T1_e2e_ping", "error": None, **parsed}


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
        "error": None,
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
        "error": None,
    }


_SNMP_ERROR_FIELDS = ("InHdrErrors", "InCsumErrors", "InErrors")
_T5_SNMP_NODES = ("r1", "r2", "r3", "r4", "hostA", "hostB")
# F4 corrupts r2→r4. An e2e ping can be dropped before any checksum
# counter moves; a hop ping lands on r4's IP stack.
_T5_HOP_PROBES = (("r2", "10.0.24.2"),)


def _snmp_table_errors(text, prefixes):
    total = 0
    pending = None
    pending_header = None
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in prefixes:
            continue
        if pending == parts[0]:
            for name, value in zip(pending_header, parts[1:]):
                if name in _SNMP_ERROR_FIELDS:
                    try:
                        total += int(value)
                    except ValueError:
                        pass
            pending = None
        else:
            pending = parts[0]
            pending_header = parts[1:]
    return total


def _snmp_input_errors(node):
    """IP/ICMP header, checksum and InErrors. netem corrupt bumps these; loss does not.

    F4 corrupts on r2's egress, so r4 (the next hop) is where InHdrErrors
    land. hostB only sees the packets r4 forwarded with a good IP checksum.
    """
    snmp = _snmp_table_errors(run_in_node(node, "cat /proc/net/snmp"), ("Ip:", "Icmp:"))
    ext = _snmp_table_errors(run_in_node(node, "cat /proc/net/netstat"), ("IpExt:",))
    return snmp + ext


def T5_if_errors(net, probe_count=0):
    t0 = time.monotonic()
    stats = {}
    for name in ("r1", "r2", "r3", "r4"):
        node = net.get(name)
        intfs = [intf.name for intf in node.intfList() if intf.name != "lo"]
        if not intfs:
            continue
        script = "; ".join(
            f'echo "{intf} '
            f'$(cat /sys/class/net/{intf}/statistics/rx_errors) '
            f'$(cat /sys/class/net/{intf}/statistics/tx_errors) '
            f'$(cat /sys/class/net/{intf}/statistics/rx_crc_errors 2>/dev/null || echo 0) '
            f'$(cat /sys/class/net/{intf}/statistics/rx_frame_errors 2>/dev/null || echo 0)"'
            for intf in intfs
        )
        for line in run_in_node(node, script).splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            rx = _counter(parts[1])
            if len(parts) > 3:
                rx += _counter(parts[3])
            if len(parts) > 4:
                rx += _counter(parts[4])
            stats[f"{name}/{parts[0]}"] = {"rx_errors": rx, "tx_errors": _counter(parts[2])}
    if probe_count:
        before = {name: _snmp_input_errors(net.get(name)) for name in _T5_SNMP_NODES}
        interval = 0.02 if probe_count >= 50 else 0.05
        for src, dst_ip in _T5_HOP_PROBES:
            _hop_ping(net, src, dst_ip, count=probe_count, interval=interval)
        _hop_ping(net, "hostA", net.get("hostB").IP(), count=probe_count, interval=interval)
        time.sleep(0.2)
        for name in _T5_SNMP_NODES:
            delta = max(0, _snmp_input_errors(net.get(name)) - before[name])
            stats[f"{name}/icmp"] = {"rx_errors": delta, "tx_errors": 0}
    hop_packets = probe_count * (1 + len(_T5_HOP_PROBES)) if probe_count else 0
    probes = [_icmp_probe(hop_packets)] if hop_packets else []
    return {
        "test_id": "T5_if_errors",
        "interfaces": stats,
        "duration_seconds": time.monotonic() - t0,
        "probes": probes,
        "error": None,
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
        "error": None,
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
        "error": None,
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
        "error": None,
    }


def _icmp_in_echos(node):
    """Echo requests this node has received (Icmp InEchos), or None if unreadable."""
    header = None
    for line in (run_in_node(node, "cat /proc/net/snmp") or "").splitlines():
        parts = line.split()
        if not parts or parts[0] != "Icmp:":
            continue
        if header is None:
            header = parts[1:]
            continue
        try:
            return int(dict(zip(header, parts[1:]))["InEchos"])
        except (KeyError, ValueError):
            return None
    return None


def _one_way_ping(net, src, dst, count, interval):
    """Round-trip ping plus loss of the requests alone, from the receiver's counter.

    A round-trip ping loses a probe on either leg, so loss in one direction
    reads the same from both ends. packet_loss_percent here is requests that
    never reached dst. With no counter it stays the round-trip figure.
    """
    dst_node = net.get(dst)
    before = _icmp_in_echos(dst_node)
    result = _hop_ping(net, src, dst_node.IP(), count=count, interval=interval)
    after = _icmp_in_echos(dst_node)
    result["round_trip_loss_percent"] = result["packet_loss_percent"]
    result["one_way"] = before is not None and after is not None and count > 0
    if result["one_way"]:
        # Duplicated packets can count twice.
        arrived = min(max(after - before, 0), int(count))
        result["requests_arrived"] = arrived
        result["packet_loss_percent"] = 100.0 * (int(count) - arrived) / int(count)
    return result


def T9_oneway_loss(net, count=10, interval=0.2):
    t0 = time.monotonic()
    a2b = _one_way_ping(net, "hostA", "hostB", count, interval)
    b2a = _one_way_ping(net, "hostB", "hostA", count, interval)
    return {
        "test_id": "T9_oneway_loss",
        "hostA_to_hostB": a2b,
        "hostB_to_hostA": b2a,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe(count * 2)],
        "error": None,
    }


_SUMMARY_LINE = re.compile(
    r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+[\d.]+\s+[KMG]?Bytes\s+([\d.]+)\s*([KMG])bits/sec"
)


def _covers(start, end, seconds):
    """The summary spans this transfer. The receiver line runs past the
    requested time while the link queue drains after the sender stops."""
    span = float(end) - float(start)
    return -0.25 <= span - float(seconds) <= 0.5


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
        if not _covers(start, end, seconds):
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
        if not _covers(start, end, seconds):
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
        f"{_IPERF_TIMEOUT} 8 iperf3 -c {dest} {port_arg}-t 1 --connect-timeout 2000",
    )
    return _parse_mbit(out), out


# SIGINT, not the default SIGTERM: iperf3 writes to a file block-buffered,
# and only an interrupt makes a hung client flush what it printed. -k
# kills a client that ignores the interrupt.
_IPERF_TIMEOUT = "timeout -s INT -k 3"


def T10_speed(net, duration=3, interval=None):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    if interval:
        mbit, out = measure_speed_once(net, port=5201)
        intervals = [mbit] if mbit is not None else []
    else:
        out = run_in_node(
            net.get("hostA"),
            f"{_IPERF_TIMEOUT} 20 iperf3 -c {hostB} -t {duration} --connect-timeout 2000",
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
        "error": None,
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
    from bench import mininet_util as mu

    if "T8_tcp_vs_ping" in ids or "T10_speed" in ids:
        mu.ensure_iperf3_server(net)
    for tid in ids:
        kwargs = {}
        if ping_count and tid in ("T1_e2e_ping", "T2_hop_ping", "T9_oneway_loss"):
            kwargs["count"] = ping_count
            kwargs["interval"] = 0.05 if ping_interval is None else ping_interval
        if error_probe and tid == "T5_if_errors":
            kwargs["probe_count"] = error_probe
        if speed_seconds and tid == "T10_speed":
            kwargs["duration"] = speed_seconds
        t0 = time.monotonic()
        try:
            if tid == "T10_speed":
                # The long-lived 5201 server stops answering after a few
                # clients (and after T8's bare connect). T10 gets its own.
                server = mu.start_one_off_iperf3(net, port=5201)
                listening = mu.wait_for_listener(net, 5201)
                try:
                    row = run_test(net, tid, **kwargs)
                finally:
                    mu.finish_one_off(server)
                    mu.free_tcp_port(net, 5201)
                if isinstance(row, dict):
                    row["server_listening"] = listening
                    if row.get("mbit_per_sec") is None:
                        row["server_log"] = mu.one_off_log_tail(5201)
            else:
                row = run_test(net, tid, **kwargs)
            if isinstance(row, dict) and "error" not in row:
                row["error"] = None
            results.append(row)
        except Exception as exc:
            results.append({
                "test_id": tid,
                "duration_seconds": time.monotonic() - t0,
                "probes": [],
                "error": f"{type(exc).__name__}: {exc}",
            })
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
    drained = "[  5]   0.00-1.26   sec  1.25 MBytes  8.32 Mbits/sec                  receiver\n"
    assert _summary_mbit(drained, 1) == 8.32
    assert _summary_mbit("[  5]   0.00-2.00   sec  2.00 MBytes  8.39 Mbits/sec  receiver\n", 1) is None
    assert _icmp_probe(5, 1400)["bytes"] == 5 * (_ICMP_HEADER + 1400)

    def _boom(_net, **_kwargs):
        raise RuntimeError("probe failed")

    saved = TEST_FUNCS["T7_route_check"]
    TEST_FUNCS["T7_route_check"] = _boom
    try:
        failed = run_all(object(), test_ids=["T7_route_check"])
        assert failed["results"][0]["error"] == "RuntimeError: probe failed"
        assert failed["results"][0]["test_id"] == "T7_route_check"
    finally:
        TEST_FUNCS["T7_route_check"] = saved

    from bench import mininet_util as mu

    calls = []
    patched = {
        "ensure_iperf3_server": lambda net, *a, **k: calls.append("ensure"),
        "start_one_off_iperf3": lambda net, *a, **k: calls.append("one_off") or "server",
        "finish_one_off": lambda proc: calls.append(f"finish:{proc}"),
        "free_tcp_port": lambda net, port, *a, **k: calls.append(f"free:{port}"),
        "wait_for_listener": lambda net, port, *a, **k: calls.append(f"listen:{port}") or True,
        "one_off_log_tail": lambda port: "",
    }
    saved_mu = {name: getattr(mu, name) for name in patched}
    saved_tests = {tid: TEST_FUNCS[tid] for tid in ("T7_route_check", "T8_tcp_vs_ping", "T10_speed")}
    for name, fn in patched.items():
        setattr(mu, name, fn)
    TEST_FUNCS["T7_route_check"] = lambda _net, **_k: calls.append("T7") or {"test_id": "T7_route_check"}
    TEST_FUNCS["T8_tcp_vs_ping"] = lambda _net, **_k: calls.append("T8") or {"test_id": "T8_tcp_vs_ping"}

    def _speed_boom(_net, **_k):
        calls.append("T10")
        raise RuntimeError("iperf3 died")

    TEST_FUNCS["T10_speed"] = _speed_boom
    try:
        run_all(object(), test_ids=["T7_route_check"])
        assert calls == ["T7"], calls
        calls.clear()
        run_all(object(), test_ids=["T8_tcp_vs_ping"])
        assert calls == ["ensure", "T8"], calls
        calls.clear()
        crashed = run_all(object(), test_ids=["T10_speed", "T8_tcp_vs_ping"])
        assert calls == ["ensure", "T8", "one_off", "listen:5201", "T10", "finish:server", "free:5201"], calls
        assert crashed["results"][-1]["error"] == "RuntimeError: iperf3 died"
        calls.clear()
        TEST_FUNCS["T10_speed"] = lambda _net, **_k: {"test_id": "T10_speed", "mbit_per_sec": None}
        silent = run_all(object(), test_ids=["T10_speed"])["results"][0]
        assert silent["server_listening"] is True and silent["server_log"] == ""
        assert silent["error"] is None
    finally:
        for name, fn in saved_mu.items():
            setattr(mu, name, fn)
        TEST_FUNCS.update(saved_tests)

    tcp = summarize_costs([{
        "test_id": "T8_tcp_vs_ping",
        "duration_seconds": 0.4,
        "probes": [_icmp_probe(5), {"kind": "tcp_connect", "packets": 1}],
    }])
    assert tcp["total_additional_packets"] == 6
    assert tcp["total_additional_bytes"] == 5 * (_ICMP_HEADER + _PING_DATA_BYTES)

    replies = "".join(
        f"64 bytes from 10.0.4.2: icmp_seq={i} ttl=61 time={t} ms\n"
        for i, t in enumerate([20.1, 20.3, 240.0, 20.2, 20.4], 1)
    )
    stalled = _parse_ping(
        replies + "5 packets transmitted, 5 received, 0% packet loss, time 812ms\n"
        "rtt min/avg/max/mdev = 20.1/64.2/240.0/87.9 ms\n"
    )
    assert stalled["latency_ms"] == 20.3 and stalled["latency_avg_ms"] == 64.2, stalled
    assert len(stalled["rtt_ms"]) == 5
    summary_only = _parse_ping("1 received, 0% packet loss\nrtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
    assert summary_only["latency_ms"] == 2.0 and summary_only["rtt_ms"] == []
    lost = _parse_ping("3 packets transmitted, 0 received, 100% packet loss, time 2030ms\n")
    assert lost["latency_ms"] is None and lost["packet_loss_percent"] == 100.0
    fraction = _parse_ping("30 packets transmitted, 20 received, 33.3333% packet loss, time 1500ms\n")
    assert fraction["packet_loss_percent"] == 33.3333, fraction

    class _SnmpNode:
        def __init__(self, blob):
            self.blob = blob

    def fake_snmp(node, command, timeout=60.0):
        if command == "cat /proc/net/netstat":
            return getattr(node, "netstat", "IpExt: InCsumErrors\nIpExt: 0\n")
        assert command == "cat /proc/net/snmp", command
        return node.blob

    saved_run = globals()["run_in_node"]
    globals()["run_in_node"] = fake_snmp
    try:
        old = (
            "Ip: Forwarding DefaultTTL InReceives InHdrErrors\n"
            "Ip: 1 64 10 0\n"
            "Icmp: InMsgs InErrors InDestUnreachs InEchos\n"
            "Icmp: 4 3 0 4\n"
        )
        new = (
            "Ip: Forwarding DefaultTTL InReceives InHdrErrors InCsumErrors\n"
            "Ip: 1 64 10 2 1\n"
            "Icmp: InMsgs InErrors InCsumErrors InEchos\n"
            "Icmp: 4 3 5 4\n"
        )
        assert _snmp_input_errors(_SnmpNode(old)) == 3, _snmp_input_errors(_SnmpNode(old))
        assert _snmp_input_errors(_SnmpNode(new)) == 2 + 1 + 3 + 5, _snmp_input_errors(_SnmpNode(new))
        ext = _SnmpNode(old)
        ext.netstat = "IpExt: InNoRoutes InCsumErrors\nIpExt: 0 7\n"
        assert _snmp_input_errors(ext) == 3 + 7, _snmp_input_errors(ext)
    finally:
        globals()["run_in_node"] = saved_run

    class _T5Net:
        def __init__(self):
            self.cmds = []

        def get(self, name):
            return _T5Node(name, self)

    class _T5Node:
        def __init__(self, name, net):
            self.name = name
            self.net = net

        def IP(self):
            return "10.0.4.2"

        def intfList(self):
            return []

    def fake_t5(node, command, timeout=60.0):
        node.net.cmds.append((node.name, command))
        if "cat /proc/net/" in command:
            return "Ip: InHdrErrors\nIp: 0\nIpExt: InCsumErrors\nIpExt: 0\n"
        return "3 packets transmitted, 3 received, 0% packet loss\n"

    globals()["run_in_node"] = fake_t5
    try:
        t5net = _T5Net()
        T5_if_errors(t5net, probe_count=3)
        pinged = [(n, c) for n, c in t5net.cmds if "ping " in c]
        assert any(n == "r2" and "10.0.24.2" in c for n, c in pinged), pinged
        assert any(n == "hostA" and "10.0.4.2" in c for n, c in pinged), pinged
    finally:
        globals()["run_in_node"] = saved_run

    _check_one_way()

    import tempfile
    import threading

    class _EarlyNode:
        """cmd() returns at once; the command finishes 0.3 s later."""
        name = "hostA"
        stdout = None

        def cmd(self, line):
            out, done = re.search(r"> (\S+\.out) 2>&1; echo \$\? > (\S+\.done)", line).groups()

            def _finish():
                time.sleep(0.3)
                Path(out).write_text("5 packets transmitted, 5 received, 0% packet loss\n", encoding="utf-8")
                Path(done).write_text("0\n", encoding="utf-8")

            threading.Thread(target=_finish, daemon=True).start()
            return ""

    saved_dir = mu.RUN_DIR
    mu.RUN_DIR = Path(tempfile.gettempdir())
    try:
        late = mu.run_in_node(_EarlyNode(), "ping -c 5 10.0.1.1", timeout=5)
        assert "0% packet loss" in late, late
        assert not list(mu.RUN_DIR.glob("adg-hostA-*")), "run_in_node left files behind"
    finally:
        mu.RUN_DIR = saved_dir


def _check_one_way():
    """Loss on the B-to-A leg only: both round trips lose 20%, only B's requests do."""
    class _Host:
        def __init__(self, name, ip):
            self.name, self._ip, self.in_echos = name, ip, 100

        def IP(self):
            return self._ip

    hosts = {"hostA": _Host("hostA", "10.0.1.2"), "hostB": _Host("hostB", "10.0.4.2")}

    class _Net:
        def get(self, name):
            return hosts[name]

    def fake_run(node, command, timeout=60.0):
        if command == "cat /proc/net/snmp":
            return (
                "Ip: Forwarding DefaultTTL\nIp: 1 64\n"
                "Icmp: InMsgs InErrors InEchos OutMsgs\n"
                f"Icmp: 0 0 {node.in_echos} 0\n"
            )
        count = int(re.search(r"-c (\d+)", command).group(1))
        dst = next(h for h in hosts.values() if h.IP() == command.split()[-1])
        delivered = count if node.name == "hostA" else count * 8 // 10
        if dst.in_echos is not None:
            dst.in_echos += delivered
        received = count * 8 // 10
        return f"{count} packets transmitted, {received} received, {100 * (count - received) // count}% packet loss\n"

    saved = globals()["run_in_node"]
    globals()["run_in_node"] = fake_run
    try:
        result = T9_oneway_loss(_Net(), count=10)
        a2b, b2a = result["hostA_to_hostB"], result["hostB_to_hostA"]
        assert a2b["round_trip_loss_percent"] == 20.0 and b2a["round_trip_loss_percent"] == 20.0, result
        assert a2b["one_way"] and a2b["packet_loss_percent"] == 0.0, a2b
        assert b2a["packet_loss_percent"] == 20.0 and b2a["requests_arrived"] == 8, b2a
        hosts["hostA"].in_echos = None
        fallback = _one_way_ping(_Net(), "hostB", "hostA", 10, 0.2)
        assert not fallback["one_way"] and fallback["packet_loss_percent"] == 20.0, fallback
    finally:
        globals()["run_in_node"] = saved


if __name__ == "__main__":
    _check_costs()
    print("test cost: PASS")
