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
    from bench.contracts import T6_QUEUE_NODES
    from bench.outcomes import grade_result, is_degraded, load_thresholds as _load_thresholds
    from bench.outcomes import using_test_thresholds
    from bench.outcomes import parse_qdisc_stats
except ImportError:
    from contracts import T6_QUEUE_NODES
    from outcomes import grade_result, is_degraded, load_thresholds as _load_thresholds
    from outcomes import using_test_thresholds
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


_PING_STATS = re.compile(r"(\d+) packets transmitted, (\d+) (?:packets )?received")
# Not a network measurement: the probe itself did not run as intended.
PING_TOOL_FAILURES = ("dns", "command_error")
PING_STATUSES = ("ok", "loss", "no_reply", "unreachable", "frag_needed") + PING_TOOL_FAILURES


def classify_ping(output):
    """What one ping run observed, or why it could not observe anything.

    Network outcomes: ok, loss (some replies), no_reply (every request timed
    out), unreachable (ICMP unreachable came back), frag_needed (DF packet
    too big for a hop or the local interface). Tool failures: dns (name did
    not resolve) and command_error (no output, no statistics line, bad
    arguments, or ping missing). A tool failure says nothing about the path.
    """
    text = output or ""
    low = text.lower()
    if not text.strip():
        return "command_error"
    if any(token in low for token in (
        "unknown host", "name or service not known", "temporary failure in name resolution",
        "no address associated",
    )):
        return "dns"
    if any(token in low for token in (
        "command not found", "usage: ping", "invalid argument", "bad number", "invalid value",
        "operation not permitted", "permission denied", "cannot assign requested address",
    )):
        return "command_error"
    stats = _PING_STATS.search(text)
    if "frag needed" in low or "message too long" in low:
        return "frag_needed"
    if not stats:
        return "command_error"
    sent, received = int(stats.group(1)), int(stats.group(2))
    if sent == 0:
        return "command_error"
    if received == 0:
        return "unreachable" if "unreachable" in low else "no_reply"
    return "loss" if received < sent else "ok"


def ping_error(status):
    """`error` text for a tool failure, else None."""
    return f"ping {status}" if status in PING_TOOL_FAILURES else None


def _parse_ping(output):
    """latency_ms is the median reply RTT. One VM scheduler stall moves the
    average of 10 pings by tens of ms; a delay fault moves every reply.
    The ping summary average is the fallback when no reply lines parse.

    ping_status separates a dead path from a ping that never ran. Loss stays
    100 for both, so graded outcomes match the frozen observability matrix;
    callers mark a tool failure with `error` so it is missing evidence.
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
        "ping_status": classify_ping(output),
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


def _as_run(run):
    """Normalize run_in_node return for structured callers (and string mocks)."""
    if isinstance(run, dict):
        return run
    return {
        "output": "" if run is None else str(run),
        "exit_code": None,
        "timed_out": False,
        "error": None,
    }


def _run_probe(node, command, timeout=60.0):
    """Prefer structured run_in_node; fall back for older string-only mocks."""
    try:
        return _as_run(run_in_node(node, command, timeout=timeout, structured=True))
    except TypeError:
        return _as_run(run_in_node(node, command, timeout=timeout))


def _probe_run_error(run):
    """Shell/measurement failure from structured run_in_node, else None."""
    run = _as_run(run)
    if run.get("timed_out"):
        return "timeout"
    return run.get("error")


def T1_e2e_ping(net, count=10, interval=0.2):
    t0 = time.monotonic()
    run = _run_probe(
        net.get("hostA"),
        f"ping -c {count} -i {interval} -W 1 {net.get('hostB').IP()}",
    )
    out = run.get("output") or ""
    parsed = _parse_ping(out)
    parsed["connectivity"] = parsed["packet_loss_percent"] < 100.0
    parsed["duration_seconds"] = time.monotonic() - t0
    parsed["probes"] = [_icmp_probe(count)]
    error = _probe_run_error(run) or ping_error(parsed["ping_status"])
    return {"test_id": "T1_e2e_ping", "error": error, **parsed}


def _hop_ping(net, src, dst_ip, count=10, interval=0.2):
    run = _run_probe(net.get(src), f"ping -c {count} -i {interval} -W 1 {dst_ip}")
    out = run.get("output") or ""
    p = _parse_ping(out)
    p["src"] = src
    p["dst_ip"] = dst_ip
    p["error"] = _probe_run_error(run) or ping_error(p["ping_status"])
    return p


def _ping_hops(net, hops, count=10, interval=0.2):
    # Sequential: Mininet node.cmd() deadlocks if called from several threads.
    return [_hop_ping(net, src, dst_ip, count=count, interval=interval) for src, _node, dst_ip in hops]


def T2_hop_ping(net, path="primary", count=10, interval=0.2):
    """Hop pings on the requested path, plus both paths for fault separation.

    Backup hops are what separate a dead backup (F12) from a healthy
    primary. `hops` stays the requested path so existing callers still
    read a single path.

    Any hop with a measurement/infrastructure `error` (timeout, read
    failure, empty output, command failure, or ping tool failure) makes
    the whole T2 row missing evidence — partial hop text is not graded.
    """
    t0 = time.monotonic()
    primary = _ping_hops(net, HOPS, count=count, interval=interval)
    backup = _ping_hops(net, HOPS_BACKUP, count=count, interval=interval)
    selected = primary if path != "backup" else backup
    failed = sorted({
        f"{hop.get('src')}->{hop.get('dst_ip')}: {hop.get('error')}"
        for hop in primary + backup
        if hop.get("error")
    })
    return {
        "test_id": "T2_hop_ping",
        "path": path,
        "hops": selected,
        "hops_primary": primary,
        "hops_backup": backup,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe((len(HOPS) + len(HOPS_BACKUP)) * count)],
        "error": ("ping tool failure: " + "; ".join(failed)) if failed else None,
    }


def T4_mtu(net):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    payload = 1400
    count = 5
    run = _run_probe(
        net.get("hostA"),
        f"ping -c {count} -i 0.2 -M do -s {payload} -W 2 {hostB}",
    )
    out = run.get("output") or ""
    # Match the loss number. A substring check treats "100% packet loss"
    # as success because it contains "0% packet loss".
    parsed = _parse_ping(out)
    loss = parsed["packet_loss_percent"]
    status = parsed["ping_status"]
    # df_ok/mtu_fail stay the graded tokens the frozen matrix was built with.
    # ping_status says why: frag_needed is MTU evidence; no_reply/unreachable
    # is a dead path that also fails a DF ping; a tool failure is no evidence.
    return {
        "test_id": "T4_mtu",
        "df_ping_ok": loss == 0.0,
        "ping_status": status,
        "mtu_evidence": status == "frag_needed",
        "raw": out,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe(count, payload)],
        "error": _probe_run_error(run) or ping_error(status),
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

    Returns (count, error_or_None). Tool failures are not zeroed into a clean reading.
    """
    snmp_run = _run_probe(node, "cat /proc/net/snmp")
    err = _probe_run_error(snmp_run)
    if err:
        return None, f"snmp: {err}"
    snmp_text = snmp_run.get("output") or ""
    if not snmp_text.strip():
        return None, "snmp: empty_output"
    snmp = _snmp_table_errors(snmp_text, ("Ip:", "Icmp:"))
    ext_run = _run_probe(node, "cat /proc/net/netstat")
    ext_err = _probe_run_error(ext_run)
    if ext_err:
        return None, f"netstat: {ext_err}"
    ext_text = ext_run.get("output") or ""
    ext = _snmp_table_errors(ext_text, ("IpExt:",)) if ext_text.strip() else 0
    return snmp + ext, None


def T5_if_errors(net, probe_count=0):
    """Interface error counters. Tool failures are explicit `error`, never clean."""
    t0 = time.monotonic()
    stats = {}
    errors = []
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
        run = _run_probe(node, script)
        err = _probe_run_error(run)
        if err:
            errors.append(f"{name}: {err}")
            continue
        out = run.get("output") or ""
        if not out.strip():
            errors.append(f"{name}: empty_output")
            continue
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            rx = _counter(parts[1])
            if len(parts) > 3:
                rx += _counter(parts[3])
            if len(parts) > 4:
                rx += _counter(parts[4])
            stats[f"{name}/{parts[0]}"] = {"rx_errors": rx, "tx_errors": _counter(parts[2])}
    if probe_count and not errors:
        before = {}
        for name in _T5_SNMP_NODES:
            count, err = _snmp_input_errors(net.get(name))
            if err:
                errors.append(f"{name}/{err}")
                continue
            before[name] = count
        if not errors:
            interval = 0.02 if probe_count >= 50 else 0.05
            for src, dst_ip in _T5_HOP_PROBES:
                hop = _hop_ping(net, src, dst_ip, count=probe_count, interval=interval)
                if hop.get("error"):
                    errors.append(f"{src}->{dst_ip}: {hop['error']}")
            end = _hop_ping(net, "hostA", net.get("hostB").IP(), count=probe_count, interval=interval)
            if end.get("error"):
                errors.append(f"hostA->hostB: {end['error']}")
            time.sleep(0.2)
            if not errors:
                for name in _T5_SNMP_NODES:
                    after, err = _snmp_input_errors(net.get(name))
                    if err:
                        errors.append(f"{name}/{err}")
                        continue
                    delta = max(0, after - before[name])
                    stats[f"{name}/icmp"] = {"rx_errors": delta, "tx_errors": 0}
    hop_packets = probe_count * (1 + len(_T5_HOP_PROBES)) if probe_count else 0
    probes = [_icmp_probe(hop_packets)] if hop_packets else []
    return {
        "test_id": "T5_if_errors",
        "interfaces": stats,
        "duration_seconds": time.monotonic() - t0,
        "probes": probes,
        "error": ("; ".join(errors)) if errors else None,
    }


def T6_queue_stats(net):
    """Queue stats for r1/r2/r3 only. r4 is not measured.

    Structured probe failures are `error`; successful empty/clear qdiscs grade.
    """
    t0 = time.monotonic()
    queues = {}
    errors = []
    for name in T6_QUEUE_NODES:
        node = net.get(name)
        intfs = [intf.name for intf in node.intfList() if intf.name != "lo"]
        if not intfs:
            continue
        script = "; ".join(f'echo "@@{intf}"; tc -s qdisc show dev {intf}' for intf in intfs)
        run = _run_probe(node, script)
        err = _probe_run_error(run)
        if err:
            errors.append(f"{name}: {err}")
            continue
        out = run.get("output") or ""
        if not out.strip():
            errors.append(f"{name}: empty_output")
            continue
        current = None
        chunk = []

        def store(intf_name, lines, _name=name):
            if not intf_name:
                return
            raw = "\n".join(lines).strip()
            parsed = parse_qdisc_stats(raw)
            parsed["raw"] = raw
            queues[f"{_name}/{intf_name}"] = parsed

        for line in out.splitlines():
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
        "error": ("; ".join(errors)) if errors else None,
    }


def T7_route_check(net):
    """Forward/return routes. Probe/tool failures are `error`, not a path grade."""
    t0 = time.monotonic()
    specs = (
        ("r1_to_dest", "r1", "ip route show 10.0.4.0/24"),
        ("r2_to_dest", "r2", "ip route show 10.0.4.0/24"),
        ("r4_to_src", "r4", "ip route show 10.0.1.0/24"),
    )
    routes = {}
    errors = []
    for key, node_name, command in specs:
        run = _run_probe(net.get(node_name), command)
        err = _probe_run_error(run)
        if err:
            errors.append(f"{node_name}: {err}")
            routes[key] = ""
            continue
        text = (run.get("output") or "").strip()
        routes[key] = text
    r1 = routes.get("r1_to_dest") or ""
    r2 = routes.get("r2_to_dest") or ""
    r4 = routes.get("r4_to_src") or ""
    active = None
    if not errors:
        if "blackhole" in r1 or "blackhole" in r2:
            active = "blackhole"
        elif "10.0.12.2" in r1:
            active = "primary"
        elif "10.0.13.2" in r1:
            active = "backup"
        else:
            try:
                active = ctl.get_active_path(net)
            except Exception as exc:
                errors.append(f"active_path: {type(exc).__name__}: {exc}")
                active = None
    return {
        "test_id": "T7_route_check",
        "r1_to_dest": r1,
        "r2_to_dest": r2,
        "r4_to_src": r4,
        "active_path": active,
        "duration_seconds": time.monotonic() - t0,
        "probes": [],
        "error": ("; ".join(errors)) if errors else None,
    }


def T8_tcp_vs_ping(net):
    t0 = time.monotonic()
    ping = T1_e2e_ping(net, count=5, interval=0.2)
    hostB = net.get("hostB").IP()
    # D2: listener absence is infrastructure, not a TCP/network fault grade.
    from bench import mininet_util as mu

    listener_up = None
    try:
        listener_up = bool(mu.wait_for_listener(net, 5201, host="hostB", timeout=0.5))
    except Exception as exc:
        listener_up = None
        listener_error = f"{type(exc).__name__}: {exc}"
    else:
        listener_error = None
    # Write-only connect. nc would print iperf3's binary handshake, and
    # Mininet's UTF-8 shell decoder crashes on the leading 0xff byte.
    tcp = _run_probe(
        net.get("hostA"),
        f"timeout 5 bash -c 'echo >/dev/tcp/{hostB}/5201' >/dev/null 2>&1; echo exit:$?",
    )
    tcp_out = tcp.get("output") or ""
    has_exit = "exit:" in tcp_out
    tcp_ok = "exit:0" in tcp_out or "succeeded" in tcp_out.lower()
    error = ping.get("error")
    if error is None:
        error = _probe_run_error(tcp)
    # Completed probe with no exit token and no usable text is not a TCP fault.
    if error is None and not has_exit and not tcp_out.strip():
        error = "empty_output"
    elif error is None and not has_exit and tcp_out.strip() and "succeeded" not in tcp_out.lower():
        error = "tcp_probe_unparseable"
    # Known infra: no listener (or listener check failed) when connect also fails.
    if error is None and not tcp_ok:
        if listener_up is False:
            error = "listener_unavailable"
        elif listener_error:
            error = f"listener_check_failed: {listener_error}"
    return {
        "test_id": "T8_tcp_vs_ping",
        "ping_ok": False if error else ping["connectivity"],
        "tcp_ok": False if error else tcp_ok,
        "tcp_raw": tcp_out,
        "listener_up": listener_up,
        "duration_seconds": time.monotonic() - t0,
        "probes": list(ping.get("probes") or []) + [{"kind": "tcp_connect", "packets": 1}],
        "error": error,
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
    failed = []
    for label, row in (("a2b", a2b), ("b2a", b2a)):
        err = row.get("error")
        if err:
            failed.append(f"{label}: {err}")
    return {
        "test_id": "T9_oneway_loss",
        "hostA_to_hostB": a2b,
        "hostB_to_hostA": b2a,
        "duration_seconds": time.monotonic() - t0,
        "probes": [_icmp_probe(count * 2)],
        "error": ("ping tool failure: " + "; ".join(failed)) if failed else None,
    }


_SUMMARY_LINE = re.compile(
    r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+[\d.]+\s+[KMG]?Bytes\s+([\d.]+)\s*([KMG])bits/sec"
)


_RECEIVER_LINE = re.compile(
    r"(\d+\.\d+)-(\d+\.\d+)\s+sec\s+[\d.]+\s+[KMG]?Bytes\s+([\d.]+)\s*([KMG])bits/sec\s+receiver\b"
)
# How long after the requested duration the receiver summary may end while
# queued data drains (episodes: median +0.16 s, max +0.55 s observed). Below
# 1 s so a 2-second run is never read as a 1-second probe.
RECEIVER_DRAIN_SECONDS = 0.75


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
    receiver = [m for m in _RECEIVER_LINE.finditer(output)]
    if receiver:
        # The receiver line is delivered throughput. The sender line counts
        # bytes still in queues, so it is never a substitute: a v3b F12 run
        # read 12.2 Mbit sender / 7.67 Mbit receiver and was graded slow.
        start, end, value, unit = receiver[-1].groups()
        span = float(end) - float(start)
        if not -0.25 <= span - float(seconds) <= RECEIVER_DRAIN_SECONDS:
            return None
        chosen = float(value) * scale[unit]
    else:
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


def _iperf_infra_error(out, mbit, run_meta=None):
    """Infra/tool failure for T10. A real slow transfer keeps error=None.

    Timeout or structured run error is always measurement failure, even when
    a partial bitrate was parsed from incomplete output.
    """
    if run_meta and run_meta.get("timed_out"):
        return "timeout"
    if run_meta and run_meta.get("error"):
        return run_meta["error"]
    text = out or ""
    low = text.lower()
    if not text.strip():
        return "empty_output"
    if "connection refused" in low or "unable to connect" in low or "no route to host" in low:
        return "iperf_connect_failed"
    if "unable to send" in low or "error -" in low:
        return "iperf_error"
    if "interrupt -" in low and mbit is None:
        return "iperf_interrupted"
    if mbit is None:
        return "iperf_unparseable"
    return None


def measure_speed_once(net, port=5201):
    """One one-second iperf3, the same shape as a working N0 T10 sample.

    A multi-second interval report makes this iperf3 close its control
    socket with "Bad file descriptor". One second, with no interval flag, does not.
    """
    dest = net.get("hostB").IP()
    port_arg = "" if int(port) == 5201 else f"-p {int(port)} "
    run = _run_probe(
        net.get("hostA"),
        f"{_IPERF_TIMEOUT} 8 iperf3 -c {dest} {port_arg}-t 1 --connect-timeout 2000",
    )
    out = run.get("output") or ""
    return _parse_mbit(out), out, run


# SIGINT, not the default SIGTERM: iperf3 writes to a file block-buffered,
# and only an interrupt makes a hung client flush what it printed. -k
# kills a client that ignores the interrupt.
_IPERF_TIMEOUT = "timeout -s INT -k 3"


def T10_speed(net, duration=3, interval=None):
    t0 = time.monotonic()
    hostB = net.get("hostB").IP()
    run_meta = None
    if interval:
        mbit, out, run_meta = measure_speed_once(net, port=5201)
        intervals = [mbit] if mbit is not None else []
    else:
        run_meta = _run_probe(
            net.get("hostA"),
            f"{_IPERF_TIMEOUT} 20 iperf3 -c {hostB} -t {duration} --connect-timeout 2000",
        )
        out = run_meta.get("output") or ""
        mbit = _summary_mbit(out, duration)
        intervals = []
    return {
        "test_id": "T10_speed",
        "mbit_per_sec": mbit,
        "intervals": intervals,
        "raw": out[-2000:] if len(out) > 2000 else out,
        "duration_seconds": time.monotonic() - t0,
        "probes": [{"kind": "iperf3", "bytes": _summary_bytes(out, 1 if interval else duration)}],
        "error": _iperf_infra_error(out, mbit, run_meta),
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
        if outcome is None:
            continue
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
    assert T6_QUEUE_NODES == ("r1", "r2", "r3")
    assert "r4" not in T6_QUEUE_NODES
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
        TEST_FUNCS["T10_speed"] = lambda _net, **_k: {
            "test_id": "T10_speed",
            "mbit_per_sec": None,
            "error": "empty_output",
        }
        silent = run_all(object(), test_ids=["T10_speed"])["results"][0]
        assert silent["server_listening"] is True and silent["server_log"] == ""
        # Missing speed reading is infra/missing evidence, not a graded slow.
        assert silent["error"] == "empty_output"
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

    def fake_snmp(node, command, timeout=60.0, structured=False):
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
        old_count, old_err = _snmp_input_errors(_SnmpNode(old))
        assert old_err is None and old_count == 3, (old_count, old_err)
        new_count, new_err = _snmp_input_errors(_SnmpNode(new))
        assert new_err is None and new_count == 2 + 1 + 3 + 5, (new_count, new_err)
        ext = _SnmpNode(old)
        ext.netstat = "IpExt: InNoRoutes InCsumErrors\nIpExt: 0 7\n"
        ext_count, ext_err = _snmp_input_errors(ext)
        assert ext_err is None and ext_count == 3 + 7, (ext_count, ext_err)
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

    def fake_t5(node, command, timeout=60.0, structured=False):
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
            out, pid, done = _parse_run_in_node_paths(line)
            if pid:
                Path(pid).write_text("4242\n", encoding="utf-8")

            def _finish():
                time.sleep(0.3)
                Path(out).write_text("5 packets transmitted, 5 received, 0% packet loss\n", encoding="utf-8")
                Path(done).write_text("0\n", encoding="utf-8")

            threading.Thread(target=_finish, daemon=True).start()
            return ""

    class _TimeoutNode:
        """Never completes; exposes popen so the timed-out PID can be killed."""
        name = "hostA"
        stdout = None
        kills = []

        def cmd(self, line):
            out, pid, done = _parse_run_in_node_paths(line)
            Path(pid).write_text("7777\n", encoding="utf-8")
            Path(out).write_text("partial\n", encoding="utf-8")
            # Leave done missing so run_in_node times out.
            return ""

        def popen(self, args, stdout=None, stderr=None):
            self.kills.append(list(args))

            class _Proc:
                def wait(self, timeout=None):
                    return 0

            return _Proc()

    saved_dir = mu.RUN_DIR
    mu.RUN_DIR = Path(tempfile.gettempdir())
    try:
        late = mu.run_in_node(_EarlyNode(), "ping -c 5 10.0.1.1", timeout=5)
        assert "0% packet loss" in late, late
        assert not list(mu.RUN_DIR.glob("adg-hostA-*")), "run_in_node left files behind"
        # D3: timeout kills only the recorded PID (TERM then KILL), not a shell-wide pattern.
        timed = mu.run_in_node(_TimeoutNode(), "sleep 30", timeout=0.2, structured=True)
        assert timed["timed_out"] and timed["error"] == "timeout", timed
        assert timed.get("killed_pid") == 7777, timed
        assert ["kill", "-TERM", "--", "7777"] in _TimeoutNode.kills, _TimeoutNode.kills
        assert ["kill", "-KILL", "--", "7777"] in _TimeoutNode.kills, _TimeoutNode.kills
        assert not any("pkill" in arg for args in _TimeoutNode.kills for arg in args), _TimeoutNode.kills
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

    def fake_run(node, command, timeout=60.0, structured=False):
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
        assert result["error"] is None, result
        hosts["hostA"].in_echos = None
        fallback = _one_way_ping(_Net(), "hostB", "hostA", 10, 0.2)
        assert not fallback["one_way"] and fallback["packet_loss_percent"] == 20.0, fallback
    finally:
        globals()["run_in_node"] = saved


def _check_ping_classification():
    """T4/T1 ping outcomes: real outputs from iputils ping on Linux."""
    ok = (
        "PING 10.0.4.2 (10.0.4.2) 56(84) bytes of data.\n"
        "64 bytes from 10.0.4.2: icmp_seq=1 ttl=62 time=40.1 ms\n"
        "--- 10.0.4.2 ping statistics ---\n"
        "5 packets transmitted, 5 received, 0% packet loss, time 801ms\n"
        "rtt min/avg/max/mdev = 40.1/41.0/42.0/0.6 ms\n"
    )
    lossy = ok.replace("5 received, 0% packet loss", "3 received, 40% packet loss")
    timeout = (
        "PING 10.0.4.2 (10.0.4.2) 1400(1428) bytes of data.\n\n--- 10.0.4.2 ping statistics ---\n"
        "5 packets transmitted, 0 received, 100% packet loss, time 4093ms\n"
    )
    unreachable = (
        "From 10.0.1.1 icmp_seq=1 Destination Net Unreachable\n--- 10.0.4.2 ping statistics ---\n"
        "5 packets transmitted, 0 received, +5 errors, 100% packet loss, time 4005ms\n"
    )
    frag = (
        "From 10.0.12.2 icmp_seq=1 Frag needed and DF set (mtu = 1000)\n--- 10.0.4.2 ping statistics ---\n"
        "5 packets transmitted, 0 received, +5 errors, 100% packet loss, time 4005ms\n"
    )
    local_mtu = "ping: local error: message too long, mtu=1000\n"
    dns = "ping: hostB: Name or service not known\n"
    missing = "bash: line 1: ping: command not found\n"
    cases = {
        ok: "ok", lossy: "loss", timeout: "no_reply", unreachable: "unreachable",
        frag: "frag_needed", local_mtu: "frag_needed", dns: "dns", missing: "command_error",
        "": "command_error", "garbage without statistics\n": "command_error",
    }
    for text, want in cases.items():
        got = classify_ping(text)
        assert got == want, (want, got, text)
        parsed = _parse_ping(text)
        assert parsed["ping_status"] == want
    # Loss stays 100 for a dead path and for a ping that never ran, so graded
    # tokens match the frozen matrix; only `error` separates them.
    assert _parse_ping(timeout)["packet_loss_percent"] == 100.0
    assert _parse_ping(missing)["packet_loss_percent"] == 100.0
    assert ping_error("no_reply") is None and ping_error("unreachable") is None
    assert ping_error("frag_needed") is None and ping_error("dns") == "ping dns"
    assert ping_error("command_error") == "ping command_error"

    class _Node:
        def __init__(self, ip):
            self._ip = ip
            self.name = "n"

        def IP(self):
            return self._ip

    class _Net:
        def get(self, name):
            return _Node("10.0.4.2")

    saved = globals()["run_in_node"]
    try:
        for text, df_ok, mtu_evidence, error in (
            (ok, True, False, None),
            (frag, False, True, None),
            (timeout, False, False, None),
            (dns, False, False, "ping dns"),
            (missing, False, False, "ping command_error"),
        ):
            globals()["run_in_node"] = (
                lambda node, cmd, timeout=60.0, structured=False, _t=text: _t
            )
            row = T4_mtu(_Net())
            assert row["df_ping_ok"] is df_ok and row["mtu_evidence"] is mtu_evidence, row
            assert row["error"] == error, row
            t1 = T1_e2e_ping(_Net())
            assert t1["error"] == error and t1["ping_status"] == classify_ping(text)
    finally:
        globals()["run_in_node"] = saved
    print("ping classification: PASS")


def _check_t10_receiver():
    """The receiver summary is the measurement; the sender line never replaces it."""
    v3b_f12 = (
        "[ ID] Interval           Transfer     Bitrate         Retr\n"
        "[  5]   0.00-3.00   sec  4.38 MBytes  12.2 Mbits/sec    0            sender\n"
        "[  5]   0.00-3.55   sec  3.25 MBytes  7.67 Mbits/sec                  receiver\n\niperf Done.\n"
    )
    assert _summary_mbit(v3b_f12, 3) == 7.67, _summary_mbit(v3b_f12, 3)
    normal = v3b_f12.replace("0.00-3.55", "0.00-3.16").replace("7.67", "8.38")
    assert _summary_mbit(normal, 3) == 8.38
    # Receiver far past the drain allowance: no reading, and no sender fallback.
    stalled = v3b_f12.replace("0.00-3.55", "0.00-5.20")
    assert _summary_mbit(stalled, 3) is None
    # A receiver line above the link rate is still refused.
    fast = v3b_f12.replace("7.67 Mbits/sec", "15.0 Mbits/sec")
    assert _summary_mbit(fast, 3) is None
    # Failures stay failures.
    assert _summary_mbit(v3b_f12 + "iperf3: error - unable to send control message\n", 3) is None
    unreachable = "iperf3: error - unable to connect to server - server may have stopped running: Network is unreachable\n"
    assert _summary_mbit(unreachable, 3) is None
    # Infra/tool failures are tagged as error (missing evidence), not graded slow.
    assert _iperf_infra_error("", None) == "empty_output"
    assert _iperf_infra_error(unreachable, None) == "iperf_connect_failed"
    assert _iperf_infra_error("iperf3: error - unable to send control message\n", None) == "iperf_error"
    assert _iperf_infra_error("interrupt -\n", None) == "iperf_interrupted"
    assert _iperf_infra_error(v3b_f12, 7.67) is None
    assert _iperf_infra_error("no summary", None, {"timed_out": True}) == "timeout"
    # Timeout/partial bitrate is measurement failure, never graded slow.
    assert _iperf_infra_error(v3b_f12, 7.67, {"timed_out": True}) == "timeout"
    assert _iperf_infra_error(v3b_f12, 7.67, {"error": "timeout", "timed_out": True}) == "timeout"
    assert _iperf_infra_error("no receiver summary\n", None) == "iperf_unparseable"
    print("T10 receiver parsing: PASS")


def _parse_run_in_node_paths(line):
    """out/pid/done paths from the current run_in_node wrapper command."""
    match = re.search(
        r"> (\S+\.out) 2>&1 & echo \$! > (\S+\.pid); wait \$\(cat \2\); echo \$\? > (\S+\.done)",
        line,
    )
    if match:
        return match.groups()
    match = re.search(r"> (\S+\.out) 2>&1; echo \$\? > (\S+\.done)", line)
    out, done = match.groups()
    return out, None, done


def _check_h2_measurement_errors():
    """Network outcomes must not share grading with measurement failures."""
    import tempfile
    from pathlib import Path

    import bench.mininet_util as mu

    class _FileNode:
        name = "hostA"
        stdout = None

        def __init__(self, writer):
            self._writer = writer

        def cmd(self, line):
            out, pid, done = _parse_run_in_node_paths(line)
            if pid:
                Path(pid).write_text("4242\n", encoding="utf-8")
            self._writer(out, done)
            return ""

        def popen(self, args, stdout=None, stderr=None):
            class _Proc:
                def wait(self, timeout=None):
                    return 0

            return _Proc()

    saved_dir = mu.RUN_DIR
    mu.RUN_DIR = Path(tempfile.gettempdir())
    try:
        def write_ok(out, done, text="", code="0"):
            Path(out).write_text(text, encoding="utf-8")
            Path(done).write_text(f"{code}\n", encoding="utf-8")

        empty = mu.run_in_node(_FileNode(lambda o, d: write_ok(o, d, "", "0")), "true", structured=True)
        assert empty["error"] == "empty_output" and empty["exit_code"] == 0, empty

        fail = mu.run_in_node(_FileNode(lambda o, d: write_ok(o, d, "", "1")), "false", structured=True)
        assert fail["error"] == "command_failure" and fail["exit_code"] == 1, fail

        # Nonzero exit with usable ping text is not a shell measurement failure.
        ping_loss = (
            "3 packets transmitted, 0 received, 100% packet loss, time 2030ms\n"
        )
        loss_run = mu.run_in_node(
            _FileNode(lambda o, d: write_ok(o, d, ping_loss, "1")),
            "ping",
            structured=True,
        )
        assert loss_run["error"] is None and loss_run["exit_code"] == 1, loss_run

        def write_timeout_partial(out, done):
            Path(out).write_text(
                "[  5]   0.00-1.00   sec  1.00 MBytes  8.00 Mbits/sec    0            sender\n",
                encoding="utf-8",
            )
            # No .done → timed_out

        timed = mu.run_in_node(
            _FileNode(write_timeout_partial),
            "iperf3",
            timeout=0.2,
            structured=True,
        )
        assert timed["timed_out"] and timed["error"] == "timeout" and timed["output"], timed
        assert timed.get("killed_pid") == 4242, timed

        def write_read_fail(out, done):
            Path(out).mkdir()
            Path(done).write_text("0\n", encoding="utf-8")

        read_fail = mu.run_in_node(
            _FileNode(write_read_fail),
            "true",
            structured=True,
        )
        assert read_fail["error"] == "read_failure", read_fail
        for leftover in mu.RUN_DIR.glob("adg-hostA-*.out"):
            if leftover.is_dir():
                leftover.rmdir()

        def write_bad_done(out, done):
            Path(out).write_text("", encoding="utf-8")
            Path(done).write_text("not-an-int\n", encoding="utf-8")

        missing = mu.run_in_node(
            _FileNode(write_bad_done),
            "true",
            structured=True,
        )
        assert missing["error"] == "missing_completion", missing
    finally:
        for leftover in Path(tempfile.gettempdir()).glob("adg-hostA-*"):
            try:
                if leftover.is_dir():
                    leftover.rmdir()
                else:
                    leftover.unlink()
            except OSError:
                pass
        mu.RUN_DIR = saved_dir

    class _Node:
        def __init__(self, name, ip):
            self.name, self._ip = name, ip

        def IP(self):
            return self._ip

    class _Net:
        def get(self, name):
            return hosts[name]

    hosts = {
        "hostA": _Node("hostA", "10.0.1.2"),
        "hostB": _Node("hostB", "10.0.4.2"),
    }
    ok_ping = (
        "5 packets transmitted, 5 received, 0% packet loss, time 801ms\n"
        "rtt min/avg/max/mdev = 40.1/41.0/42.0/0.6 ms\n"
    )
    tool_fail = "bash: line 1: ping: command not found\n"
    lossy = "10 packets transmitted, 8 received, 20% packet loss, time 1800ms\n"

    def fake_t9(node, command, timeout=60.0, structured=False):
        if command == "cat /proc/net/snmp":
            return "Ip: Forwarding\nIp: 1\nIcmp: InEchos\nIcmp: 0\n"
        if node.name == "hostA":
            return tool_fail
        return ok_ping

    saved = globals()["run_in_node"]
    globals()["run_in_node"] = fake_t9
    try:
        t9 = T9_oneway_loss(_Net(), count=5)
        assert t9["error"], t9
        assert "a2b" in t9["error"] and "ping command_error" in t9["error"], t9
        assert t9["hostA_to_hostB"]["error"] == "ping command_error", t9["hostA_to_hostB"]
        assert t9["hostB_to_hostA"]["error"] is None, t9["hostB_to_hostA"]
    finally:
        globals()["run_in_node"] = saved

    # Asymmetric network loss is still graded network evidence, not a tool error.
    def fake_asym(node, command, timeout=60.0, structured=False):
        if command == "cat /proc/net/snmp":
            return "unreadable"
        if node.name == "hostA":
            return ok_ping
        return lossy

    globals()["run_in_node"] = fake_asym
    try:
        asym = T9_oneway_loss(_Net(), count=10)
        assert asym["error"] is None, asym
        assert asym["hostA_to_hostB"]["packet_loss_percent"] == 0.0, asym
        assert asym["hostB_to_hostA"]["packet_loss_percent"] == 20.0, asym
        assert asym["hostA_to_hostB"]["error"] is None
        assert asym["hostB_to_hostA"]["error"] is None
    finally:
        globals()["run_in_node"] = saved

    # T8: empty / timeout are measurement failures, not ping_ok_tcp_fail.
    # D2: listener down + connect fail is infrastructure, not TCP network fault.
    def fake_t8(node, command, timeout=60.0, structured=False, mode="empty"):
        if "ping " in command:
            return ok_ping
        if mode == "timeout":
            return {"output": "", "exit_code": None, "timed_out": True, "error": "timeout"}
        if mode == "empty":
            return {"output": "", "exit_code": 0, "timed_out": False, "error": "empty_output"}
        if mode == "tcp_fail":
            return {"output": "exit:1\n", "exit_code": 0, "timed_out": False, "error": None}
        return {"output": "exit:0\n", "exit_code": 0, "timed_out": False, "error": None}

    import bench.mininet_util as mu_t8

    saved_listener = mu_t8.wait_for_listener
    for mode, want_err, want_tcp, listener_up in (
        ("empty", "empty_output", False, True),
        ("timeout", "timeout", False, True),
        ("tcp_fail", None, False, True),
        ("tcp_fail", "listener_unavailable", False, False),
        ("ok", None, True, True),
    ):
        globals()["run_in_node"] = (
            lambda node, command, timeout=60.0, structured=False, _m=mode: fake_t8(
                node, command, timeout=timeout, structured=structured, mode=_m
            )
        )
        mu_t8.wait_for_listener = lambda net, port, host="hostB", timeout=0.5, _up=listener_up: _up
        try:
            row = T8_tcp_vs_ping(_Net())
            assert row["error"] == want_err, (mode, listener_up, row)
            assert row["tcp_ok"] is want_tcp, (mode, listener_up, row)
            assert row.get("listener_up") is listener_up, row
            if want_err:
                assert row["ping_ok"] is False, row
        finally:
            globals()["run_in_node"] = saved
            mu_t8.wait_for_listener = saved_listener

    # T10: timeout with a partial bitrate stays measurement failure.
    partial = (
        "[ ID] Interval           Transfer     Bitrate         Retr\n"
        "[  5]   0.00-3.00   sec  4.38 MBytes  12.2 Mbits/sec    0            sender\n"
        "[  5]   0.00-3.16   sec  3.25 MBytes  8.38 Mbits/sec                  receiver\n"
    )

    def fake_t10(node, command, timeout=60.0, structured=False):
        return {
            "output": partial,
            "exit_code": None,
            "timed_out": True,
            "error": "timeout",
        }

    globals()["run_in_node"] = fake_t10
    try:
        t10 = T10_speed(_Net(), duration=3)
        assert t10["error"] == "timeout", t10
        assert t10["mbit_per_sec"] == 8.38, t10
    finally:
        globals()["run_in_node"] = saved

    def fake_t10_connect(node, command, timeout=60.0, structured=False):
        text = "iperf3: error - unable to connect to server: Connection refused\n"
        return {"output": text, "exit_code": 1, "timed_out": False, "error": None}

    globals()["run_in_node"] = fake_t10_connect
    try:
        t10c = T10_speed(_Net(), duration=3)
        assert t10c["error"] == "iperf_connect_failed", t10c
        assert t10c["mbit_per_sec"] is None, t10c
    finally:
        globals()["run_in_node"] = saved

    # D1: T2 parent error from any hop structured measurement failure.
    t2_nodes = {
        name: _Node(name, "10.0.4.2" if name == "hostB" else "10.0.1.1")
        for name in ("hostA", "r1", "r2", "r3", "r4", "hostB")
    }

    class _T2Net:
        def get(self, name):
            return t2_nodes[name]

    ok_hop = (
        "5 packets transmitted, 5 received, 0% packet loss, time 801ms\n"
        "rtt min/avg/max/mdev = 1.0/1.0/1.0/0.0 ms\n"
    )
    loss_hop = "10 packets transmitted, 8 received, 20% packet loss, time 1800ms\n"
    partial_timeout = (
        "PING 10.0.24.2 (10.0.24.2) 56(84) bytes of data.\n"
        "64 bytes from 10.0.24.2: icmp_seq=1 ttl=63 time=1.0 ms\n"
        "3 packets transmitted, 1 received, 66% packet loss, time 2000ms\n"
    )

    def _structured(output="", exit_code=0, timed_out=False, error=None):
        return {
            "output": output,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "error": error,
        }

    def fake_t2(node, command, timeout=60.0, structured=False, by_dst=None):
        dst = command.split()[-1]
        meta = (by_dst or {}).get(dst)
        if meta is None:
            meta = _structured(ok_hop)
        if structured:
            return meta
        return meta.get("output") or ""

    def run_t2(by_dst):
        globals()["run_in_node"] = (
            lambda node, command, timeout=60.0, structured=False, _b=by_dst: fake_t2(
                node, command, timeout=timeout, structured=structured, by_dst=_b
            )
        )
        try:
            return T2_hop_ping(_T2Net(), count=5)
        finally:
            globals()["run_in_node"] = saved

    # Timeout with partial stats on one hop → T2 error, not graded loss.
    t2_timeout = run_t2({
        "10.0.24.2": _structured(partial_timeout, timed_out=True, error="timeout"),
    })
    assert t2_timeout["error"] and "timeout" in t2_timeout["error"], t2_timeout
    assert any(h.get("error") == "timeout" for h in t2_timeout["hops_primary"]), t2_timeout

    t2_read = run_t2({"10.0.12.2": _structured("", error="read_failure")})
    assert t2_read["error"] and "read_failure" in t2_read["error"], t2_read

    t2_empty = run_t2({
        "10.0.13.2": _structured("", exit_code=0, error="empty_output"),
    })
    assert t2_empty["error"] and "empty_output" in t2_empty["error"], t2_empty

    t2_cmd = run_t2({
        "10.0.34.2": _structured("", exit_code=1, error="command_failure"),
    })
    assert t2_cmd["error"] and "command_failure" in t2_cmd["error"], t2_cmd

    # One ok path + one errored hop still fails the whole T2 row.
    mixed = run_t2({
        "10.0.4.2": _structured("", exit_code=0, error="empty_output"),
    })
    assert mixed["error"] and "empty_output" in mixed["error"], mixed
    assert any(
        h.get("error") is None and h.get("ping_status") == "ok"
        for h in mixed["hops_primary"]
    ), mixed

    healthy_t2 = run_t2({})
    assert healthy_t2["error"] is None, healthy_t2
    assert all(h.get("error") is None for h in healthy_t2["hops_primary"] + healthy_t2["hops_backup"])

    lossy_t2 = run_t2({"10.0.24.2": _structured(loss_hop)})
    assert lossy_t2["error"] is None, lossy_t2
    bad = next(h for h in lossy_t2["hops_primary"] if h.get("dst_ip") == "10.0.24.2")
    assert bad.get("packet_loss_percent") == 20.0 and bad.get("error") is None, bad

    print("H2 measurement errors: PASS")


def _check_e1_t567_measurement_errors():
    """E1: T5/T6/T7 propagate structured tool failures; never invent clean grades."""
    from bench.outcomes import grade_result

    class _Intf:
        def __init__(self, name):
            self.name = name

    class _Node:
        def __init__(self, name, intfs):
            self.name = name
            self._intfs = intfs

        def intfList(self):
            return [ _Intf(n) for n in self._intfs ]

        def IP(self):
            return "10.0.4.2"

    class _Net:
        def get(self, name):
            if name in ("r1", "r2", "r3"):
                return _Node(name, [f"{name}-eth0"])
            if name == "r4":
                return _Node(name, ["r4-eth0"])
            return _Node(name, [])

    def _structured(output="", exit_code=0, timed_out=False, error=None):
        return {
            "output": output,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "error": error,
        }

    saved = globals()["run_in_node"]

    def fail_t5(node, command, timeout=60.0, structured=False):
        meta = _structured("", timed_out=True, error="timeout")
        return meta if structured else meta["output"]

    globals()["run_in_node"] = fail_t5
    try:
        t5 = T5_if_errors(_Net())
        assert t5["error"] and "timeout" in t5["error"], t5
        assert grade_result(t5) is None, grade_result(t5)
    finally:
        globals()["run_in_node"] = saved

    def fail_t6(node, command, timeout=60.0, structured=False):
        meta = _structured("", exit_code=1, error="command_failure")
        return meta if structured else meta["output"]

    globals()["run_in_node"] = fail_t6
    try:
        t6 = T6_queue_stats(_Net())
        assert t6["error"] and "command_failure" in t6["error"], t6
        assert grade_result(t6) is None
    finally:
        globals()["run_in_node"] = saved

    def empty_t6(node, command, timeout=60.0, structured=False):
        meta = _structured("", exit_code=0, error=None)
        return meta if structured else ""

    globals()["run_in_node"] = empty_t6
    try:
        t6e = T6_queue_stats(_Net())
        assert t6e["error"] and "empty_output" in t6e["error"], t6e
        assert grade_result(t6e) is None
    finally:
        globals()["run_in_node"] = saved

    def fail_t7(node, command, timeout=60.0, structured=False):
        meta = _structured("", exit_code=0, error="read_failure")
        return meta if structured else ""

    globals()["run_in_node"] = fail_t7
    try:
        t7 = T7_route_check(_Net())
        assert t7["error"] and "read_failure" in t7["error"], t7
        assert t7.get("active_path") is None
        assert grade_result(t7) is None
    finally:
        globals()["run_in_node"] = saved

    # Legitimate empty route text (tool ok) is still a network observation path.
    def ok_empty_route(node, command, timeout=60.0, structured=False):
        meta = _structured("\n", exit_code=0, error=None)
        return meta if structured else "\n"

    globals()["run_in_node"] = ok_empty_route
    try:
        t7_ok = T7_route_check(_Net())
        # No tool error; active_path may fall through to controller.
        assert t7_ok["error"] is None or "active_path" in (t7_ok["error"] or ""), t7_ok
    finally:
        globals()["run_in_node"] = saved

    print("E1 T5/T6/T7 measurement errors: PASS")


@using_test_thresholds
def _offline_checks():
    _check_costs()
    print("test cost: PASS")
    _check_ping_classification()
    _check_t10_receiver()
    _check_h2_measurement_errors()
    _check_e1_t567_measurement_errors()


if __name__ == "__main__":
    _offline_checks()
