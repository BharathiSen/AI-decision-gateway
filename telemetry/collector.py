"""

collect_current_state() re-reads the actual running network every time
it is called -- it never returns a cached or static snapshot. Per-link
latency/loss are measured by pinging across that link's own two
endpoint IPs (not just the two edge hosts), so a specific degraded hop
shows up in the data instead of only the end-to-end summary.

The same snapshot also includes the static routes the diamond depends
on, interface error/drop counters, and parsed queue drops/backlog.
"""

import re
from datetime import datetime, timezone

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

try:
    from bench.mininet_util import run_in_node
    from bench.outcomes import parse_qdisc_stats
except ImportError:
    from mininet_util import run_in_node
    from outcomes import parse_qdisc_stats

ROUTERS = ["r1", "r2", "r3", "r4"]
HOSTS = ["hostA", "hostB"]
TRANSIT_LINKS = ["hostA-r1", "r1-r2", "r1-r3", "r2-r4", "r3-r4", "r4-hostB"]
ROUTE_QUERIES = [
    ("r1", "10.0.4.0/24"),
    ("r1", "10.0.1.0/24"),
    ("r2", "10.0.4.0/24"),
    ("r2", "10.0.1.0/24"),
    ("r4", "10.0.4.0/24"),
    ("r4", "10.0.1.0/24"),
]
QUEUE_NODES = ("r1", "r2", "r3")
COUNTER_NAMES = ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")


def _node_ips(node):
    ips = []
    for intf in node.intfList():
        if intf.name == "lo":
            continue
        ip = intf.IP()
        if ip:
            ips.append({"intf": intf.name, "ip": ip})
    return ips


def _parse_ping(output):
    loss_match = re.search(r"(\d+)% packet loss", output)
    rtt_match = re.search(r"= [\d.]+/([\d.]+)/", output)

    return {
        "packet_loss_percent": float(loss_match.group(1)) if loss_match else 100.0,
        "latency_ms": float(rtt_match.group(1)) if rtt_match else None,
    }


def _ping_metrics(net, src_name="hostA", dst_name="hostB", count=4):
    src = net.get(src_name)
    dst = net.get(dst_name)
    result = _parse_ping(src.cmd(f"ping -c {count} -W 1 {dst.IP()}"))
    result["connectivity"] = result["packet_loss_percent"] < 100.0
    return result


def _read_int(node, path):
    raw = run_in_node(node, f"cat {path}")
    nums = re.findall(r"\d+", raw or "")
    return int(nums[-1]) if nums else 0


def _collect_routes(net):
    routes = []
    for node_name, prefix in ROUTE_QUERIES:
        route = run_in_node(net.get(node_name), f"ip route show {prefix}").strip()
        routes.append({"node": node_name, "prefix": prefix, "route": route})
    return routes


def _collect_interface_counters(net):
    counters = {}
    for name in ROUTERS:
        node = net.get(name)
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            stats = {}
            for field in COUNTER_NAMES:
                stats[field] = _read_int(node, f"/sys/class/net/{intf.name}/statistics/{field}")
            counters[f"{name}/{intf.name}"] = stats
    return counters


def _collect_queues(net):
    queues = {}
    for name in QUEUE_NODES:
        node = net.get(name)
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            raw = run_in_node(node, f"tc -s qdisc show dev {intf.name}").strip()
            parsed = parse_qdisc_stats(raw)
            parsed["raw"] = raw
            queues[f"{name}/{intf.name}"] = parsed
    return queues


def _measure_link(link):
    """Real, measured (not configured) latency/loss for one hop -- ping
    across the link's own two endpoint IPs rather than the end nodes'
    default addresses, so this reflects that specific link even for a
    multi-homed router. Skips the ping (straight 100% loss) when the
    link is administratively down instead of waiting out a timeout.
    """
    if not (link.intf1.isUp() and link.intf2.isUp()):
        return {"latency_ms": None, "packet_loss_percent": 100.0}

    dst_ip = link.intf2.IP()
    if not dst_ip:
        return {"latency_ms": None, "packet_loss_percent": 100.0}

    return _parse_ping(link.intf1.node.cmd(f"ping -c 2 -W 1 {dst_ip}"))


def collect_current_state(net, links=None):
    """Read the live network and return a plain-dict snapshot:

    {
      "timestamp": "...",
      "nodes": [{"name", "type", "ips": [...]}],
      "links": [{"name", "up", "endpoints", "latency_ms",
                 "packet_loss_percent", "role"}],
      "routes": [{"node", "prefix", "route"}],
      "interface_counters": {"router/intf": {rx_errors, tx_errors,
                             rx_dropped, tx_dropped}},
      "queues": {"router/intf": {drops, backlog_bytes, backlog_packets, raw}},
      "metrics": {"latency_ms": ..., "packet_loss_percent": ...,
                  "connectivity": ..., "active_path": ...}
    }
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    nodes = []
    for name in HOSTS + ROUTERS:
        node = net.get(name)
        nodes.append({
            "name": name,
            "type": "router" if name in ROUTERS else "host",
            "ips": _node_ips(node),
        })

    link_states = []
    for name in TRANSIT_LINKS:
        status = ctl.link_status(net, name, links)
        status.update(_measure_link(ctl.get_link(net, name, links)))
        if name == "r1-r2":
            status["role"] = "primary"
        elif name == "r1-r3":
            status["role"] = "backup"
        link_states.append(status)

    metrics = _ping_metrics(net)
    metrics["active_path"] = ctl.get_active_path(net)

    return {
        "timestamp": timestamp,
        "nodes": nodes,
        "links": link_states,
        "routes": _collect_routes(net),
        "interface_counters": _collect_interface_counters(net),
        "queues": _collect_queues(net),
        "metrics": metrics,
    }
