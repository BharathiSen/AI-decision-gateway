"""
Live telemetry / topology collector.

collect_current_state() re-reads the actual running network every time
it is called -- it never returns a cached or static snapshot. This is
the minimum needed to satisfy Phase 1's success condition (the network
state can genuinely be observed from Python); Phase 2 formalizes and
extends this into the full telemetry pipeline.
"""

import re
from datetime import datetime, timezone

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

ROUTERS = ["r1", "r2", "r3", "r4"]
HOSTS = ["hostA", "hostB"]
TRANSIT_LINKS = ["hostA-r1", "r1-r2", "r1-r3", "r2-r4", "r3-r4", "r4-hostB"]


def _node_ips(node):
    ips = []
    for intf in node.intfList():
        if intf.name == "lo":
            continue
        ip = intf.IP()
        if ip:
            ips.append({"intf": intf.name, "ip": ip})
    return ips


def _ping_metrics(net, src_name="hostA", dst_name="hostB", count=4):
    src = net.get(src_name)
    dst = net.get(dst_name)
    output = src.cmd(f"ping -c {count} -W 1 {dst.IP()}")

    loss_match = re.search(r"(\d+)% packet loss", output)
    rtt_match = re.search(r"= [\d.]+/([\d.]+)/", output)

    packet_loss = float(loss_match.group(1)) if loss_match else 100.0
    latency_ms = float(rtt_match.group(1)) if rtt_match else None

    return {
        "packet_loss_percent": packet_loss,
        "latency_ms": latency_ms,
        "connectivity": packet_loss < 100.0,
    }


def collect_current_state(net, links=None):
    """Read the live network and return a plain-dict snapshot:

    {
      "timestamp": "...",
      "nodes": [...],
      "links": [...],
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
        "metrics": metrics,
    }
