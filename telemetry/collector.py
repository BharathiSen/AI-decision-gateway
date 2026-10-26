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
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    result = _parse_ping(run_in_node(src, f"ping -c {count} -W 1 {dst.IP()}"))
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


def _measure_link(link, up):
    """Real, measured (not configured) latency/loss for one hop -- ping
    across the link's own two endpoint IPs rather than the end nodes'
    default addresses, so this reflects that specific link even for a
    multi-homed router. Skips the ping (straight 100% loss) when the
    link is down instead of waiting out a timeout. `up` is the sysfs
    operstate from link_status; intf.isUp() needs ifconfig.
    """
    if not up:
        return {"latency_ms": None, "packet_loss_percent": 100.0}

    dst_ip = link.intf2.IP()
    if not dst_ip:
        return {"latency_ms": None, "packet_loss_percent": 100.0}

    return _parse_ping(run_in_node(link.intf1.node, f"ping -c 2 -W 1 {dst_ip}"))


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
        status.update(_measure_link(ctl.get_link(net, name, links), status["up"]))
        if name == "r1-r2":
            status["role"] = "primary"
        elif name == "r1-r3":
            status["role"] = "backup"
        link_states.append(status)

    routes = _collect_routes(net)
    r1_route = next(
        (item["route"] for item in routes if item["node"] == "r1" and item["prefix"] == "10.0.4.0/24"),
        "",
    )
    metrics = _ping_metrics(net)
    metrics["active_path"] = ctl.path_from_route("r1", r1_route)

    return {
        "timestamp": timestamp,
        "nodes": nodes,
        "links": link_states,
        "routes": routes,
        "interface_counters": _collect_interface_counters(net),
        "queues": _collect_queues(net),
        "metrics": metrics,
    }


def _self_check():
    """Fake network. Checks which reads go through run_in_node and sysfs."""
    module = sys.modules[__name__]
    shell_calls = []
    file_calls = []

    class Intf:
        def __init__(self, node, name, ip):
            self.node, self.name, self._ip = node, name, ip

        def IP(self):
            return self._ip

        def isUp(self):
            raise AssertionError("isUp() needs ifconfig; use sysfs operstate")

    class Node:
        def __init__(self, name):
            self.name = name
            self.intfs = []

        def intfList(self):
            return self.intfs

        def IP(self):
            return self.intfs[0].IP() if self.intfs else None

        def cmd(self, command):
            shell_calls.append((self.name, command))
            if "operstate" in command:
                return "down" if self.name == "r3" and "r3-eth1" in command else "up"
            return ""

    class Link:
        def __init__(self, a, b):
            self.intf1, self.intf2 = a, b

    names = HOSTS + ROUTERS
    nodes = {name: Node(name) for name in names}
    links = {}
    for index, link_name in enumerate(TRANSIT_LINKS):
        left, right = link_name.split("-")
        a = Intf(nodes[left], f"{left}-eth{len(nodes[left].intfs)}", f"10.9.{index}.1")
        b = Intf(nodes[right], f"{right}-eth{len(nodes[right].intfs)}", f"10.9.{index}.2")
        nodes[left].intfs.append(a)
        nodes[right].intfs.append(b)
        links[link_name] = Link(a, b)

    class Net:
        def get(self, name):
            return nodes[name]

    def fake_run(node, command):
        file_calls.append((node.name, command))
        if command.startswith("ping"):
            return "2 packets transmitted, 2 received, 0% packet loss\nrtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms"
        if command == "ip route show 10.0.4.0/24" and node.name == "r1":
            return "10.0.4.0/24 via 10.0.13.2 dev r1-eth2"
        return "0"

    saved = module.run_in_node
    module.run_in_node = fake_run
    try:
        snap = collect_current_state(Net(), links)
    finally:
        module.run_in_node = saved

    assert not any(cmd.startswith("ping") for _name, cmd in shell_calls), shell_calls
    assert not any("ip route" in cmd for _name, cmd in shell_calls), shell_calls
    assert snap["metrics"]["active_path"] == "backup"
    assert snap["metrics"]["connectivity"] is True
    by_name = {link["name"]: link for link in snap["links"]}
    assert by_name["r3-r4"]["up"] is False and by_name["r3-r4"]["packet_loss_percent"] == 100.0
    assert by_name["r1-r2"]["up"] is True and by_name["r1-r2"]["latency_ms"] == 2.0
    from bench.contracts import validate_record

    assert validate_record("telemetry", snap)[0], validate_record("telemetry", snap)[1]
    assert ctl.path_from_route("r1", "10.0.4.0/24 via 10.0.12.20 dev x") == "none"
    assert ctl.path_from_route("r4", "10.0.1.0/24 via 10.0.24.1 dev r4-eth0") == "primary"
    print("telemetry collector: PASS")


if __name__ == "__main__":
    _self_check()
