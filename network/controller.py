"""
Network control API for the emulated network.

This is the single place that knows how to read and change the state of
the running Mininet network: which path (primary/backup) is currently
active, whether a link is up or down, and its current latency/loss/
bandwidth. Fault injection, telemetry collection, and the verified-
decision controller (once built) all go through here instead of poking
Mininet objects directly.
"""

import time

# Which next-hop/interface r1 and r4 use to reach the *far* edge LAN,
# for each of the two paths. This -- not any pre-installed backup route
# -- is what "reroute traffic" means in this project: replacing these
# routes on demand.
ROUTE_TARGETS = {
    "r1": {
        "dest": "10.0.4.0/24",
        "primary": {"via": "10.0.12.2", "dev": "r1-eth1"},
        "backup": {"via": "10.0.13.2", "dev": "r1-eth2"},
    },
    "r4": {
        "dest": "10.0.1.0/24",
        "primary": {"via": "10.0.24.1", "dev": "r4-eth0"},
        "backup": {"via": "10.0.34.1", "dev": "r4-eth1"},
    },
}

BASELINE_LINK_PARAMS = dict(bw=10, delay="5ms", loss=0)


def get_link(net, link_name, links=None):
    """Resolve a link name (e.g. 'r1-r2') to its Mininet Link object."""
    if links is not None and link_name in links:
        return links[link_name]

    parts = link_name.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"{link_name!r} isn't a valid link name -- expected the form "
            f"'node1-node2', e.g. 'r1-r2'"
        )

    a, b = parts
    node_a, node_b = net.get(a), net.get(b)
    found = net.linksBetween(node_a, node_b)
    if not found:
        raise ValueError(f"No link found between {a} and {b}")
    return found[0]


def set_active_path(net, path):
    """Point r1/r4's route to the far LAN at either the primary or
    backup next hop. This is the single 'reroute traffic' primitive the
    AI agent's proposed action will call, once it has been independently
    verified.
    """
    if path not in ("primary", "backup"):
        raise ValueError("path must be 'primary' or 'backup'")

    for router_name, cfg in ROUTE_TARGETS.items():
        router = net.get(router_name)
        hop = cfg[path]
        router.cmd(f"ip route replace {cfg['dest']} via {hop['via']} dev {hop['dev']}")


def path_from_route(router_name, route):
    """primary, backup, or none for one router's route text to the far LAN."""
    cfg = ROUTE_TARGETS[router_name]
    route = route or ""
    if f"via {cfg['primary']['via']} " in route + " ":
        return "primary"
    if f"via {cfg['backup']['via']} " in route + " ":
        return "backup"
    return "none"


def get_active_path(net):
    """Inspect r1's routing table to report which path is active."""
    r1 = net.get("r1")
    route = r1.cmd(f"ip route show {ROUTE_TARGETS['r1']['dest']}")
    return path_from_route("r1", route)


def _intf_is_up(intf):
    """Read the kernel's own link-state for one interface via sysfs,
    rather than Mininet's isUp()/ifconfig() helpers -- those shell out
    to the legacy net-tools `ifconfig` binary, which recent Ubuntu
    releases don't install by default. If it's missing, isUp() silently
    reports "down" forever (no error, no crash -- it just never finds
    the string "UP" in an "ifconfig: command not found" message), which
    would make every link look permanently down in telemetry even after
    a real restore. /sys/class/net/<intf>/operstate has no such
    dependency: it's the kernel's own state, always present.
    """
    return intf_operstate(intf) == "up"


def intf_operstate(intf):
    """sysfs operstate, read in its own process when the node supports it.

    node.cmd() shares the node's interactive shell and can hand back
    output left over from an earlier command.
    """
    path = f"/sys/class/net/{intf.name}/operstate"
    if hasattr(intf.node, "pexec"):
        out, _err, _code = intf.node.pexec(["cat", path])
    else:
        out = intf.node.cmd(f"cat {path}")
    lines = (out or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def is_link_up(link):
    """True if both ends of a link report their interface as up."""
    return _intf_is_up(link.intf1) and _intf_is_up(link.intf2)


def wait_link_up(link, timeout=3.0):
    """Poll until both ends report up. The kernel's linkwatch updates
    operstate up to about a second after the interface is set up, so a
    read right after net.start() or link_up() can still say "down".
    """
    deadline = time.monotonic() + timeout
    while not is_link_up(link):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def link_down(link):
    link.intf1.node.cmd(f"ip link set dev {link.intf1.name} down")
    link.intf2.node.cmd(f"ip link set dev {link.intf2.name} down")


def link_up(link):
    link.intf1.node.cmd(f"ip link set dev {link.intf1.name} up")
    link.intf2.node.cmd(f"ip link set dev {link.intf2.name} up")
    wait_link_up(link)


def configure_link(link, bw=None, delay=None, loss=None):
    """Live-update a TCLink's bandwidth/delay/loss. Used for the
    congestion, latency, and packet-loss fault scenarios. Safe to call
    repeatedly -- Mininet's TCIntf.config() clears any existing tc
    qdisc before applying the new one, so settings don't stack.
    """
    params = {}
    if bw is not None:
        params["bw"] = bw
    if delay is not None:
        params["delay"] = delay
    if loss is not None:
        params["loss"] = loss

    link.intf1.config(**params)
    link.intf2.config(**params)


def reset_link(link):
    """Restore a link to its baseline bandwidth/delay/loss and bring it
    back up if it had been failed."""
    configure_link(link, **BASELINE_LINK_PARAMS)
    link_up(link)


def link_status(net, link_name, links=None):
    link = get_link(net, link_name, links)
    return {
        "name": link_name,
        "up": is_link_up(link),
        "endpoints": [link.intf1.name, link.intf2.name],
    }
