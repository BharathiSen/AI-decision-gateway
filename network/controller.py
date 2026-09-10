"""
Network control API for the Phase 1 emulated network.

This is the single place that knows how to read and change the state of
the running Mininet network: which path (primary/backup) is currently
active, whether a link is up or down, and its current latency/loss/
bandwidth. Fault injection, telemetry collection, and (in later phases)
the verified-decision controller all go through here instead of poking
Mininet objects directly.
"""

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

    a, b = link_name.split("-")
    node_a, node_b = net.get(a), net.get(b)
    found = net.linksBetween(node_a, node_b)
    if not found:
        raise ValueError(f"No link found between {a} and {b}")
    return found[0]


def set_active_path(net, path):
    """Point r1/r4's route to the far LAN at either the primary or
    backup next hop. This is the single 'reroute traffic' primitive
    later phases (the AI agent's proposed action, executed only after
    verification) will call.
    """
    if path not in ("primary", "backup"):
        raise ValueError("path must be 'primary' or 'backup'")

    for router_name, cfg in ROUTE_TARGETS.items():
        router = net.get(router_name)
        hop = cfg[path]
        router.cmd(f"ip route replace {cfg['dest']} via {hop['via']} dev {hop['dev']}")


def get_active_path(net):
    """Inspect r1's routing table to report which path is active."""
    r1 = net.get("r1")
    route = r1.cmd(f"ip route show {ROUTE_TARGETS['r1']['dest']}")

    if ROUTE_TARGETS["r1"]["primary"]["via"] in route:
        return "primary"
    if ROUTE_TARGETS["r1"]["backup"]["via"] in route:
        return "backup"
    return "none"


def is_link_up(link):
    """True if both ends of a link report their interface as up."""
    return link.intf1.isUp() and link.intf2.isUp()


def link_down(link):
    link.intf1.ifconfig("down")
    link.intf2.ifconfig("down")


def link_up(link):
    link.intf1.ifconfig("up")
    link.intf2.ifconfig("up")


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
