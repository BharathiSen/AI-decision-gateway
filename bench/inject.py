"""
Phase 4 fault-injection primitives.

Deliberately separate from network.controller's link_down()/link_up()
(both-ended, used by Phase 1's interactive fault injector in
network/fault_injector.py). That existing behavior is left untouched
here -- fail_primary/fail_backup in the main.py CLI still bring down
both ends of a link, exactly as they always have.

The benchmark's own fault semantics are single-ended: a link failure
here means exactly one named endpoint's interface goes administratively
down. Its peer's own admin state is never touched -- only its carrier
drops, as a normal consequence of that peer being the other end of the
same veth pair. That's a materially different (and more realistic:
most single link failures aren't symmetric on both routers at once)
failure mode than taking both ends down together, so it gets its own
mechanism rather than silently changing what Phase 1 already does and
has already been validated against.

Interface names come from network.interfaces.get_interface() -- the
frozen, verified map -- not from a live net.linksBetween() lookup.
"""

try:
    from network.interfaces import get_interface
except ImportError:
    from interfaces import get_interface


def link_down_one_end(net, link_name, node):
    """Bring down exactly one endpoint of a named link.

    Only `node`'s interface goes administratively down. The peer is
    left alone -- not brought down, not reconfigured.
    """
    intf = get_interface(link_name, node)
    net.get(node).cmd(f"ip link set dev {intf} down")
    return {"link": link_name, "node": node, "interface": intf, "action": "down"}


def link_up_one_end(net, link_name, node):
    """Restore exactly the endpoint link_down_one_end() brought down."""
    intf = get_interface(link_name, node)
    net.get(node).cmd(f"ip link set dev {intf} up")
    return {"link": link_name, "node": node, "interface": intf, "action": "up"}


def _operstate(net, link_name, node):
    intf = get_interface(link_name, node)
    return net.get(node).cmd(f"cat /sys/class/net/{intf}/operstate").strip()


def _peer_ipv4(net, link_name, peer_node):
    intf = get_interface(link_name, peer_node)
    out = net.get(peer_node).cmd(f"ip -4 -o addr show {intf}")
    # e.g. "3: r2-eth1    inet 10.0.24.1/24 brd ... scope global r2-eth1"
    return out.split("inet ")[1].split("/")[0].split()[0]


def verify_restore(net, link_name, node, peer_node):
    """Inject a single-ended down/up cycle on (link_name, node) and
    confirm the link returns to exactly its pre-injection state:

    - both endpoints' operstate match what they were before injection
    - the link passes a real ping test afterward (not just "up" on paper)

    Returns a dict describing the result. Raises AssertionError with a
    specific, actionable message if restoration isn't exact -- this is
    meant to be run once per fault type before it's trusted for real
    experiments, per "test that cleanup restores the exact pre-injection
    state."
    """
    before = {
        node: _operstate(net, link_name, node),
        peer_node: _operstate(net, link_name, peer_node),
    }

    link_down_one_end(net, link_name, node)
    link_up_one_end(net, link_name, node)

    after = {
        node: _operstate(net, link_name, node),
        peer_node: _operstate(net, link_name, peer_node),
    }

    if after != before:
        raise AssertionError(
            f"{link_name}/{node}: operstate after restore {after} "
            f"does not match state before injection {before}"
        )

    peer_ip = _peer_ipv4(net, link_name, peer_node)
    ping_output = net.get(node).cmd(f"ping -c 2 -W 1 {peer_ip}")
    if "0% packet loss" not in ping_output:
        raise AssertionError(
            f"{link_name}/{node}: interfaces report up but the link failed "
            f"a real connectivity check against {peer_node} ({peer_ip}):\n"
            f"{ping_output}"
        )

    return {"link": link_name, "node": node, "peer": peer_node, "restored": True, "state": after}
