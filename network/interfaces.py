"""
Central interface mapping for the Mininet diamond topology.

Phase 4's fault-injection harness needs real interface names (rX-ethY) to
target with tc/iptables, keyed by a stable semantic name for each link
rather than a raw node pair. build_interface_map() derives those names
from the live Mininet network instead of assuming a fixed numbering, so
it stays correct even if link creation order in network/topology.py
ever changes.

This is independent of network/controller.py's get_link() (which
resolves a link name to the Mininet Link object itself, for live
operations like configure_link()/link_up()/link_down()) -- this module
answers a different question: "what is this node's interface name on
this specific link."
"""

LINKS = {
    "host_access": ("hostA", "r1"),
    "primary_r1_r2": ("r1", "r2"),
    "primary_r2_r4": ("r2", "r4"),
    "backup_r1_r3": ("r1", "r3"),
    "backup_r3_r4": ("r3", "r4"),
    "destination_access": ("r4", "hostB"),
}


def build_interface_map(net):
    """Build a bidirectional interface map from the live Mininet network.

    net must be an already-started Mininet network (e.g. the one
    network.topology.create_network() returns) -- this function only
    reads its existing link objects, it does not build or start a
    network itself. Call it after topology construction, while that
    net object is still in scope.

    Raises RuntimeError if any mapped link is missing or duplicated
    (linksBetween() returning other than exactly one match), or if its
    endpoints don't orient the way LINKS declares -- fail fast rather
    than hand the fault-injection harness a silently wrong interface.
    """
    result = {}

    for link_name, (node_a, node_b) in LINKS.items():
        a = net.get(node_a)
        b = net.get(node_b)

        matches = net.linksBetween(a, b)
        if len(matches) != 1:
            raise RuntimeError(
                f"{link_name}: expected exactly one link between "
                f"{node_a} and {node_b}, found {len(matches)}"
            )

        link = matches[0]
        intf_a = link.intf1
        intf_b = link.intf2

        # Verify that the returned link endpoints match our node order.
        if intf_a.node != a:
            intf_a, intf_b = intf_b, intf_a

        if intf_a.node != a or intf_b.node != b:
            raise RuntimeError(f"{link_name}: unexpected link orientation")

        result[link_name] = {
            node_a: intf_a.name,
            node_b: intf_b.name,
        }

    return result
