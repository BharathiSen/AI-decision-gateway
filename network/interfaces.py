"""Verified interface mapping for the Mininet diamond topology.

These interface names were confirmed against the live network's own
`net`/`links` output (see network/topology.py's git history for the
verification pass) and are now frozen as a static map -- Phase 4's
fault-injection harness needs real interface names (rX-ethY) to target
with tc/iptables, keyed by a stable semantic name for each link rather
than a raw node pair.

This is independent of network/controller.py's get_link() (which
resolves a link name to the Mininet Link object itself, for live
operations like configure_link()/link_up()/link_down()) -- this module
answers a different question: "what is this node's interface name on
this specific link."
"""

LINKS = {
    "host_access": {
        "hostA": "hostA-eth0",
        "r1": "r1-eth0",
    },
    "primary_r1_r2": {
        "r1": "r1-eth1",
        "r2": "r2-eth0",
    },
    "primary_r2_r4": {
        "r2": "r2-eth1",
        "r4": "r4-eth0",
    },
    "backup_r1_r3": {
        "r1": "r1-eth2",
        "r3": "r3-eth0",
    },
    "backup_r3_r4": {
        "r3": "r3-eth1",
        "r4": "r4-eth1",
    },
    "destination_access": {
        "r4": "r4-eth2",
        "hostB": "hostB-eth0",
    },
}


def get_interface(link_name, node):
    """Return the interface for a node on a named link."""
    try:
        return LINKS[link_name][node]
    except KeyError as exc:
        raise ValueError(
            f"Unknown link or endpoint: {link_name}, {node}"
        ) from exc
