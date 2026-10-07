"""
Topology 2: 16-router grid (4×4) with two hosts — held-out generalization topology.

    sudo python3 network/topology_large.py
"""

from mininet.net import Mininet
from mininet.node import Node
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

try:
    from network.topology import LinuxRouter, BASELINE_LINK_PARAMS
except ImportError:
    from topology import LinuxRouter, BASELINE_LINK_PARAMS


def create_large_network():
    """4×4 router grid r0_0 .. r3_3; hostA on r0_0, hostB on r3_3."""
    net = Mininet(controller=None, link=TCLink)
    routers = {}
    for row in range(4):
        for col in range(4):
            name = f"r{row}_{col}"
            routers[name] = net.addHost(name, cls=LinuxRouter, ip=None)

    hostA = net.addHost("hostA", ip=None)
    hostB = net.addHost("hostB", ip=None)
    links = {}

    def _add(a, b, ip_a, ip_b, n1, n2):
        key = f"{a}-{b}"
        links[key] = net.addLink(
            net.get(a), net.get(b),
            intfName1=n1, intfName2=n2,
            params1={"ip": ip_a},
            params2={"ip": ip_b},
            **BASELINE_LINK_PARAMS,
        )

    # Grid horizontal + vertical links
    subnet = 10
    for row in range(4):
        for col in range(4):
            name = f"r{row}_{col}"
            if col < 3:
                right = f"r{row}_{col+1}"
                base = subnet + row * 10 + col
                _add(name, right, f"10.{base}.1/24", f"10.{base}.2/24", f"{name}-eth1", f"{right}-eth0")
            if row < 3:
                down = f"r{row+1}_{col}"
                base = subnet + 40 + row * 10 + col
                _add(name, down, f"10.{base}.1/24", f"10.{base}.2/24", f"{name}-eth2", f"{down}-eth0")

    _add("hostA", "r0_0", "10.200.1.2/24", "10.200.1.1/24", "hostA-eth0", "r0_0-eth3")
    _add("r3_3", "hostB", "10.200.2.1/24", "10.200.2.2/24", "r3_3-eth3", "hostB-eth0")

    net.start()
    for name in list(routers) + ["hostA", "hostB"]:
        node = net.get(name)
        if name.startswith("r"):
            node.cmd("sysctl -w net.ipv4.ip_forward=1")
            node.cmd("sysctl -w net.ipv4.conf.all.rp_filter=0")

    hostA.cmd("ip route add default via 10.200.1.1")
    hostB.cmd("ip route add default via 10.200.2.1")

    # Simple static routes: east then south path
    for row in range(4):
        for col in range(4):
            r = net.get(f"r{row}_{col}")
            if col < 3:
                r.cmd("ip route add 10.200.2.0/24 via 10.%d.2 dev r%d_%d-eth1" % (10 + row * 10 + col, row, col))
            elif row < 3:
                r.cmd("ip route add 10.200.2.0/24 via 10.%d.2 dev r%d_%d-eth2" % (50 + row * 10 + col, row, col))

    return {"net": net, "links": links, "topology": "grid_4x4"}


if __name__ == "__main__":
    setLogLevel("info")
    state = create_large_network()
    info("*** Large topology up (16 routers).\n")
    CLI(state["net"])
    state["net"].stop()
