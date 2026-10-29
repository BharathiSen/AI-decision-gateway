"""
Topology 2: 16-router grid (4×4) with two hosts — held-out generalization topology.

    sudo python3 network/topology_large.py
"""

from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

try:
    from network.topology import LinuxRouter, BASELINE_LINK_PARAMS
except ImportError:
    from topology import LinuxRouter, BASELINE_LINK_PARAMS

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl


def _h_base(row, col):
    return 10 + row * 10 + col


def _v_base(row, col):
    return 50 + row * 10 + col


def _addr(cidr):
    return cidr.split("/", 1)[0]


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
    eth_n = {}
    adj = {}

    def _next_eth(node):
        n = eth_n.get(node, 0)
        eth_n[node] = n + 1
        return f"{node}-eth{n}"

    def _add(a, b, ip_a, ip_b):
        name1 = _next_eth(a)
        name2 = _next_eth(b)
        key = f"{a}-{b}"
        link = net.addLink(
            net.get(a),
            net.get(b),
            intfName1=name1,
            intfName2=name2,
            params1={"ip": ip_a},
            params2={"ip": ip_b},
            **BASELINE_LINK_PARAMS,
        )
        links[key] = link
        adj[(a, b)] = {"dev": link.intf1.name, "peer": _addr(ip_b)}
        adj[(b, a)] = {"dev": link.intf2.name, "peer": _addr(ip_a)}

    for row in range(4):
        for col in range(4):
            name = f"r{row}_{col}"
            if col < 3:
                right = f"r{row}_{col + 1}"
                base = _h_base(row, col)
                _add(name, right, f"10.{base}.0.1/24", f"10.{base}.0.2/24")
            if row < 3:
                down = f"r{row + 1}_{col}"
                base = _v_base(row, col)
                _add(name, down, f"10.{base}.0.1/24", f"10.{base}.0.2/24")

    _add("hostA", "r0_0", "10.200.1.2/24", "10.200.1.1/24")
    _add("r3_3", "hostB", "10.200.2.1/24", "10.200.2.2/24")

    net.start()
    for link in links.values():
        ctl.wait_link_up(link)

    for name in list(routers):
        node = net.get(name)
        node.cmd("sysctl -w net.ipv4.ip_forward=1")
        node.cmd("sysctl -w net.ipv4.conf.all.rp_filter=0")
        node.cmd("sysctl -w net.ipv4.conf.default.rp_filter=0")
        for intf in node.intfList():
            if intf.name != "lo":
                node.cmd(f"sysctl -w net.ipv4.conf.{intf.name}.rp_filter=0")

    hostA.cmd("ip route add default via 10.200.1.1")
    hostB.cmd("ip route add default via 10.200.2.1")

    for row in range(4):
        for col in range(4):
            name = f"r{row}_{col}"
            r = net.get(name)
            if col < 3:
                hop = adj[(name, f"r{row}_{col + 1}")]
                r.cmd(f"ip route add 10.200.2.0/24 via {hop['peer']} dev {hop['dev']}")
            elif row < 3:
                hop = adj[(name, f"r{row + 1}_{col}")]
                r.cmd(f"ip route add 10.200.2.0/24 via {hop['peer']} dev {hop['dev']}")
            if col > 0:
                hop = adj[(name, f"r{row}_{col - 1}")]
                r.cmd(f"ip route add 10.200.1.0/24 via {hop['peer']} dev {hop['dev']}")
            elif row > 0:
                hop = adj[(name, f"r{row - 1}_{col}")]
                r.cmd(f"ip route add 10.200.1.0/24 via {hop['peer']} dev {hop['dev']}")

    return {"net": net, "links": links, "topology": "grid_4x4"}


if __name__ == "__main__":
    setLogLevel("info")
    state = create_large_network()
    info("*** Large topology up (16 routers).\n")
    CLI(state["net"])
    state["net"].stop()
