"""
Phase 1 network topology: a dynamic emulated network with a primary and
backup path between two hosts, built from real Linux routers so the
network state (routing tables, interface status, link conditions) is
something a telemetry collector can actually observe -- not simulated.

                    r2
                  /    \\
    hostA -- r1              r4 -- hostB
                  \\    /
                    r3

Primary path : hostA -> r1 -> r2 -> r4 -> hostB
Backup path  : hostA -> r1 -> r3 -> r4 -> hostB

Run standalone for manual poking around:

    sudo python3 network/topology.py

Or via the full Phase 1 CLI (recommended, adds fault-injection commands):

    sudo python3 main.py
"""

from mininet.net import Mininet
from mininet.node import Node
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl


BASELINE_LINK_PARAMS = dict(bw=10, delay="5ms", loss=0)


class LinuxRouter(Node):
    """A Mininet Node with IP forwarding enabled -- a real L3 router."""

    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        self.cmd("sysctl -w net.ipv4.ip_forward=1")

    def terminate(self):
        self.cmd("sysctl -w net.ipv4.ip_forward=0")
        super(LinuxRouter, self).terminate()


def create_network():
    """Build and start the Phase 1 emulated network.

    Returns {"net": <Mininet>, "links": {name: <Link>}} so other modules
    (fault injector, controller, telemetry collector) can reference each
    hop by name instead of re-discovering topology internals.
    """

    net = Mininet(controller=None, link=TCLink)

    hostA = net.addHost("hostA")
    hostB = net.addHost("hostB")

    r1 = net.addHost("r1", cls=LinuxRouter, ip=None)
    r2 = net.addHost("r2", cls=LinuxRouter, ip=None)
    r3 = net.addHost("r3", cls=LinuxRouter, ip=None)
    r4 = net.addHost("r4", cls=LinuxRouter, ip=None)

    links = {}

    links["hostA-r1"] = net.addLink(
        hostA, r1,
        intfName1="hostA-eth0", intfName2="r1-eth0",
        params1={"ip": "10.0.1.2/24"},
        params2={"ip": "10.0.1.1/24"},
        **BASELINE_LINK_PARAMS,
    )

    links["r1-r2"] = net.addLink(
        r1, r2,
        intfName1="r1-eth1", intfName2="r2-eth0",
        params1={"ip": "10.0.12.1/24"},
        params2={"ip": "10.0.12.2/24"},
        **BASELINE_LINK_PARAMS,
    )

    links["r1-r3"] = net.addLink(
        r1, r3,
        intfName1="r1-eth2", intfName2="r3-eth0",
        params1={"ip": "10.0.13.1/24"},
        params2={"ip": "10.0.13.2/24"},
        **BASELINE_LINK_PARAMS,
    )

    links["r2-r4"] = net.addLink(
        r2, r4,
        intfName1="r2-eth1", intfName2="r4-eth0",
        params1={"ip": "10.0.24.1/24"},
        params2={"ip": "10.0.24.2/24"},
        **BASELINE_LINK_PARAMS,
    )

    links["r3-r4"] = net.addLink(
        r3, r4,
        intfName1="r3-eth1", intfName2="r4-eth1",
        params1={"ip": "10.0.34.1/24"},
        params2={"ip": "10.0.34.2/24"},
        **BASELINE_LINK_PARAMS,
    )

    links["r4-hostB"] = net.addLink(
        r4, hostB,
        intfName1="r4-eth2", intfName2="hostB-eth0",
        params1={"ip": "10.0.4.1/24"},
        params2={"ip": "10.0.4.2/24"},
        **BASELINE_LINK_PARAMS,
    )

    net.start()

    # IP forwarding is already enabled per-router via LinuxRouter.config()
    # above, but that relies on Mininet's build()/configDefault() dispatch
    # actually invoking it. Set it explicitly here too so it's impossible
    # to miss when auditing/debugging, and disable reverse-path filtering:
    # r1 and r4 are each multi-homed (two router-facing links), and Linux's
    # strict rp_filter can silently drop forwarded packets on a multi-homed
    # node even when routing and ip_forward are both correct.
    for router in (r1, r2, r3, r4):
        router.cmd("sysctl -w net.ipv4.ip_forward=1")
        router.cmd("sysctl -w net.ipv4.conf.all.rp_filter=0")
        router.cmd("sysctl -w net.ipv4.conf.default.rp_filter=0")
        for intf in router.intfList():
            if intf.name != "lo":
                router.cmd(f"sysctl -w net.ipv4.conf.{intf.name}.rp_filter=0")

    hostA.cmd("ip route add default via 10.0.1.1")
    hostB.cmd("ip route add default via 10.0.4.1")

    # r2 and r3 each sit on exactly one path, so they just need routes
    # back to both edge LANs.
    r2.cmd("ip route add 10.0.1.0/24 via 10.0.12.1 dev r2-eth0")
    r2.cmd("ip route add 10.0.4.0/24 via 10.0.24.2 dev r2-eth1")

    r3.cmd("ip route add 10.0.1.0/24 via 10.0.13.1 dev r3-eth0")
    r3.cmd("ip route add 10.0.4.0/24 via 10.0.34.2 dev r3-eth1")

    # r1/r4 are where path selection actually happens. Start on primary.
    ctl.set_active_path(net, "primary")

    return {"net": net, "links": links}


if __name__ == "__main__":
    setLogLevel("info")

    state = create_network()
    net = state["net"]

    info("\n*** Phase 1 network is up.\n")
    info("*** Primary path: hostA -> r1 -> r2 -> r4 -> hostB\n")
    info("*** Backup path : hostA -> r1 -> r3 -> r4 -> hostB\n")
    info("*** (Run 'sudo python3 main.py' instead for fault-injection commands.)\n\n")

    CLI(net)

    net.stop()
