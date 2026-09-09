from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI


def create_network():

    net = Mininet(
        controller=Controller,
        link=TCLink
    )

    # Controller
    net.addController("c0")

    # Hosts
    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")

    # Switches
    s1 = net.addSwitch("s1")
    s2 = net.addSwitch("s2")
    s3 = net.addSwitch("s3")
    s4 = net.addSwitch("s4")

    # Host connections
    net.addLink(h1, s1)
    net.addLink(s4, h2)

    # PRIMARY PATH
    # h1 -> s1 -> s2 -> s4 -> h2

    net.addLink(
        s1, s2,
        bw=10,
        delay="5ms",
        loss=0
    )

    net.addLink(
        s2, s4,
        bw=10,
        delay="5ms",
        loss=0
    )

    # BACKUP PATH
    # h1 -> s1 -> s3 -> s4 -> h2

    net.addLink(
        s1, s3,
        bw=10,
        delay="5ms",
        loss=0
    )

    net.addLink(
        s3, s4,
        bw=10,
        delay="5ms",
        loss=0
    )

    # Start network
    net.start()

    return net


if __name__ == "__main__":

    setLogLevel("info")

    net = create_network()

    print("\nNetwork started successfully!")

    # Open Mininet CLI
    CLI(net)

    # Stop network when CLI exits
    net.stop()