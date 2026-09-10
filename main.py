"""
Phase 1 entry point: build the emulated network, then drop into an
interactive Mininet CLI extended with commands for triggering each
fault scenario and inspecting the live network state.

Must be run as root inside the Mininet VM (Mininet requires raw
network-namespace/OVS access):

    sudo python3 main.py

Extra CLI commands (on top of the usual Mininet ones like pingall):

    status                        show live topology + telemetry as JSON
    congestion [link] [bw_mbps]   starve a link's bandwidth (default r1-r2 0.5)
    latency [link] [delay_ms]     add delay to a link (default r1-r2 200)
    packet_loss [link] [percent]  add random loss to a link (default r1-r2 20)
    fail_primary                  bring down the primary path (r1-r2)
    fail_backup                   bring down the backup path (r1-r3)
    restore <link|all>            reset one link, or every link, to baseline
    reroute <primary|backup>      switch the active hostA<->hostB path
"""

import json

from mininet.cli import CLI
from mininet.log import setLogLevel, info

from network.topology import create_network
from network.fault_injector import FaultInjector
from network import controller as ctl
from telemetry.collector import collect_current_state


class GatewayCLI(CLI):
    """Mininet CLI extended with Phase 1 fault-injection commands."""

    def __init__(self, net, links, injector, *args, **kwargs):
        self.links = links
        self.injector = injector
        CLI.__init__(self, net, *args, **kwargs)

    def do_status(self, _line):
        "Print the live network state (nodes, links, metrics) as JSON."
        state = collect_current_state(self.mn, self.links)
        print(json.dumps(state, indent=2))

    def do_congestion(self, line):
        "congestion [link] [bw_mbps] -- starve a link's bandwidth. Default: r1-r2 0.5"
        args = line.split()
        link_name = args[0] if len(args) > 0 else "r1-r2"
        bw = float(args[1]) if len(args) > 1 else 0.5
        print(self.injector.induce_congestion(link_name, bw))

    def do_latency(self, line):
        "latency [link] [delay_ms] -- add delay to a link. Default: r1-r2 200"
        args = line.split()
        link_name = args[0] if len(args) > 0 else "r1-r2"
        delay_ms = int(args[1]) if len(args) > 1 else 200
        print(self.injector.induce_latency(link_name, delay_ms))

    def do_packet_loss(self, line):
        "packet_loss [link] [percent] -- add random loss to a link. Default: r1-r2 20"
        args = line.split()
        link_name = args[0] if len(args) > 0 else "r1-r2"
        loss = float(args[1]) if len(args) > 1 else 20
        print(self.injector.induce_packet_loss(link_name, loss))

    def do_fail_primary(self, _line):
        "Bring down the primary path link (r1-r2)."
        print(self.injector.fail_primary_link())

    def do_fail_backup(self, _line):
        "Bring down the backup path link (r1-r3)."
        print(self.injector.fail_backup_link())

    def do_restore(self, line):
        "restore <link|all> -- reset one link, or every link, back to baseline."
        link_name = line.strip()
        if not link_name or link_name == "all":
            print(self.injector.reset_all())
        else:
            print(self.injector.restore_link(link_name))

    def do_reroute(self, line):
        "reroute <primary|backup> -- switch the active hostA<->hostB path."
        path = line.strip()
        ctl.set_active_path(self.mn, path)
        print(f"active path is now: {ctl.get_active_path(self.mn)}")


def main():
    setLogLevel("info")

    state = create_network()
    net, links = state["net"], state["links"]
    injector = FaultInjector(net, links)

    info("\n*** Phase 1 network is up.\n")
    info("*** Primary path: hostA -> r1 -> r2 -> r4 -> hostB\n")
    info("*** Backup path : hostA -> r1 -> r3 -> r4 -> hostB\n")
    info("*** Extra commands: status, congestion, latency, packet_loss, "
         "fail_primary, fail_backup, restore, reroute\n\n")

    GatewayCLI(net, links, injector)

    net.stop()


if __name__ == "__main__":
    main()
