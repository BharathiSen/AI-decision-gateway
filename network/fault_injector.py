"""
Runtime fault injection: congestion, latency, packet loss, and primary/backup link failure.

Each method mutates the already-running Mininet network in place, so a
telemetry collector polling the network afterwards observes the actual
new state -- nothing here writes to a telemetry file directly.

"""

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

PRIMARY_LINK = "r1-r2"
BACKUP_LINK = "r1-r3"

ALL_LINKS = ["hostA-r1", "r1-r2", "r1-r3", "r2-r4", "r3-r4", "r4-hostB"]


class FaultInjector:
    def __init__(self, net, links=None):
        self.net = net
        self.links = links

    def _link(self, name):
        return ctl.get_link(self.net, name, self.links)

    def induce_congestion(self, link_name=PRIMARY_LINK, bw_mbps=0.5):
        """Starve a link's bandwidth to simulate congestion."""
        link = self._link(link_name)
        ctl.configure_link(link, bw=bw_mbps)
        return {"fault": "congestion", "link": link_name, "bw_mbps": bw_mbps}

    def induce_latency(self, link_name=PRIMARY_LINK, delay_ms=200):
        """Add propagation delay to a link."""
        link = self._link(link_name)
        ctl.configure_link(link, delay=f"{delay_ms}ms")
        return {"fault": "latency", "link": link_name, "delay_ms": delay_ms}

    def induce_packet_loss(self, link_name=PRIMARY_LINK, loss_percent=20):
        """Add random packet loss to a link."""
        link = self._link(link_name)
        ctl.configure_link(link, loss=loss_percent)
        return {"fault": "packet_loss", "link": link_name, "loss_percent": loss_percent}

    def fail_primary_link(self):
        """Bring down the r1-r2 hop, breaking the primary path."""
        link = self._link(PRIMARY_LINK)
        ctl.link_down(link)
        return {"fault": "primary_link_failure", "link": PRIMARY_LINK}

    def fail_backup_link(self):
        """Bring down the r1-r3 hop, breaking the backup path."""
        link = self._link(BACKUP_LINK)
        ctl.link_down(link)
        return {"fault": "backup_link_failure", "link": BACKUP_LINK}

    def restore_link(self, link_name):
        """Reset one link back to baseline bandwidth/delay/loss and up."""
        link = self._link(link_name)
        ctl.reset_link(link)
        return {"restored": link_name}

    def reset_all(self):
        """Clear every fault and restore the complete network state."""

        for name in ALL_LINKS:
            self.restore_link(name)

        # Restore static routes that depend on links which may have
        # disappeared when those interfaces were brought down.
        r2 = self.net.get("r2")
        r3 = self.net.get("r3")

        r2.cmd(
            "ip route replace 10.0.1.0/24 "
            "via 10.0.12.1 dev r2-eth0"
        )
        r2.cmd(
            "ip route replace 10.0.4.0/24 "
            "via 10.0.24.2 dev r2-eth1"
        )

        r3.cmd(
            "ip route replace 10.0.1.0/24 "
            "via 10.0.13.1 dev r3-eth0"
        )
        r3.cmd(
            "ip route replace 10.0.4.0/24 "
            "via 10.0.34.2 dev r3-eth1"
        )

        # Restore the selected primary path on r1 and r4.
        ctl.set_active_path(self.net, "primary")

        return {
            "reset": ALL_LINKS,
            "active_path": "primary"
        }