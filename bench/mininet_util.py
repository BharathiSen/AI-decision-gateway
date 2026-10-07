"""Shared Mininet VM helpers."""

import shutil
import subprocess


def mn_cleanup():
    """Remove stale namespaces/OVS state left after a crashed run."""
    mn = shutil.which("mn")
    if mn:
        subprocess.run([mn, "-c"], check=False, capture_output=True)


def ensure_iperf3_server(net, host="hostB"):
    """Start iperf3 in the background. Do not use -D: Mininet cmd() waits on the pty and hangs."""
    node = net.get(host)
    node.cmd(
        "sh -c 'pgrep -f \"iperf3 -s\" >/dev/null || "
        "iperf3 -s > /tmp/iperf3_server.log 2>&1 & echo $!'"
    )
