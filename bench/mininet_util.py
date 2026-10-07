"""Shared Mininet VM helpers."""

import shutil
import subprocess


def mn_cleanup():
    """Remove stale namespaces/OVS state left after a crashed run."""
    mn = shutil.which("mn")
    if mn:
        subprocess.run([mn, "-c"], check=False, capture_output=True)


def ensure_iperf3_server(net, host="hostB"):
    """Start iperf3 in server mode if not already listening (for T10 / F6)."""
    node = net.get(host)
    check = node.cmd("pgrep -x iperf3 || true").strip()
    if check:
        return
    node.cmd("iperf3 -s -D 2>/dev/null")
