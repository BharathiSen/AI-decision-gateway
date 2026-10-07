"""Shared Mininet VM helpers."""

import shutil
import subprocess


def mn_cleanup():
    """Remove stale namespaces/OVS state left after a crashed run."""
    mn = shutil.which("mn")
    if mn:
        subprocess.run([mn, "-c"], check=False, capture_output=True)


def ensure_iperf3_server(net, host="hostB"):
    """Start iperf3 with Node.popen so the server outlives the Mininet shell.

    A background `iperf3 -s &` inside node.cmd() is killed when that shell
    exits, which leaves every T10 sample empty.
    """
    node = net.get(host)
    proc = getattr(node, "_iperf3_proc", None)
    if proc is not None and proc.poll() is None:
        return proc
    proc = node.popen(
        ["iperf3", "-s"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    node._iperf3_proc = proc
    return proc
