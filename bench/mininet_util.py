"""Shared Mininet VM helpers."""

import codecs
import os
import shutil
import subprocess


def recover_shell(node):
    """Unstick a Mininet shell after a UnicodeDecodeError on binary output.

    node.cmd() leaves waiting=True when the UTF-8 decoder raises, and the
    next command then asserts. Drain leftover bytes and reset that flag.
    """
    node.waiting = False
    node.readbuf = ""
    try:
        node.decoder = codecs.getincrementaldecoder("utf-8")("replace")
    except Exception:
        pass
    stdout = getattr(node, "stdout", None)
    if stdout is None:
        return
    fd = stdout.fileno()
    try:
        import fcntl

        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except (BlockingIOError, OSError):
                    break
                if not chunk:
                    break
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    except Exception:
        pass


def recover_net(net, names=("hostA", "hostB", "r1", "r2", "r3", "r4")):
    for name in names:
        try:
            recover_shell(net.get(name))
        except Exception:
            pass


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
