"""Shared Mininet VM helpers."""

import codecs
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path


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


def run_in_node(node, command):
    """Run a command and return its output from a file, not the Mininet PTY.

    node.cmd() sometimes returns leftover output from the previous command
    (a ping still in the shell buffer). The command's own stdout is written
    to /tmp and read from the host, which shares that directory with the node.
    """
    path = Path(f"/tmp/adg-{node.name}-{uuid.uuid4().hex}.out")
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    recover_shell(node)
    quoted = shlex.quote(command)
    try:
        node.cmd(f"bash -c {quoted} > {path} 2>&1")
    except UnicodeDecodeError:
        recover_shell(node)
        node.cmd(f"bash -c {quoted} > {path} 2>&1")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def stop_iperf_clients(net, host="hostA"):
    """Kill iperf3 clients without using the node shell. Servers stay up."""
    node = net.get(host)
    if not hasattr(node, "popen"):
        return
    node.popen(
        ["pkill", "-f", "iperf3 -c"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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


def ensure_iperf3_server(net, host="hostB", port=5201):
    """Start iperf3 with Node.popen so the server outlives the Mininet shell.

    A background `iperf3 -s &` inside node.cmd() is killed when that shell
    exits, which leaves every T10 sample empty. Port 5201 is the speed and
    TCP test. Another port can serve background traffic at the same time.
    """
    node = net.get(host)
    procs = getattr(node, "_iperf3_procs", None)
    if procs is None:
        procs = {}
        node._iperf3_procs = procs
    if port == 5201 and port not in procs:
        legacy = getattr(node, "_iperf3_proc", None)
        if legacy is not None:
            procs[port] = legacy
    proc = procs.get(port)
    if proc is not None and proc.poll() is None:
        return proc
    cmd = ["iperf3", "-s"] if int(port) == 5201 else ["iperf3", "-s", "-p", str(port)]
    proc = node.popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs[port] = proc
    if int(port) == 5201:
        node._iperf3_proc = proc
    return proc


def _release_stdio(proc):
    if proc is None:
        return
    for attr in ("_stdin_keep", "_log"):
        handle = getattr(proc, attr, None)
        if handle is None:
            continue
        try:
            if hasattr(handle, "close"):
                handle.close()
            else:
                os.close(handle)
        except OSError:
            pass


def stop_iperf3_server(net, host="hostB", port=5201):
    """Stop a tracked server so its port can be bound again."""
    node = net.get(host)
    procs = getattr(node, "_iperf3_procs", None) or {}
    proc = procs.pop(port, None)
    if port == 5201:
        node._iperf3_proc = None
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    _release_stdio(proc)


def free_tcp_port(net, port, host="hostB"):
    """Kill whatever is still listening on this port inside the node.

    Killing the Mininet popen handle often leaves iperf3 itself running.
    The next client then connects to that stuck server and gets
    "Bad file descriptor".
    """
    node = net.get(host)
    recover_shell(node)
    node.cmd(
        f"fuser -k {int(port)}/tcp >/dev/null 2>&1; "
        f"ss -lptnH 'sport = :{int(port)}' 2>/dev/null "
        f"| sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | sort -u | xargs -r kill >/dev/null 2>&1; "
        f"true"
    )


def start_one_off_iperf3(net, host="hostB", port=5201):
    """Serve a single client, then exit.

    A long-lived iperf3 server closes its control socket after a few tests.
    Later clients connect and then fail with "Bad file descriptor". stdin
    stays open so the server does not treat a closed fd 0 as a hangup.
    """
    stop_iperf3_server(net, host, port)
    free_tcp_port(net, port, host)
    time.sleep(0.2)
    node = net.get(host)
    cmd = ["iperf3", "-s", "-1"]
    if int(port) != 5201:
        cmd += ["-p", str(port)]
    read_fd, write_fd = os.pipe()
    log = open(f"/tmp/iperf3-oneoff-{int(port)}.log", "wb", buffering=0)
    try:
        proc = node.popen(
            cmd,
            stdin=read_fd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    finally:
        os.close(read_fd)
    proc._stdin_keep = write_fd
    proc._log = log
    return proc


def wait_for_listener(net, port, host="hostB", timeout=3.0):
    """True once something listens on this TCP port inside the node."""
    node = net.get(host)
    deadline = time.monotonic() + timeout
    while True:
        if run_in_node(node, f"ss -ltnH 'sport = :{int(port)}'").strip():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def one_off_log_tail(port, lines=8):
    """Last lines of the one-off server's log, for a transfer that failed."""
    try:
        text = Path(f"/tmp/iperf3-oneoff-{int(port)}.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def finish_one_off(proc):
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    _release_stdio(proc)


def restart_iperf3_server(net, host="hostB", port=5201):
    """Replace a server left stuck after a client died mid-test."""
    node = net.get(host)
    procs = getattr(node, "_iperf3_procs", None) or {}
    proc = procs.pop(port, None)
    if port == 5201:
        node._iperf3_proc = None
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    return ensure_iperf3_server(net, host, port)
