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


RUN_DIR = Path("/tmp")
_SHARED_RUN_DIR = RUN_DIR
_private_run_dir = None


def make_private_run_dir(base):
    """Create an owner-only directory under base for command output files.

    run_in_node trusts the exit code and the PID it reads back from these
    files, and signals that PID as root. In a world-writable directory any
    local user could plant them.
    """
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="adg-run-", dir=str(base)))
    os.chmod(path, 0o700)
    return path


def run_dir():
    """Where run_in_node keeps its files. The node shares the host filesystem.

    A caller that sets RUN_DIR (offline self-checks) gets that directory as is.
    Otherwise one private directory is created on first use and removed at exit.
    """
    global _private_run_dir
    if RUN_DIR != _SHARED_RUN_DIR:
        return RUN_DIR
    if _private_run_dir is None or not _private_run_dir.is_dir():
        import atexit

        _private_run_dir = make_private_run_dir(RUN_DIR)
        atexit.register(shutil.rmtree, str(_private_run_dir), True)
    return _private_run_dir


def _kill_node_pid(node, pid):
    """Terminate one timed-out command PID inside the node (not shell-wide)."""
    if pid is None or not hasattr(node, "popen"):
        return
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 1:
        return
    for args in (
        ["kill", "-TERM", "--", str(pid)],
        ["kill", "-KILL", "--", str(pid)],
    ):
        try:
            proc = node.popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except Exception:
            pass
        if args[1] == "-TERM":
            time.sleep(0.05)


def run_in_node(node, command, timeout=60.0, structured=False):
    """Run a command and return its output from a file, not the Mininet PTY.

    node.cmd() can return on the previous command's prompt while this one
    is still running, so the output file is read empty. Completion is the
    marker file the shell writes after the command exits, not the prompt.
    The host shares /tmp with the node.

    On timeout, only the recorded command PID is signalled (TERM then KILL).
    No shell-wide pkill/killall.

    When structured=True, return
    {"output", "exit_code", "timed_out", "error"} so callers can tell
    measurement failures from observed network outcomes:
      timeout / read_failure / missing_completion / empty_output /
      command_failure (nonzero exit with no output).
    Nonzero exit with usable output leaves error=None; callers grade the
    network from the text (ping loss often exits 1).
    """
    stem = (run_dir() / f"adg-{node.name}-{uuid.uuid4().hex}").as_posix()
    path = Path(f"{stem}.out")
    done = Path(f"{stem}.done")
    pid_path = Path(f"{stem}.pid")
    recover_shell(node)
    # Record the command PID so a timeout can kill only that process.
    line = (
        f"bash -c {shlex.quote(command)} > {path} 2>&1 & "
        f"echo $! > {pid_path}; "
        f"wait $(cat {pid_path}); "
        f"echo $? > {done}"
    )
    try:
        node.cmd(line)
    except UnicodeDecodeError:
        recover_shell(node)
        node.cmd(line)
    deadline = time.monotonic() + timeout
    while not done.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    timed_out = not done.exists()
    timed_out_pid = None
    if timed_out:
        try:
            timed_out_pid = int(pid_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            timed_out_pid = None
        _kill_node_pid(node, timed_out_pid)
        # Brief wait for the shell to write done/exit after kill.
        kill_deadline = time.monotonic() + 0.5
        while not done.exists() and time.monotonic() < kill_deadline:
            time.sleep(0.05)
    exit_code = None
    if done.exists():
        try:
            raw_code = done.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0].strip()
            exit_code = int(raw_code)
        except (OSError, ValueError, IndexError):
            exit_code = None
    read_failed = False
    try:
        output = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
        read_failed = True
    for leftover in (path, done, pid_path):
        try:
            leftover.unlink(missing_ok=True)
        except OSError:
            pass
    if not structured:
        return output
    blank = not (output or "").strip()
    error = None
    if timed_out:
        error = "timeout"
    elif read_failed:
        error = "read_failure"
    elif blank and exit_code == 0:
        error = "empty_output"
    elif blank and exit_code is None:
        error = "missing_completion"
    elif blank and exit_code not in (0, None):
        error = "command_failure"
    result = {
        "output": output,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "error": error,
    }
    if timed_out and timed_out_pid is not None:
        result["killed_pid"] = timed_out_pid
    return result


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
