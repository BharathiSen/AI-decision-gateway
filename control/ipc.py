"""Framed JSON messages between the orchestrator, gate service and executor service.

Frame: 4-byte big-endian length, then UTF-8 JSON (one object), at most
MAX_FRAME bytes. Transports:

  Unix-domain socket (deployment, Linux): the server checks every peer with
  SO_PEERCRED and serves only the uids it was told to trust. The socket file
  is created with mode 0660 inside a directory the operator restricts.
  stdio pipes (offline checks, any OS): the peer is the parent process that
  spawned the service; there is no credential check on this transport.

Nothing here decides what is authorized. Handlers do.
"""

import json
import os
import socket
import struct
import subprocess
import sys

MAX_FRAME = 1 << 20


class FrameError(RuntimeError):
    """A frame is truncated, oversized or not a JSON object."""


def _read_exact(stream, size):
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            if not data:
                return None
            raise FrameError("truncated frame")
        data += chunk
    return data


def send(stream, message):
    # NaN and Infinity pass through: both ends are Python, and evidence digests
    # (version 2) distinguish them, so they must arrive unchanged.
    body = json.dumps(message, sort_keys=True).encode("utf-8")
    if len(body) > MAX_FRAME:
        raise FrameError(f"frame of {len(body)} bytes exceeds {MAX_FRAME}")
    stream.write(struct.pack(">I", len(body)) + body)
    stream.flush()


def recv(stream):
    """Next message, or None at a clean end of stream."""
    header = _read_exact(stream, 4)
    if header is None:
        return None
    (size,) = struct.unpack(">I", header)
    if size > MAX_FRAME:
        raise FrameError(f"frame of {size} bytes exceeds {MAX_FRAME}")
    body = _read_exact(stream, size)
    if body is None:
        raise FrameError("truncated frame")
    try:
        message = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FrameError(f"frame is not JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise FrameError("frame is not a JSON object")
    return message


def _safe_handle(handler, message):
    try:
        reply = handler(message)
    except Exception as exc:  # a handler bug is a refusal, never a crash of the service
        return {"ok": False, "error": f"service error: {type(exc).__name__}: {exc}"}
    return reply if isinstance(reply, dict) else {"ok": False, "error": "service returned no reply"}


def serve_stream(reader, writer, handler):
    """Answer frames until the peer closes. A malformed frame ends the session."""
    while True:
        try:
            message = recv(reader)
        except FrameError as exc:
            try:
                send(writer, {"ok": False, "error": str(exc)})
            except Exception:
                pass
            return
        if message is None:
            return
        send(writer, _safe_handle(handler, message))


def serve_stdio(handler):
    serve_stream(sys.stdin.buffer, sys.stdout.buffer, handler)


def peer_credentials(conn):
    """(pid, uid, gid) of the connected peer. Linux SO_PEERCRED only."""
    if not hasattr(socket, "SO_PEERCRED"):
        raise OSError("peer credentials are not supported on this platform")
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", raw)


def peer_allowed(credentials, allowed_uids):
    """Policy only: the peer's uid must be listed. Root is not implied."""
    _pid, uid, _gid = credentials
    return uid in set(allowed_uids)


def serve_unix(path, handler, allowed_uids, once=False):
    """Serve on a Unix-domain socket, one connection at a time (single flight)."""
    if not hasattr(socket, "AF_UNIX") or not hasattr(socket, "SO_PEERCRED"):
        raise OSError("Unix-domain sockets with SO_PEERCRED are required (Linux)")
    if not allowed_uids:
        raise ValueError("refusing to serve without an explicit list of allowed peer uids")
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o117)
    try:
        server.bind(path)
    finally:
        os.umask(old_umask)
    os.chmod(path, 0o660)
    server.listen(4)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    creds = peer_credentials(conn)
                except OSError:
                    creds = None
                reader, writer = conn.makefile("rb"), conn.makefile("wb")
                if creds is None or not peer_allowed(creds, allowed_uids):
                    send(writer, {"ok": False, "error": f"peer not allowed: {creds}"})
                else:
                    serve_stream(reader, writer, handler)
            if once:
                return
    finally:
        server.close()
        if os.path.exists(path):
            os.unlink(path)


class UnixClient:
    def __init__(self, path, timeout=30.0):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(path)
        self._reader, self._writer = self._sock.makefile("rb"), self._sock.makefile("wb")

    def call(self, message):
        send(self._writer, message)
        reply = recv(self._reader)
        if reply is None:
            raise FrameError("service closed the connection")
        return reply

    def close(self):
        self._sock.close()


class StdioClient:
    """Spawn a service as a child process and talk to it over its stdin/stdout."""

    def __init__(self, argv, cwd=None, env=None):
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def call(self, message):
        send(self.proc.stdin, message)
        reply = recv(self.proc.stdout)
        if reply is None:
            raise FrameError("service exited: " + self.proc.stderr.read().decode("utf-8", "replace")[-2000:])
        return reply

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        finally:
            for stream in (self.proc.stdout, self.proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


def assert_unprivileged(role):
    """Refuse to run an unprivileged role (orchestrator, model caller) as root."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise PermissionError(f"{role} must not run as root in the separated deployment")
