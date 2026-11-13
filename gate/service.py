"""Gate service: the only process that holds the Ed25519 signing key (Stage 2).

    python -m gate.service --key /etc/verifix/gate_ed25519.pem --socket /run/verifix/gate.sock --allow-uid 1001
    python -m gate.service --key KEY --stdio          # offline checks only

Requests (framed JSON, control/ipc.py):
  {"op": "decide", "proposal", "snapshot", "evidence", "mode", "noise_id", "path_qdiscs", "ttl_seconds"?}
      -> {"ok": true, "decision": <gate.decide record with its authorization>}
  {"op": "public_key"} -> {"ok": true, "public_key_pem": "..."}

The service evaluates only what it is sent (run_missing_tests=False, no
network). It cannot prove who measured the evidence: a compromised
orchestrator can still send false evidence. What it guarantees is that only
this process can produce an authorization the executor will accept, and that
the authorization is bound to exactly what was evaluated.

Key handling: the key file must be a regular file readable only by its owner
(mode 0600, POSIX). The orchestrator and the executor must not be able to
read it. ttl_seconds can only shorten the validity window.
"""

import argparse
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate import authority  # noqa: E402

MODES = ("full", "pilot_v0")


def check_key_file(path):
    """Refuse a signing key others can read or write (POSIX)."""
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"signing key {path} is not a regular file")
    if os.name == "posix":
        if info.st_mode & 0o077:
            raise PermissionError(f"signing key {path} must be mode 0600 (is {oct(info.st_mode & 0o777)})")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise PermissionError(f"signing key {path} is not owned by the gate service user")


class GateService:
    def __init__(self, private_key, offline_test_thresholds=False):
        self._private_key = private_key
        self._test_thresholds = offline_test_thresholds
        authority.install_signer(private_key)

    def handle(self, request):
        op = request.get("op")
        if op == "public_key":
            from cryptography.hazmat.primitives import serialization

            pem = self._private_key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return {"ok": True, "public_key_pem": pem.decode("ascii")}
        if op != "decide":
            return {"ok": False, "error": f"unknown op {op!r}"}
        mode = request.get("mode", "full")
        if mode not in MODES:
            return {"ok": False, "error": f"mode must be one of {MODES}"}
        ttl = request.get("ttl_seconds")
        if ttl is not None and (
            type(ttl) not in (int, float) or not 0 < ttl <= authority.AUTHORIZATION_TTL_SECONDS
        ):
            return {"ok": False, "error": "ttl_seconds must be in (0, AUTHORIZATION_TTL_SECONDS]"}
        from gate.decide import decide

        def _run():
            return decide(
                None,
                request.get("proposal"),
                request.get("snapshot"),
                request.get("evidence"),
                mode=mode,
                run_missing_tests=False,
                noise_id=request.get("noise_id"),
                authorization_ttl=ttl,
                path_qdiscs=request.get("path_qdiscs"),
            )

        if self._test_thresholds:
            from bench.outcomes import test_thresholds

            with test_thresholds():
                record = _run()
        else:
            record = _run()
        return {"ok": True, "decision": record}


def main(argv=None):
    parser = argparse.ArgumentParser(description="VeriFix gate service (holds the signing key)")
    parser.add_argument("--key", required=True, help="Ed25519 private key, PKCS#8 PEM, mode 0600")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--socket", help="Unix-domain socket path (Linux)")
    transport.add_argument("--stdio", action="store_true", help="serve on stdin/stdout (offline checks)")
    parser.add_argument("--allow-uid", type=int, action="append", default=[], help="peer uid allowed to connect")
    parser.add_argument(
        "--offline-test-thresholds",
        action="store_true",
        help="grade with bench/test_thresholds.yaml (offline self-checks only; refused with --socket)",
    )
    args = parser.parse_args(argv)
    if args.offline_test_thresholds and args.socket:
        parser.error("--offline-test-thresholds is for offline checks over --stdio only")
    check_key_file(args.key)
    service = GateService(authority.load_private_key(args.key), offline_test_thresholds=args.offline_test_thresholds)
    from control import ipc

    if args.stdio:
        ipc.serve_stdio(service.handle)
    else:
        ipc.serve_unix(args.socket, service.handle, args.allow_uid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
