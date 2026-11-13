"""Executor service: the only process that changes routes in the separated deployment.

    python -m control.executor_service --public-key /etc/verifix/gate_ed25519.pub \\
        --ledger /var/lib/verifix/ledger.jsonl --socket /run/verifix/executor.sock \\
        --allow-uid 1001 --backend mininet
    python -m control.executor_service ... --stdio --backend fake --fake-state STATE.json   # offline

It holds ONLY the gate's public key. For every request it independently:
  1. revalidates the proposal (agent.validate + record schema);
  2. verifies the gate-issued Ed25519 authorization (gate/authority.py) against
     the proposal, the gate's snapshot, the evidence, the action, target,
     path, gate config hash and validity window;
  3. consumes the decision id in a durable ledger (refused if seen before,
     also after a restart);
  4. reads the routes and path links itself and rechecks them against the
     snapshot the gate signed;
  5. runs the action with checked commands and compensates a partial one
     (control/gated.run_lifecycle), immediately: there is no research
     harness here, so post-action measurement is the caller's job afterwards.
An unverified compensation latches the service as contaminated (a file next
to the ledger, so a restart keeps it) and every later execute is refused
until an operator removes the latch after repairing the network.

Requests:
  {"op": "execute", "decision", "proposal", "evidence", "gate_snapshot"}
  {"op": "status"}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import gated  # noqa: E402
from control.ledger import FileLedger, LedgerError  # noqa: E402
from control.routes import RouteReadError, specs_from_snapshot  # noqa: E402

PATH_LINK_INTERFACES = {
    "r1-r2": (("r1", "r1-eth1"), ("r2", "r2-eth0")),
    "r2-r4": (("r2", "r2-eth1"), ("r4", "r4-eth0")),
    "r1-r3": (("r1", "r1-eth2"), ("r3", "r3-eth0")),
    "r3-r4": (("r3", "r3-eth1"), ("r4", "r4-eth1")),
}


class ExecutorService:
    def __init__(self, public_key, ledger, backend, latch_path=None, clock=time.time):
        if public_key is None:
            raise ValueError("executor needs the gate public key")
        self.public_key = public_key
        self.ledger = ledger
        self.backend = backend
        self.latch_path = Path(latch_path) if latch_path else None
        self.clock = clock
        self._latched_in_memory = False

    def contaminated(self):
        return self._latched_in_memory or bool(self.latch_path and self.latch_path.exists())

    def _latch(self, outcome):
        """Latch in memory first (cannot fail), then durably. A failed write keeps the memory latch."""
        self._latched_in_memory = True
        if not self.latch_path:
            return None
        tmp = self.latch_path.with_name(self.latch_path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome, sort_keys=True, default=str))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.latch_path)
        except OSError as exc:
            return f"latch file not written ({exc}); latched in memory until restart"
        return None

    def handle(self, request):
        op = request.get("op")
        if op == "status":
            return {"ok": True, "contaminated": self.contaminated()}
        if op != "execute":
            return {"ok": False, "error": f"unknown op {op!r}"}
        return self._execute(request)

    def _refuse(self, error, **extra):
        return dict({"ok": False, "executed": False, "error": error}, **extra)

    def _execute(self, request):
        from bench.contracts import validate_record
        from control.recheck import assess_recheck
        from gate.authority import authorization_errors

        if self.contaminated():
            return self._refuse("executor is latched contaminated; repair the network and clear the latch")
        decision, proposal = request.get("decision"), request.get("proposal")
        evidence, gate_snapshot = request.get("evidence"), request.get("gate_snapshot")
        if not isinstance(decision, dict) or not isinstance(proposal, dict) or not isinstance(gate_snapshot, dict):
            return self._refuse("execute needs decision, proposal and gate_snapshot")
        from agent.validate import validate_proposal

        ok, errors = validate_proposal(proposal)
        if ok:
            ok, errors = validate_record("proposal", proposal)
        if not ok:
            return self._refuse("invalid proposal: " + "; ".join(errors))
        started = time.perf_counter()
        signed = decision.get("authorization") if isinstance(decision.get("authorization"), dict) else {}
        path_qdiscs = signed.get("path_qdiscs")
        errors = authorization_errors(
            decision, gate_snapshot, proposal, evidence, now=self.clock(), public_key=self.public_key,
            path_qdiscs=path_qdiscs,
        )
        if not errors and not isinstance(path_qdiscs, dict):
            errors = ["authorization does not bind the path qdisc baseline"]
        verify_seconds = time.perf_counter() - started
        if errors:
            return self._refuse(errors[0], authorization_verify_seconds=verify_seconds)
        authorization = decision["authorization"]
        try:
            with gated.single_flight():
                try:
                    used = self.ledger.consume(authorization["decision_id"], authorization["expires_at"])
                except LedgerError as exc:
                    used = f"replay ledger unavailable: {exc}"
                if used:
                    return self._refuse(used, authorization_verify_seconds=verify_seconds)
                try:
                    current = self.backend.read_state()
                except Exception as exc:
                    return self._refuse(f"state recheck failed: {type(exc).__name__}: {exc}")
                # Approved = the gate's snapshot plus the signed qdisc baseline.
                assessment = assess_recheck(dict(gate_snapshot, qdiscs=path_qdiscs), current, proposal)
                if not assessment["state_verified"]:
                    return self._refuse(assessment["reason"])
                try:
                    routes_before = specs_from_snapshot(current)
                except RouteReadError as exc:
                    return self._refuse(f"route capture failed: {exc}")
                capability = gated.mint(authorization, proposal)
                try:
                    outcome = gated.run_lifecycle(
                        capability,
                        routes_before,
                        execute=self.backend.execute,
                        read_routes=lambda: specs_from_snapshot(self.backend.read_state()),
                        restore_route=self.backend.restore,
                    )
                except gated.CompensationFailed as exc:
                    latch_error = self._latch(exc.outcome)
                    return self._refuse(
                        f"contaminated: {exc}", status="contaminated", outcome=exc.outcome,
                        authorization_verify_seconds=verify_seconds, latch_error=latch_error,
                    )
        except gated.ExecutionBusy as exc:
            return self._refuse(str(exc))
        result = outcome.get("execution_result") or {}
        return {
            "ok": True,
            "executed": isinstance(result, dict) and result.get("executed") is True,
            "outcome": outcome,
            "authorization_verify_seconds": verify_seconds,
        }


class FakeBackend:
    """Offline backend: router state in a JSON file the test can read and script.

    State: {"routes": {r1, r4}, "links": {name: bool}, "plan": [...], "restore_plan": [...]}.
    plan steps per router: "ok" | "raise" | "silent"; restore steps: "ok" | "raise" | "wrong".
    """

    def __init__(self, state_path):
        self.path = Path(state_path)

    def _load(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state):
        self.path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def read_state(self):
        state = self._load()
        from network import controller as ctl

        return {
            "routes": [
                {"node": router, "prefix": cfg["dest"], "route": state["routes"][router]}
                for router, cfg in ctl.ROUTE_TARGETS.items()
            ],
            "links": [{"name": name, "up": up} for name, up in state["links"].items()],
            "qdiscs": state.get("qdiscs"),
        }

    def execute(self, capability):
        from network import controller as ctl

        state = self._load()
        state.setdefault("log", []).append(["execute", capability.decision_id, capability.action])
        if capability.action != "REROUTE":
            self._save(state)
            return {"executed": False, "action": capability.action}
        path = capability.reroute_path
        try:
            for router, cfg in ctl.ROUTE_TARGETS.items():
                step = state["plan"].pop(0) if state.get("plan") else "ok"
                if step == "raise":
                    raise RuntimeError(f"{router}: `ip route replace` exited 2")
                if step == "ok":
                    hop = cfg[path]
                    state["routes"][router] = f"{cfg['dest']} via {hop['via']} dev {hop['dev']}"
        finally:
            self._save(state)
        return {"executed": True, "action": "REROUTE", "path": path}

    def restore(self, router, spec):
        state = self._load()
        state.setdefault("log", []).append(["restore", router])
        step = state["restore_plan"].pop(0) if state.get("restore_plan") else "ok"
        try:
            if step == "raise":
                raise RuntimeError(f"{router}: restore exited 2")
            if step == "ok":
                state["routes"][router] = spec
        finally:
            self._save(state)


class MininetBackend:
    """Acts on a running Mininet topology from a separate root process (Linux, VM).

    Nodes are found in /proc by their shells' command line (`mininet:<name>`),
    never by a pid the orchestrator supplies. Because any process can set its
    command line, a candidate must also run as uid 0 and live in a network
    namespace other than init's; its start time is recorded and re-checked
    before every use, so a restarted topology or a reused pid is refused.
    Commands run in the node through `mnexec -a <pid>`, exit status checked.
    Still trusts root-owned processes on the host: run on a single-tenant VM.
    NOT RUN offline: needs the VM (the discovery rules are checked on a fake /proc).
    """

    NODES = ("r1", "r2", "r3", "r4")

    def __init__(self, proc_root="/proc"):
        self.proc = Path(proc_root)
        self._nodes = self._discover()

    def _identity(self, pid):
        """(name, start time) of a trusted Mininet node shell, else None."""
        base = self.proc / str(pid)
        try:
            cmdline = (base / "cmdline").read_bytes().split(b"\0")
            status = (base / "status").read_text(encoding="ascii", errors="replace")
            stat_text = (base / "stat").read_text(encoding="ascii", errors="replace")
            net_ns = os.stat(base / "ns" / "net").st_ino
            init_ns = os.stat(self.proc / "1" / "ns" / "net").st_ino
        except OSError:
            return None
        names = [part.split(b":", 1)[1].decode("ascii", "replace") for part in cmdline if part.startswith(b"mininet:")]
        if len(names) != 1 or names[0] not in self.NODES:
            return None
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        if uid_line.split()[1:2] != ["0"]:
            return None
        if net_ns == init_ns:
            return None
        fields = stat_text.rsplit(")", 1)[-1].split()
        if len(fields) < 20:
            return None
        return names[0], fields[19]

    def _discover(self):
        found = {}
        for entry in self.proc.iterdir():
            if not entry.name.isdigit():
                continue
            identity = self._identity(int(entry.name))
            if identity is None:
                continue
            name, started = identity
            if name in found:
                raise RuntimeError(f"two Mininet shells claim {name}")
            found[name] = (int(entry.name), started)
        missing = [name for name in self.NODES if name not in found]
        if missing:
            raise RuntimeError(f"Mininet nodes not found: {missing}")
        return found

    class _Node:
        def __init__(self, name, pid):
            self.name, self.pid = name, pid

        def pexec(self, argv):
            import subprocess

            done = subprocess.run(["mnexec", "-a", str(self.pid)] + list(argv), capture_output=True, text=True, timeout=30)
            return done.stdout, done.stderr, done.returncode

    def get(self, name):
        pid, started = self._nodes[name]
        if self._identity(pid) != (name, started):
            raise RuntimeError(f"{name}: node process {pid} changed since discovery; refusing to act")
        return self._Node(name, pid)

    def read_state(self):
        from control.recheck import PATH_INTERFACES, qdisc_config
        from control.routes import read_specs
        from network import controller as ctl

        specs = read_specs(self)
        links = []
        for name, ends in PATH_LINK_INTERFACES.items():
            up = True
            for node, intf in ends:
                out, _err, code = self.get(node).pexec(["cat", f"/sys/class/net/{intf}/operstate"])
                up = up and code == 0 and out.strip() == "up"
            links.append({"name": name, "up": up})
        qdiscs = {}
        for node, intf in PATH_INTERFACES:
            out, err, code = self.get(node).pexec(["tc", "qdisc", "show", "dev", intf])
            if code != 0:
                raise RuntimeError(f"{node}: tc qdisc show dev {intf} exited {code}: {err.strip()}")
            qdiscs[f"{node}/{intf}"] = qdisc_config(out)
        return {
            "routes": [{"node": r, "prefix": ctl.ROUTE_TARGETS[r]["dest"], "route": specs[r]} for r in ctl.ROUTE_TARGETS],
            "links": links,
            "qdiscs": qdiscs,
        }

    def execute(self, capability):
        from control.actions import execute_gated

        return execute_gated(self, capability)

    def restore(self, router, spec):
        from control.routes import restore_spec_checked

        restore_spec_checked(self, router, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description="VeriFix executor service (holds only the gate public key)")
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--ledger", required=True, help="durable replay ledger (JSON lines)")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--socket")
    transport.add_argument("--stdio", action="store_true")
    parser.add_argument("--allow-uid", type=int, action="append", default=[])
    parser.add_argument("--backend", choices=("mininet", "fake"), required=True)
    parser.add_argument("--fake-state", help="state file for --backend fake")
    args = parser.parse_args(argv)
    from gate.authority import SIGNING_KEY_ENV, load_public_key

    if os.environ.get(SIGNING_KEY_ENV):
        parser.error(f"{SIGNING_KEY_ENV} is set: the executor must never see the gate signing key")
    if args.backend == "mininet":
        if args.stdio:
            parser.error("--backend mininet is served on --socket only")
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            parser.error("--backend mininet needs root (CAP_NET_ADMIN) on the Mininet VM")
        backend = MininetBackend()
    else:
        if args.socket:
            parser.error("--backend fake is for offline checks over --stdio only")
        backend = FakeBackend(args.fake_state)
    service = ExecutorService(
        load_public_key(args.public_key),
        FileLedger(args.ledger),
        backend,
        latch_path=Path(args.ledger).with_name(Path(args.ledger).name + ".contaminated"),
    )
    from control import ipc

    if args.stdio:
        ipc.serve_stdio(service.handle)
    else:
        ipc.serve_unix(args.socket, service.handle, args.allow_uid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
