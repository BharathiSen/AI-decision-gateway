"""Offline self-check for the gated executor and the Stage 2 services.

    python -m control

1. Route helpers: flags, exit status, checked commands, exact restore guard.
2. Import allowlist: who may call execute_ungated / issue authorizations.
3. ExecutorService in-process: verification, ledger, compensation, latch.
4. Separate processes over stdio pipes: a gate service process holding the
   private key, an executor service process holding only the public key, and
   this process (the "orchestrator") holding neither. Forgeries from this
   process are refused by the executor process.
5. Unix socket + SO_PEERCRED and the root requirement: Linux only (NOT RUN
   elsewhere, reported as such).
No Mininet, no network commands, no model calls.
"""

import copy
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRIMARY = {"r1": "10.0.4.0/24 via 10.0.12.2 dev r1-eth1", "r4": "10.0.1.0/24 via 10.0.24.1 dev r4-eth0"}
BACKUP = {"r1": "10.0.4.0/24 via 10.0.13.2 dev r1-eth2", "r4": "10.0.1.0/24 via 10.0.34.1 dev r4-eth1"}
LINKS_UP = {"r1-r2": True, "r2-r4": True, "r1-r3": True, "r3-r4": True}
QDISCS = {"r1/r1-eth1": "qdisc netem 10: parent 5:1 limit 1000 delay 5ms", "r1/r1-eth2": "qdisc netem 10: parent 5:1 limit 1000 delay 5ms"}
REROUTE = {"action": "REROUTE", "target": "HostA_to_HostB", "reason": "check", "proposed_path": ["r1", "r3", "r4"]}
CALM = {
    "action": "DO_NOTHING",
    "target": "HostA_to_HostB",
    "reason": "healthy",
    "proposed_path": ["r1", "r2", "r4"],
    "causes": [{"cause_id": "H0", "predicted_tests": []}],
}
NOOP_EVIDENCE = [
    {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "connectivity": True, "error": None},
    {"test_id": "T4_mtu", "df_ping_ok": True, "error": None},
    {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None},
    {"test_id": "T10_speed", "mbit_per_sec": 50.0, "error": None},
]


def _snapshot(routes=None):
    from bench.run_episode import _snapshot as base

    snap = base()
    for item in snap["routes"]:
        item["route"] = (routes or PRIMARY)[item["node"]]
    return snap


def _check_routes():
    from bench import inject
    from control import routes

    assert routes.ROUTE_FLAGS == frozenset(inject._ROUTE_FLAGS), "route flag lists drifted apart"
    assert routes.specs_from_route_text(PRIMARY["r1"] + " linkdown onlink\n") == PRIMARY["r1"]
    assert routes.specs_from_route_text("") == ""
    try:
        routes.specs_from_route_text(PRIMARY["r1"] + "\n" + BACKUP["r1"])
    except routes.RouteReadError:
        pass
    else:
        raise AssertionError("two routes read as one")

    class CmdNode:
        def __init__(self, name, codes):
            self.name, self.codes, self.commands = name, list(codes), []

        def cmd(self, command):
            self.commands.append(command)
            code = self.codes.pop(0) if self.codes else 0
            return f"some output\n__VERIFIX_RC={code}\n"

    class Net:
        def __init__(self, r1=(), r4=()):
            self.nodes = {"r1": CmdNode("r1", r1), "r4": CmdNode("r4", r4)}

        def get(self, name):
            return self.nodes[name]

    assert routes.run_checked(CmdNode("x", [3]), "false") == ("some output\n", 3)
    assert routes.run_checked(type("N", (), {"cmd": lambda self, c: "no marker"})(), "x")[1] is None
    net = Net(r4=[2])
    try:
        routes.apply_path_checked(net, "backup")
    except routes.RouteCommandError as exc:
        assert exc.router == "r4", exc
    else:
        raise AssertionError("r4 failure not raised")
    assert len(net.nodes["r1"].commands) == 1 and len(net.nodes["r4"].commands) == 1
    try:
        routes.apply_path_checked(Net(r1=[1]), "backup")
    except routes.RouteCommandError as exc:
        assert exc.router == "r1"
    else:
        raise AssertionError("r1 failure not raised")
    # Unknown exit status (no marker) is a failure, never success.
    try:
        routes.apply_path_checked(type("N", (), {"get": lambda self, n: type("X", (), {"cmd": lambda s, c: "", "name": n})()})(), "backup")
    except routes.RouteCommandError:
        pass
    else:
        raise AssertionError("unknown exit status treated as success")
    for bad in ("", "10.9.9.0/24 via 1.2.3.4 dev x", "blackhole"):
        try:
            routes.restore_spec_checked(Net(), "r1", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"restore accepted {bad!r}")
    good = Net()
    routes.restore_spec_checked(good, "r1", PRIMARY["r1"] + " metric 7")
    assert good.nodes["r1"].commands[0].startswith("ip route replace " + PRIMARY["r1"] + " metric 7")
    print("  PASS  route helpers")


def _check_allowlists():
    from control.actions import UNGATED_CALLERS

    ungated = set()
    issuers = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith((".git/", ".claude/")):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bexecute_ungated\b", text):
            ungated.add(rel)
        if re.search(r"\b(issue_into|_issue|install_signer)\b", text) and "authority" in text:
            issuers.add(rel)
    allowed_ungated = set(UNGATED_CALLERS) | {"control/actions.py", "control/__init__.py", "control/__main__.py"}
    extra = sorted(ungated - allowed_ungated)
    assert not extra, f"execute_ungated used outside the allowlist: {extra}"
    allowed_issuers = {
        "gate/authority.py", "gate/decide.py", "gate/service.py",
        "bench/run_episode.py",  # _gate_stand_in: self-checks and live smokes only
        "control/__main__.py",
    }
    extra = sorted(issuers - allowed_issuers)
    assert not extra, f"authorizations issued outside the gate: {extra}"
    from control import actions

    assert not hasattr(actions, "execute"), "the old unnamed execute() must not come back"
    print("  PASS  ungated and issuing allowlists (accident guard, not a security boundary)")


def _fake_state(tmp, **extra):
    path = Path(tmp) / f"state-{time.perf_counter_ns()}.json"
    path.write_text(json.dumps(dict({"routes": dict(PRIMARY), "links": dict(LINKS_UP), "qdiscs": dict(QDISCS)}, **extra)), encoding="utf-8")
    return path


def _check_service_in_process(tmp):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from control.executor_service import ExecutorService, FakeBackend
    from control.ledger import FileLedger
    from gate import authority

    gate_key = Ed25519PrivateKey.generate()
    snap = _snapshot()

    def issue(verdict="allow", proposal=REROUTE, **kw):
        record = {"decision": verdict, "reason": "t", "proposal_action": proposal["action"], "tests_run": [], "checks": {}}
        p, s, e = authority.entry_copies(proposal, snap, [])
        kw.setdefault("path_qdiscs", QDISCS)
        return authority.issue_into(record, p, s, e, "full", private_key=gate_key, **kw)

    def service(state, ledger_name="ledger.jsonl"):
        ledger = Path(tmp) / ledger_name
        return ExecutorService(
            gate_key.public_key(), FileLedger(ledger), FakeBackend(state),
            latch_path=ledger.with_name(ledger.name + ".contaminated"),
        )

    def run(svc, decision, proposal=REROUTE):
        return svc.handle({"op": "execute", "decision": decision, "proposal": proposal, "evidence": [], "gate_snapshot": snap})

    state = _fake_state(tmp)
    svc = service(state)
    good = issue()
    reply = run(svc, good)
    assert reply["ok"] and reply["executed"] and reply["outcome"]["compensation"] is None, reply
    assert json.loads(state.read_text())["routes"] == BACKUP
    assert run(svc, good)["error"] == "decision already used"
    assert run(service(state), good)["error"] == "decision already used", "replay after restart"
    flipped = issue("block")
    flipped["decision"] = "allow"
    assert run(svc, flipped)["error"] == "verdict differs from the gate-issued authorization"
    other = Ed25519PrivateKey.generate()
    p, s, e = authority.entry_copies(REROUTE, snap, [])
    forged = authority.issue_into({"decision": "allow"}, p, s, e, "full", private_key=other)
    assert run(svc, forged)["error"] == "authorization was signed by an unknown key"
    assert run(svc, issue(now=time.time() - 1000))["error"] == "authorization has expired"
    assert run(svc, {"decision": "allow"})["error"] == "decision has no gate-issued authorization"
    assert svc.handle({"op": "execute"})["error"].startswith("execute needs")
    assert svc.handle({"op": "set_route", "router": "r1"})["error"].startswith("unknown op")
    bad_sig = copy.deepcopy(issue())
    bad_sig["authorization"]["signature"] = "é" * 128
    assert run(svc, bad_sig)["error"] == "authorization signature is invalid"
    # State changed since the gate's snapshot (the executor read it itself).
    moved_state = _fake_state(tmp, routes=dict(BACKUP))
    assert run(service(moved_state, "l2.jsonl"), issue())["error"] == "route changed since the gate decision"
    # Partial (r4 fails after r1 changed): compensated on r1 only, verified.
    partial_state = _fake_state(tmp, plan=["ok", "raise"])
    partial = run(service(partial_state, "l3.jsonl"), issue())
    out = partial["outcome"]
    assert partial["ok"] and out["partial_application"] is True and out["compensation"]["routers"] == ["r1"], partial
    assert out["compensation"]["verified"] is True and out["execution_error"].startswith("RuntimeError: r1") is False
    assert json.loads(partial_state.read_text())["routes"] == PRIMARY
    # Failed compensation latches the executor, also across a restart.
    stuck_state = _fake_state(tmp, plan=["ok", "raise"], restore_plan=["wrong"])
    stuck = run(service(stuck_state, "l4.jsonl"), issue())
    assert stuck["status"] == "contaminated" and stuck["outcome"]["compensation"]["verified"] is False, stuck
    again = run(service(stuck_state, "l4.jsonl"), issue())
    assert again["error"].startswith("executor is latched contaminated"), again
    assert service(stuck_state, "l4.jsonl").handle({"op": "status"})["contaminated"] is True
    # BLOCK never executes.
    assert run(service(_fake_state(tmp), "l5.jsonl"), issue("block"))["error"] == "gate did not authorize execution"
    # Path qdiscs: the signed baseline is required and rechecked against the executor's own read.
    assert run(service(_fake_state(tmp), "l6.jsonl"), issue(path_qdiscs=None))["error"] == (
        "authorization does not bind the path qdisc baseline"
    )
    shaped = _fake_state(tmp, qdiscs=dict(QDISCS, **{"r1/r1-eth2": "qdisc netem 10: parent 5:1 loss 20%"}))
    assert run(service(shaped, "l7.jsonl"), issue())["error"] == "path qdisc changed since the gate decision"
    unread_q = _fake_state(tmp, qdiscs=None)
    assert run(service(unread_q, "l8.jsonl"), issue())["error"] == "execution recheck could not read path qdiscs"
    # A latch that cannot be written still latches (in memory) and refuses the next execute.
    blocked_dir = Path(tmp) / "latch-is-a-dir.jsonl.contaminated"
    blocked_dir.mkdir()
    (blocked_dir / "keep").write_text("x", encoding="utf-8")
    memory_svc = ExecutorService(
        gate_key.public_key(), FileLedger(Path(tmp) / "latch-is-a-dir.jsonl"),
        FakeBackend(_fake_state(tmp, plan=["ok", "raise"], restore_plan=["wrong"])),
        latch_path=Path(tmp) / "latch-is-a-dir.jsonl.contaminated",
    )
    memory_svc.latch_path = Path(tmp) / "no-such-dir" / "latch"
    first_fail = run(memory_svc, issue())
    assert first_fail["status"] == "contaminated" and first_fail["latch_error"], first_fail
    assert run(memory_svc, issue())["error"].startswith("executor is latched contaminated")
    # Two ledger instances on one file (two executor processes): the second sees the first's entry.
    shared = Path(tmp) / "shared.jsonl"
    first_ledger, second_ledger = FileLedger(shared), FileLedger(shared)
    assert first_ledger.consume("abc", time.time() + 60) is None
    assert second_ledger.consume("abc", time.time() + 60) == "decision already used"
    print("  PASS  executor service (in process): verification, ledger, compensation, latch")


def _spawn(argv, env=None):
    from control.ipc import StdioClient

    return StdioClient([sys.executable, "-B"] + argv, cwd=str(ROOT), env=env)


def _check_separate_processes(tmp):
    from gate import authority

    private_pem, public_pem = authority.generate_keypair_pem()
    key_path, pub_path = Path(tmp) / "gate.pem", Path(tmp) / "gate.pub"
    key_path.write_bytes(private_pem)
    pub_path.write_bytes(public_pem)
    if os.name == "posix":
        os.chmod(key_path, 0o600)
    del private_pem  # this process plays the orchestrator: it keeps no signing key
    env = {k: v for k, v in os.environ.items() if not k.startswith("VERIFIX_GATE_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    state = _fake_state(tmp)
    ledger = Path(tmp) / "proc-ledger.jsonl"
    gate = _spawn(["-m", "gate.service", "--key", str(key_path), "--stdio", "--offline-test-thresholds"], env=env)
    executor_argv = [
        "-m", "control.executor_service", "--public-key", str(pub_path), "--ledger", str(ledger),
        "--stdio", "--backend", "fake", "--fake-state", str(state),
    ]
    executor = _spawn(executor_argv, env=env)
    try:
        assert gate.proc.pid != executor.proc.pid != os.getpid()
        snap = _snapshot()
        reply = gate.call({"op": "decide", "proposal": CALM, "snapshot": snap, "evidence": NOOP_EVIDENCE, "noise_id": "N0", "path_qdiscs": QDISCS})
        assert reply["ok"], reply
        allow = reply["decision"]
        assert allow["decision"] == "allow" and allow["authorization"]["issuer"] == "verifix-gate", allow.get("reason")

        def execute(decision, proposal=CALM, evidence=NOOP_EVIDENCE):
            return executor.call({
                "op": "execute", "decision": decision, "proposal": proposal,
                "evidence": evidence, "gate_snapshot": snap,
            })

        first = execute(allow)
        assert first["ok"] and first["outcome"]["decision_id"] == allow["authorization"]["decision_id"], first
        assert execute(allow)["error"] == "decision already used"
        # This process has no gate key: whatever it signs is refused.
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        p, s, e = authority.entry_copies(REROUTE, snap, [])
        forged = authority.issue_into({"decision": "allow"}, p, s, e, "full", private_key=Ed25519PrivateKey.generate())
        assert execute(forged, REROUTE, [])["error"] == "authorization was signed by an unknown key"
        # A hand-written allow with no authorization.
        assert execute({"decision": "allow"}, REROUTE, [])["error"] == "decision has no gate-issued authorization"
        # A gate BLOCK flipped by this process.
        blocked = gate.call({"op": "decide", "proposal": REROUTE, "snapshot": snap, "evidence": [], "noise_id": "N0"})["decision"]
        assert blocked["decision"] != "allow", blocked
        flipped = copy.deepcopy(blocked)
        flipped["decision"] = "allow"
        assert execute(flipped, REROUTE, [])["error"] == "verdict differs from the gate-issued authorization"
        flipped["authorization"]["decision"] = "allow"
        assert execute(flipped, REROUTE, [])["error"] == "authorization signature is invalid"
        # Evidence swapped after the decision.
        fresh = gate.call({"op": "decide", "proposal": CALM, "snapshot": snap, "evidence": NOOP_EVIDENCE, "noise_id": "N0", "path_qdiscs": QDISCS})["decision"]
        swapped = [dict(NOOP_EVIDENCE[0], packet_loss_percent=0.5)] + NOOP_EVIDENCE[1:]
        assert execute(fresh, CALM, swapped)["error"] == "evidence changed since the gate decision"
        # Expiry: the gate may only shorten the window.
        assert gate.call({"op": "decide", "proposal": CALM, "snapshot": snap, "evidence": [], "noise_id": "N0", "ttl_seconds": 9999})["ok"] is False
        short = gate.call({"op": "decide", "proposal": CALM, "snapshot": snap, "evidence": NOOP_EVIDENCE, "noise_id": "N0", "ttl_seconds": 1, "path_qdiscs": QDISCS})["decision"]
        time.sleep(1.3)
        assert execute(short)["error"] == "authorization has expired"
        # Unknown operation: the executor has no other way to change routes.
        assert executor.call({"op": "restore", "router": "r1", "spec": BACKUP["r1"]})["error"].startswith("unknown op")
    finally:
        executor.close()
    # Restarted executor process: the durable ledger still refuses the replay.
    executor = _spawn(executor_argv, env=env)
    try:
        replay = executor.call({"op": "execute", "decision": allow, "proposal": CALM, "evidence": NOOP_EVIDENCE, "gate_snapshot": _snapshot()})
        assert replay["error"] == "decision already used", replay
    finally:
        executor.close()
        gate.close()
    # The executor refuses to start when it can see the signing key.
    leaky = dict(env, VERIFIX_GATE_SIGNING_KEY=str(key_path))
    done = subprocess.run([sys.executable, "-B"] + executor_argv, cwd=str(ROOT), env=leaky, capture_output=True, text=True, input="")
    assert done.returncode != 0 and "must never see the gate signing key" in done.stderr, done.stderr
    # The gate refuses test thresholds outside the offline stdio transport.
    done = subprocess.run(
        [sys.executable, "-B", "-m", "gate.service", "--key", str(key_path), "--socket", str(Path(tmp) / "g.sock"), "--offline-test-thresholds"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    assert done.returncode != 0
    print("  PASS  separate processes: gate (private key) / executor (public key) / orchestrator (no key)")


def _check_discovery(tmp):
    """MininetBackend's /proc rules on a fake /proc (the real one needs the VM)."""
    from control.executor_service import MininetBackend

    proc = Path(tmp) / "proc"

    def node(pid, cmd, uid="0", start="111", shared_ns=False):
        base = proc / str(pid)
        (base / "ns").mkdir(parents=True)
        (base / "cmdline").write_bytes(b"bash\0--norc\0-is\0" + cmd.encode() + b"\0")
        (base / "status").write_text(f"Name:\tbash\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n", encoding="ascii")
        (base / "stat").write_text(f"{pid} (bash) S 1 " + " ".join(["0"] * 17) + f" {start} 0 0\n", encoding="ascii")
        if shared_ns:
            os.link(proc / "1" / "ns" / "net", base / "ns" / "net")
        else:
            (base / "ns" / "net").write_text(str(pid), encoding="ascii")

    node(1, "init")
    for offset, name in enumerate(("r1", "r2", "r3", "r4")):
        node(100 + offset, f"mininet:{name}")
    node(200, "mininet:r1", uid="1001")            # spoofed name, not root
    node(201, "mininet:r2", shared_ns=True)        # spoofed name, host namespace
    backend = MininetBackend(proc_root=proc)
    assert {name: pid for name, (pid, _s) in backend._nodes.items()} == {"r1": 100, "r2": 101, "r3": 102, "r4": 103}
    assert backend.get("r1").pid == 100
    # The node process changed after discovery (restart / pid reuse): refused.
    (proc / "100" / "stat").write_text("100 (bash) S 1 " + " ".join(["0"] * 17) + " 999 0 0\n", encoding="ascii")
    try:
        backend.get("r1")
    except RuntimeError as exc:
        assert "changed since discovery" in str(exc)
    else:
        raise AssertionError("changed node process accepted")
    # A second root shell in its own namespace claiming r3: discovery refuses to guess.
    node(202, "mininet:r3")
    try:
        MininetBackend(proc_root=proc)
    except RuntimeError as exc:
        assert "two Mininet shells claim r3" in str(exc)
    else:
        raise AssertionError("ambiguous node accepted")
    print("  PASS  Mininet node discovery rules (fake /proc): uid 0, own netns, start time re-checked")


def _check_linux_only(tmp):
    from control import ipc

    assert ipc.peer_allowed((1, 1001, 1001), [1001]) and not ipc.peer_allowed((1, 0, 0), [1001])
    try:
        ipc.serve_unix(str(Path(tmp) / "x.sock"), lambda m: m, [])
    except (ValueError, OSError):
        pass  # no allowed uids (Linux) or no Unix sockets (elsewhere): refused either way
    else:
        raise AssertionError("served without a peer allowlist")
    if not (hasattr(socket, "AF_UNIX") and hasattr(socket, "SO_PEERCRED")):
        print("  NOT RUN  Unix socket + SO_PEERCRED (needs Linux)")
        print("  NOT RUN  privilege checks: executor needs root, orchestrator must not be root (needs Linux)")
        return
    me = os.geteuid()
    for allowed, expect_ok in (([me], True), ([me + 4242], False)):
        path = str(Path(tmp) / f"s{expect_ok}.sock")
        thread = threading.Thread(target=ipc.serve_unix, args=(path, lambda m: {"ok": True, "echo": m}, allowed), kwargs={"once": True})
        thread.start()
        for _ in range(100):
            if os.path.exists(path):
                break
            time.sleep(0.02)
        assert oct(os.stat(path).st_mode & 0o777) == "0o660"
        client = ipc.UnixClient(path)
        reply = client.call({"op": "status"})
        client.close()
        thread.join(5)
        assert reply.get("ok") is expect_ok, (allowed, reply)
    print("  PASS  Unix socket + SO_PEERCRED: listed uid served, other uid refused")
    if me != 0:
        done = subprocess.run(
            [sys.executable, "-B", "-m", "control.executor_service", "--public-key", "x", "--ledger", str(Path(tmp) / "l"),
             "--socket", str(Path(tmp) / "e.sock"), "--backend", "mininet", "--allow-uid", str(me)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        assert done.returncode != 0 and "needs root" in done.stderr, done.stderr
        print("  PASS  executor refuses the Mininet backend without root")
    else:
        try:
            ipc.assert_unprivileged("orchestrator")
        except PermissionError:
            print("  PASS  orchestrator refuses to run as root")
        else:
            raise AssertionError("orchestrator ran as root")


def main():
    from bench.outcomes import test_thresholds

    tmp = tempfile.mkdtemp(prefix="verifix-control-")
    try:
        with test_thresholds():
            _check_routes()
            _check_allowlists()
            _check_service_in_process(tmp)
            _check_separate_processes(tmp)
            _check_discovery(tmp)
            _check_linux_only(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("control (gated executor, Stage 2 services): PASS")


if __name__ == "__main__":
    main()
