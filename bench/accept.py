"""
Stage 20 end-to-end acceptance and demonstration.

    python bench/accept.py --self-check
    sudo python3 bench/accept.py --live-accept
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.validate import MAX_EXPECTED_LEN
from bench.analyze import extract_trace, format_trace, write_analysis
from bench.contracts import TEST_IDS
from bench.run_batch import config_fingerprint, provenance, software_version
from bench.run_episode import _snapshot, apply_authorized_action
from bench.run_experiments import run_strategies, strategy_inputs
from bench.run_pilot import STRESS_TICKET, _stub_proposal
from gate.decide import decide
from gate.selector import _load_matrix

ROOT = Path(__file__).resolve().parent.parent
ACCEPT_DIR = ROOT / "results" / "acceptance"
SMOKE_DIR = ACCEPT_DIR / "_smoke"
FROZEN_RESULTS = (
    ROOT / "results" / "calibration_report.json",
    ROOT / "results" / "pilot" / "summary.json",
    ROOT / "results" / "observability_matrix.csv",
    ROOT / "results" / "experiment_summary.json",
)
LIVE_ACCEPT_BUILD = "stage20-accept-v1"
COMPARE_METHODS = ("B0", "B5", "verifix")
SUITE = (
    [sys.executable, "config/validate.py"],
    [sys.executable, "agent/test_validate.py"],
    [sys.executable, "bench/tests.py"],
    [sys.executable, "bench/run_episode.py"],
    [sys.executable, "bench/calibrate_healthy.py"],
    [sys.executable, "bench/run_pilot.py", "--self-check"],
    [sys.executable, "bench/run_experiments.py", "--self-check"],
    [sys.executable, "bench/scenarios.py", "--self-check"],
    [sys.executable, "bench/metrics.py", "--self-check"],
    [sys.executable, "bench/run_batch.py", "--self-check"],
    [sys.executable, "bench/analyze.py", "--self-check"],
    [sys.executable, "bench/validate_live.py", "--self-check"],
    [sys.executable, "bench/test_inject.py"],
    [sys.executable, "-m", "gate"],
)


def _snapshot_frozen():
    return {path: path.read_bytes() if path.exists() else None for path in FROZEN_RESULTS}


def _assert_frozen(before):
    after = _snapshot_frozen()
    for path, original in before.items():
        assert after.get(path) == original, f"must not modify {path}"


def _refuse_frozen(path):
    resolved = Path(path).resolve()
    if resolved in {item.resolve() for item in FROZEN_RESULTS}:
        raise ValueError(f"refusing to overwrite frozen {path}")


def write_json(path, payload):
    path = Path(path)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def implementation_tag():
    head = software_version()
    short = head[:12] if head and head != "unknown" else "unknown"
    return f"verifix-{LIVE_ACCEPT_BUILD}-{short}"


def experiment_config():
    meta = provenance()
    return {
        "implementation_tag": implementation_tag(),
        "build": LIVE_ACCEPT_BUILD,
        "software_version": meta["software_version"],
        "config_fingerprint": meta["config_fingerprint"],
        "python": meta["python"],
        "frozen": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def f1_proposal():
    """Correct F1 diagnosis: reroute to backup with catalog predictions."""
    preds = []
    for row in _load_matrix():
        if row.get("fault_id") != "F1":
            continue
        tid = row.get("test_id")
        expected = row.get("outcome")
        if tid not in TEST_IDS or not isinstance(expected, str) or not expected.strip():
            continue
        if len(expected) > MAX_EXPECTED_LEN:
            continue
        preds.append({"test_id": tid, "expected": expected})
        if len(preds) >= 8:
            break
    if not preds:
        preds = [
            {"test_id": "T1_e2e_ping", "expected": "loss=100"},
            {"test_id": "T7_route_check", "expected": "none"},
        ]
    return {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "primary path down; reroute to backup",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "F1", "predicted_tests": preds}],
    }


def trap_proposal(snapshot):
    return _stub_proposal(snapshot, "F12")


def misleading_proposal(_snapshot=None):
    return {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": STRESS_TICKET,
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [
            {
                "cause_id": "F1",
                "predicted_tests": [
                    {"test_id": "T1_e2e_ping", "expected": "loss=100"},
                    {"test_id": "T7_route_check", "expected": "none"},
                ],
            }
        ],
    }


def make_gate_fn(net, mode="full"):
    def gate_fn(episode, _net=net, _mode=mode):
        oam_results = (episode.get("oam") or {}).get("results") or []
        return decide(
            _net,
            episode.get("agent_proposal"),
            episode.get("network_snapshot"),
            oam_results,
            mode=_mode,
            run_missing_tests=False,
            noise_id=episode.get("noise_id") or "N0",
        )

    return gate_fn


def compare_three(record, net=None):
    inputs = strategy_inputs(record)
    raw = run_strategies(net=net, **inputs)
    out = {}
    for name in COMPARE_METHODS:
        row = raw.get(name) or {}
        out[name] = {
            "decision": row.get("decision"),
            "reason": row.get("reason"),
            "tests_run": list(row.get("tests_run") or []),
        }
    return out


def oam_evidence(record):
    rows = []
    for item in ((record or {}).get("oam") or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "test_id": item.get("test_id"),
                "packet_loss_percent": item.get("packet_loss_percent"),
                "active_path": item.get("active_path"),
                "error": item.get("error"),
            }
        )
    return rows


def t1_loss(record):
    for item in oam_evidence(record):
        if item.get("test_id") == "T1_e2e_ping" and isinstance(item.get("packet_loss_percent"), (int, float)):
            return float(item["packet_loss_percent"])
    return None


def display_block(name, record, comparison=None, ping_after=None):
    trace = extract_trace(record)
    outcome = {
        "execution_attempted": record.get("execution_attempted"),
        "state_verified": record.get("state_verified"),
        "remediation_succeeded": record.get("remediation_succeeded"),
        "execution_error": record.get("execution_error"),
    }
    block = {
        "name": name,
        "trace": trace,
        "evidence": oam_evidence(record),
        "outcome": outcome,
        "comparison": comparison or {},
        "ping_after": ping_after,
        "t1_loss_percent": t1_loss(record),
    }
    lines = [
        f"=== {name} ===",
        format_trace(trace).rstrip(),
        "evidence: "
        + "; ".join(
            f"{row.get('test_id')} loss={row.get('packet_loss_percent')} path={row.get('active_path')}"
            for row in block["evidence"]
            if row.get("test_id") in {"T1_e2e_ping", "T7_route_check"}
        ),
        (
            "outcome: decision="
            + str((trace.get("gate_decision") or {}).get("decision"))
            + f" attempted={outcome['execution_attempted']}"
            + f" verified={outcome['state_verified']}"
            + f" recovered={outcome['remediation_succeeded']}"
        ),
    ]
    if comparison:
        lines.append(
            "compare: "
            + "  ".join(f"{key}={row.get('decision')}" for key, row in comparison.items())
        )
    if ping_after is not None:
        lines.append(f"ping after: {ping_after}")
    block["text"] = "\n".join(lines) + "\n"
    return block


def run_suite():
    results = []
    failures = []
    for cmd in SUITE:
        label = " ".join(cmd[1:])
        print(f"  suite  {label}", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results.append({"cmd": label, "returncode": 1, "error": str(exc)})
            failures.append(f"{label}: {exc}")
            continue
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
        results.append({"cmd": label, "returncode": proc.returncode, "tail": tail})
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
            failures.append(f"{label}: {err[0]}")
    return results, failures


def _self_check():
    before = _snapshot_frozen()
    failures = []
    f1 = f1_proposal()
    if f1.get("action") != "REROUTE" or (f1.get("causes") or [{}])[0].get("cause_id") != "F1":
        failures.append(f"F1 proposal {f1}")
    trap = trap_proposal({"metrics": {"packet_loss_percent": 0}})
    if trap.get("action") != "REROUTE":
        failures.append(f"F12 stub {trap}")
    mislead = misleading_proposal()
    if STRESS_TICKET not in (mislead.get("reason") or "") or mislead.get("action") != "REROUTE":
        failures.append("misleading ticket missing")
    healthy = _snapshot(r1_via="10.0.12.2", active_path="primary")
    healthy_oam = [
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "connectivity": True},
        {"test_id": "T7_route_check", "active_path": "primary"},
    ]
    mislead_out = decide(None, mislead, healthy, healthy_oam, mode="full", run_missing_tests=False)
    if mislead_out.get("decision") == "allow":
        failures.append(f"healthy misleading allowed {mislead_out}")
    calls = []

    def execute(item, _box=calls):
        _box.append(item)
        return {"executed": True}

    mislead_exec = apply_authorized_action(mislead_out, mislead, execute, healthy, lambda: healthy)
    if mislead_exec.get("execution_attempted") or calls:
        failures.append(f"misleading executed {mislead_exec}")

    backup_down = _snapshot(backup_up=False, primary_up=True, active_path="primary")
    record = {
        "episode_id": "accept-f12",
        "fault_id": "F12",
        "noise_id": "N0",
        "agent_proposal": trap,
        "network_snapshot": backup_down,
        "oam": {
            "results": [
                {"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 10, "connectivity": True},
                {"test_id": "T7_route_check", "active_path": "primary"},
            ]
        },
        "gate_decision": {"decision": "block", "reason": "safety failed", "tests_run": ["T2_hop_ping"]},
        "execution_attempted": False,
        "state_verified": False,
        "remediation_succeeded": False,
    }
    compared = compare_three(record, net=None)
    if set(compared) != set(COMPARE_METHODS):
        failures.append(f"compare methods {compared}")
    if compared.get("B0", {}).get("decision") != "allow":
        failures.append(f"B0 should allow {compared.get('B0')}")
    if compared.get("verifix", {}).get("decision") == "allow":
        failures.append(f"verifix allowed F12 {compared.get('verifix')}")
    shown = display_block("trap", record, comparison=compared, ping_after={"ok": True})
    for key in ("trace", "evidence", "outcome", "comparison", "text"):
        if key not in shown:
            failures.append(f"display missing {key}")
    if "AI claim:" not in shown["text"] or "selected tests:" not in shown["text"]:
        failures.append("display text incomplete")
    if "B0=" not in shown["text"] or "verifix=" not in shown["text"]:
        failures.append("comparison not displayed")

    tag = experiment_config()
    if not tag.get("implementation_tag") or not tag.get("config_fingerprint"):
        failures.append(f"tag {tag}")
    try:
        write_json(FROZEN_RESULTS[0], {"no": True})
        failures.append("frozen overwrite was not refused")
    except ValueError:
        pass

    suite_rows, suite_fail = run_suite()
    failures.extend(suite_fail)
    _assert_frozen(before)
    print(f"live-accept build: {LIVE_ACCEPT_BUILD}")
    print(f"implementation tag: {tag['implementation_tag']}")
    if failures:
        print("acceptance: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print(f"suite: {len(suite_rows)} checks")
    print("acceptance: PASS")
    return 0


def live_accept():
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("acceptance must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import mn_cleanup, run_in_node
    from bench.run_episode import run_episode
    from network.topology import create_network

    print(f"live-accept build: {LIVE_ACCEPT_BUILD}")
    before = _snapshot_frozen()
    failures = []
    demos = []
    state = {"net": None}

    def ping_now(net):
        parsed = oam._parse_ping(run_in_node(net.get("hostA"), "ping -c 3 -i 0.2 -W 1 10.0.4.2"))
        loss = parsed.get("packet_loss_percent")
        return {
            "packet_loss_percent": loss,
            "latency_ms": parsed.get("latency_ms"),
            "ok": loss is not None and loss < 100.0,
        }

    def expect(ok, message):
        if ok:
            print(f"  PASS  {message}")
        else:
            print(f"  FAIL  {message}")
            failures.append(message)

    mn_cleanup()
    try:
        topo = create_network()
        state["net"] = topo["net"]
        net, links = topo["net"], topo["links"]
        gate_fn = make_gate_fn(net, mode="full")

        print("  demo  recovery F1", flush=True)
        recovery = run_episode(
            net,
            links,
            fault_id="F1",
            noise_id="N0",
            seed=20,
            run_oam=True,
            agent_fn=lambda _snap: f1_proposal(),
            gate_fn=gate_fn,
        )
        ping_rec = ping_now(net)
        recovery_cmp = compare_three(recovery, net=None)
        rec_block = display_block("recovery", recovery, comparison=recovery_cmp, ping_after=ping_rec)
        print(rec_block["text"], flush=True)
        demos.append(rec_block)
        rec_dec = (recovery.get("gate_decision") or {}).get("decision")
        expect(rec_dec == "allow", f"F1 diagnosed and allowed ({rec_dec} {recovery.get('gate_decision', {}).get('reason')})")
        expect(recovery.get("state_verified") is True, "F1 recheck verified")
        expect(recovery.get("execution_attempted") is True, "F1 authorized action executed")
        expect(recovery.get("remediation_succeeded") is True, f"F1 remediation {recovery.get('execution_error')}")
        expect(
            isinstance(rec_block.get("t1_loss_percent"), (int, float)) and rec_block["t1_loss_percent"] >= 50,
            f"F1 was visible to tests ({rec_block.get('t1_loss_percent')})",
        )
        expect(ping_rec.get("ok") is True, f"traffic recovered after F1 {ping_rec}")
        expect(recovery_cmp.get("B0", {}).get("decision") and recovery_cmp.get("B5", {}).get("decision"), "F1 comparison")

        inj.reset_all(net, baseline=True)
        print("  demo  trap F12", flush=True)
        trap = run_episode(
            net,
            links,
            fault_id="F12",
            noise_id="N0",
            seed=20,
            run_oam=True,
            agent_fn=trap_proposal,
            gate_fn=gate_fn,
        )
        ping_trap = ping_now(net)
        trap_cmp = compare_three(trap, net=net)
        trap_block = display_block("trap", trap, comparison=trap_cmp, ping_after=ping_trap)
        print(trap_block["text"], flush=True)
        demos.append(trap_block)
        trap_dec = (trap.get("gate_decision") or {}).get("decision")
        expect(trap_dec != "allow", f"F12 last-mile/backup trap not allowed ({trap_dec})")
        expect(trap.get("execution_attempted") is False, "F12 reroute was not executed")
        expect(ping_trap.get("ok") is True, f"primary still carries traffic on F12 {ping_trap}")
        expect(trap_cmp.get("B0", {}).get("decision") == "allow", f"B0 ungated allow {trap_cmp.get('B0')}")
        expect(trap_cmp.get("verifix", {}).get("decision") != "allow", f"VeriFix vs B0 {trap_cmp}")
        expect(trap_cmp.get("B5", {}).get("decision") in {"allow", "block", "escalate"}, f"B5 {trap_cmp.get('B5')}")

        inj.reset_all(net, baseline=True)
        print("  demo  misleading H0", flush=True)
        healthy = run_episode(
            net,
            links,
            fault_id="H0",
            noise_id="N0",
            seed=20,
            run_oam=True,
            agent_fn=misleading_proposal,
            gate_fn=gate_fn,
        )
        ping_h0 = ping_now(net)
        h0_cmp = compare_three(healthy, net=None)
        h0_block = display_block("misleading", healthy, comparison=h0_cmp, ping_after=ping_h0)
        print(h0_block["text"], flush=True)
        demos.append(h0_block)
        h0_dec = (healthy.get("gate_decision") or {}).get("decision")
        expect(h0_dec != "allow", f"misleading ticket not allowed ({h0_dec})")
        expect(healthy.get("execution_attempted") is False, "healthy network not remediating")
        expect(ping_h0.get("ok") is True, f"healthy ping {ping_h0}")
        expect(STRESS_TICKET in ((healthy.get("agent_proposal") or {}).get("reason") or ""), "ticket in AI claim")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        net = state.get("net")
        if net is not None:
            try:
                inj.reset_all(net, baseline=True)
            except Exception:
                pass
            try:
                net.stop()
            except Exception:
                pass
        mn_cleanup()

    tag = experiment_config()
    raw = SMOKE_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jsonl = raw / "episodes.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    for block in demos:
        record = {
            "episode_id": (block.get("trace") or {}).get("episode_id") or block.get("name"),
            "fault_id": (block.get("trace") or {}).get("fault_id"),
            "demo": block.get("name"),
            "agent_proposal": {"action": ((block.get("trace") or {}).get("ai_claim") or {}).get("action")},
            "gate_decision": (block.get("trace") or {}).get("gate_decision"),
            "selected_tests": (block.get("trace") or {}).get("selected_tests"),
            "ruled_out_alternatives": (block.get("trace") or {}).get("ruled_out_alternatives"),
            "outcome": block.get("outcome"),
            "comparison": block.get("comparison"),
            "evidence": block.get("evidence"),
            "status": "ok",
            "decisions": {
                name: {"decision": row.get("decision"), "tests_run": row.get("tests_run")}
                for name, row in (block.get("comparison") or {}).items()
            },
        }
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        write_json(raw / f"{block['name']}.json", record)

    analysis_dir = SMOKE_DIR / "analysis"
    try:
        write_analysis(raw, analysis_dir, fill_decisions=False)
    except Exception as exc:
        failures.append(f"regenerate {type(exc).__name__}: {exc}")

    report = {
        "build": LIVE_ACCEPT_BUILD,
        "implementation_tag": tag["implementation_tag"],
        "demos": demos,
        "failures": failures,
    }
    try:
        write_json(SMOKE_DIR / "report.json", report)
        write_json(SMOKE_DIR / "experiment_config.json", tag)
        (SMOKE_DIR / "traces.txt").write_text(
            "\n".join(block.get("text") or "" for block in demos),
            encoding="utf-8",
        )
        _assert_frozen(before)
    except Exception as exc:
        failures.append(f"report {type(exc).__name__}: {exc}")

    if failures:
        print("acceptance: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print(f"implementation tag: {tag['implementation_tag']}")
    print("acceptance: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--live-accept", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.live_accept:
        return live_accept()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
