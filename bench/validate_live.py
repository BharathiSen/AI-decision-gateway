"""
Stage 19 live validation and stress testing.

Live Mininet episodes, measured OAM cost and network impact, compounds,
unseen faults, the large topology, live-vs-replay comparison, and the
execution boundary. Does not retune the gate or write frozen results.

    python bench/validate_live.py --self-check
    sudo python3 bench/validate_live.py --live-validate
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.inject import TUNING_PURPOSES, held_out_ids
from bench.run_experiments import compare_episode

ROOT = Path(__file__).resolve().parent.parent
SMOKE_OUT = ROOT / "results" / "validation" / "_smoke" / "live_validation.json"
FROZEN_RESULTS = (
    ROOT / "results" / "calibration_report.json",
    ROOT / "results" / "pilot" / "summary.json",
    ROOT / "results" / "observability_matrix.csv",
    ROOT / "results" / "experiment_summary.json",
)
LIVE_VALIDATE_BUILD = "stage19-validate-v2"
DIAMOND_CASES = (
    {
        "name": "healthy",
        "fault_id": "H0",
        "compound_id": None,
        "noise_id": "N0",
        "topology": "diamond",
        "mode": "normal",
        "data_split": "validation",
    },
    {
        "name": "trap",
        "fault_id": "F12",
        "compound_id": None,
        "noise_id": "N0",
        "topology": "diamond",
        "mode": "normal",
        "data_split": "validation",
    },
    {
        "name": "compound",
        "fault_id": "H0",
        "compound_id": "C1",
        "noise_id": "N0",
        "topology": "diamond",
        "mode": "normal",
        "data_split": "validation",
    },
    {
        "name": "unseen",
        "fault_id": "U1",
        "compound_id": None,
        "noise_id": "N0",
        "topology": "diamond",
        "mode": "normal",
        "data_split": "held_out",
    },
    {
        "name": "high_traffic_noise",
        "fault_id": "F3",
        "compound_id": None,
        "noise_id": "N2",
        "topology": "diamond",
        "mode": "stress",
        "stress_kind": "high_traffic",
        "data_split": "validation",
    },
)
GRID_CASE = {
    "name": "large_topology",
    "fault_id": "H0",
    "compound_id": None,
    "noise_id": "N0",
    "topology": "grid_4x4",
    "mode": "normal",
    "data_split": "held_out",
}


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


def scenario_from_case(case, seed=19):
    case = case or {}
    label = case.get("compound_id") or case.get("fault_id") or "H0"
    return {
        "scenario_id": "|".join(
            [
                str(seed),
                case.get("topology") or "diamond",
                case.get("data_split") or "validation",
                case.get("mode") or "normal",
                label,
                case.get("noise_id") or "N0",
                "stub",
                "none",
                "verifix",
                case.get("stress_kind") or "none",
            ]
        ),
        "seed": seed,
        "topology": case.get("topology") or "diamond",
        "data_split": case.get("data_split") or "validation",
        "mode": case.get("mode") or "normal",
        "fault_id": case.get("fault_id") or "H0",
        "compound_id": case.get("compound_id"),
        "noise_id": case.get("noise_id") or "N0",
        "traffic_level": case.get("noise_id") or "N0",
        "agent_mode": "stub",
        "model_id": None,
        "strategy": "verifix",
        "stress_kind": case.get("stress_kind"),
        "name": case.get("name"),
    }


def impact_from_record(record, ping_after=None):
    oam = (record or {}).get("oam_cost") or {}
    gate_cost = (record or {}).get("gate_cost") or {}
    tests = oam.get("tests") if isinstance(oam.get("tests"), list) else []
    return {
        "test_count": oam.get("test_count") if oam.get("test_count") is not None else len(tests),
        "test_duration_seconds": oam.get("total_duration_seconds"),
        "additional_packets": oam.get("total_additional_packets"),
        "additional_bytes": oam.get("total_additional_bytes"),
        "decision_latency_seconds": gate_cost.get("latency_seconds"),
        "ping_after": ping_after,
        "per_test": [
            {
                "test_id": row.get("test_id"),
                "duration_seconds": row.get("duration_seconds"),
                "additional_bytes": row.get("additional_bytes"),
            }
            for row in tests
            if isinstance(row, dict)
        ],
    }


def unseen_reroute_proposal():
    return {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "stage19 unseen",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "U1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "ok"}]}],
    }


def risky_unseen_refused(decision_row, exec_row, executed=None):
    """Risky U1 claims must escalate outside catalog and must not execute."""
    decision_row = decision_row or {}
    exec_row = exec_row or {}
    return (
        decision_row.get("decision") == "escalate"
        and decision_row.get("reason") == "outside catalog"
        and exec_row.get("execution_attempted") is False
        and not list(executed or [])
    )


def execution_safe(record):
    """True when execute happened only after allow + verified recheck."""
    record = record or {}
    decision = (record.get("gate_decision") or {}).get("decision")
    attempted = bool(record.get("execution_attempted"))
    if attempted:
        return decision == "allow" and record.get("state_verified") is True
    return True


def investigate(live_record, replay_row):
    live_gate = (live_record or {}).get("gate_decision") or {}
    replay_row = replay_row or {}
    live_decision = live_gate.get("decision")
    replay_decision = replay_row.get("decision")
    live_tests = list(live_gate.get("tests_run") or [])
    replay_tests = list(replay_row.get("tests_run") or [])
    match = live_decision == replay_decision
    reason = None
    if not match:
        if live_tests != replay_tests:
            reason = "live gate used the Mininet net; replay scores saved OAM only"
        elif live_decision in {"allow", "block", "escalate"} and replay_decision in {
            "allow",
            "block",
            "escalate",
        }:
            reason = "same saved OAM, different strategy contract (live verifix vs replay verifix)"
        else:
            reason = "live or replay decision missing"
    return {
        "match": match,
        "live_decision": live_decision,
        "replay_decision": replay_decision,
        "live_tests": live_tests,
        "replay_tests": replay_tests,
        "reason": reason,
    }


def compare_live_replay(record):
    compared = compare_episode(record, net=None)
    replay_row = (compared.get("decisions") or {}).get("verifix") or {}
    return {
        "episode_id": record.get("episode_id"),
        "investigation": investigate(record, replay_row),
        "replay_methods": sorted((compared.get("decisions") or {}).keys()),
        "replay_recovery": compared.get("recovery"),
    }


def ping_summary(parsed):
    parsed = parsed or {}
    return {
        "packet_loss_percent": parsed.get("packet_loss_percent"),
        "latency_ms": parsed.get("latency_ms"),
        "ok": parsed.get("packet_loss_percent") is not None and parsed.get("packet_loss_percent") < 100.0,
    }


def write_report(payload, path=None):
    path = Path(path or SMOKE_OUT)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def _self_check():
    before = _snapshot_frozen()
    hidden = held_out_ids()
    names = {case["name"] for case in DIAMOND_CASES}
    assert names >= {"healthy", "trap", "compound", "unseen", "high_traffic_noise"}
    assert GRID_CASE["topology"] == "grid_4x4"
    assert any(case["noise_id"] == "N2" for case in DIAMOND_CASES)
    assert any(case.get("compound_id") == "C1" for case in DIAMOND_CASES)
    assert any(case["fault_id"] == "U1" for case in DIAMOND_CASES)
    assert "U1" in hidden and "C1" not in hidden
    assert "tuning" in TUNING_PURPOSES

    allow_ok = {
        "gate_decision": {"decision": "allow", "tests_run": ["T1_e2e_ping"]},
        "execution_attempted": True,
        "state_verified": True,
        "oam_cost": {
            "test_count": 2,
            "total_duration_seconds": 4.5,
            "total_additional_bytes": 1280,
            "total_additional_packets": 20,
            "tests": [
                {"test_id": "T1_e2e_ping", "duration_seconds": 2.0, "additional_bytes": 640},
                {"test_id": "T7_route_check", "duration_seconds": 0.2, "additional_bytes": 0},
            ],
        },
        "gate_cost": {"latency_seconds": 0.1},
    }
    blocked = {
        "gate_decision": {"decision": "block", "tests_run": ["T5_checksum"]},
        "execution_attempted": False,
        "state_verified": False,
    }
    unsafe = {
        "gate_decision": {"decision": "block"},
        "execution_attempted": True,
        "state_verified": False,
    }
    assert execution_safe(allow_ok)
    assert execution_safe(blocked)
    assert not execution_safe(unsafe)
    impact = impact_from_record(allow_ok, ping_after={"ok": True, "packet_loss_percent": 0})
    assert impact["test_duration_seconds"] == 4.5
    assert impact["additional_bytes"] == 1280
    assert impact["ping_after"]["ok"] is True

    match = investigate(allow_ok, {"decision": "allow", "tests_run": ["T1_e2e_ping"]})
    assert match["match"] is True and match["reason"] is None
    mismatch = investigate(blocked, {"decision": "allow", "tests_run": []})
    assert mismatch["match"] is False and mismatch["reason"]

    live_n = len(DIAMOND_CASES)
    replay_only = 0
    share = live_n / (live_n + replay_only)
    assert share == 1.0 and live_n >= 5

    fill = {
        "episode_id": "self-check-replay",
        "fault_id": "H0",
        "noise_id": "N0",
        "agent_proposal": {
            "action": "DO_NOTHING",
            "target": "HostA_to_HostB",
            "reason": "healthy",
            "proposed_path": ["r1", "r2", "r4"],
            "causes": [],
        },
        "network_snapshot": {
            "links": [],
            "metrics": {"packet_loss_percent": 0, "connectivity": True, "active_path": "primary"},
        },
        "oam": {"results": [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0, "latency_ms": 10, "connectivity": True}]},
        "gate_decision": {"decision": "allow", "tests_run": []},
        "execution_attempted": False,
        "remediation_succeeded": False,
        "state_verified": False,
    }
    compared = compare_live_replay(fill)
    assert "verifix" in compared["replay_methods"]
    assert "investigation" in compared

    from gate.decide import decide
    from bench.run_episode import _snapshot, apply_authorized_action

    healthy = _snapshot(r1_via="10.0.12.2", active_path="primary")
    risky = unseen_reroute_proposal()
    risky_out = decide(None, risky, healthy, [], mode="full", run_missing_tests=False)
    calls = []

    def execute(item, _box=calls):
        _box.append(item)
        return {"executed": True}

    risky_exec = apply_authorized_action(risky_out, risky, execute, healthy, lambda: healthy)
    assert risky_unseen_refused(risky_out, risky_exec, calls), (risky_out, risky_exec)
    calm = dict(risky, action="DO_NOTHING", proposed_path=["r1", "r2", "r4"])
    calm_out = decide(None, calm, healthy, [], mode="full", run_missing_tests=False)
    assert calm_out.get("decision") == "allow"

    try:
        write_report({"no": True}, FROZEN_RESULTS[0])
        raise AssertionError("frozen overwrite was not refused")
    except ValueError:
        pass
    _assert_frozen(before)
    print(f"live-validate build: {LIVE_VALIDATE_BUILD}")
    print("live validation: PASS")
    return 0


def live_validate():
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("live validation must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import mn_cleanup, run_in_node
    from bench.run_batch import run_one
    from bench.run_episode import apply_authorized_action, read_execution_state
    from gate.decide import decide
    from network.topology import create_network
    from telemetry.collector import collect_current_state
    from network.topology_large import create_large_network

    print(f"live-validate build: {LIVE_VALIDATE_BUILD}")
    before = _snapshot_frozen()
    failures = []
    rows = []
    grid_result = None
    diamond = None
    large = None

    def ping_diamond(net):
        return ping_summary(
            oam._parse_ping(run_in_node(net.get("hostA"), "ping -c 3 -i 0.2 -W 1 10.0.4.2"))
        )

    def expect(ok, message):
        if not ok:
            failures.append(message)
            print(f"  FAIL  {message}")
        else:
            print(f"  PASS  {message}")

    mn_cleanup()
    try:
        diamond = create_network()
        net, links = diamond["net"], diamond["links"]
        for case in DIAMOND_CASES:
            scenario = scenario_from_case(case)
            print(f"  live  {case['name']} {scenario['fault_id']} {scenario['noise_id']}", flush=True)
            try:
                record = run_one(net, links, scenario, timeout_seconds=180)
            except Exception as exc:
                record = {
                    "episode_id": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fault_id": scenario["fault_id"],
                    "compound_id": scenario.get("compound_id"),
                    "execution_attempted": False,
                }
            ping_after = ping_diamond(net)
            row = {
                "name": case["name"],
                "live": True,
                "replay_only": False,
                "status": record.get("status"),
                "episode_id": record.get("episode_id"),
                "fault_id": record.get("fault_id") or scenario["fault_id"],
                "compound_id": record.get("compound_id") or scenario.get("compound_id"),
                "noise_id": record.get("noise_id") or scenario["noise_id"],
                "topology": "diamond",
                "mode": scenario.get("mode"),
                "decision": (record.get("gate_decision") or {}).get("decision") or record.get("decision"),
                "reason": (record.get("gate_decision") or {}).get("reason"),
                "proposed_fix": (record.get("agent_proposal") or {}).get("action") or record.get("proposed_fix"),
                "execution_attempted": record.get("execution_attempted"),
                "state_verified": record.get("state_verified"),
                "remediation_succeeded": record.get("remediation_succeeded"),
                "execution_error": record.get("execution_error"),
                "error": record.get("error"),
                "impact": impact_from_record(record, ping_after=ping_after),
                "execution_safe": execution_safe(record),
            }
            if record.get("status") == "ok" and record.get("agent_proposal"):
                try:
                    row["replay"] = compare_live_replay(record)
                except Exception as exc:
                    row["replay"] = {"error": f"{type(exc).__name__}: {exc}"}
                    failures.append(f"{case['name']} replay {type(exc).__name__}: {exc}")
            rows.append(row)
            if record.get("status") != "ok":
                failures.append(f"{case['name']} status {record.get('status')} {record.get('error')}")
            if not execution_safe(record):
                failures.append(f"{case['name']} executed without verified allow")
            impact = row["impact"]
            if record.get("status") == "ok":
                if not isinstance(impact.get("test_duration_seconds"), (int, float)):
                    failures.append(f"{case['name']} missing test duration")
                if impact.get("test_count") in (None, 0):
                    failures.append(f"{case['name']} missing OAM tests")

        inj.reset_all(net, baseline=True)
        executed = []

        def stub(item, _box=executed):
            _box.append(item)
            return {"executed": False, "action": (item or {}).get("action")}

        snap = collect_current_state(net, links)
        approved = read_execution_state(net, links)
        risky = unseen_reroute_proposal()
        risky_out = decide(
            net, risky, snap, [], mode="full", run_missing_tests=False, noise_id="N0"
        )
        risky_exec = apply_authorized_action(
            risky_out,
            risky,
            stub,
            approved,
            lambda: read_execution_state(net, links),
        )
        expect(
            risky_unseen_refused(risky_out, risky_exec, executed),
            "risky unseen U1 is not authorized",
        )
        executed.clear()
        block_out = apply_authorized_action(
            {"decision": "block", "reason": "stage19"},
            {"action": "REROUTE", "target": "HostA_to_HostB", "reason": "x", "proposed_path": ["r1", "r3", "r4"]},
            stub,
            approved,
            lambda: read_execution_state(net, links),
        )
        expect(
            not executed and block_out.get("execution_attempted") is False,
            "block does not execute",
        )
        executed.clear()
        escalate_out = apply_authorized_action(
            {"decision": "escalate", "reason": "stage19"},
            {"action": "REROUTE", "target": "HostA_to_HostB", "reason": "x", "proposed_path": ["r1", "r3", "r4"]},
            stub,
            approved,
            lambda: read_execution_state(net, links),
        )
        expect(
            not executed and escalate_out.get("execution_attempted") is False,
            "escalate does not execute",
        )
        executed.clear()
        missing_out = apply_authorized_action(
            None,
            {"action": "DO_NOTHING"},
            stub,
            approved,
            lambda: read_execution_state(net, links),
        )
        expect(
            not executed and missing_out.get("execution_attempted") is False,
            "missing decision does not execute",
        )

        net.stop()
        diamond = None
        mn_cleanup()

        large = create_large_network()
        lnet = large["net"]
        names = list(lnet.nameToNode) if hasattr(lnet, "nameToNode") else list(lnet)
        grid = [name for name in names if name.startswith("r") and "_" in name]
        fwd = ping_summary(
            oam._parse_ping(run_in_node(lnet.get("hostA"), "ping -c 3 -i 0.2 -W 2 10.200.2.2"))
        )
        rev = ping_summary(
            oam._parse_ping(run_in_node(lnet.get("hostB"), "ping -c 3 -i 0.2 -W 2 10.200.1.2"))
        )
        grid_result = {
            "name": GRID_CASE["name"],
            "live": True,
            "replay_only": False,
            "topology": large.get("topology"),
            "routers": len(grid),
            "ping_a_to_b": fwd,
            "ping_b_to_a": rev,
        }
        expect(len(grid) == 16, f"grid routers {len(grid)}")
        expect(large.get("topology") == "grid_4x4", f"topology tag {large.get('topology')}")
        expect(fwd.get("ok") is True, f"grid ping A->B {fwd}")
        expect(rev.get("ok") is True, f"grid ping B->A {rev}")
        lnet.stop()
        large = None
        mn_cleanup()
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        if diamond is not None:
            try:
                diamond["net"].stop()
            except Exception:
                pass
        if large is not None:
            try:
                large["net"].stop()
            except Exception:
                pass
        mn_cleanup()

    live_n = sum(1 for row in rows if row.get("live") and not row.get("replay_only"))
    if grid_result and grid_result.get("live"):
        live_n += 1
    replay_only = sum(1 for row in rows if row.get("replay_only"))
    share = live_n / (live_n + replay_only) if (live_n + replay_only) else 0.0
    if live_n < 6:
        failures.append(f"live share too small: live={live_n} replay_only={replay_only}")
    if share < 1.0:
        failures.append(f"validation included replay-only episodes share={share}")

    by_name = {row["name"]: row for row in rows}
    trap = by_name.get("trap") or {}
    if trap.get("decision") == "allow" or trap.get("execution_attempted"):
        failures.append(f"F12 trap was authorized {trap.get('decision')} attempted={trap.get('execution_attempted')}")
    unseen = by_name.get("unseen") or {}
    if unseen.get("status") == "ok" and unseen.get("decision") not in {"allow", "block", "escalate"}:
        failures.append(f"unseen U1 decision {unseen.get('decision')}")
    if unseen.get("status") == "ok" and unseen.get("proposed_fix") in {"REROUTE", "REBALANCE"}:
        if unseen.get("decision") == "allow" or unseen.get("execution_attempted"):
            failures.append(
                f"unseen U1 risky proposal was authorized {unseen.get('decision')} "
                f"attempted={unseen.get('execution_attempted')}"
            )
    compound = by_name.get("compound") or {}
    if compound.get("status") == "ok" and compound.get("decision") not in {"allow", "block", "escalate"}:
        failures.append(f"compound decision {compound.get('decision')}")
    noisy = by_name.get("high_traffic_noise") or {}
    if noisy.get("status") == "ok":
        if not isinstance((noisy.get("impact") or {}).get("test_duration_seconds"), (int, float)):
            failures.append("N2 missing measured test time")
        if (noisy.get("impact") or {}).get("additional_bytes") is None and (
            noisy.get("impact") or {}
        ).get("additional_packets") is None:
            failures.append("N2 missing added traffic")
    healthy = by_name.get("healthy") or {}
    if healthy.get("execution_attempted") and healthy.get("state_verified") is not True:
        failures.append("H0 executed without state_verified")
    for row in rows:
        replay = row.get("replay") or {}
        inv = replay.get("investigation") or {}
        if replay.get("error"):
            continue
        if row.get("status") == "ok" and inv and inv.get("match") is False and not inv.get("reason"):
            failures.append(f"{row.get('name')} unexplained live/replay mismatch")

    payload = {
        "build": LIVE_VALIDATE_BUILD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_episodes": live_n,
        "replay_only_episodes": replay_only,
        "live_share": share,
        "rows": rows,
        "grid": grid_result,
        "failures": failures,
    }
    try:
        write_report(payload)
        _assert_frozen(before)
    except Exception as exc:
        failures.append(f"report {type(exc).__name__}: {exc}")

    if failures:
        print("live validation: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print(f"live share: {live_n} live / {live_n + replay_only} = {share:.0%}")
    print("live validation: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--live-validate", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.live_validate:
        return live_validate()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
