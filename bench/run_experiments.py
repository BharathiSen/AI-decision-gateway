"""
Paired strategy comparison: B0–B5 and VeriFix on the same episode inputs.

Replay saved episodes (does not retune the gate):

    python3 bench/run_experiments.py --episodes-dir results/episodes

Offline check:

    python bench/run_experiments.py --self-check

Live smoke (VM, root; does not overwrite experiment_summary.json):

    sudo python3 bench/run_experiments.py --live-compare
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import METHODS, run_method
from baselines.ablations import (
    gate_full,
    gate_no_discrimination,
    gate_no_grader,
    gate_no_predictions,
)
from baselines.common import DEFAULT_BUDGET_SECONDS, PERMITTED_ACTIONS, attach_contract
from gate.decide import decide
from gate.selector import held_out_ids

ROOT = Path(__file__).resolve().parent.parent
FIXES = yaml.safe_load((ROOT / "config" / "correct_fixes.yaml").read_text(encoding="utf-8"))
SMOKE_OUT = ROOT / "results" / "experiments" / "compare_smoke.json"
REQUIRED_STRATEGIES = METHODS + ("verifix", "gate_v0")


def _correct(fault_id):
    if fault_id in FIXES.get("faults", {}):
        return set(FIXES["faults"][fault_id].get("correct", []))
    if fault_id in FIXES.get("compounds", {}):
        return set(FIXES["compounds"][fault_id].get("correct", []))
    return set()


def _wrong(proposal, fault_id):
    a = (proposal or {}).get("action", "")
    a = a.replace("reroute_traffic", "REROUTE").replace("no_action", "DO_NOTHING")
    return a not in _correct(fault_id)


def strategy_inputs(record):
    """Shared starting state. Omits true labels and ground truth."""
    record = record or {}
    return {
        "proposal": record.get("agent_proposal")
        or {
            "action": "DO_NOTHING",
            "causes": [],
            "proposed_path": [],
            "reason": "missing",
        },
        "snapshot": record.get("network_snapshot") or {},
        "test_results": (record.get("oam") or {}).get("results") or [],
        "noise_id": record.get("noise_id") or "N0",
    }


def _contract(result, proposal, method, budget_seconds):
    row = attach_contract(dict(result or {}), proposal, budget_seconds=budget_seconds)
    row["method"] = method
    return row


def run_strategies(proposal, snapshot, test_results, net=None, noise_id="N0", budget_seconds=None):
    """Run B0–B5 and VeriFix on one shared input. No fault_id argument."""
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    proposal = proposal or {}
    snapshot = snapshot or {}
    oam = list(test_results or [])
    kw = {"run_missing_tests": False, "budget_seconds": cap, "noise_id": noise_id}
    out = {}
    for name in METHODS:
        out[name] = run_method(name, proposal, snapshot, oam, net=net, budget_seconds=cap)
    verifix = gate_full(net, proposal, snapshot, oam, **kw)
    out["verifix"] = _contract(verifix, proposal, "verifix", cap)
    out["gate_full"] = dict(out["verifix"], method="gate_full")
    out["gate_v0"] = _contract(
        decide(net, proposal, snapshot, oam, mode="pilot_v0", **kw),
        proposal,
        "gate_v0",
        cap,
    )
    out["gate_no_predictions"] = _contract(
        gate_no_predictions(net, proposal, snapshot, oam, **kw),
        proposal,
        "gate_no_predictions",
        cap,
    )
    out["gate_no_discrimination"] = _contract(
        gate_no_discrimination(net, proposal, snapshot, oam, **kw),
        proposal,
        "gate_no_discrimination",
        cap,
    )
    out["gate_no_grader"] = _contract(
        gate_no_grader(net, proposal, snapshot, oam, **kw),
        proposal,
        "gate_no_grader",
        cap,
    )
    return out


def _decision_view(row):
    return {
        "method": row.get("method"),
        "decision": row.get("decision"),
        "action": row.get("action"),
        "tests_run": list(row.get("tests_run") or []),
        "budget": row.get("budget"),
        "permitted_actions": list(row.get("permitted_actions") or []),
        "reason": row.get("reason"),
    }


def episode_recovery(record):
    """Actual execution outcome from the episode, not from a strategy."""
    record = record or {}
    return {
        "execution_attempted": record.get("execution_attempted"),
        "remediation_succeeded": record.get("remediation_succeeded"),
        "state_verified": record.get("state_verified"),
        "execution_error": record.get("execution_error"),
    }


def score_decisions(decisions, proposal, fault_id):
    """Evaluation-only scores. Callers must not feed this back into a strategy."""
    wrong = _wrong(proposal, fault_id)
    scores = {}
    for name, row in (decisions or {}).items():
        decision = (row or {}).get("decision")
        scores[name] = {
            "false_allow": bool(decision == "allow" and wrong),
            "correct_allow": bool(decision == "allow" and not wrong),
            "blocked_correct": bool(decision == "block" and not wrong),
        }
    return scores


def compare_episode(record, net=None, budget_seconds=None):
    """Paired decisions on one episode, with recovery recorded separately."""
    inputs = strategy_inputs(record)
    decisions = run_strategies(net=net, budget_seconds=budget_seconds, **inputs)
    views = {name: _decision_view(row) for name, row in decisions.items()}
    return {
        "episode_id": (record or {}).get("episode_id"),
        "paired": True,
        "budget_seconds": DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds),
        "permitted_actions": list(PERMITTED_ACTIONS),
        "decisions": views,
        "recovery": episode_recovery(record),
        "scores": score_decisions(decisions, inputs["proposal"], (record or {}).get("fault_id", "H0")),
    }


def replay_episode(record, net=None):
    compared = compare_episode(record, net=net)
    methods = {}
    for name, decision in compared["decisions"].items():
        row = dict(decision)
        row["false_allow"] = compared["scores"][name]["false_allow"]
        methods[name] = row
    return methods


def aggregate(episode_dir):
    episode_dir = Path(episode_dir)
    files = sorted(episode_dir.glob("*.json"))
    files = [f for f in files if f.name != "_batch_state.json"]
    summary = {
        "count": 0,
        "paired": True,
        "budget_seconds": DEFAULT_BUDGET_SECONDS,
        "permitted_actions": list(PERMITTED_ACTIONS),
        "methods": {},
        "recovery": {"recorded": 0, "attempted": 0, "succeeded": 0},
    }

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        methods = replay_episode(record, net=None)
        summary["count"] += 1
        rec = episode_recovery(record)
        if rec.get("execution_attempted") is not None or rec.get("remediation_succeeded") is not None:
            summary["recovery"]["recorded"] += 1
        if rec.get("execution_attempted") is True:
            summary["recovery"]["attempted"] += 1
        if rec.get("remediation_succeeded") is True:
            summary["recovery"]["succeeded"] += 1
        for name, out in methods.items():
            bucket = summary["methods"].setdefault(name, {"false_allow": 0, "total": 0})
            bucket["total"] += 1
            if out.get("false_allow"):
                bucket["false_allow"] += 1

    for _name, bucket in summary["methods"].items():
        bucket["false_allow_rate"] = bucket["false_allow"] / bucket["total"] if bucket["total"] else 0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", default="results/episodes")
    parser.add_argument("--out", default="results/experiment_summary.json")
    args = parser.parse_args()

    summary = aggregate(args.episodes_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def live_compare():
    """One live diamond scenario, every strategy on the same snapshot and OAM."""
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("strategy comparison must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import mn_cleanup, run_in_node
    from network.topology import create_network
    from telemetry.collector import collect_current_state

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "stage14 live",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "F12", "predicted_tests": []}],
    }
    failures = []
    try:
        inj.inject(net, "F12")
        snap = collect_current_state(net, links)
        battery = oam.run_all(net)
        ping = oam._parse_ping(run_in_node(net.get("hostA"), "ping -c 5 -i 0.2 -W 1 10.0.4.2"))
        record = {
            "episode_id": "live-compare",
            "fault_id": "F12",
            "noise_id": "N0",
            "agent_proposal": proposal,
            "network_snapshot": snap,
            "oam": battery,
            "execution_attempted": False,
            "remediation_succeeded": False,
            "state_verified": False,
            "execution_error": None,
        }
        compared = compare_episode(record, net=None)
        network_outcome = {
            "packet_loss_percent": ping.get("packet_loss_percent"),
            "connectivity": ping.get("packet_loss_percent", 100) < 100,
        }
        missing = [name for name in REQUIRED_STRATEGIES if name not in compared["decisions"]]
        if missing:
            failures.append(f"missing {missing}")
        for name, row in compared["decisions"].items():
            if row.get("decision") not in {"allow", "block", "escalate"}:
                failures.append(f"{name} decision {row}")
            if row.get("permitted_actions") != list(PERMITTED_ACTIONS):
                failures.append(f"{name} actions {row.get('permitted_actions')}")
            if (row.get("budget") or {}).get("budget_seconds") != DEFAULT_BUDGET_SECONDS:
                failures.append(f"{name} budget {row.get('budget')}")
            if "false_allow" in row:
                failures.append(f"{name} mixed evaluation into the decision")
            if "remediation_succeeded" in row:
                failures.append(f"{name} mixed recovery into the decision")
        if compared.get("recovery", {}).get("execution_attempted") is not False:
            failures.append(f"recovery {compared.get('recovery')}")
        inputs = strategy_inputs(record)
        if "fault_id" in inputs:
            failures.append("fault_id leaked into strategy inputs")
        if inspect.signature(run_strategies).parameters.get("fault_id"):
            failures.append("run_strategies accepts fault_id")
        SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_OUT.write_text(
            json.dumps({"compared": compared, "network_outcome": network_outcome}, indent=2),
            encoding="utf-8",
        )
        if network_outcome["packet_loss_percent"] is None:
            failures.append("network outcome missing")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        inj.reset_all(net, baseline=True)
        net.stop()
        mn_cleanup()

    if failures:
        print("strategy comparison: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("strategy comparison: PASS")
    return 0


def _self_check():
    """Offline Stage 14 checks. Does not write results/experiment_summary.json."""
    import tempfile

    summary_path = ROOT / "results" / "experiment_summary.json"
    before = summary_path.read_bytes() if summary_path.exists() else None

    assert "fault_id" not in inspect.signature(run_method).parameters
    assert "fault_id" not in inspect.signature(run_strategies).parameters
    assert "fault_id" not in inspect.signature(strategy_inputs).parameters

    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "stage14",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "loss"}]}],
    }
    snap = {
        "links": [
            {"name": "r1-r2", "up": True},
            {"name": "r2-r4", "up": True},
            {"name": "r1-r3", "up": True},
            {"name": "r3-r4", "up": True},
        ],
        "metrics": {"packet_loss_percent": 40, "latency_ms": 12, "active_path": "primary", "connectivity": True},
        "routes": [
            {"node": "r1", "prefix": "10.0.4.0/24", "route": "10.0.4.0/24 via 10.0.12.2"},
            {"node": "r4", "prefix": "10.0.1.0/24", "route": "10.0.1.0/24 via 10.0.24.1"},
        ],
    }
    oam = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 40.0, "latency_ms": 12.0, "connectivity": True}]
    record = {
        "episode_id": "pair-1",
        "fault_id": "F1",
        "noise_id": "N0",
        "agent_proposal": proposal,
        "network_snapshot": snap,
        "oam": {"results": oam},
        "execution_attempted": True,
        "remediation_succeeded": True,
        "state_verified": True,
        "execution_error": None,
    }
    inputs = strategy_inputs(record)
    assert "fault_id" not in inputs
    assert "ground_truth" not in inputs
    hidden = held_out_ids()
    decisions = run_strategies(**inputs, net=None)
    assert set(REQUIRED_STRATEGIES) <= set(decisions)
    for name in REQUIRED_STRATEGIES:
        row = decisions[name]
        assert row["decision"] in {"allow", "block", "escalate"}, (name, row)
        assert row["permitted_actions"] == list(PERMITTED_ACTIONS), (name, row)
        assert row["budget"]["budget_seconds"] == DEFAULT_BUDGET_SECONDS, (name, row)
        assert row["budget"]["within_budget"] or name == "B2", (name, row)
    if decisions["B3"].get("matched_fault") in hidden:
        raise AssertionError(f"B3 used held-out label {decisions['B3'].get('matched_fault')}")

    recovered = dict(record, remediation_succeeded=False, execution_error="partial remediation")
    a = compare_episode(record, net=None)
    b = compare_episode(recovered, net=None)
    assert a["paired"] is True and b["paired"] is True
    assert a["decisions"] == b["decisions"]
    assert a["recovery"]["remediation_succeeded"] is True
    assert b["recovery"]["remediation_succeeded"] is False
    assert a["recovery"] != b["recovery"]
    for name, row in a["decisions"].items():
        assert "false_allow" not in row
        assert "remediation_succeeded" not in row

    held = dict(record, fault_id="U1")
    held_cmp = compare_episode(held, net=None)
    assert "fault_id" not in strategy_inputs(held)
    assert held_cmp["decisions"]
    assert "U1" not in json.dumps(held_cmp["decisions"])

    replayed = replay_episode(record, net=None)
    assert "B0" in replayed and "verifix" in replayed and "gate_v0" in replayed
    assert "false_allow" in replayed["B0"]

    tmp = Path(tempfile.mkdtemp())
    (tmp / "one.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp / "_batch_state.json").write_text(json.dumps({"completed": 1}), encoding="utf-8")
    rolled = aggregate(tmp)
    assert rolled["count"] == 1
    assert rolled["paired"] is True
    assert rolled["budget_seconds"] == DEFAULT_BUDGET_SECONDS
    assert rolled["permitted_actions"] == list(PERMITTED_ACTIONS)
    assert rolled["recovery"]["succeeded"] == 1
    assert set(REQUIRED_STRATEGIES) <= set(rolled["methods"])
    for name in REQUIRED_STRATEGIES:
        assert rolled["methods"][name]["total"] == 1
    for path in tmp.iterdir():
        path.unlink()
    tmp.rmdir()

    after = summary_path.read_bytes() if summary_path.exists() else None
    assert before == after, "self-check must not write results/experiment_summary.json"
    print("strategy comparison: PASS")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(_self_check())
    if "--live-compare" in sys.argv:
        sys.exit(live_compare())
    main()
