"""
Regenerate experiment tables from saved episodes (replay mode).

    python3 bench/run_experiments.py --episodes-dir results/episodes
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.ablations import gate_full, gate_no_discrimination
from baselines.b5_adaptive import decide_b5
from gate.decide import decide

ROOT = Path(__file__).resolve().parent.parent
FIXES = yaml.safe_load((ROOT / "config" / "correct_fixes.yaml").read_text(encoding="utf-8"))


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


def replay_episode(record, net=None):
    proposal = record.get("agent_proposal") or {
        "action": "DO_NOTHING",
        "causes": [],
        "proposed_path": [],
        "reason": "missing",
    }
    oam = (record.get("oam") or {}).get("results") or []
    snap = record.get("network_snapshot") or {}
    fault_id = record.get("fault_id", "H0")

    methods = {}
    if net:
        methods["gate_full"] = gate_full(net, proposal, snap, oam, run_missing_tests=False)
        methods["gate_v0"] = decide(net, proposal, snap, oam, mode="pilot_v0", run_missing_tests=False)
        methods["B5"] = decide_b5(net, proposal, snap, oam, fault_id, run_tests=False)
    else:
        methods["replay_only"] = {"decision": "n/a", "fault_id": fault_id}

    wrong = _wrong(proposal, fault_id)
    for name, out in methods.items():
        if out.get("decision") == "allow" and wrong:
            out["false_allow"] = True
        else:
            out["false_allow"] = False
    return methods


def aggregate(episode_dir):
    episode_dir = Path(episode_dir)
    files = sorted(episode_dir.glob("*.json"))
    files = [f for f in files if f.name != "_batch_state.json"]
    summary = {"count": 0, "methods": {}}

    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        methods = replay_episode(record, net=None)
        summary["count"] += 1
        for m, out in methods.items():
            bucket = summary["methods"].setdefault(m, {"false_allow": 0, "total": 0})
            bucket["total"] += 1
            if out.get("false_allow"):
                bucket["false_allow"] += 1

    for m, b in summary["methods"].items():
        b["false_allow_rate"] = b["false_allow"] / b["total"] if b["total"] else 0
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


if __name__ == "__main__":
    main()
