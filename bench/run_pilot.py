"""
Phase 9 pilot runner.

Uses the OpenRouter agent when OPENROUTER_API_KEY is set. Otherwise it
uses a local stub and says so at startup.

    sudo --preserve-env=OPENROUTER_API_KEY python3 bench/run_pilot.py --episodes 5
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench.run_episode import run_episode
from bench import inject as inj
from bench import tests as oam
from gate.decide import decide
from baselines.b5_adaptive import decide_b5

ROOT = Path(__file__).resolve().parent.parent
FIXES = yaml.safe_load((ROOT / "config" / "correct_fixes.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "pilot"


def _correct_set(fault_id):
    if fault_id in FIXES.get("faults", {}):
        return set(FIXES["faults"][fault_id].get("correct", []))
    if fault_id in FIXES.get("compounds", {}):
        return set(FIXES["compounds"][fault_id].get("correct", []))
    return set()


def _stub_proposal(snapshot, fault_id):
    """Deterministic stub when no API key."""
    metrics = snapshot.get("metrics", {})
    loss = metrics.get("packet_loss_percent", 0)
    action = "REROUTE" if loss > 10 else "DO_NOTHING"
    if fault_id == "F12":
        action = "REROUTE"  # intentional unsafe stub for pilot stats
    path = ["r1", "r3", "r4"] if action == "REROUTE" else ["r1", "r2", "r4"]
    return {
        "action": action,
        "target": "HostA_to_HostB",
        "proposed_path": path,
        "reason": f"stub agent loss={loss}",
        "causes": [
            {
                "cause_id": fault_id,
                "predicted_tests": [
                    {"test_id": "T1_e2e_ping", "expected": f"loss={loss}"},
                    {"test_id": "T7_route_check", "expected": "primary"},
                ],
            }
        ],
    }


def _propose(snapshot, fault_id, use_model):
    if not use_model:
        return _stub_proposal(snapshot, fault_id), "stub"
    from agent.decision_agent import propose_with_predictions

    return propose_with_predictions(snapshot), "model"


def _is_wrong(proposal, fault_id):
    action = proposal.get("action", "").replace("reroute_traffic", "REROUTE").replace("no_action", "DO_NOTHING")
    return action not in _correct_set(fault_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("run_pilot.py requires root on Mininet VM.", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    faults = [f for f in inj._FAULTS]
    use_model = bool(os.environ.get("OPENROUTER_API_KEY"))
    print(
        "agent: OpenRouter model" if use_model else "agent: local stub (OPENROUTER_API_KEY is not set)",
        flush=True,
    )
    stats = {
        "episodes": args.episodes,
        "agent": "model" if use_model else "stub",
        "model_errors": 0,
        "wrong_proposals": 0,
        "gate_v0_unsafe": 0,
        "b5_unsafe": 0,
        "gate_v0_cost": 0.0,
        "b5_cost": 0.0,
    }

    topo = create_network()
    net, links = topo["net"], topo["links"]
    costs = yaml.safe_load((ROOT / "config" / "costs.yaml").read_text())["tests"]

    try:
        for i in range(args.episodes):
            fault_id = "H0" if rng.random() < 0.3 else rng.choice([f for f in faults if f != "H0"])
            print(f"episode {i + 1}/{args.episodes} fault {fault_id} ...", flush=True)
            record = run_episode(net, links, fault_id=fault_id, seed=rng.randint(0, 2**20), run_oam=True)
            try:
                proposal, source = _propose(record["network_snapshot"], fault_id, use_model)
            except Exception as exc:
                stats["model_errors"] += 1
                print(f"  model error: {exc}", flush=True)
                continue
            print(f"  {source} proposed {proposal.get('action')}", flush=True)
            if fault_id != "H0" and _is_wrong(proposal, fault_id):
                stats["wrong_proposals"] += 1

            oam_results = record["oam"]["results"]
            g = decide(net, proposal, record["network_snapshot"], oam_results, mode="pilot_v0", run_missing_tests=False)
            b = decide_b5(net, proposal, record["network_snapshot"], oam_results, fault_id)

            if g["decision"] == "allow" and _is_wrong(proposal, fault_id):
                stats["gate_v0_unsafe"] += 1
            if b["decision"] == "allow" and _is_wrong(proposal, fault_id):
                stats["b5_unsafe"] += 1

            for tid in g["tests_run"]:
                for k, v in costs.items():
                    if k.startswith(tid[:2]):
                        stats["gate_v0_cost"] += v.get("time_seconds", 0)
                        break
            for tid in b["tests_run"]:
                for k, v in costs.items():
                    if k.startswith(tid[:2]):
                        stats["b5_cost"] += v.get("time_seconds", 0)
                        break
    finally:
        inj.reset_all(net)
        net.stop()

    n = max(stats["episodes"], 1)
    stats["wrong_rate"] = stats["wrong_proposals"] / n
    stats["gate_v0_unsafe_rate"] = stats["gate_v0_unsafe"] / n
    stats["b5_unsafe_rate"] = stats["b5_unsafe"] / n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
