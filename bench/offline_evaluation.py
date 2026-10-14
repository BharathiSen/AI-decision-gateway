"""
Offline pilot + episode generation (no Mininet).

Uses the observability matrix and the same stub agent / gate logic as
bench/run_pilot.py. Numbers are useful for CI and dev replay; paper runs
must use live Mininet (scripts/finish_evaluation.py --mode live).
"""

import csv
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from baselines.b5_adaptive import decide_b5
from bench import inject as inj
from bench.run_episode import EPISODE_DIR, save_episode
from gate.decide import decide

ROOT = Path(__file__).resolve().parent.parent
FIXES = yaml.safe_load((ROOT / "config" / "correct_fixes.yaml").read_text(encoding="utf-8"))
COSTS = yaml.safe_load((ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))["tests"]
MATRIX_PATH = ROOT / "results" / "observability_matrix.csv"
PILOT_OUT = ROOT / "results" / "pilot"


def _load_matrix():
    if not MATRIX_PATH.exists():
        return []
    rows = []
    with MATRIX_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _outcome(fault_id, test_id, matrix_rows):
    for row in matrix_rows:
        if row["fault_id"] == fault_id and row["test_id"] == test_id:
            return row["outcome"]
    if fault_id == "H0":
        if test_id == "T1_e2e_ping":
            return "loss=0.0,lat=10.0"
        if test_id == "T7_route_check":
            return "primary"
    if test_id == "T1_e2e_ping":
        return "loss=15.0,lat=12.0"
    if test_id == "T7_route_check":
        return "primary"
    return "ok"


def _oam_result_from_outcome(test_id, outcome):
    if test_id == "T1_e2e_ping":
        loss, lat = 15.0, 12.0
        if "loss=" in outcome:
            loss = float(outcome.split("loss=")[1].split(",")[0])
        if "lat=" in outcome:
            lat = float(outcome.split("lat=")[1])
        return {
            "test_id": "T1_e2e_ping",
            "packet_loss_percent": loss,
            "latency_ms": lat,
            "connectivity": loss < 100.0,
        }
    if test_id == "T7_route_check":
        return {"test_id": "T7_route_check", "active_path": outcome.strip()}
    return {"test_id": test_id}


def _build_oam_bundle(fault_id, matrix_rows):
    test_ids = ["T1_e2e_ping", "T7_route_check"]
    results = [
        _oam_result_from_outcome(tid, _outcome(fault_id, tid, matrix_rows)) for tid in test_ids
    ]
    now = datetime.now(timezone.utc).isoformat()
    return {"started_at": now, "completed_at": now, "results": results}


def _build_snapshot(oam_bundle):
    t1 = next((r for r in oam_bundle["results"] if r["test_id"] == "T1_e2e_ping"), {})
    loss = t1.get("packet_loss_percent", 0)
    return {
        "metrics": {
            "packet_loss_percent": loss,
            "latency_ms": t1.get("latency_ms"),
            "connectivity": t1.get("connectivity", True),
        },
        "links": [{"name": "r1-r2", "up": True, "latency_ms": 5}],
    }


def _correct_set(fault_id):
    if fault_id in FIXES.get("faults", {}):
        return set(FIXES["faults"][fault_id].get("correct", []))
    if fault_id in FIXES.get("compounds", {}):
        return set(FIXES["compounds"][fault_id].get("correct", []))
    return set()


def _stub_proposal(snapshot, fault_id):
    metrics = snapshot.get("metrics", {})
    loss = metrics.get("packet_loss_percent", 0)
    action = "REROUTE" if loss > 10 else "DO_NOTHING"
    if fault_id == "F12":
        action = "REROUTE"
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


def _is_wrong(proposal, fault_id):
    action = proposal.get("action", "").replace("reroute_traffic", "REROUTE").replace("no_action", "DO_NOTHING")
    return action not in _correct_set(fault_id)


def _cost_for_tests(test_ids):
    total = 0.0
    for tid in test_ids:
        for key, val in COSTS.items():
            if key.startswith(tid[:2]):
                total += val.get("time_seconds", 0)
                break
    return total


def _evaluate_go_nogo(stats):
    """Apply frozen criteria from results/pilot_criteria.md (approximate for offline)."""
    n = max(stats["episodes"], 1)
    non_h0 = max(stats["episodes"] - stats.get("h0_episodes", 0), 1)
    wrong_non_h0 = stats.get("wrong_proposals_non_h0", 0) / non_h0
    gate_unsafe = stats["gate_v0_unsafe_rate"]
    b5_unsafe = stats["b5_unsafe_rate"]
    diff = abs(gate_unsafe - b5_unsafe)
    continue_main = wrong_non_h0 >= 0.15 and (
        gate_unsafe < b5_unsafe or stats["gate_v0_cost"] < stats["b5_cost"]
    )
    pivot = diff < 0.02 and stats["episodes"] >= 400
    return {
        "wrong_rate_non_h0": wrong_non_h0,
        "unsafe_rate_delta_pp": diff * 100,
        "recommendation": (
            "continue_main_track"
            if continue_main
            else ("pivot_measurement" if pivot else "inconclusive_or_collect_more")
        ),
        "criteria_doc": "results/pilot_criteria.md",
    }


def run_offline_pilot(episodes=500, seed=1, save_episodes=True, episodes_dir=None):
    rng = random.Random(seed)
    matrix_rows = _load_matrix()
    faults = [f for f in inj._FAULTS]

    stats = {
        "episodes": episodes,
        "environment": "offline_simulation",
        "agent": "stub",
        "seed": seed,
        "model_errors": 0,
        "wrong_proposals": 0,
        "wrong_proposals_non_h0": 0,
        "h0_episodes": 0,
        "gate_v0_unsafe": 0,
        "b5_unsafe": 0,
        "gate_v0_escalate": 0,
        "b5_escalate": 0,
        "gate_v0_cost": 0.0,
        "b5_cost": 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Simulated from observability matrix + stub agent. "
            "Run scripts/finish_evaluation.py --mode live on a Mininet VM for paper numbers."
        ),
    }

    ep_dir = Path(episodes_dir) if episodes_dir else EPISODE_DIR
    if save_episodes:
        ep_dir.mkdir(parents=True, exist_ok=True)
        for old in ep_dir.glob("*.json"):
            if old.name != "_batch_state.json":
                old.unlink()

    for i in range(episodes):
        fault_id = "H0" if rng.random() < 0.3 else rng.choice([f for f in faults if f != "H0"])
        if fault_id == "H0":
            stats["h0_episodes"] += 1

        oam_bundle = _build_oam_bundle(fault_id, matrix_rows)
        snapshot = _build_snapshot(oam_bundle)
        proposal = _stub_proposal(snapshot, fault_id)

        if fault_id != "H0" and _is_wrong(proposal, fault_id):
            stats["wrong_proposals"] += 1
            stats["wrong_proposals_non_h0"] += 1

        oam_results = oam_bundle["results"]
        g = decide(None, proposal, snapshot, oam_results, mode="pilot_v0", run_missing_tests=False)
        b = decide_b5(None, proposal, snapshot, oam_results, fault_id, run_tests=False)

        if g["decision"] == "allow" and _is_wrong(proposal, fault_id):
            stats["gate_v0_unsafe"] += 1
        if b["decision"] == "allow" and _is_wrong(proposal, fault_id):
            stats["b5_unsafe"] += 1
        if g["decision"] == "escalate":
            stats["gate_v0_escalate"] += 1
        if b["decision"] == "escalate":
            stats["b5_escalate"] += 1

        stats["gate_v0_cost"] += _cost_for_tests(g["tests_run"])
        stats["b5_cost"] += _cost_for_tests(b["tests_run"])

        if save_episodes:
            record = {
                "episode_id": str(uuid.uuid4()),
                "started_at": stats["generated_at"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "fault_id": fault_id,
                "noise_id": "N0",
                "seed": rng.randint(0, 2**20),
                "correlation_ids": [],
                "network_snapshot": snapshot,
                "oam": oam_bundle,
                "agent_proposal": proposal,
                "gate_decision": g,
                "offline_simulation": True,
            }
            save_episode(record, directory=ep_dir)

    n = max(stats["episodes"], 1)
    stats["wrong_rate"] = stats["wrong_proposals"] / n
    stats["gate_v0_unsafe_rate"] = stats["gate_v0_unsafe"] / n
    stats["b5_unsafe_rate"] = stats["b5_unsafe"] / n
    stats["gate_v0_escalation_rate"] = stats["gate_v0_escalate"] / n
    stats["b5_escalation_rate"] = stats["b5_escalate"] / n
    stats["go_nogo"] = _evaluate_go_nogo(stats)

    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    (PILOT_OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
