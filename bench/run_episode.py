"""
Run one benchmark episode: fault → telemetry → OAM → optional agent/gate.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench import inject as inj
from bench import tests as oam
from bench.traffic import apply_noise, clear_noise
from telemetry.collector import collect_current_state

EPISODE_DIR = Path(__file__).resolve().parent.parent / "results" / "episodes"


def run_episode(
    net,
    links,
    fault_id="H0",
    noise_id="N0",
    seed=None,
    compound_id=None,
    run_oam=True,
    agent_fn=None,
    gate_fn=None,
):
    """
    Execute one episode and return a serializable record.
    agent_fn(state) -> proposal dict or None
    gate_fn(episode_so_far) -> gate decision or None
    """
    episode_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    noise_applied = apply_noise(net, noise_id, links)
    correlation_ids = []

    try:
        inj.reset_all(net)
        if compound_id:
            comp = inj.inject_compound(net, compound_id, seed=seed)
            correlation_ids = comp["correlation_ids"]
            fault_label = compound_id
        elif fault_id != "H0":
            correlation_ids = [inj.inject(net, fault_id, seed=seed)]
            fault_label = fault_id
        else:
            fault_label = "H0"

        snapshot = collect_current_state(net, links)
        oam_bundle = oam.run_all(net) if run_oam else None

        record = {
            "episode_id": episode_id,
            "started_at": started,
            "fault_id": fault_label,
            "noise_id": noise_id,
            "seed": seed,
            "correlation_ids": correlation_ids,
            "network_snapshot": snapshot,
            "oam": oam_bundle,
            "agent_proposal": None,
            "gate_decision": None,
        }

        if agent_fn:
            record["agent_proposal"] = agent_fn(snapshot)

        if gate_fn:
            record["gate_decision"] = gate_fn(record)

        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        return record
    finally:
        clear_noise(net, noise_applied, links)
        inj.reset_all(net)


def save_episode(record, directory=None):
    directory = Path(directory or EPISODE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['episode_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
