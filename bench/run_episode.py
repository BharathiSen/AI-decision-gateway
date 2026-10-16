"""
Run one benchmark episode: fault → telemetry → OAM → optional agent/gate.

    python bench/run_episode.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import uuid
from datetime import datetime, timezone

from bench import inject as inj
from bench import tests as oam
from bench.traffic import apply_noise, clear_noise
from telemetry.collector import collect_current_state

EPISODE_DIR = Path(__file__).resolve().parent.parent / "results" / "episodes"

GATE_DECISIONS = {"allow", "block", "escalate"}
CONTROL_ROUTES = (("r1", "10.0.4.0/24"), ("r4", "10.0.1.0/24"))
PATH_LINKS = ("r1-r2", "r2-r4", "r1-r3", "r3-r4")
BACKUP_LINKS = ("r1-r3", "r3-r4")


def normalize_gate_decision(gate_decision):
    """Return allow, block, or escalate. Anything else is not actionable."""
    if not isinstance(gate_decision, dict):
        return None
    raw = gate_decision.get("decision")
    if not isinstance(raw, str):
        return None
    decision = raw.strip().lower()
    if decision not in GATE_DECISIONS:
        return None
    return decision


def _route_text(route):
    return " ".join((route or "").split())


def execution_fingerprint(snapshot):
    """Routes and path-link state that an allow decision is bound to."""
    wanted = set(CONTROL_ROUTES)
    routes = {}
    for item in (snapshot or {}).get("routes") or []:
        key = (item.get("node"), item.get("prefix"))
        if key in wanted:
            routes[f"{key[0]} {key[1]}"] = _route_text(item.get("route"))
    links = {}
    for link in (snapshot or {}).get("links") or []:
        name = link.get("name")
        if name in PATH_LINKS:
            links[name] = bool(link.get("up"))
    return {"routes": routes, "links": links}


def proposal_uses_backup(proposal):
    proposal = proposal or {}
    label = proposal.get("path") or proposal.get("proposed_path_label")
    if label == "backup":
        return True
    if label == "primary":
        return False
    return "r3" in (proposal.get("proposed_path") or [])


def assess_recheck(approved_snapshot, current_snapshot, proposal):
    """Compare the gate snapshot with a fresh read. Does not trust a caller flag."""
    approved = execution_fingerprint(approved_snapshot)
    current = execution_fingerprint(current_snapshot)
    expected = {f"{node} {prefix}" for node, prefix in CONTROL_ROUTES}
    if set(approved["routes"]) != expected or any(not approved["routes"][key] for key in expected):
        return {"state_verified": False, "reason": "gate snapshot has no route fingerprint to recheck"}
    if set(current["routes"]) != expected or any(not current["routes"][key] for key in expected):
        return {"state_verified": False, "reason": "execution recheck could not read routes"}
    if approved["routes"] != current["routes"]:
        return {"state_verified": False, "reason": "route changed since the gate decision"}
    if approved["links"] != current["links"]:
        return {"state_verified": False, "reason": "path state changed since the gate decision"}
    if proposal_uses_backup(proposal):
        backup_up = all(current["links"].get(name) is True for name in BACKUP_LINKS)
        if not backup_up:
            return {"state_verified": False, "reason": "backup path is down"}
    return {"state_verified": True, "reason": None}


def read_execution_state(net, links=None):
    """Re-read the routes and path links a remediation is allowed to depend on."""
    from bench.mininet_util import run_in_node
    from network import controller as ctl

    routes = []
    for node_name, prefix in CONTROL_ROUTES:
        route = run_in_node(net.get(node_name), f"ip route show {prefix}").strip()
        routes.append({"node": node_name, "prefix": prefix, "route": route})
    link_states = [ctl.link_status(net, name, links) for name in PATH_LINKS]
    return {
        "routes": routes,
        "links": link_states,
        "metrics": {"active_path": ctl.get_active_path(net)},
    }


def apply_authorized_action(gate_decision, proposal, execute_fn, approved_snapshot, read_state):
    """Single boundary between a gate decision and remediation.

    On allow, read_state() is called and compared with approved_snapshot.
    execute_fn runs only when that comparison succeeds. state_verified is
    recorded from the comparison. Callers cannot pass it in.
    """
    outcome = {
        "execution_attempted": False,
        "execution_result": None,
        "execution_error": None,
        "remediation_succeeded": False,
        "safeguard_missing": None,
        "state_verified": False,
    }
    decision = normalize_gate_decision(gate_decision)
    if decision != "allow":
        if decision is None:
            outcome["execution_error"] = "missing or invalid gate decision"
        return outcome
    try:
        current_snapshot = read_state()
    except Exception as exc:
        outcome["execution_error"] = f"state recheck failed: {type(exc).__name__}: {exc}"
        return outcome
    assessment = assess_recheck(approved_snapshot, current_snapshot, proposal)
    outcome["state_verified"] = assessment["state_verified"]
    if not assessment["state_verified"]:
        outcome["execution_error"] = assessment["reason"]
        return outcome
    outcome["execution_attempted"] = True
    try:
        result = execute_fn(proposal)
    except Exception as exc:
        outcome["execution_error"] = f"{type(exc).__name__}: {exc}"
        outcome["remediation_succeeded"] = False
        return outcome
    outcome["execution_result"] = result
    outcome["remediation_succeeded"] = bool(
        isinstance(result, dict) and result.get("executed") is True
    )
    return outcome


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
    execute_fn=None,
):
    """
    Execute one episode and return a serializable record.
    agent_fn(state) -> proposal dict or None
    gate_fn(episode_so_far) -> gate decision or None
    execute_fn(proposal) -> execution result. Invoked only after allow
        and a fresh route/path recheck against network_snapshot.
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
            "agent_error": None,
            "gate_decision": None,
            "gate_error": None,
            "execution_attempted": False,
            "execution_result": None,
            "execution_error": None,
            "remediation_succeeded": False,
            "safeguard_missing": None,
            "state_verified": False,
        }

        if agent_fn:
            try:
                record["agent_proposal"] = agent_fn(snapshot)
            except Exception as exc:
                record["agent_error"] = f"{type(exc).__name__}: {exc}"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record

        if gate_fn:
            try:
                record["gate_decision"] = gate_fn(record)
            except Exception as exc:
                record["gate_error"] = f"{type(exc).__name__}: {exc}"
                record["execution_error"] = f"gate error: {record['gate_error']}"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record

        if execute_fn is None:
            def execute_fn(proposal, _net=net):
                from control.actions import execute, proposal_to_action

                action, payload = proposal_to_action(proposal or {})
                return execute(_net, action, payload)

        record.update(apply_authorized_action(
            record.get("gate_decision"),
            record.get("agent_proposal") or {},
            execute_fn,
            record.get("network_snapshot") or {},
            lambda: read_execution_state(net, links),
        ))
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


def _snapshot(r1_via="10.0.12.2", backup_up=True):
    return {
        "routes": [
            {"node": "r1", "prefix": "10.0.4.0/24", "route": f"10.0.4.0/24 via {r1_via} dev r1-eth1"},
            {"node": "r4", "prefix": "10.0.1.0/24", "route": "10.0.1.0/24 via 10.0.24.1 dev r4-eth0"},
        ],
        "links": [
            {"name": "r1-r2", "up": True},
            {"name": "r2-r4", "up": True},
            {"name": "r1-r3", "up": backup_up},
            {"name": "r3-r4", "up": backup_up},
        ],
    }


def _self_check():
    """Offline check of the execution boundary. Does not touch Mininet."""
    import inspect

    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "check",
        "proposed_path": ["r1", "r3", "r4"],
    }
    calls = []

    def execute(item):
        calls.append(item)
        return {"executed": True, "action": "REROUTE"}

    def once(decision, approved, current=None):
        calls.clear()
        current = approved if current is None else current
        return apply_authorized_action(
            {"decision": decision} if decision is not None else None,
            proposal,
            execute,
            approved,
            lambda: current,
        )

    healthy = _snapshot()
    allowed = once("allow", healthy)
    assert calls == [proposal], calls
    assert allowed["state_verified"] is True and allowed["remediation_succeeded"] is True

    for decision in ("block", "escalate", None, "approve"):
        outcome = once(decision, healthy)
        assert calls == [], decision
        assert outcome["remediation_succeeded"] is False

    changed_route = once("allow", healthy, _snapshot(r1_via="10.0.13.2"))
    assert calls == [] and changed_route["execution_error"] == "route changed since the gate decision"

    changed_backup = once("allow", healthy, _snapshot(backup_up=False))
    assert calls == [] and changed_backup["execution_error"] == "path state changed since the gate decision"

    down = _snapshot(backup_up=False)
    already_down = once("allow", down, down)
    assert calls == [] and already_down["execution_error"] == "backup path is down"
    assert already_down["state_verified"] is False

    def boom(_proposal):
        calls.append(_proposal)
        raise RuntimeError("link update failed")

    calls.clear()
    failed = apply_authorized_action({"decision": "allow"}, proposal, boom, healthy, lambda: healthy)
    assert len(calls) == 1 and failed["remediation_succeeded"] is False
    assert failed["execution_error"] == "RuntimeError: link update failed"

    assert "state_verified" not in inspect.signature(apply_authorized_action).parameters
    assert "state_verified" not in inspect.signature(run_episode).parameters
    print("execution boundary: PASS")


if __name__ == "__main__":
    _self_check()
