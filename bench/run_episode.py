"""
Run one benchmark episode: fault → telemetry → OAM → optional agent/gate.

    python bench/run_episode.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
import time
import uuid
from datetime import datetime, timezone

from bench import inject as inj
from bench import tests as oam
from bench.contracts import validate_record
from bench.traffic import apply_noise, clear_noise
from telemetry.collector import collect_current_state

EPISODE_DIR = Path(__file__).resolve().parent.parent / "results" / "episodes"

GATE_DECISIONS = {"allow", "block", "escalate"}
CONTROL_ROUTES = (("r1", "10.0.4.0/24"), ("r4", "10.0.1.0/24"))
PATH_LINKS = ("r1-r2", "r2-r4", "r1-r3", "r3-r4")
PATH_INTERFACES = (
    ("r1", "r1-eth1"),
    ("r2", "r2-eth0"),
    ("r2", "r2-eth1"),
    ("r4", "r4-eth0"),
    ("r1", "r1-eth2"),
    ("r3", "r3-eth0"),
    ("r3", "r3-eth1"),
    ("r4", "r4-eth1"),
)
_QDISC_COUNTERS = re.compile(r"\b(direct_packets_stat|refcnt|seed)\s+\d+")


def qdisc_config(text):
    """The configured qdisc lines of `tc qdisc show`, without counters."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("qdisc "):
            lines.append(" ".join(_QDISC_COUNTERS.sub("", line).split()))
    return "|".join(lines)


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
    qdiscs = (snapshot or {}).get("qdiscs")
    return {"routes": routes, "links": links, "qdiscs": qdiscs if isinstance(qdiscs, dict) else None}


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
    if current["qdiscs"] is not None and approved["qdiscs"] is None:
        return {"state_verified": False, "reason": "gate snapshot has no path qdisc fingerprint"}
    if approved["qdiscs"] is not None and current["qdiscs"] is None:
        return {"state_verified": False, "reason": "execution recheck could not read path qdiscs"}
    if approved["qdiscs"] != current["qdiscs"]:
        return {"state_verified": False, "reason": "path qdisc changed since the gate decision"}
    from gate.checks import path_links_up, reroute_choice

    choice = reroute_choice(proposal)
    action = (proposal or {}).get("action")
    if choice is None and action in ("REROUTE", "REBALANCE", "reroute_traffic", "rebalance"):
        return {"state_verified": False, "reason": "path not in policy-approved routes"}
    if choice is not None:
        up, _down = path_links_up(current_snapshot, choice)
        if not up:
            return {"state_verified": False, "reason": f"{choice} path is down"}
    return {"state_verified": True, "reason": None}


def _call_timed(fn):
    started = time.monotonic()
    try:
        return fn(), time.monotonic() - started, None
    except Exception as exc:
        return None, time.monotonic() - started, exc


def _agent_cost(proposal, elapsed):
    """Wall time of the agent call, plus token counts when the model returned them."""
    raw = {}
    if isinstance(proposal, dict):
        found = proposal.pop("_cost", None)
        if isinstance(found, dict):
            raw = found
    return {
        "latency_seconds": elapsed,
        "model_latency_seconds": raw.get("latency_seconds"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
    }


def bind_proposal_observation(proposal, snapshot):
    """Bind a proposal to the telemetry snapshot used to produce it."""
    if not isinstance(proposal, dict):
        return proposal
    bound = dict(proposal)
    timestamp = (snapshot or {}).get("timestamp")
    if timestamp:
        bound["observed_at"] = timestamp
    return bound


def screen_proposal(proposal, snapshot=None):
    """Reject a proposal before the gate. None means the proposal may proceed."""
    from agent.validate import proposal_freshness_errors, validate_proposal

    ok, errors = validate_proposal(proposal)
    if ok:
        ok, errors = validate_record("proposal", proposal)
    if ok and snapshot is not None:
        fresh = proposal_freshness_errors(proposal, snapshot)
        if fresh:
            ok = False
            errors = fresh
    if ok:
        return None
    return {
        "agent_error": "; ".join(errors),
        "execution_error": "invalid proposal",
        "execution_attempted": False,
    }


def read_execution_state(net, links=None):
    """Re-read the routes, path links and path qdiscs a remediation depends on."""
    from bench.mininet_util import run_in_node
    from network import controller as ctl

    routes = []
    for node_name, prefix in CONTROL_ROUTES:
        route = run_in_node(net.get(node_name), f"ip route show {prefix}").strip()
        routes.append({"node": node_name, "prefix": prefix, "route": route})
    link_states = [ctl.link_status(net, name, links) for name in PATH_LINKS]
    qdiscs = {
        f"{node_name}/{intf}": qdisc_config(run_in_node(net.get(node_name), f"tc qdisc show dev {intf}"))
        for node_name, intf in PATH_INTERFACES
    }
    return {
        "routes": routes,
        "links": link_states,
        "qdiscs": qdiscs,
        "metrics": {"active_path": ctl.path_from_route("r1", routes[0]["route"])},
    }


def _routes_on_path(snapshot, path):
    """True when both r1 and r4 point at this path's next hops."""
    from network import controller as ctl

    found = {
        item.get("node"): ctl.path_from_route(item.get("node"), item.get("route"))
        for item in (snapshot or {}).get("routes") or []
        if (item.get("node"), item.get("prefix")) in CONTROL_ROUTES
    }
    return found.get("r1") == path and found.get("r4") == path


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
    succeeded = isinstance(result, dict) and result.get("executed") is True
    outcome["remediation_succeeded"] = succeeded
    if succeeded:
        from gate.checks import reroute_choice

        requested = reroute_choice(proposal)
        if requested is not None:
            try:
                after = read_state()
            except Exception as exc:
                outcome["remediation_succeeded"] = False
                outcome["execution_error"] = f"partial remediation: {type(exc).__name__}: {exc}"
                return outcome
            if not _routes_on_path(after, requested):
                outcome["remediation_succeeded"] = False
                outcome["execution_error"] = "partial remediation"
    return outcome


def default_components():
    """The pieces one episode is allowed to call."""
    from control.actions import execute
    from gate.decide import decide

    return {
        "apply_noise": apply_noise,
        "clear_noise": clear_noise,
        "reset_all": inj.reset_all,
        "inject": inj.inject,
        "inject_compound": inj.inject_compound,
        "telemetry": collect_current_state,
        "tests": oam.run_all,
        "read_state": read_execution_state,
        "gate": decide,
        "execute": execute,
        "log": save_episode,
    }


def _execution_view(record):
    return {
        key: record.get(key)
        for key in (
            "execution_attempted",
            "execution_result",
            "execution_error",
            "remediation_succeeded",
            "safeguard_missing",
            "state_verified",
        )
    }


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
    _components=None,
):
    """
    Execute one episode and return a serializable record.
    agent_fn(state) -> proposal dict or None
    gate_fn(episode_so_far) -> gate decision or None
    execute_fn(proposal) -> execution result. Invoked only after allow
        and a fresh route/path recheck against network_snapshot.
    """
    parts = dict(default_components())
    if _components:
        parts.update(_components)
    episode_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    noise_applied = parts["apply_noise"](net, noise_id, links)
    correlation_ids = []

    try:
        parts["reset_all"](net)
        if compound_id:
            comp = parts["inject_compound"](net, compound_id, seed=seed)
            correlation_ids = comp["correlation_ids"]
            fault_label = compound_id
        elif fault_id != "H0":
            correlation_ids = [parts["inject"](net, fault_id, seed=seed)]
            fault_label = fault_id
        else:
            fault_label = "H0"

        snapshot = parts["telemetry"](net, links)
        try:
            baseline = parts["read_state"](net, links)
            baseline_error = None
        except Exception as exc:
            baseline = None
            baseline_error = f"{type(exc).__name__}: {exc}"
        oam_bundle = parts["tests"](net) if run_oam else None

        record = {
            "episode_id": episode_id,
            "started_at": started,
            "fault_id": fault_label,
            "noise_id": noise_id,
            "seed": seed,
            "correlation_ids": correlation_ids,
            "network_snapshot": snapshot,
            "execution_baseline": {
                "qdiscs": (baseline or {}).get("qdiscs") if isinstance(baseline, dict) else None,
                "error": baseline_error,
            },
            "oam": oam_bundle,
            "oam_cost": oam.summarize_costs(
                oam_bundle.get("results") if isinstance(oam_bundle, dict) else None
            ),
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

        telemetry_ok, telemetry_errors = validate_record("telemetry", snapshot)
        if not telemetry_ok:
            record["execution_error"] = "invalid telemetry"
            record["agent_error"] = "; ".join(telemetry_errors)
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            return record

        if agent_fn:
            proposal, elapsed, agent_exc = _call_timed(lambda: agent_fn(snapshot))
            record["agent_cost"] = _agent_cost(proposal, elapsed)
            if agent_exc is not None:
                record["agent_error"] = f"{type(agent_exc).__name__}: {agent_exc}"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record
            record["agent_proposal"] = bind_proposal_observation(proposal, snapshot)
            rejection = screen_proposal(record["agent_proposal"], snapshot)
            if rejection:
                record.update(rejection)
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record

        if gate_fn:
            decision, elapsed, gate_exc = _call_timed(lambda: gate_fn(record))
            record["gate_cost"] = {"latency_seconds": elapsed}
            if gate_exc is not None:
                record["gate_error"] = f"{type(gate_exc).__name__}: {gate_exc}"
                record["execution_error"] = f"gate error: {record['gate_error']}"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record
            record["gate_decision"] = decision
            if record["gate_decision"] is not None:
                gate_ok, gate_errors = validate_record("gate_decision", record["gate_decision"])
                if not gate_ok:
                    record["gate_error"] = "; ".join(gate_errors)
                    record["execution_error"] = f"gate error: {record['gate_error']}"
                    record["completed_at"] = datetime.now(timezone.utc).isoformat()
                    return record

        if execute_fn is None:
            controller = parts["execute"]

            def execute_fn(proposal, _net=net, _execute=controller):
                from control.actions import proposal_to_action

                action, payload = proposal_to_action(proposal or {})
                return _execute(_net, action, payload)

        approved = dict(record.get("network_snapshot") or {})
        approved["qdiscs"] = record["execution_baseline"]["qdiscs"]
        record.update(apply_authorized_action(
            record.get("gate_decision"),
            record.get("agent_proposal") or {},
            execute_fn,
            approved,
            lambda: parts["read_state"](net, links),
        ))
        result_ok, result_errors = validate_record("execution_result", _execution_view(record))
        if not result_ok:
            record["remediation_succeeded"] = False
            record["execution_error"] = "invalid execution result: " + "; ".join(result_errors)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        return record
    finally:
        parts["clear_noise"](net, noise_applied, links)
        parts["reset_all"](net, baseline=True)


def aggregate_episode_costs(directory=None):
    """Sum oam_cost, agent_cost and gate_cost across saved episode JSON files.

    Each block is summed over the episodes that recorded it (`episodes`).
    Each field is summed over the episodes where it was measured, and
    `<field>_known` says how many that was. An unmeasured value is not
    counted as zero.
    """
    directory = Path(directory or EPISODE_DIR)
    if not directory.is_dir():
        return {"episodes": 0, "oam": None, "agent": None, "gate": None}

    battery_fields = ("total_duration_seconds", "total_additional_packets", "total_additional_bytes")
    blocks = {"oam": [], "agent": [], "gate": [], "full_battery": [], "selected_tests": []}
    episodes = 0
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or "episode_id" not in record:
            continue
        episodes += 1
        for name, block in (
            ("oam", record.get("oam_cost")),
            ("agent", record.get("agent_cost")),
            ("gate", record.get("gate_cost")),
        ):
            if isinstance(block, dict):
                blocks[name].append(block)
        selection = (record.get("gate_decision") or {}).get("cost")
        if isinstance(selection, dict):
            for name in ("full_battery", "selected_tests"):
                if isinstance(selection.get(name), dict):
                    blocks[name].append(selection[name])

    def rollup(found, fields):
        if not found:
            return None
        out = {"episodes": len(found)}
        for field in fields:
            values = [
                block.get(field)
                for block in found
                if isinstance(block.get(field), (int, float)) and not isinstance(block.get(field), bool)
            ]
            out[field] = sum(values) if values else None
            out[f"{field}_known"] = len(values)
        return out

    return {
        "episodes": episodes,
        "oam": rollup(blocks["oam"], battery_fields),
        "agent": rollup(blocks["agent"], ("latency_seconds", "model_latency_seconds", "prompt_tokens", "completion_tokens")),
        "gate": rollup(blocks["gate"], ("latency_seconds",)),
        "gate_test_selection": {
            "full_battery": rollup(blocks["full_battery"], battery_fields),
            "selected_tests": rollup(blocks["selected_tests"], battery_fields),
        },
    }


def save_episode(record, directory=None):
    directory = Path(directory or EPISODE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['episode_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _snapshot(r1_via="10.0.12.2", backup_up=True, primary_up=True, active_path="backup", r4_via="10.0.24.1", qdiscs=None):
    snap = {
        "routes": [
            {"node": "r1", "prefix": "10.0.4.0/24", "route": f"10.0.4.0/24 via {r1_via} dev r1-eth1"},
            {"node": "r4", "prefix": "10.0.1.0/24", "route": f"10.0.1.0/24 via {r4_via} dev r4-eth0"},
        ],
        "links": [
            {"name": "r1-r2", "up": primary_up},
            {"name": "r2-r4", "up": primary_up},
            {"name": "r1-r3", "up": backup_up},
            {"name": "r3-r4", "up": backup_up},
        ],
        "metrics": {"active_path": active_path},
    }
    if qdiscs is not None:
        snap["qdiscs"] = qdiscs
    return snap


def _reads(*snapshots):
    """read_state stand-in: each call returns the next snapshot, then the last one."""
    queue = list(snapshots)

    def read():
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return read


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

    def once(decision, approved, current=None, after=None):
        calls.clear()
        current = approved if current is None else current
        return apply_authorized_action(
            {"decision": decision} if decision is not None else None,
            proposal,
            execute,
            approved,
            _reads(current, after if after is not None else current),
        )

    healthy = _snapshot()
    on_backup = _snapshot(r1_via="10.0.13.2", r4_via="10.0.34.1")
    allowed = once("allow", healthy, after=on_backup)
    assert calls == [proposal], calls
    assert allowed["state_verified"] is True and allowed["remediation_succeeded"] is True, allowed

    one_sided = once("allow", healthy, after=_snapshot(r1_via="10.0.13.2"))
    assert one_sided["execution_attempted"] is True and one_sided["remediation_succeeded"] is False
    assert one_sided["execution_error"] == "partial remediation"

    shaped = {"r1/r1-eth2": "qdisc netem 10: parent 5:1 limit 1000 delay 5ms"}
    lossy = {"r1/r1-eth2": "qdisc netem 10: parent 5:1 limit 1000 delay 5ms loss 20%"}
    changed_qdisc = once("allow", _snapshot(qdiscs=shaped), _snapshot(qdiscs=lossy))
    assert calls == [] and changed_qdisc["execution_error"] == "path qdisc changed since the gate decision"
    no_fingerprint = once("allow", healthy, _snapshot(qdiscs=shaped))
    assert calls == [] and no_fingerprint["execution_error"] == "gate snapshot has no path qdisc fingerprint"
    unread = once("allow", _snapshot(qdiscs=shaped), healthy)
    assert calls == [] and unread["execution_error"] == "execution recheck could not read path qdiscs"
    same_qdisc = once("allow", _snapshot(qdiscs=shaped), _snapshot(qdiscs=dict(shaped)), after=on_backup)
    assert same_qdisc["remediation_succeeded"] is True, same_qdisc

    htb = "qdisc htb 5: root refcnt 2 r2q 10 default 0x1 direct_packets_stat {} direct_qlen 1000\n Sent 9 bytes 1 pkt"
    assert qdisc_config(htb.format(0)) == qdisc_config(htb.format(812))
    assert "Sent" not in qdisc_config(htb.format(0))
    assert qdisc_config("qdisc netem 10: delay 5ms") != qdisc_config("qdisc netem 10: delay 5ms loss 5%")

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

    rejected = screen_proposal({"action": "reboot", "target": "HostA_to_HostB", "reason": "no"})
    assert rejected["execution_attempted"] is False
    assert rejected["execution_error"] == "invalid proposal"

    from gate.checks import check_safety

    backup = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "check",
        "proposed_path": ["r1", "r3", "r4"],
    }
    primary = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "check",
        "proposed_path": ["r1", "r2", "r4"],
    }
    assert check_safety(backup, _snapshot(backup_up=False))["passed"] is False
    assert check_safety(primary, _snapshot(primary_up=False))["passed"] is False
    assert check_safety(backup, healthy)["passed"] is True
    from gate.checks import reroute_choice

    conflicted = dict(backup, path="backup", proposed_path=["r1", "r2", "r4"])
    assert reroute_choice(conflicted) is None
    assert check_safety(conflicted, healthy)["passed"] is False
    assert reroute_choice(dict(backup, path="backup")) == "backup"
    assert reroute_choice({"action": "REROUTE", "path": "sideways"}) is None
    blackholed = _snapshot()
    for link in blackholed["links"]:
        link["packet_loss_percent"] = 100.0 if link["name"] == "r3-r4" else 0.0
    dead = check_safety(backup, blackholed)
    assert dead["passed"] is False and dead["reason"] == "backup path is unreachable", dead
    assert check_safety(primary, blackholed)["passed"] is True

    wrong_flow = screen_proposal({"action": "DO_NOTHING", "target": "HostC_to_HostD", "reason": "x"})
    assert wrong_flow and "not a supported flow" in wrong_flow["agent_error"]

    calls.clear()
    primary_down = _snapshot(primary_up=False, active_path="primary")
    calls.clear()
    blocked_primary = apply_authorized_action(
        {"decision": "allow"},
        primary,
        execute,
        primary_down,
        lambda: primary_down,
    )
    assert calls == []
    assert blocked_primary["execution_error"] == "primary path is down"

    calls.clear()
    stayed = apply_authorized_action(
        {"decision": "allow"},
        backup,
        execute,
        healthy,
        lambda: _snapshot(active_path="primary"),
    )
    assert stayed["execution_attempted"] is True
    assert stayed["remediation_succeeded"] is False
    assert stayed["execution_error"] == "partial remediation"

    import ast
    from bench.tests import TEST_FUNCS
    from baselines import METHODS as BASELINE_METHODS
    from baselines import _self_check as baselines_self_check
    from baselines import run_method

    banned = ("network.controller", "network.fault_injector", "bench.inject", "control.actions")
    agent_dir = Path(__file__).resolve().parent.parent / "agent"
    for path in agent_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for imported in names:
                assert not any(imported == item or imported.startswith(item + ".") for item in banned), (path.name, imported)

    required_tests = {
        "T1_e2e_ping",
        "T2_hop_ping",
        "T4_mtu",
        "T5_if_errors",
        "T6_queue_stats",
        "T7_route_check",
        "T8_tcp_vs_ping",
        "T9_oneway_loss",
        "T10_speed",
    }
    assert required_tests <= set(TEST_FUNCS)

    from agent.validate import validate_proposal

    unknown = {
        "action": "DO_NOTHING",
        "target": "HostA_to_HostB",
        "reason": "check",
        "causes": [{"cause_id": "NOT_A_FAULT", "predicted_tests": []}],
    }
    assert validate_proposal(unknown)[0] is False

    shared = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "check",
        "proposed_path": ["r1", "r3", "r4"],
    }
    for method in BASELINE_METHODS:
        out = run_method(method, shared, healthy, [])
        assert out["decision"] in {"allow", "block", "escalate"}, out
        assert "action" in out and out["method"] == method
        assert "tests_run" in out and "budget" in out, out
        assert out["permitted_actions"] == ["REROUTE", "DO_NOTHING", "ESCALATE"]
    baselines_self_check()
    try:
        run_method("B0", shared, healthy, [], fault_id="F1")
        raise AssertionError("fault_id was accepted")
    except TypeError:
        pass

    from control.actions import execute as controller_execute
    from gate.decide import decide as gate_decide

    wired = default_components()
    assert wired["telemetry"] is collect_current_state
    assert wired["tests"] is oam.run_all
    assert wired["inject"] is inj.inject
    assert wired["reset_all"] is inj.reset_all
    assert wired["read_state"] is read_execution_state
    assert wired["execute"] is controller_execute
    assert wired["gate"] is gate_decide
    assert wired["log"] is save_episode

    full = _snapshot(active_path="primary")
    for link in full["links"]:
        link["endpoints"] = ["a", "b"]
        link["latency_ms"] = 5.0
        link["packet_loss_percent"] = 0.0
    full = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": [
            {"name": "hostA", "type": "host", "ips": [{"intf": "hostA-eth0", "ip": "10.0.1.2"}]},
            {"name": "r1", "type": "router", "ips": [{"intf": "r1-eth1", "ip": "10.0.12.1"}]},
        ],
        "links": full["links"],
        "routes": full["routes"],
        "interface_counters": {"r1/r1-eth1": {"rx_errors": 0, "tx_errors": 0, "rx_dropped": 0, "tx_dropped": 0}},
        "queues": {"r1/r1-eth1": {"drops": 0, "backlog_bytes": 0, "backlog_packets": 0, "raw": ""}},
        "metrics": {
            "latency_ms": 20.0,
            "packet_loss_percent": 0.0,
            "connectivity": True,
            "active_path": "primary",
        },
    }
    assert validate_record("telemetry", full)[0]
    assert validate_record("telemetry", {"timestamp": "x"})[0] is False
    assert validate_record("predicted_test", {"test_id": "T1_e2e_ping", "expected": "loss"})[0]
    assert validate_record("predicted_test", {"test_id": "T3_missing", "expected": "loss"})[0] is False
    assert validate_record("gate_decision", {"decision": "approve"})[0] is False

    events = []

    def quiet_noise(net, noise_id, links=None):
        events.append("noise")
        return {}

    def quiet_clear(net, applied, links=None):
        events.append("clear")

    def quiet_reset(net, baseline=False):
        events.append("reset+baseline" if baseline else "reset")

    def quiet_inject(net, fault_id, seed=None):
        events.append("inject")
        return "corr-1"

    def quiet_telemetry(net, links=None):
        events.append("telemetry")
        return full

    def quiet_tests(net):
        events.append("tests")
        return {"results": [{"test_id": "T7_route_check"}]}

    def quiet_read(net, links=None):
        return full

    fakes = {
        "apply_noise": quiet_noise,
        "clear_noise": quiet_clear,
        "reset_all": quiet_reset,
        "inject": quiet_inject,
        "telemetry": quiet_telemetry,
        "tests": quiet_tests,
        "read_state": quiet_read,
    }

    def local_agent(state):
        events.append("agent")
        return {
            "action": "DO_NOTHING",
            "target": "HostA_to_HostB",
            "reason": "healthy",
            "proposed_path": ["r1", "r2", "r4"],
        }

    def local_gate(record):
        events.append("gate")
        return gate_decide(
            None,
            record["agent_proposal"],
            record["network_snapshot"],
            (record.get("oam") or {}).get("results") or [],
            mode="pilot_v0",
            run_missing_tests=False,
            noise_id=record["noise_id"],
        )

    def local_execute(proposal):
        events.append("execute")
        return {"executed": False, "action": proposal.get("action")}

    connected = run_episode(
        object(),
        {},
        fault_id="F1",
        run_oam=True,
        agent_fn=local_agent,
        gate_fn=local_gate,
        execute_fn=local_execute,
        _components=fakes,
    )
    assert connected["gate_decision"]["decision"] == "allow", connected
    assert connected["execution_attempted"] is True and connected["state_verified"] is True
    assert validate_record("proposal", connected["agent_proposal"])[0]
    assert validate_record("gate_decision", connected["gate_decision"])[0]
    assert validate_record("execution_result", _execution_view(connected))[0]
    walked = [item for item in events if item in ("inject", "telemetry", "tests", "agent", "gate", "execute")]
    assert walked == ["inject", "telemetry", "tests", "agent", "gate", "execute"], walked
    assert events[:2] == ["noise", "reset"], events
    assert events[-2:] == ["clear", "reset+baseline"], events

    import tempfile

    logged = save_episode(connected, tempfile.mkdtemp())
    loaded = json.loads(Path(logged).read_text(encoding="utf-8"))
    assert loaded["episode_id"] == connected["episode_id"]
    Path(logged).unlink()
    Path(logged).parent.rmdir()

    skipped = []
    invalid_telemetry = run_episode(
        object(),
        {},
        fault_id="H0",
        run_oam=False,
        agent_fn=lambda state: skipped.append("agent"),
        gate_fn=lambda record: skipped.append("gate"),
        execute_fn=lambda proposal: skipped.append("execute"),
        _components=dict(fakes, telemetry=lambda net, links=None: {"timestamp": "x"}),
    )
    assert skipped == []
    assert invalid_telemetry["execution_error"] == "invalid telemetry"

    invalid_gate = run_episode(
        object(),
        {},
        fault_id="H0",
        run_oam=False,
        agent_fn=local_agent,
        gate_fn=lambda record: {"decision": "approve"},
        execute_fn=lambda proposal: skipped.append("execute"),
        _components=fakes,
    )
    assert skipped == []
    assert invalid_gate["execution_attempted"] is False
    assert str(invalid_gate["execution_error"]).startswith("gate error:")

    def reroute_agent(state):
        return {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "primary degraded",
            "proposed_path": ["r1", "r3", "r4"],
        }

    def allow_gate(record):
        return {"decision": "allow", "reason": "test", "proposal_action": "REROUTE", "tests_run": [], "checks": {}}

    def stateful(*snapshots):
        read = _reads(*snapshots)
        return lambda net, links=None: read()

    with_q = dict(full, qdiscs=shaped)
    injected_between = run_episode(
        object(), {}, fault_id="H0", run_oam=False,
        agent_fn=reroute_agent, gate_fn=allow_gate,
        execute_fn=lambda proposal: skipped.append("execute"),
        _components=dict(fakes, read_state=stateful(with_q, dict(full, qdiscs=lossy))),
    )
    assert skipped == [] and injected_between["execution_attempted"] is False
    assert injected_between["execution_error"] == "path qdisc changed since the gate decision"
    assert injected_between["execution_baseline"]["qdiscs"] == shaped

    def broken_read(net, links=None):
        raise RuntimeError("tc missing")

    reads_left = [broken_read, lambda net, links=None: with_q]

    def flaky_read(net, links=None):
        return reads_left.pop(0)(net, links)

    no_baseline = run_episode(
        object(), {}, fault_id="H0", run_oam=False,
        agent_fn=reroute_agent, gate_fn=allow_gate,
        execute_fn=lambda proposal: skipped.append("execute"),
        _components=dict(fakes, read_state=flaky_read),
    )
    assert skipped == [] and no_baseline["execution_attempted"] is False
    assert no_baseline["execution_baseline"]["error"] == "RuntimeError: tc missing"
    assert no_baseline["execution_error"] == "gate snapshot has no path qdisc fingerprint"

    rerouted = run_episode(
        object(), {}, fault_id="H0", run_oam=False,
        agent_fn=reroute_agent, gate_fn=allow_gate,
        execute_fn=lambda proposal: {"executed": True, "action": "REROUTE"},
        _components=dict(fakes, read_state=stateful(
            with_q, with_q, dict(full, qdiscs=shaped, routes=on_backup["routes"]),
        )),
    )
    assert rerouted["state_verified"] is True and rerouted["remediation_succeeded"] is True, rerouted

    import bench.outcomes as outcomes

    predicted_ok = dict(
        backup,
        causes=[{"cause_id": "F3", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "ok"}]}],
    )
    slow_ping = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 180.0}]
    saved_report = outcomes._REPORT
    outcomes._REPORT = Path(tempfile.gettempdir()) / f"absent-{uuid.uuid4().hex}.json"
    try:
        limits = {level: outcomes.load_yaml_thresholds(level)["T1_e2e_ping"]["max_latency_ms"] for level in ("N0", "N2")}
        assert limits["N0"] < 180.0 < limits["N2"], limits
        at_n0 = gate_decide(None, predicted_ok, healthy, slow_ping, mode="pilot_v0", run_missing_tests=False)
        at_n2 = gate_decide(None, predicted_ok, healthy, slow_ping, mode="pilot_v0", run_missing_tests=False, noise_id="N2")
    finally:
        outcomes._REPORT = saved_report
    assert at_n0["decision"] == "block" and at_n0["reason"] == "consistency failed", at_n0
    assert at_n2["decision"] == "allow", at_n2

    oam._check_costs()
    priced = {
        "action": "DO_NOTHING",
        "target": "HostA_to_HostB",
        "reason": "priced",
        "_cost": {
            "latency_seconds": 0.25,
            "prompt_tokens": 11,
            "completion_tokens": 4,
        },
    }
    priced_cost = _agent_cost(priced, 0.4)
    assert "_cost" not in priced
    assert priced_cost["latency_seconds"] == 0.4
    assert priced_cost["model_latency_seconds"] == 0.25
    assert priced_cost["prompt_tokens"] == 11 and priced_cost["completion_tokens"] == 4
    bare = _agent_cost({"action": "DO_NOTHING"}, 0.1)
    assert bare["prompt_tokens"] is None and bare["model_latency_seconds"] is None

    assert connected["agent_cost"]["prompt_tokens"] is None
    assert connected["agent_cost"]["latency_seconds"] >= 0
    assert connected["gate_cost"]["latency_seconds"] >= 0
    assert connected["oam_cost"]["test_count"] == 1
    assert connected["oam_cost"]["total_additional_bytes"] is None
    compared = connected["gate_decision"]["cost"]
    assert compared["full_battery"]["test_count"] == 1
    assert compared["selected_tests"]["test_count"] == 0
    assert compared["selected_tests"]["total_duration_seconds"] == 0.0
    assert compared["tests_executed"]["total_duration_seconds"] is None

    stale = screen_proposal(
        {
            "action": "DO_NOTHING",
            "target": "HostA_to_HostB",
            "reason": "old",
            "observed_at": "2020-01-01T00:00:00+00:00",
        },
        {"timestamp": full["timestamp"]},
    )
    assert stale and "does not match" in stale["agent_error"], stale

    rolled = aggregate_episode_costs(tempfile.mkdtemp())
    assert rolled["episodes"] == 0 and rolled["oam"] is None

    rollup_dir = Path(tempfile.mkdtemp())
    save_episode(connected, rollup_dir)
    save_episode(dict(connected, episode_id="second"), rollup_dir)
    (rollup_dir / "_batch_state.json").write_text(json.dumps({"completed": 2}), encoding="utf-8")
    batch_only = {
        "episode_id": "batch",
        "oam_cost": {"total_duration_seconds": 20.0, "total_additional_packets": 300, "total_additional_bytes": None},
    }
    save_episode(batch_only, rollup_dir)
    rolled = aggregate_episode_costs(rollup_dir)
    assert rolled["episodes"] == 3, rolled
    assert rolled["oam"]["episodes"] == 3
    assert rolled["oam"]["total_duration_seconds"] == 20.0
    assert rolled["oam"]["total_duration_seconds_known"] == 1
    assert rolled["oam"]["total_additional_bytes"] is None
    assert rolled["agent"]["episodes"] == 2 and rolled["agent"]["latency_seconds_known"] == 2
    assert rolled["agent"]["prompt_tokens"] is None
    assert rolled["gate"]["episodes"] == 2 and rolled["gate"]["latency_seconds"] is not None
    selection = rolled["gate_test_selection"]
    assert selection["full_battery"]["episodes"] == 2
    assert selection["full_battery"]["total_additional_packets"] is None
    assert selection["selected_tests"]["total_duration_seconds"] == 0.0
    assert selection["selected_tests"]["total_additional_bytes"] == 0
    for item in rollup_dir.iterdir():
        item.unlink()
    rollup_dir.rmdir()

    print("execution boundary: PASS")


def live_oam_cost():
    """Full OAM battery on the diamond; checks measured cost fields.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-cost
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("live OAM cost check must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench.mininet_util import mn_cleanup, stop_iperf3_server
    from network.topology import create_network

    mn_cleanup()
    state = create_network()
    net = state["net"]
    try:
        bundle = oam.run_all(net)
        cost = bundle.get("cost") or {}
        results = {row.get("test_id"): row for row in (bundle.get("results") or [])}
    finally:
        try:
            stop_iperf3_server(net)
        finally:
            net.stop()
            mn_cleanup()

    failures = []
    if cost.get("test_count") != 9:
        failures.append(f"test_count={cost.get('test_count')}")
    for test_id, row in results.items():
        if row.get("error"):
            failures.append(f"{test_id} error={row.get('error')}")
        if not isinstance(row.get("duration_seconds"), (int, float)):
            failures.append(f"{test_id} has no duration")
    t8 = results.get("T8_tcp_vs_ping") or {}
    if t8.get("tcp_ok") is not True:
        failures.append("T8 TCP connect failed (no iperf3 listener on 5201)")
    t10 = results.get("T10_speed") or {}
    if (t10.get("duration_seconds") or 0) < 1.0:
        failures.append("T10 finished too quickly (iperf likely failed)")
    probes = t10.get("probes") or []
    iperf_bytes = probes[0].get("bytes") if probes else None
    if not isinstance(iperf_bytes, int) or iperf_bytes <= 0:
        failures.append("T10 additional_bytes missing")
    if cost.get("total_additional_packets") is None:
        failures.append("total_additional_packets unset")
    if cost.get("total_additional_bytes") is None:
        failures.append("total_additional_bytes unset")
    if failures:
        print("live OAM cost: FAIL " + "; ".join(failures))
        print(f"  T10 mbit={t10.get('mbit_per_sec')} duration={t10.get('duration_seconds')} "
              f"server_listening={t10.get('server_listening')}")
        tail = [line for line in (t10.get("raw") or "").strip().splitlines() if line.strip()][-6:]
        print("  T10 client output:" + ("\n    " + "\n    ".join(tail) if tail else " (empty)"))
        if t10.get("server_log"):
            print("  T10 server log:\n    " + t10["server_log"].replace("\n", "\n    "))
        return 1
    evidence = Path(__file__).resolve().parent.parent / "results" / "live_oam_cost.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps({"status": "pass", "cost": cost, "T10": t10}, indent=2),
        encoding="utf-8",
    )
    print("live OAM cost: PASS")
    return 0


def live_diamond():
    """One healthy pass on the diamond topology.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("diamond integration must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench.mininet_util import mn_cleanup
    from network.topology import create_network

    parts = default_components()
    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    try:
        def agent_fn(_snapshot):
            return {
                "action": "DO_NOTHING",
                "target": "HostA_to_HostB",
                "reason": "integration",
                "proposed_path": ["r1", "r2", "r4"],
            }

        def gate_fn(record, _net=net):
            oam_results = (record.get("oam") or {}).get("results") or []
            return parts["gate"](
                _net,
                record["agent_proposal"],
                record["network_snapshot"],
                oam_results,
                mode="pilot_v0",
                run_missing_tests=False,
                noise_id=record["noise_id"],
            )

        record = run_episode(
            net,
            links,
            fault_id="H0",
            run_oam=True,
            agent_fn=agent_fn,
            gate_fn=gate_fn,
            _components={"tests": lambda _net: {"results": [oam.run_test(_net, "T7_route_check")]}},
        )
    finally:
        net.stop()

    failures = []
    if not validate_record("telemetry", record.get("network_snapshot"))[0]:
        failures.append("telemetry")
    names = {node.get("name") for node in (record.get("network_snapshot") or {}).get("nodes") or []}
    if not {"hostA", "hostB", "r1", "r2", "r3", "r4"} <= names:
        failures.append("diamond nodes")
    if not validate_record("proposal", record.get("agent_proposal"))[0]:
        failures.append("proposal")
    if (record.get("gate_decision") or {}).get("decision") != "allow":
        failures.append("gate")
    if not validate_record("execution_result", _execution_view(record))[0]:
        failures.append("execution")
    if record.get("execution_attempted") is not True or record.get("state_verified") is not True:
        failures.append(record.get("execution_error") or "execution withheld")
    oam_ids = [item.get("test_id") for item in (record.get("oam") or {}).get("results") or []]
    if oam_ids != ["T7_route_check"]:
        failures.append("test runner")
    if failures:
        print("diamond integration: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("diamond integration: PASS")
    return 0


def live_recheck():
    """Execution recheck against the live diamond, one case per stale-state kind.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-recheck

    Link-down cases run last: taking an interface down deletes the static
    routes through it, and the earlier cases need the backup path to carry traffic.
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("live recheck must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench.mininet_util import mn_cleanup, run_in_node
    from control.actions import execute, proposal_to_action
    from network import controller as ctl
    from network.topology import create_network

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "live recheck",
        "proposed_path": ["r1", "r3", "r4"],
    }
    allow = {"decision": "allow"}
    executed = []

    def read():
        return read_execution_state(net, links)

    def controller(item):
        executed.append(item)
        action, payload = proposal_to_action(item)
        return execute(net, action, payload)

    def one_sided(item):
        executed.append(item)
        run_in_node(net.get("r1"), "ip route replace 10.0.4.0/24 via 10.0.13.2 dev r1-eth2")
        return {"executed": True, "action": "REROUTE", "path": "backup"}

    def attempt(execute_fn=controller, approved=None):
        executed.clear()
        return apply_authorized_action(allow, proposal, execute_fn, approved or read(), read)

    failures = []

    def expect(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    try:
        out = attempt()
        loss = oam._parse_ping(run_in_node(net.get("hostA"), "ping -c 5 -i 0.2 -W 1 10.0.4.2"))["packet_loss_percent"]
        expect(
            "allow executes and both routers move to backup",
            out["state_verified"] and out["remediation_succeeded"] and _routes_on_path(read(), "backup"),
            str(out),
        )
        expect("traffic flows on backup", loss < 100.0, f"loss {loss}%")
        ctl.set_active_path(net, "primary")

        approved = read()
        ctl.set_active_path(net, "backup")
        out = attempt(approved=approved)
        expect("stale route blocks execution", not executed and out["execution_error"] == "route changed since the gate decision", str(out))
        ctl.set_active_path(net, "primary")

        approved = read()
        lossy = ctl.get_link(net, "r1-r3", links)
        ctl.configure_link(lossy, loss=20)
        out = attempt(approved=approved)
        expect("qdisc change blocks execution", not executed and out["execution_error"] == "path qdisc changed since the gate decision", str(out))
        ctl.configure_link(lossy, **ctl.BASELINE_LINK_PARAMS)

        out = attempt(one_sided)
        expect(
            "one-sided reroute is recorded as partial",
            len(executed) == 1 and out["remediation_succeeded"] is False and out["execution_error"] == "partial remediation",
            str(out),
        )
        ctl.set_active_path(net, "primary")
        expect("primary restored", _routes_on_path(read(), "primary"))

        approved = read()
        changed = ctl.get_link(net, "r3-r4", links)
        ctl.link_down(changed)
        out = attempt(approved=approved)
        expect("backup link change blocks execution", not executed and out["execution_error"] == "path state changed since the gate decision", str(out))
        ctl.link_up(changed)

        down = ctl.get_link(net, "r1-r3", links)
        ctl.link_down(down)
        out = attempt()
        expect("backup already down blocks execution", not executed and out["execution_error"] == "backup path is down", str(out))
        ctl.link_up(down)
        expect("routes still on primary after blocked cases", _routes_on_path(read(), "primary"))
    finally:
        net.stop()
        mn_cleanup()

    if failures:
        print("live recheck: FAIL " + "; ".join(failures))
        return 1
    print("live recheck: PASS")
    return 0


if __name__ == "__main__":
    if "--live-recheck" in sys.argv:
        sys.exit(live_recheck())
    if "--live" in sys.argv:
        sys.exit(live_diamond())
    if "--live-cost" in sys.argv:
        sys.exit(live_oam_cost())
    if "--aggregate-costs" in sys.argv:
        print(json.dumps(aggregate_episode_costs(), indent=2))
        sys.exit(0)
    _self_check()
