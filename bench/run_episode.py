"""
Run one benchmark episode: fault → telemetry → OAM → optional agent/gate.

    python bench/run_episode.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import signal
import time
import uuid
from datetime import datetime, timezone

from bench import inject as inj
from bench import tests as oam
from bench.contracts import atomic_write_text, refuse_protected_path, validate_record
from bench.traffic import apply_noise, clear_noise
from telemetry.collector import collect_current_state

EPISODE_DIR = Path(__file__).resolve().parent.parent / "results" / "episodes"

GATE_DECISIONS = {"allow", "block", "escalate"}
from control.recheck import (  # noqa: E402  (moved; re-exported for callers and tests)
    CONTROL_ROUTES,
    PATH_INTERFACES,
    PATH_LINKS,
    assess_recheck,
    execution_fingerprint,
    qdisc_config,
)


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
        "agent_error_class": "invalid_output",
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


# Independent post-action recovery statuses. Command success is never one of these.
RECOVERY_RECOVERED = "recovered"
RECOVERY_FAILED = "failed"
RECOVERY_PARTIAL = "partial"
RECOVERY_UNAVAILABLE = "measurement_unavailable"
RECOVERY_NOT_ATTEMPTED = "not_attempted"


def _row_by_id(results, test_id):
    for row in results or []:
        if isinstance(row, dict) and row.get("test_id") == test_id:
            return row
    return None


def traffic_conditions(results):
    """Connectivity and performance fields from an OAM result list.

    Errored or missing probes yield None for that metric and are listed under
    infrastructure_errors. A missing metric is not treated as healthy.
    """
    infra = {}
    for row in results or []:
        if isinstance(row, dict) and row.get("test_id") and row.get("error"):
            infra[row["test_id"]] = row["error"]
    t1 = _row_by_id(results, "T1_e2e_ping")
    t10 = _row_by_id(results, "T10_speed")
    connectivity = latency_ms = loss = throughput = None
    if isinstance(t1, dict) and not t1.get("error"):
        connectivity = t1.get("connectivity")
        if isinstance(t1.get("latency_ms"), (int, float)):
            latency_ms = float(t1["latency_ms"])
        if isinstance(t1.get("packet_loss_percent"), (int, float)):
            loss = float(t1["packet_loss_percent"])
    if isinstance(t10, dict) and not t10.get("error"):
        try:
            throughput = float(t10.get("mbit_per_sec"))
        except (TypeError, ValueError):
            throughput = None
    return {
        "connectivity": connectivity,
        "latency_ms": latency_ms,
        "packet_loss_percent": loss,
        "throughput_mbit": throughput,
        "infrastructure_errors": infra,
    }


def pre_remediation_observations(oam_bundle):
    """Snapshot of pre-action OAM used for recovery comparison."""
    results = (oam_bundle or {}).get("results") if isinstance(oam_bundle, dict) else None
    if results is None and isinstance(oam_bundle, list):
        results = oam_bundle
    conditions = traffic_conditions(results or [])
    return {
        "recorded": bool(results),
        "oam_started_at": (oam_bundle or {}).get("started_at") if isinstance(oam_bundle, dict) else None,
        "oam_completed_at": (oam_bundle or {}).get("completed_at") if isinstance(oam_bundle, dict) else None,
        **conditions,
    }


def _classify_recovery(results, noise_id="N0"):
    """Map measured results to recovery_status without using command success.

    Required recovery tests that are absent, errored or ungradable (e.g. T1
    loss None) are measurement_unavailable, never recovered, partial or
    failed. Partial means connectivity is back but performance is not.
    """
    from bench.signature_checks import (
        RECOVERY_T10_MIN_MBIT,
        degraded_tests,
        missing_recovery_evidence,
    )
    from bench.outcomes import signature

    results = list(results or [])
    by_id = {
        row.get("test_id"): row
        for row in results
        if isinstance(row, dict) and row.get("test_id")
    }
    infra = {
        tid: row.get("error")
        for tid, row in by_id.items()
        if row.get("error")
    }
    conditions = traffic_conditions(results)
    outcomes = signature(results, noise_id=noise_id)
    missing_required = missing_recovery_evidence(results, outcomes)
    t10 = by_id.get("T10_speed")
    samples = []
    if isinstance(t10, dict) and not t10.get("error"):
        try:
            mbit = float(t10.get("mbit_per_sec"))
        except (TypeError, ValueError):
            mbit = None
        if mbit is not None:
            samples = [mbit]
    degraded = degraded_tests(outcomes, t10_samples=samples)
    if missing_required:
        status = RECOVERY_UNAVAILABLE
        traffic_recovered = None
    elif not degraded:
        status = RECOVERY_RECOVERED
        traffic_recovered = True
    elif conditions.get("connectivity") is True or (
        outcomes.get("T1_e2e_ping") == "ok" and "T1_e2e_ping" not in degraded
    ):
        status = RECOVERY_PARTIAL
        traffic_recovered = False
    else:
        status = RECOVERY_FAILED
        traffic_recovered = False
    return {
        "results": results,
        "outcomes": outcomes,
        "degraded": degraded,
        "missing_evidence": missing_required,
        "infrastructure_errors": infra,
        "t10_min_mbit": RECOVERY_T10_MIN_MBIT,
        "post_action": {
            "connectivity": conditions["connectivity"],
            "latency_ms": conditions["latency_ms"],
            "packet_loss_percent": conditions["packet_loss_percent"],
            "throughput_mbit": conditions["throughput_mbit"],
        },
        "recovery_status": status,
        "traffic_recovered": traffic_recovered,
    }


def measure_traffic_recovery(net, noise_id="N0", ping_count=None, speed_seconds=None):
    """Independent post-remediation OAM. Command success is not recovery.

    Reuses the recovery test set from the observability matrix (T1/T4/T8
    healthy and T10 ≥ 4 Mbit when measured). Infrastructure failures yield
    measurement_unavailable with traffic_recovered=None.
    """
    from bench.signature_checks import RECOVERY_TESTS

    if ping_count is None:
        ping_count = getattr(oam, "EPISODE_PING_COUNT", 10)
    if speed_seconds is None:
        speed_seconds = getattr(oam, "EPISODE_SPEED_SECONDS", 3)
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    try:
        battery = oam.run_all(
            net,
            test_ids=list(RECOVERY_TESTS),
            ping_count=ping_count,
            speed_seconds=speed_seconds,
        )
    except Exception as exc:
        duration = time.monotonic() - t0
        completed_at = datetime.now(timezone.utc).isoformat()
        err = f"{type(exc).__name__}: {exc}"
        return {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration,
            "results": [],
            "outcomes": {},
            "degraded": list(RECOVERY_TESTS),
            "missing_evidence": list(RECOVERY_TESTS),
            "infrastructure_errors": {"battery": err},
            "t10_min_mbit": None,
            "post_action": {
                "connectivity": None,
                "latency_ms": None,
                "packet_loss_percent": None,
                "throughput_mbit": None,
            },
            "recovery_status": RECOVERY_UNAVAILABLE,
            "traffic_recovered": None,
            "error": err,
        }
    duration = time.monotonic() - t0
    completed_at = datetime.now(timezone.utc).isoformat()
    results = list(battery.get("results") or [])
    classified = _classify_recovery(results, noise_id=noise_id)
    classified.update({
        "started_at": battery.get("started_at") or started_at,
        "completed_at": battery.get("completed_at") or completed_at,
        "duration_seconds": duration,
        "battery_cost": battery.get("cost"),
    })
    return classified


def _iso_span_seconds(started_at, completed_at):
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        ended = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (ended - started).total_seconds()


def route_state_result(before_snapshot, after_snapshot):
    """Whether path routes/links/qdiscs changed between two execution snapshots."""
    if before_snapshot is None or after_snapshot is None:
        return {
            "route_state_changed": None,
            "before": None,
            "after": None,
            "error": "missing snapshot for route-state comparison",
        }
    before = execution_fingerprint(before_snapshot)
    after = execution_fingerprint(after_snapshot)
    return {
        "route_state_changed": before != after,
        "before": before,
        "after": after,
        "error": None,
    }


def attach_post_action_recovery(
    record,
    net,
    *,
    read_state=None,
    links=None,
    noise_id=None,
    measure_fn=None,
):
    """Record independent recovery after an authorized execution attempt.

    Runs for every execution_attempted episode (including DO_NOTHING, where
    executed=False). Never sets traffic_recovered from remediation_succeeded.
    """
    record = record if isinstance(record, dict) else {}
    measure_fn = measure_fn or measure_traffic_recovery
    pre = record.get("pre_remediation")
    if pre is None:
        record["pre_remediation"] = pre_remediation_observations(record.get("oam"))
    if not record.get("execution_attempted"):
        record["recovery_status"] = RECOVERY_NOT_ATTEMPTED
        record["traffic_recovered"] = None
        record["route_state_changed"] = None
        record["recovery_time_seconds"] = None
        record["recovery_measurement_duration_seconds"] = None
        record.setdefault("post_oam", None)
        return record

    noise_id = noise_id or record.get("noise_id")
    if not noise_id:
        raise ValueError("attach_post_action_recovery requires noise_id (N0/N1/N2)")

    live_net = net is not None and callable(getattr(net, "get", None))
    after_snapshot = None
    route_error = None
    if live_net and callable(read_state):
        try:
            after_snapshot = read_state(net, links) if links is not None else read_state(net)
        except TypeError:
            try:
                after_snapshot = read_state()
            except Exception as exc:
                route_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            route_error = f"{type(exc).__name__}: {exc}"
    approved = dict(record.get("network_snapshot") or {})
    baseline = record.get("execution_baseline") or {}
    if baseline.get("qdiscs") is not None:
        approved["qdiscs"] = baseline.get("qdiscs")
    route = route_state_result(approved if approved.get("routes") else None, after_snapshot)
    if route_error and route.get("error"):
        route["error"] = route_error
    elif route_error:
        route["error"] = route_error
        route["route_state_changed"] = None
    record["route_state"] = route
    record["route_state_changed"] = route.get("route_state_changed")

    if not live_net:
        record["post_oam"] = {
            "recovery_status": RECOVERY_UNAVAILABLE,
            "traffic_recovered": None,
            "infrastructure_errors": {"net": "post-action measurement net unavailable"},
            "error": "post-action measurement net unavailable",
        }
        record["recovery_status"] = RECOVERY_UNAVAILABLE
        record["traffic_recovered"] = None
        record["recovery_time_seconds"] = None
        record["recovery_measurement_duration_seconds"] = None
        return record

    try:
        post = measure_fn(net, noise_id=noise_id)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        post = {
            "recovery_status": RECOVERY_UNAVAILABLE,
            "traffic_recovered": None,
            "infrastructure_errors": {"measure": err},
            "error": err,
            "post_action": {
                "connectivity": None,
                "latency_ms": None,
                "packet_loss_percent": None,
                "throughput_mbit": None,
            },
        }
    record["post_oam"] = post
    record["recovery_status"] = post.get("recovery_status")
    record["traffic_recovered"] = post.get("traffic_recovered")
    record["recovery_measurement_duration_seconds"] = post.get("duration_seconds")
    if post.get("recovery_status") == RECOVERY_RECOVERED:
        span = _iso_span_seconds(
            record.get("remediation_started_at") or post.get("started_at"),
            post.get("completed_at"),
        )
        record["recovery_time_seconds"] = span if span is not None else post.get("duration_seconds")
    else:
        record["recovery_time_seconds"] = None
    return record


def _execute_with_capability(execute_fn, proposal, capability):
    """Call execute_fn with the capability when it declares one (the gated executor)."""
    import inspect

    try:
        params = inspect.signature(execute_fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "capability" in params:
        return execute_fn(proposal, capability=capability)
    return execute_fn(proposal)


def apply_authorized_action(
    gate_decision,
    proposal,
    execute_fn,
    approved_snapshot,
    read_state,
    evidence=None,
    *,
    gate_snapshot=None,
    route_control=None,
    measure_partial=None,
    ledger=None,
    public_key=None,
):
    """Single in-process boundary between a gate decision and remediation.

    Stage 1 integrity boundary, NOT a security boundary: code in this process
    can reach the gate key. The process boundary is control/executor_service.

    On allow: the proposal is revalidated; the gate-issued authorization
    (gate/authority.py) must be sealed by the gate key, unexpired, and match
    the verdict, action, target, reroute path, proposal, the snapshot the gate
    evaluated (gate_snapshot, default approved_snapshot), the evidence rows
    (None means none) and the current gate config. The decision id is then
    consumed in the replay ledger. read_state() is compared with
    approved_snapshot; that fresh read is also the exact route capture used
    for compensation. The action runs through control.gated.run_lifecycle:
    routes are always read afterwards; a partial action is measured
    (measure_partial) and then compensated through route_control, verified
    twice; an unverified compensation raises TopologyContaminated with this
    outcome attached. Only one authorized execution runs at a time.
    """
    import time as _time

    from control import gated
    from control.ledger import LedgerError, default_ledger
    from control.routes import RouteReadError, specs_from_snapshot
    from gate.authority import AuthorizationKeyError, authorization_errors

    outcome = {
        "execution_attempted": False,
        "execution_result": None,
        "execution_error": None,
        "remediation_succeeded": False,
        "safeguard_missing": None,
        "state_verified": False,
        "authorization_verify_seconds": None,
        "routes_before": None,
        "routes_after": None,
        "partial_application": None,
        "compensation": None,
    }
    decision = normalize_gate_decision(gate_decision)
    if decision != "allow":
        if decision is None:
            outcome["execution_error"] = "missing or invalid gate decision"
        return outcome

    snap_for_freshness = approved_snapshot if (approved_snapshot or {}).get("timestamp") else None
    rejection = screen_proposal(proposal, snap_for_freshness)
    if rejection:
        outcome["execution_error"] = rejection["execution_error"]
        return outcome
    started = _time.perf_counter()
    try:
        auth_errs = authorization_errors(
            gate_decision,
            approved_snapshot if gate_snapshot is None else gate_snapshot,
            proposal,
            evidence,
            public_key=public_key,
            path_qdiscs=(approved_snapshot or {}).get("qdiscs"),
        )
    except AuthorizationKeyError as exc:
        auth_errs = [str(exc)]
    outcome["authorization_verify_seconds"] = _time.perf_counter() - started
    if auth_errs:
        outcome["execution_error"] = auth_errs[0]
        return outcome
    authorization = gate_decision["authorization"]
    try:
        with gated.single_flight():
            ledger = default_ledger() if ledger is None else ledger
            try:
                used = ledger.consume(authorization["decision_id"], authorization["expires_at"])
            except LedgerError as exc:
                used = f"replay ledger unavailable: {exc}"
            if used:
                outcome["execution_error"] = used
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
            try:
                routes_before = specs_from_snapshot(current_snapshot)
            except RouteReadError as exc:
                outcome["execution_error"] = f"route capture failed: {exc}"
                return outcome
            capability = gated.mint(authorization, proposal)
            outcome["execution_attempted"] = True

            def _restore(router, spec):
                if route_control is None:
                    raise RuntimeError("no route control: compensation cannot run")
                route_control.restore(router, spec)

            try:
                life = gated.run_lifecycle(
                    capability,
                    routes_before,
                    execute=lambda cap: _execute_with_capability(execute_fn, proposal, cap),
                    read_routes=lambda: specs_from_snapshot(read_state()),
                    restore_route=_restore,
                    measure_partial=measure_partial,
                )
            except gated.CompensationFailed as exc:
                outcome.update(_lifecycle_view(exc.outcome))
                outcome["remediation_succeeded"] = False
                if not outcome["execution_error"]:
                    outcome["execution_error"] = "partial remediation"
                raise TopologyContaminated(
                    f"gated execution: {exc}", None, record=dict(outcome)
                ) from exc
    except gated.ExecutionBusy as exc:
        outcome["execution_error"] = str(exc)
        return outcome
    outcome.update(_lifecycle_view(life))
    result = life["execution_result"]
    succeeded = isinstance(result, dict) and result.get("executed") is True and not life["execution_error"]
    if succeeded and capability.action in gated.ROUTE_CHANGING:
        if life["routes_after"] is None:
            succeeded = False
            outcome["execution_error"] = f"partial remediation: {life['route_read_error']}"
        elif not life["fully_applied"]:
            succeeded = False
            outcome["execution_error"] = "partial remediation"
    outcome["remediation_succeeded"] = succeeded
    return outcome


def _lifecycle_view(life):
    return {
        "execution_result": life.get("execution_result"),
        "execution_error": life.get("execution_error"),
        "routes_before": life.get("routes_before"),
        "routes_after": life.get("routes_after"),
        "partial_application": life.get("partial_application"),
        "compensation": life.get("compensation"),
    }


def default_components():
    """The pieces one episode is allowed to call."""
    from control.actions import execute_gated
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
        "execute": execute_gated,
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
    noise_id=None,
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
    noise_id must be the episode traffic level (N0/N1/N2); no silent default.
    """
    if not noise_id:
        raise ValueError("run_episode requires noise_id (N0/N1/N2)")
    parts = dict(default_components())
    if _components:
        parts.update(_components)
    real_reset = parts["reset_all"] is inj.reset_all
    episode_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    noise_applied = None
    correlation_ids = []
    record = None

    try:
        # Noise is inside the try so a partial/failed setup still hits cleanup.
        noise_applied = parts["apply_noise"](net, noise_id, links)
        try:
            pre_reset = parts["reset_all"](net)
        except TopologyContaminated:
            raise
        except Exception as exc:
            raise TopologyContaminated(
                f"pre-episode reset: {type(exc).__name__}: {exc}",
                None,
                record=record,
            ) from exc
        # Background traffic is on here, so only errors and residual count.
        require_clean_topology(
            pre_reset,
            "pre-episode reset",
            real_reset=real_reset,
            check_health=False,
        )
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

        from agent.decision_agent import PROMPT_SHA256, PROMPT_VERSION

        record = {
            "episode_id": episode_id,
            "started_at": started,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
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
            "agent_error_class": None,
            "agent_raw": None,
            "request_fingerprint": None,
            "gate_decision": None,
            "gate_error": None,
            "execution_attempted": False,
            "execution_result": None,
            "execution_error": None,
            "remediation_succeeded": False,
            "safeguard_missing": None,
            "state_verified": False,
            "pre_remediation": pre_remediation_observations(oam_bundle),
            "traffic_recovered": None,
            "recovery_status": RECOVERY_NOT_ATTEMPTED,
            "route_state_changed": None,
            "recovery_time_seconds": None,
            "recovery_measurement_duration_seconds": None,
            "post_oam": None,
        }

        telemetry_ok, telemetry_errors = validate_record("telemetry", snapshot)
        if not telemetry_ok:
            record["execution_error"] = "invalid telemetry"
            record["agent_error"] = "; ".join(telemetry_errors)
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            return record

        if agent_fn:
            from agent.decision_agent import AgentOutputError

            proposal, elapsed, agent_exc = _call_timed(lambda: agent_fn(snapshot))
            if isinstance(proposal, dict):
                record["agent_raw"] = proposal.pop("_raw", None)
                record["request_fingerprint"] = proposal.pop("_fingerprint", None)
            record["agent_cost"] = _agent_cost(proposal, elapsed)
            if agent_exc is not None:
                record["request_fingerprint"] = getattr(agent_exc, "request_fingerprint", None)
                record["agent_error"] = f"{type(agent_exc).__name__}: {agent_exc}"
                if isinstance(agent_exc, AgentOutputError):
                    record["agent_error_class"] = agent_exc.error_class or "invalid_output"
                    record["agent_raw"] = {
                        "content": agent_exc.raw,
                        "parsed": agent_exc.parsed,
                        "validation_errors": agent_exc.errors,
                    }
                else:
                    record["agent_error_class"] = "model_error"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record
            record["agent_error_class"] = None
            record["agent_proposal"] = bind_proposal_observation(proposal, snapshot)
            rejection = screen_proposal(record["agent_proposal"], snapshot)
            if rejection:
                record.update(rejection)
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record

        # gate.decide copies the evidence when it is entered and seals that
        # copy, so a row changed during or after the gate call no longer
        # matches at the execution boundary.
        if gate_fn:
            decision, elapsed, gate_exc = _call_timed(lambda: gate_fn(record))
            record["gate_cost"] = {
                "latency_seconds": elapsed,
                # Included in latency_seconds; reported apart (gate-issued authorization).
                "authorization_seconds": ((decision or {}).get("authorization") or {}).get("issue_seconds")
                if isinstance(decision, dict) else None,
            }
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

            def execute_fn(proposal, capability=None, _net=net, _execute=controller):
                # The gated executor: refuses without a verified capability.
                return _execute(_net, capability)

        approved = dict(record.get("network_snapshot") or {})
        approved["qdiscs"] = record["execution_baseline"]["qdiscs"]
        oam_results = (record.get("oam") or {}).get("results") or []
        # No binding is created here: the authorization was issued by the gate
        # inside gate_fn (gate.decide) and is verified at the boundary.
        measured_partial = []

        def _measure_partial(view):
            # Research semantics: a partial action is measured as it was left,
            # before compensation restores the routes.
            record.update(_execution_view(dict(record, **view, execution_attempted=True)))
            attach_post_action_recovery(
                record,
                net,
                read_state=parts["read_state"],
                links=links,
                noise_id=record.get("noise_id") or noise_id,
            )
            measured_partial.append(True)

        route_control = None
        if net is not None and callable(getattr(net, "get", None)):
            from control.routes import RouteControl

            route_control = RouteControl(net)
        record["remediation_started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            record.update(apply_authorized_action(
                record.get("gate_decision"),
                record.get("agent_proposal") or {},
                execute_fn,
                approved,
                lambda: parts["read_state"](net, links),
                evidence=oam_results,
                gate_snapshot=record.get("network_snapshot") or {},
                route_control=route_control,
                measure_partial=_measure_partial,
            ))
        except TopologyContaminated as exc:
            record.update(exc.record or {})
            record["remediation_completed_at"] = datetime.now(timezone.utc).isoformat()
            exc.record = record
            raise
        except BaseException as exc:
            # A timeout delivered when the gated section ends: keep what happened.
            view = getattr(exc, "gated_outcome", None)
            if isinstance(view, dict):
                record.update(_lifecycle_view(view))
                record["execution_attempted"] = True
                record["remediation_completed_at"] = datetime.now(timezone.utc).isoformat()
            raise
        record["remediation_completed_at"] = datetime.now(timezone.utc).isoformat()
        result_ok, result_errors = validate_record("execution_result", _execution_view(record))
        if not result_ok:
            record["remediation_succeeded"] = False
            record["execution_error"] = "invalid execution result: " + "; ".join(result_errors)
        # Independent post-action OAM: executed=True is never traffic recovery.
        # A compensated partial action was already measured before compensation.
        if not measured_partial:
            attach_post_action_recovery(
                record,
                net,
                read_state=parts["read_state"],
                links=links,
                noise_id=record.get("noise_id") or noise_id,
            )
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        return record
    finally:
        # E6: cancel batch/episode SIGALRM before teardown so a timeout cannot
        # interrupt cleanup and let the next episode run on a dirty topology.
        if hasattr(signal, "SIGALRM"):
            try:
                signal.alarm(0)
            except Exception:
                pass
        clear_error = None
        teardown = None
        try:
            parts["clear_noise"](net, noise_applied or {}, links)
        except Exception as exc:
            clear_error = f"clear_noise: {type(exc).__name__}: {exc}"
        try:
            teardown = parts["reset_all"](net, baseline=True)
        except TopologyContaminated:
            raise
        except TimeoutError as exc:
            problem = f"timeout during teardown: {exc}"
            if clear_error:
                problem = f"{problem}; {clear_error}"
            if isinstance(record, dict):
                record["teardown"] = teardown_summary(
                    {"errors": [{"error": problem}], "residual": {}, "healthy": False},
                    problem,
                )
            raise TopologyContaminated(
                f"episode teardown: {problem}",
                None,
                record=record,
            ) from exc
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
            if clear_error:
                problem = f"{problem}; {clear_error}"
            if isinstance(record, dict):
                record["teardown"] = teardown_summary(
                    {"errors": [{"error": problem}], "residual": {}, "healthy": False},
                    problem,
                )
            raise TopologyContaminated(
                f"episode teardown: {problem}",
                None,
                record=record,
            ) from exc
        finally:
            # Mark after teardown attempt so post-episode B5 cannot run live OAM
            # on a reset net (pilot previously passed net=net after return).
            try:
                setattr(net, "_verifix_teardown_complete", True)
            except Exception:
                pass
        problem = teardown_problem(teardown, real_reset, check_health=True)
        if clear_error:
            problem = f"{problem}; {clear_error}" if problem else clear_error
        if isinstance(record, dict):
            record["teardown"] = teardown_summary(teardown, problem)
        if problem:
            # Never run another episode on a topology that may still carry
            # this one's fault. Preserve the partial episode record.
            raise TopologyContaminated(f"episode teardown: {problem}", teardown, record=record)


class TopologyContaminated(RuntimeError):
    """Reset or cleanup failed. The topology may still carry a fault.

    Runners must stop instead of recording an infra error and continuing.
    """

    def __init__(self, message, report=None, record=None):
        super().__init__(message)
        self.report = report
        self.record = record


def teardown_problem(report, real_reset, check_health=True):
    """Why a reset report shows a possibly contaminated topology, or None.

    Explicit errors or residual state always fail closed, including stub
    resets used in offline tests. Missing reports and missing health checks
    fail only for the authoritative injector reset.
    """
    if not isinstance(report, dict):
        return "reset returned no report" if real_reset else None
    problems = []
    if report.get("errors"):
        problems.append("reset errors " + json.dumps(report["errors"], default=str)[:300])
    if report.get("residual"):
        problems.append("residual changes " + json.dumps(report["residual"], default=str)[:300])
    if check_health and report.get("healthy") is False:
        problems.append(f"hostA->hostB loss {report.get('loss_percent')}% after reset")
    if check_health and real_reset and "healthy" not in report:
        problems.append("reset report has no health check")
    return "; ".join(problems) or None


def require_clean_topology(report, where, *, real_reset=True, check_health=True, record=None):
    """Raise TopologyContaminated when a reset/cleanup report is unsafe."""
    problem = teardown_problem(report, real_reset, check_health=check_health)
    if problem:
        raise TopologyContaminated(f"{where}: {problem}", report, record=record)
    return report


def teardown_summary(report, problem):
    report = report if isinstance(report, dict) else {}
    return {
        "clean": problem is None,
        "problem": problem,
        "cleared": len(report.get("cleared") or []),
        "errors": report.get("errors") or [],
        "residual": report.get("residual") or {},
        "healthy": report.get("healthy"),
        "loss_percent": report.get("loss_percent"),
    }


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


def save_episode(record, directory):
    directory = Path(directory)
    path = refuse_protected_path(directory / f"{record['episode_id']}.json")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2))
    return path


def infra_error_record(fault_id, noise_id, seed, exc, compound_id=None):
    """Attempted episode that failed before a proposal was recorded."""
    from agent.decision_agent import PROMPT_SHA256, PROMPT_VERSION

    now = datetime.now(timezone.utc).isoformat()
    return {
        "episode_id": str(uuid.uuid4()),
        "started_at": now,
        "completed_at": now,
        "status": "infra_error",
        "infra_error": f"{type(exc).__name__}: {exc}",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "request_fingerprint": None,
        "fault_id": compound_id or fault_id,
        "noise_id": noise_id,
        "seed": seed,
        "agent_proposal": None,
        "agent_error": None,
        "agent_error_class": None,
        "agent_raw": None,
        "gate_decision": None,
        "gate_error": None,
        "execution_attempted": False,
        "execution_result": None,
        "execution_error": None,
        "remediation_succeeded": False,
        "state_verified": False,
    }


def _cause_proposal(action, cause_id, path=None, expected="ok"):
    """Minimal catalog proposal used by Stage 12 negative cases."""
    if path is None:
        path = ["r1", "r3", "r4"] if action == "REROUTE" else ["r1", "r2", "r4"]
    return {
        "action": action,
        "target": "HostA_to_HostB",
        "reason": "stage12",
        "proposed_path": path,
        "causes": [{"cause_id": cause_id, "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": expected}]}],
    }


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


def _gate_stand_in(decision, snapshot, proposal, evidence=None, mode="full", now=None, ttl=None, private_key=None, path_qdiscs=None):
    """Self-checks and live smokes ONLY: play gate.decide's issuing step for a chosen verdict.

    Production code gets authorizations from gate.decide. This exists so the
    boundary can be tested with verdicts the gate would not produce here.
    """
    from gate.authority import entry_copies, issue_into

    sealed_proposal, sealed_snapshot, sealed_evidence = entry_copies(proposal, snapshot, evidence)
    return issue_into(
        dict(decision), sealed_proposal, sealed_snapshot, sealed_evidence, mode,
        now=now, ttl=ttl, private_key=private_key, path_qdiscs=path_qdiscs,
    )


def _reads(*snapshots):
    """read_state stand-in: each call returns the next snapshot, then the last one."""
    queue = list(snapshots)

    def read():
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return read


from bench.outcomes import using_test_thresholds  # offline checks grade with test-only limits


@using_test_thresholds
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
        gate_decision = {"decision": decision} if decision is not None else None
        if decision == "allow":

            gate_decision = _gate_stand_in(gate_decision, approved, proposal, [], path_qdiscs=approved.get("qdiscs"))
        return apply_authorized_action(
            gate_decision,
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
    # C4: command success is not traffic recovery; statuses stay distinct.
    saved_run_all = oam.run_all

    def _recovery_rows(mbit=8.0, loss=0.0, connectivity=True, t1_error=None, t10_error=None, omit=()):
        rows = []
        if "T1_e2e_ping" not in omit:
            rows.append({
                "test_id": "T1_e2e_ping",
                "packet_loss_percent": loss,
                "latency_ms": 10.0,
                "connectivity": connectivity,
                "error": t1_error,
            })
        if "T4_mtu" not in omit:
            rows.append({"test_id": "T4_mtu", "df_ping_ok": True, "error": None})
        if "T8_tcp_vs_ping" not in omit:
            rows.append({"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None})
        if "T10_speed" not in omit:
            rows.append({"test_id": "T10_speed", "mbit_per_sec": mbit, "error": t10_error})
        return rows

    def _fake_recovery(_net, test_ids=None, **_kwargs):
        return {
            "started_at": "2026-01-01T00:00:01+00:00",
            "completed_at": "2026-01-01T00:00:04+00:00",
            "results": _recovery_rows(),
            "cost": {"total_duration_seconds": 3.0},
        }

    oam.run_all = _fake_recovery
    try:
        recovered = measure_traffic_recovery(object(), noise_id="N0")
        assert recovered["traffic_recovered"] is True and recovered["recovery_status"] == RECOVERY_RECOVERED, recovered
        assert recovered["post_action"]["connectivity"] is True
        assert recovered["post_action"]["throughput_mbit"] == 8.0
        assert recovered["post_action"]["packet_loss_percent"] == 0.0
        assert isinstance(recovered.get("duration_seconds"), (int, float))
        # Slow T10 with healthy connectivity → partial, not command-based success.
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {
            "results": _recovery_rows(mbit=1.0),
            "started_at": "2026-01-01T00:00:01+00:00",
            "completed_at": "2026-01-01T00:00:04+00:00",
        }
        partial = measure_traffic_recovery(object(), noise_id="N0")
        assert partial["recovery_status"] == RECOVERY_PARTIAL and partial["traffic_recovered"] is False, partial
        assert "T10_speed" in partial["degraded"]
        # Hard loss / no connectivity → failed recovery.
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {
            "results": _recovery_rows(loss=100.0, connectivity=False, mbit=0.0),
        }
        failed = measure_traffic_recovery(object(), noise_id="N0")
        assert failed["recovery_status"] == RECOVERY_FAILED and failed["traffic_recovered"] is False, failed
        # Infra / timeout on a required probe → unavailable, never recovered.
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {
            "results": _recovery_rows(t1_error="TimeoutError: ping timed out"),
        }
        timed_out = measure_traffic_recovery(object(), noise_id="N0")
        assert timed_out["recovery_status"] == RECOVERY_UNAVAILABLE, timed_out
        assert timed_out["traffic_recovered"] is None
        assert "T1_e2e_ping" in timed_out["missing_evidence"]
        assert "T1_e2e_ping" in timed_out["infrastructure_errors"]
        # Missing required evidence → unavailable.
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {
            "results": _recovery_rows(omit=("T8_tcp_vs_ping",)),
        }
        missing = measure_traffic_recovery(object(), noise_id="N0")
        assert missing["recovery_status"] == RECOVERY_UNAVAILABLE and missing["traffic_recovered"] is None, missing
        assert "T8_tcp_vs_ping" in missing["missing_evidence"]
        # T1 measured with no error but loss None / non-finite → unavailable,
        # never partial/failed/recovered (P2 closure).
        for bad_loss in (None, float("nan"), float("inf")):
            oam.run_all = lambda _net, test_ids=None, _l=bad_loss, **_kwargs: {
                "results": _recovery_rows(loss=_l),
            }
            no_loss = measure_traffic_recovery(object(), noise_id="N0")
            assert no_loss["recovery_status"] == RECOVERY_UNAVAILABLE, (bad_loss, no_loss)
            assert no_loss["traffic_recovered"] is None, (bad_loss, no_loss)
            assert no_loss["missing_evidence"] == ["T1_e2e_ping"], (bad_loss, no_loss)
        # Degraded but finite T1 (40% loss, connectivity) is a measured outcome, not unavailable.
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {
            "results": _recovery_rows(loss=40.0),
        }
        lossy = measure_traffic_recovery(object(), noise_id="N0")
        assert lossy["recovery_status"] in (RECOVERY_PARTIAL, RECOVERY_FAILED), lossy
        assert lossy["traffic_recovered"] is False and lossy["missing_evidence"] == [], lossy
        assert "T1_e2e_ping" in lossy["degraded"], lossy
        # T10 is optional: absent or rate None does not block a recovered verdict.
        for t10_rows in (_recovery_rows(omit=("T10_speed",)), _recovery_rows(mbit=None)):
            oam.run_all = lambda _net, test_ids=None, _r=t10_rows, **_kwargs: {"results": _r}
            no_t10 = measure_traffic_recovery(object(), noise_id="N0")
            assert no_t10["recovery_status"] == RECOVERY_RECOVERED, no_t10
            assert no_t10["traffic_recovered"] is True and no_t10["missing_evidence"] == [], no_t10
        # Battery exception → unavailable (not failed recovery).
        def _boom(_net, test_ids=None, **_kwargs):
            raise RuntimeError("iperf server down")

        oam.run_all = _boom
        crashed = measure_traffic_recovery(object(), noise_id="N0")
        assert crashed["recovery_status"] == RECOVERY_UNAVAILABLE and crashed["traffic_recovered"] is None, crashed
        assert crashed.get("error")
        # Integration: attach_post_action_recovery keeps remediation vs traffic distinct.
        oam.run_all = _fake_recovery

        class _Net:
            def get(self, _name):
                return object()

        synthetic = {
            "noise_id": "N0",
            "execution_attempted": True,
            "remediation_succeeded": True,
            "remediation_started_at": "2026-01-01T00:00:00+00:00",
            "oam": {"results": _recovery_rows(mbit=0.5, loss=40.0, connectivity=True)},
            "network_snapshot": _snapshot(),
            "execution_baseline": {"qdiscs": None, "error": None},
        }
        attach_post_action_recovery(
            synthetic,
            _Net(),
            read_state=lambda *_a, **_k: _snapshot(r1_via="10.0.13.2", r4_via="10.0.34.1"),
        )
        assert synthetic["remediation_succeeded"] is True
        assert synthetic["traffic_recovered"] is True
        assert synthetic["recovery_status"] == RECOVERY_RECOVERED
        assert synthetic["route_state_changed"] is True
        assert synthetic["pre_remediation"]["connectivity"] is True
        assert synthetic["pre_remediation"]["throughput_mbit"] == 0.5
        assert isinstance(synthetic.get("recovery_time_seconds"), (int, float))
        assert synthetic["post_oam"]["post_action"]["throughput_mbit"] == 8.0
        # Failed remediation still preserves an independent post measurement.
        broken = {
            "noise_id": "N0",
            "execution_attempted": True,
            "remediation_succeeded": False,
            "execution_error": "partial remediation",
            "remediation_started_at": "2026-01-01T00:00:00+00:00",
            "oam": {"results": _recovery_rows()},
            "network_snapshot": _snapshot(),
            "execution_baseline": {"qdiscs": None},
        }
        oam.run_all = lambda _net, test_ids=None, **_kwargs: {"results": _recovery_rows(loss=100.0, connectivity=False)}
        attach_post_action_recovery(broken, _Net(), read_state=lambda *_a, **_k: _snapshot())
        assert broken["remediation_succeeded"] is False
        assert broken["recovery_status"] == RECOVERY_FAILED
        assert broken["traffic_recovered"] is False
        # No execution attempt → not_attempted; never invent recovery.
        idle = {"execution_attempted": False, "oam": {"results": _recovery_rows()}}
        attach_post_action_recovery(idle, _Net(), read_state=lambda *_a, **_k: _snapshot())
        assert idle["recovery_status"] == RECOVERY_NOT_ATTEMPTED
        assert idle["traffic_recovered"] is None
        # Infra failure during attach must not become traffic_recovered=False.
        oam.run_all = _boom
        unknown = {
            "noise_id": "N0",
            "execution_attempted": True,
            "remediation_succeeded": True,
            "network_snapshot": _snapshot(),
            "execution_baseline": {"qdiscs": None},
        }
        attach_post_action_recovery(unknown, _Net(), read_state=lambda *_a, **_k: _snapshot())
        assert unknown["traffic_recovered"] is None
        assert unknown["recovery_status"] == RECOVERY_UNAVAILABLE
    finally:
        oam.run_all = saved_run_all

    # ---- Gated partial-reroute compensation (control/gated.run_lifecycle) ----
    from control import gated as _gated_mod

    _PRIMARY = {"r1": "10.0.4.0/24 via 10.0.12.2 dev r1-eth1", "r4": "10.0.1.0/24 via 10.0.24.1 dev r4-eth0"}
    _BACKUP = {"r1": "10.0.4.0/24 via 10.0.13.2 dev r1-eth2", "r4": "10.0.1.0/24 via 10.0.34.1 dev r4-eth1"}

    class _GatedWorld:
        """Fake r1/r4: route reads, a per-router action plan and restore plan."""

        def __init__(self, plan=(), restore_plan=(), read_fail=(), stale=(), texts=None):
            self.routes = dict(_PRIMARY, **(texts or {}))
            self.plan, self.restore_plan = list(plan), list(restore_plan)
            self.read_fail, self.stale = set(read_fail), set(stale)
            self.reads, self.previous, self.events = 0, None, []

        def snapshot(self):
            self.reads += 1
            if self.reads in self.read_fail:
                raise OSError("route read failed")
            shown = self.previous if (self.reads in self.stale and self.previous) else dict(self.routes)
            self.previous = dict(self.routes)
            snap = _snapshot()
            for item in snap["routes"]:
                item["route"] = shown[item["node"]]
            return snap

        def execute(self, item, capability=None):
            self.events.append("execute")
            assert capability is not None and capability.decision_id
            for router in ("r1", "r4"):
                step = self.plan.pop(0) if self.plan else "ok"
                if step == "raise":
                    raise RuntimeError(f"{router}: RTNETLINK answers: Network is unreachable")
                if step == "ok":
                    self.routes[router] = _BACKUP[router]
            if self.plan and self.plan[0] == "raise_after":
                raise RuntimeError("controller lost after both commands")
            return {"executed": True, "action": "REROUTE", "path": "backup"}

        def restore(self, router, spec):
            self.events.append(("restore", router))
            step = self.restore_plan.pop(0) if self.restore_plan else "ok"
            if step == "raise":
                raise RuntimeError(f"{router}: restore exited 2")
            if step == "wrong":
                return
            if step == "strip":
                spec = " ".join(spec.split()[:5])  # same next hop, extra text lost
            self.routes[router] = spec

    def _gated(world, measured=None, control=True, start=None):
        token = _gate_stand_in({"decision": "allow"}, healthy, proposal, [])
        return apply_authorized_action(
            token,
            proposal,
            world.execute,
            healthy if start is None else start,
            world.snapshot,
            gate_snapshot=healthy,
            route_control=world if control else None,
            measure_partial=(lambda view: (world.events.append("measure"), measured.append(view))) if measured is not None else None,
        )

    def _contaminated(world, **kw):
        try:
            _gated(world, **kw)
        except TopologyContaminated as exc:
            return exc.record
        raise AssertionError("unverified compensation did not stop the run")

    # Fully applied: no compensation.
    ok_world = _GatedWorld()
    ok_run = _gated(ok_world, measured=[])
    assert ok_run["remediation_succeeded"] is True and ok_run["compensation"] is None, ok_run
    assert ok_run["partial_application"] is False and ok_run["routes_after"] == _BACKUP, ok_run
    assert ok_run["routes_before"] == _PRIMARY and "measure" not in ok_world.events
    # First-router failure: nothing changed; error preserved; no compensation.
    first_world = _GatedWorld(plan=["raise"])
    first_run = _gated(first_world, measured=[])
    assert first_run["execution_error"].startswith("RuntimeError: r1: RTNETLINK"), first_run
    assert first_run["partial_application"] is False and first_run["compensation"] is None
    assert first_run["remediation_succeeded"] is False and first_world.routes == _PRIMARY
    assert first_world.reads == 3, "after-read confirmed by a second read"
    # Second-router failure (executor raises after r1 changed): measured, then r1 restored.
    second_measured = []
    second_world = _GatedWorld(plan=["ok", "raise"])
    second_run = _gated(second_world, measured=second_measured)
    assert second_run["execution_error"].startswith("RuntimeError: r4: RTNETLINK"), "original error preserved"
    assert second_run["partial_application"] is True and second_run["routes_after"]["r1"] == _BACKUP["r1"]
    comp = second_run["compensation"]
    assert comp["verified"] is True and comp["routers"] == ["r1"] and comp["routes_restored"] == _PRIMARY, comp
    token_id = comp["decision_id"]
    assert token_id and second_measured and second_measured[0]["decision_id"] == token_id
    assert second_world.events == ["execute", "measure", ("restore", "r1")], second_world.events
    assert second_world.routes == _PRIMARY and second_run["remediation_succeeded"] is False
    # Silent r4 failure: executed=True, r4 unchanged: partial, r1 restored.
    silent_run = _gated(_GatedWorld(plan=["ok", "silent"]))
    assert silent_run["execution_error"] == "partial remediation" and silent_run["partial_application"] is True
    assert silent_run["compensation"]["routers"] == ["r1"] and silent_run["compensation"]["verified"] is True
    # Raise after both changed: both restored.
    both_world = _GatedWorld(plan=["ok", "ok", "raise_after"])
    both_run = _gated(both_world)
    assert both_run["compensation"]["routers"] == ["r1", "r4"] and both_world.routes == _PRIMARY, both_run
    # Post-action read fails once: retried. Fails twice: state unknown, all routers restored and verified.
    retry_run = _gated(_GatedWorld(read_fail={2}))
    assert retry_run["remediation_succeeded"] is True and retry_run["compensation"] is None, retry_run
    blind_world = _GatedWorld(read_fail={2, 3})
    blind_run = _gated(blind_world)
    assert blind_run["routes_after"] is None and blind_run["partial_application"] is None, blind_run
    assert blind_run["compensation"]["routers"] == ["r1", "r4"] and blind_run["compensation"]["verified"] is True
    assert blind_run["execution_error"].startswith("partial remediation: OSError") and blind_world.routes == _PRIMARY
    # Stale after-read: the confirming read sees the change.
    stale_ok = _gated(_GatedWorld(stale={2}))
    assert stale_ok["remediation_succeeded"] is True and stale_ok["compensation"] is None, stale_ok
    stale_partial_world = _GatedWorld(plan=["ok", "silent"], stale={2})
    stale_partial = _gated(stale_partial_world)
    assert stale_partial["compensation"]["routers"] == ["r1"] and stale_partial_world.routes == _PRIMARY
    # Exact routes: a captured metric survives the restore; a restore that drops it is not verified.
    metric = {"r1": _PRIMARY["r1"] + " proto static metric 100"}
    metric_start = _snapshot()
    metric_start["routes"][0]["route"] = metric["r1"]
    exact_world = _GatedWorld(plan=["ok", "raise"], texts=metric)
    exact_run = _gated(exact_world, start=metric_start)
    assert exact_run["routes_before"]["r1"] == metric["r1"] and exact_world.routes["r1"] == metric["r1"]
    assert exact_run["compensation"]["verified"] is True
    stripped = _contaminated(_GatedWorld(plan=["ok", "raise"], restore_plan=["strip"], texts=metric), start=metric_start)
    assert stripped["compensation"]["verified"] is False and stripped["partial_application"] is True, stripped
    # Incorrect restoration, failed restoration, failed verification read, no route control.
    wrong = _contaminated(_GatedWorld(plan=["ok", "raise"], restore_plan=["wrong"]))
    assert wrong["compensation"]["routes_restored"]["r1"] == _BACKUP["r1"] and wrong["remediation_succeeded"] is False
    raised = _contaminated(_GatedWorld(plan=["ok", "raise"], restore_plan=["raise"]))
    assert raised["compensation"]["errors"] and raised["execution_error"].startswith("RuntimeError: r4"), raised
    unread_verify = _contaminated(_GatedWorld(plan=["ok", "raise"], read_fail={4, 5}))
    assert unread_verify["compensation"]["verified"] is False
    no_control = _contaminated(_GatedWorld(plan=["ok", "raise"]), control=False)
    assert "no route control" in no_control["compensation"]["errors"][0], no_control
    # Exactly one restore pass: a failed restore is not retried.
    once_world = _GatedWorld(plan=["ok", "raise"], restore_plan=["raise", "ok"])
    _contaminated(once_world)
    assert once_world.events.count(("restore", "r1")) == 1, once_world.events
    # SIGALRM: blocked from execute to verified compensation, delivered after.
    class _FakeSignal:
        SIGALRM, SIG_BLOCK, SIG_SETMASK = 14, 0, 2

        def __init__(self, pending=False):
            self.log, self.pending = [], pending

        def pthread_sigmask(self, how, signals):
            self.log.append(("mask", how))
            return set()

        def sigpending(self):
            return {self.SIGALRM} if self.pending else set()

        def sigtimedwait(self, signals, timeout):
            self.log.append("discard")
            self.pending = False

    fake_sig = _FakeSignal()
    alarm_world = _GatedWorld(plan=["ok", "raise"])
    alarm_world_execute = alarm_world.execute

    def _alarm_execute(item, capability=None):
        fake_sig.log.append("execute")
        return alarm_world_execute(item, capability=capability)

    alarm_world.execute = _alarm_execute
    alarm_restore = alarm_world.restore

    def _alarm_restore(router, spec):
        fake_sig.log.append("restore")
        return alarm_restore(router, spec)

    alarm_world.restore = _alarm_restore
    _cap = _gated_mod.mint(_gate_stand_in({"decision": "allow"}, healthy, proposal, [])["authorization"], proposal)
    _gated_mod.run_lifecycle(
        _cap,
        dict(_PRIMARY),
        execute=lambda cap: alarm_world.execute(proposal, capability=cap),
        read_routes=lambda: __import__("control.routes", fromlist=["x"]).specs_from_snapshot(alarm_world.snapshot()),
        restore_route=alarm_world.restore,
        mask=lambda: _gated_mod.deferred_alarm(fake_sig),
    )
    assert fake_sig.log == [("mask", 0), "execute", "restore", ("mask", 2)], fake_sig.log
    pending_sig = _FakeSignal(pending=True)
    try:
        _gated_mod.run_lifecycle(
            _cap,
            dict(_PRIMARY),
            execute=lambda cap: {"executed": True},
            read_routes=lambda: dict(_PRIMARY, r1=_BACKUP["r1"]),
            restore_route=lambda router, spec: None,
            mask=lambda: _gated_mod.deferred_alarm(pending_sig),
        )
    except _gated_mod.CompensationFailed:
        pass
    else:
        raise AssertionError("unrestorable state was not reported")
    assert pending_sig.log == [("mask", 0), "discard", ("mask", 2)], "a pending timeout cannot replace the stop"
    # Handler-level deferral: wherever the kernel delivers SIGALRM (e.g. to a
    # flap thread), the Python handler runs in the main thread and defers.
    from bench.flap import block_alarm_in_this_thread

    class _RaisingSignal(_FakeSignal):
        def __init__(self):
            super().__init__()
            self.handler = None

        def raise_signal(self, signum):
            self.log.append("redeliver")
            self.handler(signum, None)

    def _batch_like_handler(_signum, _frame):
        if _gated_mod.alarm_should_defer():
            return
        raise TimeoutError("episode timed out")

    for compensated in (True, False):
        raising = _RaisingSignal()
        raising.handler = _batch_like_handler
        hl_world = _GatedWorld(plan=["ok", "raise"], restore_plan=[] if compensated else ["wrong"])
        hl_cap = _gated_mod.mint(_gate_stand_in({"decision": "allow"}, healthy, proposal, [])["authorization"], proposal)

        def _execute_then_alarm(cap, _world=hl_world):
            try:
                return _world.execute(proposal, capability=cap)
            finally:
                _batch_like_handler(14, None)  # the alarm arrives mid-action

        try:
            _gated_mod.run_lifecycle(
                hl_cap,
                dict(_PRIMARY),
                execute=_execute_then_alarm,
                read_routes=lambda _world=hl_world: __import__("control.routes", fromlist=["x"]).specs_from_snapshot(_world.snapshot()),
                restore_route=hl_world.restore,
                mask=lambda _sig=raising: _gated_mod.deferred_alarm(_sig),
            )
        except TimeoutError as exc:
            assert compensated, "timeout replaced the contamination stop"
            assert hl_world.routes == _PRIMARY and exc.gated_outcome["compensation"]["verified"] is True
            assert raising.log[-1] == "redeliver"
        except _gated_mod.CompensationFailed:
            assert not compensated and "redeliver" not in raising.log, raising.log
        else:
            raise AssertionError("deferred alarm was lost")
    assert _gated_mod.alarm_should_defer() is False, "deferral state leaked out of the section"
    flap_sig = _FakeSignal()
    assert block_alarm_in_this_thread(flap_sig) is True and flap_sig.log == [("mask", 0)]

    if hasattr(signal, "SIGALRM") and hasattr(signal, "pthread_sigmask"):
        # Real signal (Linux/VM only): the alarm fires inside the section and is delivered after it.
        def _late(*_args):
            raise TimeoutError("episode timed out")

        _prev_handler = signal.signal(signal.SIGALRM, _late)
        real_world = _GatedWorld(plan=["ok", "raise"])
        real_execute = real_world.execute

        def _slow_execute(item, capability=None):
            signal.setitimer(signal.ITIMER_REAL, 0.05)
            time.sleep(0.2)
            return real_execute(item, capability=capability)

        real_world.execute = _slow_execute
        import threading as _threading

        # A second, unmasked thread (as a flap thread would be) while the alarm fires.
        _bystander = _threading.Thread(target=time.sleep, args=(0.5,), daemon=True)
        _bystander.start()
        try:
            _gated(real_world)
        except TimeoutError:
            assert real_world.routes == _PRIMARY and ("restore", "r1") in real_world.events
        else:
            raise AssertionError("masked alarm was never delivered")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, _prev_handler)
        print("gated SIGALRM (real signal): PASS")
    else:
        print("gated SIGALRM (real signal): NOT RUN (no SIGALRM on this platform; fake-signal checks passed)")

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

    failed = apply_authorized_action(
        _gate_stand_in({"decision": "allow"}, healthy, proposal, []),
        proposal,
        boom,
        healthy,
        lambda: healthy,
    )
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
        _gate_stand_in({"decision": "allow"}, primary_down, primary, []),
        primary,
        execute,
        primary_down,
        lambda: primary_down,
    )
    assert calls == []
    assert blocked_primary["execution_error"] == "primary path is down"

    calls.clear()
    stayed = apply_authorized_action(
        _gate_stand_in({"decision": "allow"}, healthy, backup, []),
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

    gate_dir = Path(__file__).resolve().parent.parent / "gate"
    for path in gate_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "control.actions":
                for alias in node.names:
                    assert alias.name != "execute", path.name

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
        out = run_method(method, shared, healthy, [], noise_id="N0")
        assert out["decision"] in {"allow", "block", "escalate"}, out
        assert "action" in out and out["method"] == method
        assert "tests_run" in out and "budget" in out, out
        assert out["permitted_actions"] == ["REROUTE", "DO_NOTHING", "ESCALATE"]
    baselines_self_check()
    try:
        run_method("B0", shared, healthy, [], fault_id="F1", noise_id="N0")
        raise AssertionError("fault_id was accepted")
    except TypeError:
        pass

    from control.actions import execute_gated as controller_execute
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
        return {"errors": [], "residual": {}, "healthy": True, "loss_percent": 0.0}

    def quiet_inject(net, fault_id, seed=None):
        events.append("inject")
        return "corr-1"

    def quiet_telemetry(net, links=None):
        events.append("telemetry")
        return full

    def quiet_tests(net):
        events.append("tests")
        return {
            "results": [
                {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "connectivity": True, "error": None},
                {"test_id": "T4_mtu", "df_ping_ok": True, "error": None},
                {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None},
                {"test_id": "T10_speed", "mbit_per_sec": 50.0, "error": None},
                {"test_id": "T7_route_check", "active": "primary", "error": None},
            ]
        }

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
            "causes": [{"cause_id": "H0", "predicted_tests": []}],
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
            path_qdiscs=record["execution_baseline"]["qdiscs"],
        )

    def local_execute(proposal):
        events.append("execute")
        return {"executed": False, "action": proposal.get("action")}

    connected = run_episode(
        object(),
        {},
        noise_id="N0", fault_id="F1",
        run_oam=True,
        agent_fn=local_agent,
        gate_fn=local_gate,
        execute_fn=local_execute,
        _components=fakes,
    )
    assert connected["gate_decision"]["decision"] == "allow", connected
    assert connected["gate_decision"].get("snapshot_fingerprint")
    assert connected["gate_decision"].get("authorization", {}).get("decision_id")
    # Issued by gate.decide itself (no orchestrator binding), sealed and signed.
    _conn_auth = connected["gate_decision"]["authorization"]
    assert _conn_auth["issuer"] == "verifix-gate" and len(_conn_auth["signature"]) == 128, _conn_auth
    assert _conn_auth["decision"] == "allow" and _conn_auth["gate_mode"] == "pilot_v0"
    assert isinstance(_conn_auth.get("issue_seconds"), float)
    assert connected["gate_cost"]["authorization_seconds"] == _conn_auth["issue_seconds"]
    assert connected["gate_cost"]["authorization_seconds"] <= connected["gate_cost"]["latency_seconds"]
    assert isinstance(connected.get("authorization_verify_seconds"), float)
    assert connected["execution_attempted"] is True and connected["state_verified"] is True
    # The saved decision is bound to the saved OAM rows, under version 2.
    from gate.integrity import evidence_digest_v2 as _digest_v2

    connected_rows = connected["oam"]["results"]
    assert connected_rows, "the connected episode must carry OAM rows"
    assert connected["gate_decision"]["authorization"]["evidence_digest_version"] == 2
    assert connected["gate_decision"]["authorization"]["evidence_digest"] == _digest_v2(connected_rows)
    assert connected["gate_decision"]["authorization"]["evidence_digest"] != _digest_v2([])

    # Rows changed during the gate call do not authorize: the decision is
    # bound to the evidence as it was handed to the gate.
    def _tamper_value(record):
        record["oam"]["results"][0]["tampered_measurement"] = 0.0

    def _tamper_drop(record):
        record["oam"]["results"].pop()

    def _tamper_replace(record):
        record["oam"]["results"] = list(reversed(record["oam"]["results"]))

    def _tamper_clear(record):
        record["oam"] = {"results": []}

    events_before_tamper = list(events)
    for tamper in (_tamper_value, _tamper_drop, _tamper_replace, _tamper_clear):
        def tampering_gate(record, _tamper=tamper):
            out = local_gate(record)
            _tamper(record)
            return out

        executed_before = events.count("execute")
        tampered = run_episode(
            object(),
            {},
            noise_id="N0", fault_id="F1",
            run_oam=True,
            agent_fn=local_agent,
            gate_fn=tampering_gate,
            execute_fn=local_execute,
            _components=fakes,
        )
        assert tampered["gate_decision"]["decision"] == "allow", tamper.__name__
        assert tampered["execution_attempted"] is False, (tamper.__name__, tampered["execution_error"])
        assert tampered["execution_error"] == "evidence changed since the gate decision", (tamper.__name__, tampered["execution_error"])
        assert events.count("execute") == executed_before, tamper.__name__
    # Leave the event log as the connected episode wrote it.
    events[:] = events_before_tamper
    assert connected.get("prompt_version") and connected.get("prompt_sha256")
    assert validate_record("proposal", connected["agent_proposal"])[0]
    assert validate_record("gate_decision", connected["gate_decision"])[0]
    assert validate_record("execution_result", _execution_view(connected))[0]
    walked = [item for item in events if item in ("inject", "telemetry", "tests", "agent", "gate", "execute")]
    assert walked == ["inject", "telemetry", "tests", "agent", "gate", "execute"], walked
    assert events[:2] == ["noise", "reset"], events
    assert events[-2:] == ["clear", "reset+baseline"], events

    # H4: failed reset/cleanup fails closed and preserves the partial record.
    assert teardown_problem({"errors": [{"error": "stuck"}], "residual": {}}, False) is not None
    assert teardown_problem({"errors": [], "residual": {}, "healthy": True}, True) is None
    dirty_calls = {"n": 0}

    def dirty_reset(net, baseline=False):
        dirty_calls["n"] += 1
        if baseline:
            return {"errors": [{"error": "link stuck"}], "residual": {"c1": ["qdisc"]}, "healthy": False, "loss_percent": 100.0}
        return {"errors": [], "residual": {}, "healthy": True, "loss_percent": 0.0}

    try:
        run_episode(
            object(),
            {},
            noise_id="N0", fault_id="H0",
            run_oam=False,
            agent_fn=local_agent,
            gate_fn=local_gate,
            execute_fn=local_execute,
            _components=dict(fakes, reset_all=dirty_reset),
        )
        raise AssertionError("contaminated teardown must raise")
    except TopologyContaminated as exc:
        assert exc.record is not None, "partial episode record must be preserved"
        assert exc.record.get("episode_id")
        assert "episode teardown" in str(exc)
        assert dirty_calls["n"] >= 2

    def dirty_pre(net, baseline=False):
        # Teardown after the failed pre-reset must still be allowed to finish
        # cleanly so the original contamination error propagates.
        if baseline:
            return {"errors": [], "residual": {}, "healthy": True, "loss_percent": 0.0}
        return {"errors": [{"error": "pre residual"}], "residual": {"x": 1}}

    try:
        run_episode(
            object(),
            {},
            noise_id="N0", fault_id="H0",
            run_oam=False,
            agent_fn=local_agent,
            gate_fn=local_gate,
            execute_fn=local_execute,
            _components=dict(fakes, reset_all=dirty_pre),
        )
        raise AssertionError("contaminated pre-reset must raise")
    except TopologyContaminated as exc:
        assert "pre-episode reset" in str(exc), exc
        assert exc.record is None

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
        noise_id="N0", fault_id="H0",
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
        noise_id="N0", fault_id="H0",
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
        # Stands in for gate.decide returning an allow, with its issued authorization.
        return _gate_stand_in(
            {"decision": "allow", "reason": "test", "proposal_action": "REROUTE", "tests_run": [], "checks": {}},
            record["network_snapshot"],
            record["agent_proposal"],
            (record.get("oam") or {}).get("results") or [],
            path_qdiscs=record["execution_baseline"]["qdiscs"],
        )

    def stateful(*snapshots):
        read = _reads(*snapshots)
        return lambda net, links=None: read()

    with_q = dict(full, qdiscs=shaped)
    injected_between = run_episode(
        object(), {}, noise_id="N0", fault_id="H0", run_oam=False,
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
        object(), {}, noise_id="N0", fault_id="H0", run_oam=False,
        agent_fn=reroute_agent, gate_fn=allow_gate,
        execute_fn=lambda proposal: skipped.append("execute"),
        _components=dict(fakes, read_state=flaky_read),
    )
    assert skipped == [] and no_baseline["execution_attempted"] is False
    assert no_baseline["execution_baseline"]["error"] == "RuntimeError: tc missing"
    assert no_baseline["execution_error"] == "gate snapshot has no path qdisc fingerprint"

    rerouted = run_episode(
        object(), {}, noise_id="N0", fault_id="H0", run_oam=False,
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
        limits = {level: outcomes.load_test_thresholds(level)["T1_e2e_ping"]["max_latency_ms"] for level in ("N0", "N2")}
        assert limits["N0"] < 180.0 < limits["N2"], limits
        at_n0 = gate_decide(None, predicted_ok, healthy, slow_ping, mode="pilot_v0", run_missing_tests=False, noise_id="N0")
        at_n2 = gate_decide(None, predicted_ok, healthy, slow_ping, mode="pilot_v0", run_missing_tests=False, noise_id="N2")
    finally:
        outcomes._REPORT = saved_report
    assert at_n0["decision"] == "block" and at_n0["reason"] == "consistency failed", at_n0
    assert at_n2["decision"] == "allow", at_n2

    from gate.inputs import (
        load_frozen_context,
        pipeline_interfaces,
        validate_proposal_inputs,
        validate_snapshot,
    )

    frozen_ok, frozen_errors, frozen = load_frozen_context("N0")
    assert frozen_ok, frozen_errors
    assert frozen["catalog"] and frozen["matrix"] and frozen["thresholds"] and frozen["costs"]
    assert "U1" in frozen["held_out"] and all(row.get("fault_id") not in frozen["held_out"] for row in frozen["matrix"])
    stages = pipeline_interfaces()
    assert stages["safety_checking"] is check_safety
    assert stages["candidate_generation"] is not None and stages["test_selection"] is not None
    unknown_noise = gate_decide(None, predicted_ok, healthy, slow_ping, mode="pilot_v0", run_missing_tests=False, noise_id="N99")
    assert unknown_noise["decision"] == "escalate" and unknown_noise["reason"].startswith("frozen inputs:"), unknown_noise
    no_hypotheses = gate_decide(None, backup, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert no_hypotheses["decision"] == "block" and no_hypotheses["reason"] == "incomplete hypotheses", no_hypotheses
    assert "AI_PRIMARY" not in str(no_hypotheses)
    empty_snap = gate_decide(None, predicted_ok, {}, slow_ping, mode="pilot_v0", run_missing_tests=False, noise_id="N0")
    assert empty_snap["decision"] == "escalate" and empty_snap["reason"].startswith("incomplete snapshot:"), empty_snap
    missing_up = _snapshot()
    del missing_up["links"][0]["up"]
    assert validate_snapshot(missing_up)
    empty_route = _snapshot()
    empty_route["routes"][0]["route"] = ""
    assert not validate_snapshot(empty_route), empty_route
    del empty_route["routes"][0]["route"]
    assert validate_snapshot(empty_route)
    bogus_pred = dict(predicted_ok, causes=[{"cause_id": "F3", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": ""}]}])
    assert validate_proposal_inputs(bogus_pred, costs=frozen["costs"])
    bad_mode = gate_decide(None, predicted_ok, healthy, slow_ping, mode="maybe", run_missing_tests=False, noise_id="N0")
    assert bad_mode["decision"] == "escalate" and bad_mode["reason"] == "invalid mode", bad_mode
    none_proposal = gate_decide(None, None, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert none_proposal["decision"] == "block", none_proposal

    cfg = frozen["gate"]
    assert cfg["frozen"] is True and cfg["learned_grader"] is False and cfg["budget_seconds"] == 8.0
    assert cfg.get("require_signatures") is False
    t1_ok = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0}]
    noop_ok = [
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "connectivity": True, "error": None},
        {"test_id": "T4_mtu", "df_ping_ok": True, "error": None},
        {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None},
        {"test_id": "T10_speed", "mbit_per_sec": 50.0, "error": None},
    ]
    vacuous = gate_decide(None, predicted_ok, healthy, t1_ok, mode="full", run_missing_tests=False, noise_id="N0")
    # consistency_scope=discriminating: no covering separator exists, so a
    # risky action escalates (protocol v1 §5 fail-safe).
    assert vacuous["decision"] == "escalate", vacuous
    assert vacuous["reason"] == "insufficient evidence: no discriminating test", vacuous
    assert vacuous.get("snapshot_fingerprint")
    assert vacuous["log"]["learned_grader"] is False
    assert vacuous["log"]["frozen"] is True
    assert "candidates" in vacuous["log"] and "rules" in vacuous["log"]
    f10 = dict(
        backup,
        causes=[{"cause_id": "F10", "predicted_tests": [{"test_id": "T8_tcp_vs_ping", "expected": "ping_ok_tcp_fail"}]}],
    )
    contradicted = gate_decide(None, f10, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert contradicted["decision"] == "block" and contradicted["reason"] == "proposal contradicted", contradicted
    held = dict(backup, causes=[{"cause_id": "U1", "predicted_tests": []}])
    outside = gate_decide(None, held, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert outside["decision"] == "escalate" and outside["reason"] == "outside catalog", outside
    fp = None
    from gate.decide import evidence_fingerprint, result_reusable

    fp = evidence_fingerprint(healthy)
    assert result_reusable({"test_id": "T1_e2e_ping"}, healthy, fp, 120)
    assert not result_reusable({"test_id": "T1_e2e_ping", "state_fingerprint": "other"}, healthy, fp, 120)
    lossy_snap = _snapshot()
    lossy_snap["metrics"] = {"active_path": "primary", "packet_loss_percent": 40.0, "connectivity": True}
    for link in lossy_snap["links"]:
        link["packet_loss_percent"] = 0.0
    tight = gate_decide(
        None,
        predicted_ok,
        lossy_snap,
        t1_ok,
        mode="full",
        run_missing_tests=False,
        budget_seconds=0.1, noise_id="N0")
    assert tight["decision"] == "escalate", tight
    assert tight["reason"] in {"no affordable discriminating test", "inseparable", "no discrimination"} or tight[
        "reason"
    ].startswith("insufficient evidence"), tight
    v0_same = gate_decide(
        None,
        predicted_ok,
        lossy_snap,
        t1_ok,
        mode="pilot_v0",
        run_missing_tests=False,
        budget_seconds=0.1, noise_id="N0")
    assert v0_same["decision"] == "allow", v0_same
    assert tight["decision"] != "allow"

    idle = _cause_proposal("DO_NOTHING", "H0")
    idle_ok = gate_decide(None, idle, healthy, noop_ok, mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_ok["decision"] == "allow" and idle_ok["reason"] == "healthy no-op", idle_ok
    t1_loss = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 30.0, "latency_ms": 40.0}]
    idle_loss = gate_decide(None, idle, lossy_snap, t1_loss, mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_loss["decision"] != "allow", idle_loss
    t8_acl = list(noop_ok)
    t8_acl[2] = {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": False, "error": None}
    idle_acl = gate_decide(None, idle, healthy, t8_acl, mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_acl["decision"] != "allow", idle_acl
    idle_snap_loss = gate_decide(None, idle, lossy_snap, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_snap_loss["decision"] != "allow", idle_snap_loss
    h0_snap = _snapshot()
    h0_snap["metrics"] = {"active_path": "primary", "packet_loss_percent": 0.0, "connectivity": True}
    # Incomplete evidence: snapshot alone must never authorize DO_NOTHING.
    idle_empty = gate_decide(None, idle, h0_snap, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_empty["decision"] == "escalate", idle_empty
    assert idle_empty["reason"].startswith("no-op not justified"), idle_empty

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
    from agent.decision_agent import AgentOutputError

    def _invalid_agent():
        raise AgentOutputError(
            "invalid agent output: ['bad T2']",
            raw='{"action":"REROUTE","causes":[]}',
            parsed={"action": "REROUTE", "causes": []},
            errors=["bad T2"],
        )

    _, _, agent_exc = _call_timed(_invalid_agent)
    assert isinstance(agent_exc, AgentOutputError)
    assert agent_exc.error_class == "invalid_output"
    assert agent_exc.raw.startswith("{") and agent_exc.parsed["action"] == "REROUTE"

    assert connected["agent_cost"]["prompt_tokens"] is None
    assert connected["agent_cost"]["latency_seconds"] >= 0
    assert connected["gate_cost"]["latency_seconds"] >= 0
    assert connected["oam_cost"]["test_count"] == 5
    assert connected["oam_cost"]["total_additional_bytes"] is None
    compared = connected["gate_decision"]["cost"]
    assert compared["full_battery"]["test_count"] == 5
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
    assert stale.get("agent_error_class") == "invalid_output", stale

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

    from gate.authority import authorization_errors

    integrity_allow = {
        "decision": "allow",
        "reason": "integrity",
        "proposal_action": "REROUTE",
        "tests_run": [],
        "checks": {},
    }
    bound_allow = _gate_stand_in(integrity_allow, healthy, proposal, [])
    assert len(bound_allow["authorization"]["signature"]) == 128
    assert bound_allow["authorization"]["decision_id"]
    assert not authorization_errors(bound_allow, healthy, proposal)

    calls.clear()
    first_bound = apply_authorized_action(bound_allow, proposal, execute, healthy, _reads(healthy, on_backup))
    assert calls == [proposal] and first_bound["state_verified"] is True and first_bound["remediation_succeeded"] is True, first_bound

    calls.clear()
    replayed = apply_authorized_action(bound_allow, proposal, execute, healthy, _reads(healthy, on_backup))
    assert calls == [] and replayed["execution_error"] == "decision already used", replayed

    stale_bound = _gate_stand_in(dict(integrity_allow), healthy, proposal, [])
    calls.clear()
    other_state = _snapshot(r1_via="10.0.13.2", r4_via="10.0.34.1")
    mismatched = apply_authorized_action(stale_bound, proposal, execute, other_state, lambda: other_state)
    assert calls == [] and mismatched["execution_error"] == "decision not bound to this snapshot", mismatched

    calls.clear()
    changed_proposal = apply_authorized_action(
        stale_bound,
        dict(proposal, action="DO_NOTHING"),
        execute,
        healthy,
        lambda: healthy,
    )
    # A changed action is reported as such; the proposal digest also differs.
    assert calls == [] and changed_proposal["execution_error"] == "action differs from the authorized action", changed_proposal
    assert "proposal changed since the gate decision" in authorization_errors(
        stale_bound, healthy, dict(proposal, action="DO_NOTHING"), []
    )

    calls.clear()
    still_valid = apply_authorized_action(stale_bound, proposal, execute, healthy, _reads(healthy, on_backup))
    assert calls == [proposal] and still_valid["state_verified"] is True and still_valid["remediation_succeeded"] is True, still_valid

    calls.clear()
    malformed = apply_authorized_action(
        _gate_stand_in(dict(integrity_allow), healthy, proposal, []),
        {"action": "reboot", "target": "HostA_to_HostB", "reason": "no"},
        execute,
        healthy,
        lambda: healthy,
    )
    assert calls == [] and malformed["execution_error"] == "invalid proposal", malformed

    # ---- S3 regression: gate-issued, Ed25519-sealed authorizations ----
    # Stage 1 is an in-process INTEGRITY control: the gate key lives in this
    # process. These cases pin what it does stop; the process boundary is
    # control/executor_service (checked by `python -m control`).
    import copy as _copy
    import os as _os
    import tempfile as _tempfile

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _Ed

    from control import actions as _actions
    from control import gated as _gated
    from control.ledger import FileLedger as _FileLedger
    from gate import authority as _authority
    from gate.checks import reroute_choice as _choice

    def _issued(verdict="allow", rows=None, **kw):
        return _gate_stand_in(dict(integrity_allow, decision=verdict), healthy, proposal, rows or [], **kw)

    def _try(decision, prop=None, ev=None, **kw):
        calls.clear()
        return apply_authorized_action(
            decision, prop or proposal, execute, healthy, _reads(healthy, on_backup), evidence=ev, **kw
        )

    # A1: a hand-written allow, never seen by the gate.
    a1 = _try(dict(integrity_allow))
    assert calls == [] and a1["execution_error"] == "decision has no gate-issued authorization", a1
    # A1b: a complete authorization signed with a key that is not the gate's.
    foreign = _gate_stand_in(dict(integrity_allow), healthy, proposal, [], private_key=_Ed.generate())
    a1b = _try(foreign)
    assert calls == [] and a1b["execution_error"] == "authorization was signed by an unknown key", a1b
    # A2: the gate said BLOCK; the decision dict is flipped to allow.
    blocked = _issued("block")
    flipped = _copy.deepcopy(blocked)
    flipped["decision"] = "allow"
    a2 = _try(flipped)
    assert calls == [] and a2["execution_error"] == "verdict differs from the gate-issued authorization", a2
    # A3: both verdicts flipped: the signature covers the sealed verdict.
    both = _copy.deepcopy(blocked)
    both["decision"] = "allow"
    both["authorization"]["decision"] = "allow"
    a3 = _try(both)
    assert calls == [] and a3["execution_error"] == "authorization signature is invalid", a3
    # Every signed field: an edit is refused by the signature.
    genuine = _issued()
    assert not authorization_errors(genuine, healthy, proposal, []), authorization_errors(genuine, healthy, proposal, [])
    for name, value in (
        ("action", "DO_NOTHING"), ("target", "elsewhere"), ("reroute_path", "primary"),
        ("proposal_digest", "0" * 64), ("snapshot_fingerprint", "0" * 64), ("evidence_digest", "0" * 64),
        ("gate_mode", "pilot_v0"), ("decision_id", "f" * 32),
        ("issued_at", genuine["authorization"]["issued_at"] - 1),
        ("expires_at", genuine["authorization"]["expires_at"] + 1),
        ("key_id", "0" * 16), ("version", 2), ("issuer", "someone"), ("gate_config_sha256", "0" * 64),
        ("path_qdiscs", {"r1/r1-eth2": "qdisc netem 10: root loss 20%"}),
    ):
        edited = _copy.deepcopy(genuine)
        edited["authorization"][name] = value
        errs = authorization_errors(edited, healthy, proposal, [])
        assert errs and errs[-1] in (
            "authorization signature is invalid",
            "authorization was signed by an unknown key",
        ), (name, errs)
    # The orchestrator's qdisc baseline must be the one the gate sealed: a
    # substituted baseline that matches the (already changed) network is refused.
    _sealed_q = {"r1/r1-eth2": "qdisc netem 10: parent 5:1 limit 1000 delay 5ms"}
    _swapped_q = {"r1/r1-eth2": "qdisc netem 10: parent 5:1 loss 20%"}
    _q_token = _gate_stand_in(dict(integrity_allow), healthy, proposal, [], path_qdiscs=_sealed_q)
    calls.clear()
    _q_run = apply_authorized_action(
        _q_token, proposal, execute, dict(healthy, qdiscs=_swapped_q), lambda: dict(healthy, qdiscs=_swapped_q),
        gate_snapshot=healthy,
    )
    assert calls == [] and _q_run["execution_error"] == "path qdisc baseline differs from the authorized one", _q_run
    # The target is compared with the proposal itself (not only signed).
    assert "target differs from the authorized target" in authorization_errors(
        genuine, healthy, dict(proposal, target="HostB_to_HostA"), []
    )
    # Evidence edited DURING the gate call: the seal covers the rows as they
    # were when decide() was entered, so the edited rows no longer match.
    _decide_mod = sys.modules["gate.decide"]

    _real_inner = _decide_mod._decide

    def _editing_gate(net, prop, snap, test_results=None, **kw):
        out = _real_inner(net, prop, snap, test_results=test_results, **kw)
        if test_results:
            test_results[0]["packet_loss_percent"] = 0.0
        return out

    entry_rows = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 100.0, "error": None}]
    _decide_mod._decide = _editing_gate
    try:
        sealed_at_entry = _decide_mod.decide(None, proposal, healthy, entry_rows, run_missing_tests=False, noise_id="N0")
    finally:
        _decide_mod._decide = _real_inner
    from gate.integrity import evidence_digest_v2 as _v2

    assert entry_rows[0]["packet_loss_percent"] == 0.0
    assert sealed_at_entry["authorization"]["evidence_digest"] == _v2([dict(entry_rows[0], packet_loss_percent=100.0)])
    assert sealed_at_entry["authorization"]["evidence_digest"] != _v2(entry_rows)
    # Invalid signatures, including non-ASCII text of the right length: refused, never raised.
    for bad in ("", "dead", "é" * 128, "zz" * 64, None, 7, "0" * 128, "０" * 128):
        edited = _copy.deepcopy(genuine)
        edited["authorization"]["signature"] = bad
        assert authorization_errors(edited, healthy, proposal, []) == ["authorization signature is invalid"], bad
        assert _try(edited)["execution_error"] == "authorization signature is invalid" and calls == []
    # Validity window: expired, issued in the future, too long.
    stale_auth = _issued(now=time.time() - 3 * _authority.AUTHORIZATION_TTL_SECONDS)
    expired = _try(stale_auth)
    assert calls == [] and expired["execution_error"] == "authorization has expired", expired
    assert "authorization is issued in the future" in authorization_errors(
        _issued(now=time.time() + 600), healthy, proposal, []
    )
    assert "authorization validity is longer than allowed" in authorization_errors(
        _issued(ttl=3600), healthy, proposal, []
    )
    # Gate configuration changed after the decision.
    _before_change = _issued()
    _real_cfg_hash = _authority.gate_config_sha256
    _authority.gate_config_sha256 = lambda: "0" * 64
    try:
        assert _try(_before_change)["execution_error"] == "gate configuration changed since the decision" and calls == []
    finally:
        _authority.gate_config_sha256 = _real_cfg_hash
    # Replay: the same token twice in one process.
    once_token = _issued()
    assert _try(once_token)["remediation_succeeded"] is True and calls == [proposal]
    assert _try(once_token)["execution_error"] == "decision already used" and calls == []
    # Replay across an executor restart: a durable ledger reloaded from disk.
    ledger_dir = Path(_tempfile.mkdtemp())
    ledger_path = ledger_dir / "ledger.jsonl"
    durable = _issued()
    assert _try(durable, ledger=_FileLedger(ledger_path))["remediation_succeeded"] is True
    assert _try(durable, ledger=_FileLedger(ledger_path))["execution_error"] == "decision already used" and calls == []
    # A torn last line (crash mid-write) is tolerated; damage elsewhere refuses to load.
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write('{"decision_id": "torn')
    assert _try(durable, ledger=_FileLedger(ledger_path))["execution_error"] == "decision already used"
    ledger_path.write_text("garbage\n" + ledger_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        _FileLedger(ledger_path)
    except Exception as exc:
        assert type(exc).__name__ == "LedgerError", exc
    else:
        raise AssertionError("damaged ledger was accepted")

    class _BrokenLedger:
        def consume(self, *_args, **_kwargs):
            from control.ledger import LedgerError

            raise LedgerError("disk full")

    broken = _try(_issued(), ledger=_BrokenLedger())
    assert calls == [] and broken["execution_error"].startswith("replay ledger unavailable"), broken
    for item in ledger_dir.iterdir():
        item.unlink()
    ledger_dir.rmdir()
    # A5: evidence swapped after the decision.
    sealed_rows = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 100.0}]
    a5 = _try(_issued(rows=sealed_rows), ev=[{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0}])
    assert calls == [] and a5["execution_error"] == "evidence changed since the gate decision", a5
    # BLOCK and ESCALATE never execute and never compensate, sealed or not.
    for verdict in ("block", "escalate"):
        held = _try(_issued(verdict))
        assert calls == [] and held["execution_attempted"] is False and held["compensation"] is None, held
    # A configured verification key that is not the gate's refuses everything.
    _key_dir = Path(_tempfile.mkdtemp())
    _other_private, _other_public = _authority.generate_keypair_pem()
    (_key_dir / "pub.pem").write_bytes(_other_public)
    _previous_pub = _os.environ.get(_authority.PUBLIC_KEY_ENV)
    _os.environ[_authority.PUBLIC_KEY_ENV] = str(_key_dir / "pub.pem")
    try:
        assert _try(_issued())["execution_error"] == "authorization was signed by an unknown key" and calls == []
        _os.environ[_authority.PUBLIC_KEY_ENV] = str(_key_dir / "missing.pem")
        assert _try(_issued())["execution_error"].startswith("verification key") and calls == []
    finally:
        if _previous_pub is None:
            _os.environ.pop(_authority.PUBLIC_KEY_ENV, None)
        else:
            _os.environ[_authority.PUBLIC_KEY_ENV] = _previous_pub
        for item in _key_dir.iterdir():
            item.unlink()
        _key_dir.rmdir()
    # A6: the gated executor refuses without a minted, unexpired capability.
    for not_a_capability in (None, {"decision_id": "x"}, proposal):
        try:
            _actions.execute_gated(None, not_a_capability)
        except PermissionError:
            pass
        else:
            raise AssertionError("gated executor ran without a capability")
    try:
        _gated.Capability("x", "REROUTE", "backup", None, time.time() + 60, dict(proposal))
    except PermissionError:
        pass
    else:
        raise AssertionError("a capability was built without verification")
    _cap_auth = dict(genuine["authorization"], expires_at=time.time() - 1)
    try:
        _actions.execute_gated(None, _gated.mint(_cap_auth, proposal))
    except PermissionError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("expired capability executed")
    _wrong_path = dict(genuine["authorization"], reroute_path="primary")
    try:
        _actions.execute_gated(None, _gated.mint(_wrong_path, proposal))
    except PermissionError as exc:
        assert "not the authorized path" in str(exc)
    else:
        raise AssertionError("capability for another path executed")
    # The ungated entry point needs an explicit reason.
    try:
        _actions.execute_ungated(None, "DO_NOTHING", {}, reason="")
    except ValueError:
        pass
    else:
        raise AssertionError("ungated execution without a reason")
    # Single flight: a second authorized execution while one runs is refused.
    with _gated.single_flight():
        busy = _try(_issued())
    assert calls == [] and busy["execution_error"] == "another authorized execution is in progress", busy

    # The controller resolves a reroute path exactly as the gate does (ungated path).
    _moved = []
    _real_set = _actions.ctl.set_active_path
    _actions.ctl.set_active_path = lambda _net, path: _moved.append(path)
    try:
        for bad in (
            {"proposed_path": ["r1", "r3", "r2", "r4"]},
            {"proposed_path": "r3"},
            {"path": "backup", "proposed_path": ["r1", "r2", "r4"]},
            {"path": "sideways"},
            {},
        ):
            assert _choice(bad) is None
            try:
                _actions.execute_ungated(None, "REROUTE", bad, reason="self-check")
            except ValueError:
                pass
            else:
                raise AssertionError(f"controller acted on a path the gate rejects: {bad!r}")
        assert _moved == [], _moved
        for good, want in (
            ({"path": "backup"}, "backup"),
            ({"proposed_path": ["r1", "r3", "r4"]}, "backup"),
            ({"proposed_path": ["r1", "r2", "r4"]}, "primary"),
            ({"path": "primary", "proposed_path": ["r1", "r2", "r4"]}, "primary"),
        ):
            assert _actions.execute_ungated(None, "REROUTE", good, reason="self-check")["path"] == want == _choice(good)
        assert _moved == ["backup", "backup", "primary", "primary"], _moved
    finally:
        _actions.ctl.set_active_path = _real_set

    def _tampered(errs):
        """A hand-edited authorization fails its specific check AND the gate signature."""
        assert errs and errs[-1] in ("authorization signature is invalid", "authorization is not canonical"), errs
        return errs[:-1]

    # B2: the decision is bound to the reroute path it resolved. A label is
    # part of the digest, and the bound path is checked on its own.
    labelled = dict(proposal)
    labelled.pop("proposed_path")
    labelled["proposed_path_label"] = "backup"
    path_bound = _gate_stand_in(dict(integrity_allow), healthy, labelled, [])
    assert path_bound["authorization"]["reroute_path"] == "backup", path_bound["authorization"]
    assert not authorization_errors(path_bound, healthy, labelled)
    swapped = dict(labelled, proposed_path_label="primary")
    swap_errors = authorization_errors(path_bound, healthy, swapped)
    assert "proposal changed since the gate decision" in swap_errors, swap_errors
    assert "reroute path changed since the gate decision" in swap_errors, swap_errors
    calls.clear()
    swap_run = apply_authorized_action(path_bound, swapped, execute, healthy, lambda: healthy)
    assert calls == [] and swap_run["execution_attempted"] is False, swap_run
    assert swap_run["execution_error"] == "proposal changed since the gate decision", swap_run
    no_path = _gate_stand_in(dict(integrity_allow), healthy, proposal, [])
    del no_path["authorization"]["reroute_path"]
    assert _tampered(authorization_errors(no_path, healthy, proposal)) == ["reroute path changed since the gate decision"]
    wrong_path = _gate_stand_in(dict(integrity_allow), healthy, proposal, [])
    wrong_path["authorization"]["reroute_path"] = "primary"
    assert _tampered(authorization_errors(wrong_path, healthy, proposal)) == ["reroute path changed since the gate decision"]
    calls.clear()
    wrong_run = apply_authorized_action(wrong_path, proposal, execute, healthy, lambda: healthy)
    assert calls == [] and wrong_run["execution_error"] == "reroute path changed since the gate decision", wrong_run
    quiet = _gate_stand_in(dict(integrity_allow), healthy, dict(proposal, action="DO_NOTHING"), [])
    assert quiet["authorization"]["reroute_path"] is None

    # B3: the decision is bound to every field of every OAM row, in order
    # (evidence digest version 2). Rows are shaped like the live battery:
    # no observed_at and no state_fingerprint, so only measured values differ.
    import copy as _copy

    from gate import integrity as _integrity

    rows_seen = [
        {
            "test_id": "T1_e2e_ping",
            "error": None,
            "packet_loss_percent": 0.0,
            "latency_ms": 42.5,
            "connectivity": True,
            "rtt_ms": [41.9, 43.1],
            "raw": "2 packets transmitted, 2 received",
        },
        {"test_id": "T8_tcp_vs_ping", "error": None, "ping_ok": True, "tcp_ok": True},
    ]
    # Known answers pin both definitions: version 1 stays readable for old
    # records, and version 2 cannot drift without this failing.
    assert _integrity.evidence_digest_v1(rows_seen) == "ebb7adbe64727729abfeec1b2c0978a7be486557d8cf181269cc0edb35c17504"
    assert _integrity.evidence_digest_v2(rows_seen) == "4551ad88f350bf8a300e3f2dae07f49aadc07d5ca69f3fc2beca9825b2b595c2"
    assert _integrity.evidence_digest(rows_seen) == _integrity.evidence_digest_v2(rows_seen)
    assert _integrity.evidence_digest(rows_seen, version=1) == _integrity.evidence_digest_v1(rows_seen)
    for unknown_version in (0, 3, True, "2", None):
        try:
            _integrity.evidence_digest(rows_seen, version=unknown_version)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unknown digest version accepted: {unknown_version!r}")

    with_rows = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
    assert with_rows["authorization"]["evidence_digest_version"] == 2
    assert with_rows["authorization"]["evidence_digest"] == _integrity.evidence_digest_v2(rows_seen)
    assert not authorization_errors(with_rows, healthy, proposal, rows_seen)
    # Same content, different key order or a fresh copy: same evidence.
    assert not authorization_errors(with_rows, healthy, proposal, _copy.deepcopy(rows_seen))
    assert not authorization_errors(with_rows, healthy, proposal, [dict(reversed(list(row.items()))) for row in rows_seen])
    assert not authorization_errors(with_rows, healthy, proposal, json.loads(json.dumps(rows_seen)))

    changed = ["evidence changed since the gate decision"]

    def _edit(index, **fields):
        rows = _copy.deepcopy(rows_seen)
        rows[index].update(fields)
        return rows

    def _drop(index, key):
        rows = _copy.deepcopy(rows_seen)
        del rows[index][key]
        return rows

    value_edits = {
        "loss 0 -> 100": _edit(0, packet_loss_percent=100.0),
        "loss 0.0 -> 0.1": _edit(0, packet_loss_percent=0.1),
        "latency changed": _edit(0, latency_ms=420.5),
        "latency null": _edit(0, latency_ms=None),
        "latency key missing": _drop(0, "latency_ms"),
        "latency nan": _edit(0, latency_ms=float("nan")),
        "latency inf": _edit(0, latency_ms=float("inf")),
        "connectivity flipped": _edit(0, connectivity=False),
        "error set": _edit(0, error="TimeoutError: ping timed out"),
        "error empty string": _edit(0, error=""),
        "error key missing": _drop(0, "error"),
        "nested sample changed": _edit(0, rtt_ms=[41.9, 43.2]),
        "nested sample dropped": _edit(0, rtt_ms=[41.9]),
        "nested samples reordered": _edit(0, rtt_ms=[43.1, 41.9]),
        "raw output changed": _edit(0, raw="2 packets transmitted, 0 received"),
        "tcp flipped": _edit(1, tcp_ok=False),
        "bool -> int": _edit(1, tcp_ok=1),
        "float -> int": _edit(0, packet_loss_percent=0),
        "float -> false": _edit(0, packet_loss_percent=False),
        "timestamp added": _edit(0, observed_at="2026-01-01T00:00:00+00:00"),
        "fingerprint added": _edit(0, state_fingerprint="fp"),
        "unit field added": _edit(0, latency_unit="s"),
        "test id renamed": _edit(1, test_id="T4_mtu"),
        "row removed": rows_seen[:1],
        "row added": rows_seen + [{"test_id": "T4_mtu", "df_ping_ok": True}],
        "row duplicated": rows_seen + [_copy.deepcopy(rows_seen[0])],
        "rows reordered": list(reversed(rows_seen)),
        "non-row added": rows_seen + [None],
        "no evidence": [],
        "evidence omitted": None,
    }
    seen_digests = {_integrity.evidence_digest_v2(rows_seen): "original"}
    for label, edited in value_edits.items():
        assert authorization_errors(with_rows, healthy, proposal, edited) == changed, label
        mark = _integrity.evidence_digest_v2(edited)
        # "no evidence" and "evidence omitted" are the one intended equal pair.
        if label != "evidence omitted":
            assert mark not in seen_digests, (label, seen_digests.get(mark))
            seen_digests[mark] = label
    # These edits keep every test id. Version 1 could not see any of them.
    v1_original = _integrity.evidence_digest_v1(rows_seen)
    for label in (
        "loss 0 -> 100",
        "latency null",
        "latency key missing",
        "connectivity flipped",
        "error set",
        "tcp flipped",
        "rows reordered",
    ):
        assert _integrity.evidence_digest_v1(value_edits[label]) == v1_original, label

    # Missing, null, unavailable and failed stay four different things.
    four_states = [
        [{"test_id": "T1_e2e_ping"}],
        [{"test_id": "T1_e2e_ping", "latency_ms": None}],
        [{"test_id": "T1_e2e_ping", "latency_ms": None, "error": "measurement_unavailable"}],
        [{"test_id": "T1_e2e_ping", "latency_ms": None, "error": "TimeoutError: ping timed out"}],
    ]
    assert len({_integrity.evidence_digest_v2(rows) for rows in four_states}) == 4
    # A tagged non-finite float never equals a look-alike mapping, list or string.
    lookalikes = [float("nan"), {"float": "nan"}, ["float", "nan"], "nan", None, float("inf"), float("-inf")]
    assert len({_integrity.evidence_digest_v2([{"test_id": "T", "v": item}]) for item in lookalikes}) == len(lookalikes)
    assert _integrity.evidence_digest_v2([{"test_id": "T", "v": [1]}]) != _integrity.evidence_digest_v2(
        [{"test_id": "T", "v": {"list": [1]}}]
    )

    # Values with no canonical form are refused at bind and at verify.
    for junk in (
        [{"test_id": "T", "v": {1, 2}}],
        [{"test_id": "T", 5: "x"}],
        [{"test_id": "T", "v": b"raw"}],
        "T1_e2e_ping",
        {"test_id": "T"},
    ):
        try:
            _integrity.evidence_digest_v2(junk)
        except _integrity.EvidenceNotCanonical:
            pass
        else:
            raise AssertionError(f"non-canonical evidence accepted: {junk!r}")
        junk_bound = _gate_stand_in(dict(integrity_allow), healthy, proposal, junk)
        assert junk_bound["authorization"]["evidence_digest"] is None
        assert authorization_errors(junk_bound, healthy, proposal, junk) == ["decision is missing evidence digest"]
        assert authorization_errors(with_rows, healthy, proposal, junk)[0].startswith("evidence cannot be canonicalized"), junk

    # Nested values (hops, probes, intervals, interfaces, queues in live
    # rows) are canonicalized at every depth.
    nested_rows = [
        {
            "test_id": "T2_hop_ping",
            "hops": [{"hop": "r1-r2", "ok": True, "loss": 0.0, "rtt": [1.5, 1.6]}, {"hop": "r2-r4", "ok": True, "loss": 0.0}],
            "interfaces": {"r1-eth1": {"rx_errors": 0, "samples": [0, 0]}},
            "raw": "r\u00e9ponse de 10.0.4.2 \u2014 0% perte",
        }
    ]
    # Known answer with nested values and non-ASCII text.
    assert _integrity.evidence_digest_v2(nested_rows) == "b8448df3004c259c5e6660cbf3c22911a3f198fcc96835ff50b204adec2b9888", _integrity.evidence_digest_v2(nested_rows)
    nested_bound = _gate_stand_in(dict(integrity_allow), healthy, proposal, nested_rows)
    assert not authorization_errors(nested_bound, healthy, proposal, json.loads(json.dumps(nested_rows)))

    def _nested(mutate):
        rows = _copy.deepcopy(nested_rows)
        mutate(rows[0])
        return rows

    nested_edits = {
        "hop ok flipped": _nested(lambda row: row["hops"][0].update(ok=False)),
        "hop loss changed": _nested(lambda row: row["hops"][1].update(loss=100.0)),
        "hop loss nan": _nested(lambda row: row["hops"][1].update(loss=float("nan"))),
        "hop loss null": _nested(lambda row: row["hops"][1].update(loss=None)),
        "hop loss missing": _nested(lambda row: row["hops"][1].pop("loss")),
        "hop sample changed": _nested(lambda row: row["hops"][0]["rtt"].__setitem__(1, 9.9)),
        "hop samples reordered": _nested(lambda row: row["hops"][0]["rtt"].reverse()),
        "hops reordered": _nested(lambda row: row["hops"].reverse()),
        "counter changed": _nested(lambda row: row["interfaces"]["r1-eth1"].update(rx_errors=1)),
        "counter sample changed": _nested(lambda row: row["interfaces"]["r1-eth1"]["samples"].__setitem__(0, 7)),
        "list vs tagged lookalike": _nested(lambda row: row["hops"][0].update(rtt=[{"list": [1.5, 1.6]}])),
        "nan vs tagged lookalike": _nested(lambda row: row["hops"][1].update(loss={"float": "nan"})),
        "accent changed": _nested(lambda row: row.update(raw="reponse de 10.0.4.2 \u2014 0% perte")),
    }
    nested_seen = {_integrity.evidence_digest_v2(nested_rows): "original"}
    for label, edited in nested_edits.items():
        assert authorization_errors(nested_bound, healthy, proposal, edited) == changed, label
        mark = _integrity.evidence_digest_v2(edited)
        assert mark not in nested_seen, (label, nested_seen.get(mark))
        nested_seen[mark] = label
    for label, mutate in (
        ("set in nested dict", lambda row: row["interfaces"]["r1-eth1"].update(samples={1, 2})),
        ("bytes in nested list", lambda row: row["hops"].append(b"raw")),
        ("non-string key in nested dict", lambda row: row["hops"][0].update({7: "x"})),
        ("object in nested list", lambda row: row["hops"][0]["rtt"].append(object())),
    ):
        try:
            _integrity.evidence_digest_v2(_nested(mutate))
        except _integrity.EvidenceNotCanonical:
            pass
        else:
            raise AssertionError(f"non-canonical nested evidence accepted: {label}")
    # Cyclic or absurdly deep evidence is refused, not a stack overflow.
    cyclic = {"test_id": "T", "v": []}
    cyclic["v"].append(cyclic)
    deep = current = {"test_id": "T"}
    for _ in range(200):
        current["v"] = {}
        current = current["v"]
    # The limit is exact, and lists count as levels just like mappings.
    def _nest(levels, kind):
        value = 0
        for _ in range(levels):
            value = {"v": value} if kind == "dict" else [value]
        return value

    for kind in ("dict", "list"):
        _integrity.evidence_digest_v2([{"test_id": "T", "v": _nest(_integrity._MAX_EVIDENCE_DEPTH - 1, kind)}])
        try:
            _integrity.evidence_digest_v2([{"test_id": "T", "v": _nest(_integrity._MAX_EVIDENCE_DEPTH, kind)}])
        except _integrity.EvidenceNotCanonical:
            pass
        else:
            raise AssertionError(f"evidence one level past the limit accepted ({kind})")
    assert _integrity._MAX_EVIDENCE_DEPTH == 32
    for runaway in ([cyclic], [deep]):
        try:
            _integrity.evidence_digest_v2(runaway)
        except _integrity.EvidenceNotCanonical:
            pass
        else:
            raise AssertionError("runaway evidence accepted")
        assert _gate_stand_in(dict(integrity_allow), healthy, proposal, runaway)["authorization"]["evidence_digest"] is None

    # Version handling: only version 2 authorizes. A version 1 digest that is
    # correct under version 1 is refused, and is never re-read as version 2.
    legacy = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
    del legacy["authorization"]["evidence_digest_version"]
    legacy["authorization"]["evidence_digest"] = _integrity.evidence_digest_v1(rows_seen)
    legacy_errors = _tampered(authorization_errors(legacy, healthy, proposal, rows_seen))
    assert len(legacy_errors) == 1 and legacy_errors[0].startswith("decision has no evidence digest version"), legacy_errors
    import decimal as _decimal
    import fractions as _fractions

    for bad_version in (1, 3, "2", True, 2.5, 2.0, _decimal.Decimal(2), _fractions.Fraction(2), 2 + 0j, [2]):
        versioned = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
        versioned["authorization"]["evidence_digest_version"] = bad_version
        version_errors = _tampered(authorization_errors(versioned, healthy, proposal, rows_seen))
        assert len(version_errors) == 1 and "is not accepted for execution" in version_errors[0], (bad_version, version_errors)
    # A binding that claims version 2 but carries the version 1 digest of the
    # same rows is refused: there is no fallback to the weaker definition.
    downgraded = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
    downgraded["authorization"]["evidence_digest"] = _integrity.evidence_digest_v1(rows_seen)
    assert _tampered(authorization_errors(downgraded, healthy, proposal, rows_seen)) == changed
    calls.clear()
    downgraded_run = apply_authorized_action(downgraded, proposal, execute, healthy, lambda: healthy, evidence=rows_seen)
    assert calls == [] and downgraded_run["execution_error"] == changed[0], downgraded_run
    unversioned_v2 = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
    del unversioned_v2["authorization"]["evidence_digest_version"]
    assert authorization_errors(unversioned_v2, healthy, proposal, rows_seen), "a version 2 digest without its version must not pass"
    no_mark = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
    no_mark["authorization"]["evidence_digest"] = None
    assert _tampered(authorization_errors(no_mark, healthy, proposal, rows_seen)) == ["decision is missing evidence digest"]
    # A malformed stored digest is a refusal, never an exception at the boundary.
    for odd_mark, expected in (
        ("", "decision is missing evidence digest"),
        ("d\u00e9j\u00e0", changed[0]),
        (b"4551ad88", "decision is missing evidence digest"),
        (4551, "decision is missing evidence digest"),
        (with_rows["authorization"]["evidence_digest"].upper(), changed[0]),
        (with_rows["authorization"]["evidence_digest"][:32], changed[0]),
    ):
        odd = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
        odd["authorization"]["evidence_digest"] = odd_mark
        assert _tampered(authorization_errors(odd, healthy, proposal, rows_seen)) == [expected], (odd_mark, authorization_errors(odd, healthy, proposal, rows_seen))
        calls.clear()
        odd_run = apply_authorized_action(odd, proposal, execute, healthy, lambda: healthy, evidence=rows_seen)
        assert calls == [] and odd_run["execution_error"] == expected, (odd_mark, odd_run)

    # The boundary refuses a changed measurement and executes nothing.
    for label in (
        "loss 0 -> 100",
        "latency key missing",
        "error set",
        "rows reordered",
        "row duplicated",
        "evidence omitted",
    ):
        fresh = _gate_stand_in(dict(integrity_allow), healthy, proposal, rows_seen)
        calls.clear()
        refused_run = apply_authorized_action(
            fresh, proposal, execute, healthy, lambda: healthy, evidence=value_edits[label]
        )
        assert calls == [] and refused_run["execution_attempted"] is False, (label, refused_run)
        assert refused_run["execution_error"] == changed[0], (label, refused_run)
    calls.clear()
    legacy_run = apply_authorized_action(legacy, proposal, execute, healthy, lambda: healthy, evidence=rows_seen)
    assert calls == [] and legacy_run["execution_error"].startswith("decision has no evidence digest version"), legacy_run
    calls.clear()
    evidence_ok = apply_authorized_action(
        with_rows, proposal, execute, healthy, _reads(healthy, on_backup), evidence=_copy.deepcopy(rows_seen)
    )
    assert calls == [proposal] and evidence_ok["remediation_succeeded"] is True, evidence_ok

    # Stored version 1 digests still verify under version 1, unchanged.
    stored_root = Path(__file__).resolve().parent.parent / "results"
    stored_files = sorted(stored_root.glob("pilot/v*_episodes/*.json")) + sorted(stored_root.glob("episodes/*.json"))
    stored_checked = 0
    for stored_file in stored_files:
        stored = json.loads(stored_file.read_text(encoding="utf-8"))
        stored_binding = (stored.get("gate_decision") or {}).get("binding")
        if not isinstance(stored_binding, dict) or "evidence_digest" not in stored_binding:
            continue
        assert "evidence_digest_version" not in stored_binding, stored_file.name
        stored_rows = (stored.get("oam") or {}).get("results") or []
        assert _integrity.evidence_digest_v1(stored_rows) == stored_binding["evidence_digest"], stored_file.name
        assert _integrity.evidence_digest_v2(stored_rows) != stored_binding["evidence_digest"], stored_file.name
        stored_checked += 1
    assert stored_checked or not stored_files, "no stored version 1 digest was checked"

    # B10a: an interrupted write leaves the previous file whole and no temp file.
    import tempfile as _tempfile

    from bench import contracts as _contracts

    atomic_dir = Path(_tempfile.mkdtemp(prefix="verifix-atomic-"))
    target = atomic_dir / "record.json"
    assert _contracts.atomic_write_text(target, "first") == target and target.read_text(encoding="utf-8") == "first"
    _real_replace = _contracts.os.replace

    def _killed(_src, _dst):
        raise OSError("simulated crash before rename")

    _contracts.os.replace = _killed
    try:
        try:
            _contracts.atomic_write_text(target, "second-and-longer")
        except OSError:
            pass
        else:
            raise AssertionError("interrupted write did not raise")
    finally:
        _contracts.os.replace = _real_replace
    assert target.read_text(encoding="utf-8") == "first"
    assert [item.name for item in atomic_dir.iterdir()] == ["record.json"], list(atomic_dir.iterdir())
    _contracts.atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    saved = save_episode({"episode_id": "atomic-check", "x": 1}, atomic_dir)
    assert json.loads(saved.read_text(encoding="utf-8")) == {"episode_id": "atomic-check", "x": 1}
    assert sorted(item.name for item in atomic_dir.iterdir()) == ["atomic-check.json", "record.json"]
    saved.unlink()
    target.unlink()
    atomic_dir.rmdir()

    from gate.checks import _prediction_matches, check_consistency
    from bench.contracts import expected_grade_ok, prediction_matches

    assert _prediction_matches("loss", "loss=15.0")
    assert _prediction_matches("ok", "ok")
    assert not _prediction_matches("ok", "both_ok")
    assert not _prediction_matches("0", "100")
    assert _prediction_matches("r2\u2011r4=down", "primary:hostA-r1=ok,r2-r4=down", test_id="T2_hop_ping")
    assert not _prediction_matches("r2-r4=ok", "primary:hostA-r1=ok,r2-r4=down", test_id="T2_hop_ping")
    assert not _prediction_matches("success", "ok")
    assert not _prediction_matches("timeout / no reply", "down")
    assert expected_grade_ok("T1_e2e_ping", "down")
    assert not expected_grade_ok("T1_e2e_ping", "timeout / no reply")

    t1_down = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 100.0, "latency_ms": None}]
    t2_f1 = {
        "test_id": "T2_hop_ping",
        "hops_primary": [
            {"dst_ip": "10.0.1.1", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.12.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.24.2", "packet_loss_percent": 100.0},
            {"dst_ip": "10.0.4.2", "packet_loss_percent": 0.0},
        ],
        "hops_backup": [
            {"dst_ip": "10.0.1.1", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.13.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.34.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.4.2", "packet_loss_percent": 0.0},
        ],
    }
    t7_primary = {
        "test_id": "T7_route_check",
        "r1_to_dest": "via 10.0.12.2",
        "r2_to_dest": "via 10.0.24.2",
        "r4_to_src": "via 10.0.24.1",
        "active_path": "primary",
    }
    t9_equal_loss = {
        "test_id": "T9_oneway_loss",
        "hostA_to_hostB": {"packet_loss_percent": 100.0},
        "hostB_to_hostA": {"packet_loss_percent": 100.0},
    }
    f1_vocab = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "primary r2-r4 down",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{
            "cause_id": "F1",
            "predicted_tests": [
                {"test_id": "T1_e2e_ping", "expected": "down"},
                {"test_id": "T2_hop_ping", "expected": "r2-r4=down"},
            ],
        }],
    }
    f1_obs = t1_down + [t2_f1, t7_primary, t9_equal_loss]
    f1_ok = check_consistency(f1_vocab, f1_obs)
    assert f1_ok["passed"], f1_ok
    f1_gate = gate_decide(None, f1_vocab, healthy, f1_obs, mode="full", run_missing_tests=False, noise_id="N0")
    assert f1_gate["reason"] != "consistency failed", f1_gate
    # Healthy snapshot: no alternative needs a different fix, so there is no
    # discriminating test; consistency_scope=discriminating escalates (fail-safe).
    f1_consistency = (f1_gate.get("checks") or {}).get("consistency", {})
    assert f1_consistency.get("scope") == "discriminating" and f1_consistency.get("discriminating_tests") == [], f1_gate
    assert f1_gate["decision"] == "escalate" and "no discriminating test" in f1_gate["reason"], f1_gate
    f1_wrong_t7 = dict(
        f1_vocab,
        causes=[{
            "cause_id": "F1",
            "predicted_tests": [
                {"test_id": "T1_e2e_ping", "expected": "down"},
                {"test_id": "T7_route_check", "expected": "blackhole"},
            ],
        }],
    )
    f1_t7 = check_consistency(f1_wrong_t7, f1_obs)
    assert not f1_t7["passed"], f1_t7
    assert any(row.get("test_id") == "T7_route_check" and row.get("observed") == "primary" for row in f1_t7["mismatches"]), f1_t7
    f1_t7_gate = gate_decide(None, f1_wrong_t7, healthy, f1_obs, mode="full", run_missing_tests=False, noise_id="N0")
    # T7 is not a discriminating test here: a wrong side-prediction cannot block.
    assert f1_t7_gate["decision"] == "escalate" and f1_t7_gate["reason"] != "consistency failed", f1_t7_gate
    f1_t9 = check_consistency(
        {
            **f1_vocab,
            "causes": [{
                "cause_id": "F1",
                "predicted_tests": [{"test_id": "T9_oneway_loss", "expected": "a2b_loss"}],
            }],
        },
        f1_obs,
    )
    assert not f1_t9["passed"] and f1_t9["mismatches"][0]["observed"] == "symmetric", f1_t9
    english_f1 = dict(
        f1_vocab,
        causes=[{"cause_id": "F1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "timeout / no reply"}]}],
    )
    english_gate = gate_decide(None, english_f1, healthy, f1_obs, mode="full", run_missing_tests=False, noise_id="N0")
    assert english_gate["decision"] == "block" and "invalid proposal" in (english_gate.get("reason") or ""), english_gate

    from bench.contracts import (
        T2_HOP_OUTCOMES,
        T2_HOPS_BACKUP,
        T2_HOPS_PRIMARY,
        T6_IFACES,
        parse_t2_prediction,
        parse_t6_prediction,
    )
    from bench.outcomes import HOP_NAMES, grade_result, load_test_thresholds
    from gate.grader import grade_test

    ip_by_hop = {name: ip for ip, name in HOP_NAMES.items()}
    n0_limits = load_test_thresholds("N0")

    def _hop_row(name, outcome="ok"):
        if outcome == "down":
            body = {"packet_loss_percent": 100.0}
        elif outcome == "loss":
            body = {"packet_loss_percent": 50.0, "latency_ms": 10.0}
        elif outcome == "delay":
            body = {"packet_loss_percent": 0.0, "latency_ms": 200.0}
        else:
            body = {"packet_loss_percent": 0.0, "latency_ms": 10.0}
        return {"dst_ip": ip_by_hop[name], **body}

    def _t2_obs(primary_outcomes=None, backup_outcomes=None, include_backup=True):
        primary_outcomes = primary_outcomes or {}
        backup_outcomes = backup_outcomes or {}
        row = {
            "test_id": "T2_hop_ping",
            "hops_primary": [_hop_row(name, primary_outcomes.get(name, "ok")) for name in T2_HOPS_PRIMARY],
        }
        if include_backup:
            row["hops_backup"] = [_hop_row(name, backup_outcomes.get(name, "ok")) for name in T2_HOPS_BACKUP]
        return row

    round_trip_pairs = 0
    for path, hops in (("primary", T2_HOPS_PRIMARY), ("backup", T2_HOPS_BACKUP)):
        for hop in hops:
            for outcome in sorted(T2_HOP_OUTCOMES):
                observed = _t2_obs(
                    primary_outcomes={hop: outcome} if path == "primary" else None,
                    backup_outcomes={hop: outcome} if path == "backup" else None,
                )
                graded = grade_result(observed, n0_limits)
                parsed = parse_t2_prediction(graded)
                assert parsed["ok"] and parsed["errors"] == [], (path, hop, outcome, parsed, graded)
                assert parsed["pairs"][(path, hop)] == outcome, (path, hop, outcome, parsed, graded)
                round_trip_pairs += 1
    assert round_trip_pairs == 32, round_trip_pairs
    healthy_t2 = grade_result(_t2_obs(), n0_limits)
    healthy_parsed = parse_t2_prediction(healthy_t2)
    assert healthy_parsed["ok"] and len(healthy_parsed["pairs"]) == 8, (healthy_t2, healthy_parsed)

    f1_grade = grade_result(t2_f1, n0_limits)
    assert prediction_matches("T2_hop_ping", "r2-r4=down", f1_grade)
    assert prediction_matches("T2_hop_ping", "  r2-r4 = down  ", f1_grade)
    assert not prediction_matches("T2_hop_ping", "r2 - r4=down", f1_grade)
    assert prediction_matches("T2_hop_ping", "primary:r2-r4=down", f1_grade)
    assert not prediction_matches("T2_hop_ping", "r1-r2=down", f1_grade)
    assert not prediction_matches("T2_hop_ping", "r3-r4=down", f1_grade)
    assert not prediction_matches("T2_hop_ping", "r2-r4=ok", f1_grade)
    assert prediction_matches("T2_hop_ping", "primary:hostA-r1=ok", f1_grade)
    assert not expected_grade_ok("T2_hop_ping", "hostA-r1=ok")
    assert not expected_grade_ok("T2_hop_ping", "hop=ok")
    assert not expected_grade_ok("T2_hop_ping", "10.0.24.2=down")

    primary_only_grade = grade_result(_t2_obs(include_backup=False, primary_outcomes={"r2-r4": "down"}), n0_limits)
    assert "backup:" not in primary_only_grade
    assert not prediction_matches("T2_hop_ping", "backup:r3-r4=ok", primary_only_grade)
    assert not prediction_matches("T2_hop_ping", "primary:r2-r4=down;backup:r3-r4=ok", primary_only_grade)
    missing_backup = check_consistency(
        {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "backup missing",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T2_hop_ping", "expected": "backup:r3-r4=ok"}]}],
        },
        [_t2_obs(include_backup=False, primary_outcomes={"r2-r4": "down"})],
    )
    assert not missing_backup["passed"], missing_backup
    assert missing_backup["predictions_checked"] == 1
    assert any("no backup section" in (row.get("reason") or "") for row in missing_backup["mismatches"]), missing_backup

    comma_parsed = parse_t2_prediction("primary:r2-r4=down,backup:r3-r4=ok")
    assert comma_parsed["ok"] and comma_parsed["delimiter_normalized"] is True
    assert comma_parsed["original"] == "primary:r2-r4=down,backup:r3-r4=ok"
    assert ";backup:" in comma_parsed["normalized"]
    hop_comma = parse_t2_prediction("primary:hostA-r1=ok,r2-r4=down")
    assert hop_comma["ok"] and hop_comma["delimiter_normalized"] is False
    assert not parse_t2_prediction("primary:r2-r4=down,not-a-path:r3-r4=ok")["ok"]

    t6_drops = {
        "test_id": "T6_queue_stats",
        "queues": {
            "r1/r1-eth1": {"drops": 4, "backlog_bytes": 0, "backlog_packets": 0},
            "r2/r2-eth1": {"drops": 2, "backlog_bytes": 9, "backlog_packets": 1},
        },
    }
    t6_grade = grade_result(t6_drops, n0_limits)
    assert t6_grade.startswith("drops:"), t6_grade
    assert prediction_matches("T6_queue_stats", "drops:r2/r2-eth1", t6_grade)
    assert prediction_matches("T6_queue_stats", "drops:r2/r2-eth1|r1/r1-eth1", t6_grade)
    assert prediction_matches("T6_queue_stats", "drops:r1/r1-eth1|r2/r2-eth1", t6_grade)
    assert not prediction_matches("T6_queue_stats", "backlog:r2/r2-eth1", t6_grade)
    assert not expected_grade_ok("T6_queue_stats", "drops:")
    assert not expected_grade_ok("T6_queue_stats", "drops:r2/r2-eth1|r2/r2-eth1")
    assert not expected_grade_ok("T6_queue_stats", "drops:r4/r4-eth2")
    assert "r4/r4-eth2" not in T6_IFACES
    t6_backlog = {
        "test_id": "T6_queue_stats",
        "queues": {"r2/r2-eth1": {"drops": 0, "backlog_bytes": 8, "backlog_packets": 1}},
    }
    assert grade_result(t6_backlog, n0_limits) == "backlog:r2/r2-eth1"
    assert prediction_matches("T6_queue_stats", "backlog:r2/r2-eth1", "backlog:r2/r2-eth1")

    empty_preds = check_consistency(
        {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "empty preds",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": []}],
        },
        f1_obs,
    )
    assert empty_preds["passed"] is False and empty_preds["reason"] == "no checked predictions", empty_preds
    assert empty_preds["predictions_total"] == empty_preds["predictions_checked"] == 0

    unrun = check_consistency(
        {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "unrun tests",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": [
                {"test_id": "T10_speed", "expected": "slow"},
                {"test_id": "T5_if_errors", "expected": "clean"},
            ]}],
        },
        f1_obs,
    )
    assert unrun["passed"] is False and unrun["reason"] == "no checked predictions", unrun
    assert unrun["predictions_total"] == 2 and unrun["predictions_checked"] == 0
    assert unrun["predictions_unchecked"] == 2
    unrun_gate = gate_decide(
        None,
        {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "unrun tests",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T10_speed", "expected": "slow"}]}],
        },
        healthy,
        f1_obs,
        mode="full",
        run_missing_tests=False, noise_id="N0")
    # Unchecked predictions are missing evidence: escalate, not block.
    assert unrun_gate["decision"] == "escalate", unrun_gate
    assert unrun_gate["reason"].startswith("insufficient evidence"), unrun_gate
    assert unrun_gate["checks"]["consistency"]["reason"] == "no checked predictions"

    from agent.validate import validate_proposal as _validate_proposal

    dup_ok, dup_errs = _validate_proposal(dict(
        f1_vocab,
        causes=[{"cause_id": "F1", "predicted_tests": [
            {"test_id": "T2_hop_ping", "expected": "r2-r4=down"},
            {"test_id": "T2_hop_ping", "expected": "r2-r4=ok"},
        ]}],
    ))
    assert not dup_ok and any("duplicate test_id" in err for err in dup_errs), dup_errs

    from bench.run_pilot import prediction_overlap

    overlap_ok, overlap_matched = prediction_overlap(
        {"causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T2_hop_ping", "expected": "r2-r4=down"}]}]},
        [t2_f1],
        thresholds=n0_limits,
    )
    overlap_wrong, overlap_wrong_matched = prediction_overlap(
        {"causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T2_hop_ping", "expected": "r2-r4=ok"}]}]},
        [t2_f1],
        thresholds=n0_limits,
    )
    assert overlap_ok == 1 and overlap_matched == 1
    assert overlap_wrong == 1 and overlap_wrong_matched == 0
    assert prediction_matches("T2_hop_ping", "r2-r4=down", f1_grade) is True
    assert grade_test("T2_hop_ping", f1_grade, "r2-r4=down", use_model=False) == 1.0
    assert grade_test("T2_hop_ping", f1_grade, "r2-r4=ok", use_model=False) == 0.0
    cons_t2 = check_consistency(
        {"causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T2_hop_ping", "expected": "r2-r4=down"}]}]},
        [t2_f1],
    )
    assert cons_t2["passed"] and cons_t2["predictions_checked"] == 1 and cons_t2["predictions_matched"] == 1

    import tempfile as _tmp

    model_dir = Path(_tmp.mkdtemp())
    model_path = model_dir / "grader_model.json"
    model_path.write_text(json.dumps({"T1_e2e_ping|loss": 0.91}), encoding="utf-8")
    assert grade_test("T1_e2e_ping", "ok", "loss", use_model=True, model_path=model_path) == 0.91
    assert grade_test("T1_e2e_ping", "ok", "loss", use_model=True, model_path=model_dir / "missing.json") == 0.0
    assert grade_test("T1_e2e_ping", "loss", "loss", use_model=True, model_path=model_path) == 1.0
    assert model_path.resolve() != (Path(__file__).resolve().parent.parent / "results" / "grader_model.json").resolve()
    repo = Path(__file__).resolve().parent.parent
    caller_sources = {
        "gate/checks.py": ("from bench.contracts import", "prediction_matches"),
        "gate/grader.py": ("from bench.contracts import", "prediction_matches"),
        "agent/validate.py": ("from bench.contracts import", "expected_grade_ok"),
        "bench/run_pilot.py": ("from bench.contracts import", "prediction_matches"),
    }
    for rel, needles in caller_sources.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, (rel, needle)
        assert "replace(\";\"" not in text and "replace(';'," not in text, rel

    t4_fail = [{"test_id": "T4_mtu", "df_ping_ok": False}]
    idle_mtu = gate_decide(None, idle, healthy, t4_fail, mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_mtu["decision"] != "allow", idle_mtu
    f12_snap = _snapshot(backup_up=False, active_path="primary")
    f12_snap["metrics"] = {"active_path": "primary", "packet_loss_percent": 0.0, "connectivity": True}
    t2_backup_down = {
        "test_id": "T2_hop_ping",
        "hops_primary": [
            {"dst_ip": "10.0.1.1", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.12.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.24.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.4.2", "packet_loss_percent": 0.0},
        ],
        "hops_backup": [
            {"dst_ip": "10.0.1.1", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.13.2", "packet_loss_percent": 0.0},
            {"dst_ip": "10.0.34.2", "packet_loss_percent": 100.0},
            {"dst_ip": "10.0.4.2", "packet_loss_percent": 0.0},
        ],
    }
    idle_f12 = gate_decide(None, idle, f12_snap, noop_ok + [t2_backup_down], mode="full", run_missing_tests=False, noise_id="N0")
    assert idle_f12["decision"] == "allow" and idle_f12["reason"] == "healthy no-op", idle_f12
    f12_reroute = gate_decide(
        None, _cause_proposal("REROUTE", "F12"), f12_snap, noop_ok + [t2_backup_down], mode="full", run_missing_tests=False, noise_id="N0")
    assert f12_reroute["decision"] == "block" and f12_reroute["reason"] == "safety failed", f12_reroute

    calls.clear()
    unbound_allow = apply_authorized_action(
        {"decision": "allow"},
        proposal,
        execute,
        healthy,
        lambda: healthy,
    )
    assert calls == [] and unbound_allow["execution_attempted"] is False
    assert unbound_allow["execution_error"] == "decision has no gate-issued authorization", unbound_allow

    from agent.validate import MAX_EXPECTED_LEN

    fabricated = _cause_proposal("REROUTE", "H0")
    fabricated_out = gate_decide(None, fabricated, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert fabricated_out["decision"] == "block" and fabricated_out["reason"] == "proposal contradicted", fabricated_out
    calls.clear()
    fabricated_exec = apply_authorized_action(fabricated_out, fabricated, execute, healthy, lambda: healthy)
    assert calls == [] and fabricated_exec["execution_attempted"] is False, fabricated_exec

    lookalike_snap = _snapshot(active_path="primary")
    lookalike_snap["metrics"] = {
        "active_path": "primary",
        "packet_loss_percent": 40.0,
        "connectivity": True,
        "latency_ms": 20.0,
    }
    for link in lookalike_snap["links"]:
        link["packet_loss_percent"] = 0.0
    lookalike = _cause_proposal("REROUTE", "F3", expected="loss")
    lookalike_out = gate_decide(None, lookalike, lookalike_snap, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert lookalike_out["decision"] in {"block", "escalate"} and lookalike_out["decision"] != "allow", lookalike_out

    hidden_snap = _snapshot(backup_up=False, active_path="primary")
    hidden = _cause_proposal("REROUTE", "F1")
    hidden_out = gate_decide(None, hidden, hidden_snap, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert hidden_out["decision"] == "block" and hidden_out["reason"] == "safety failed", hidden_out
    calls.clear()
    hidden_exec = apply_authorized_action(hidden_out, hidden, execute, hidden_snap, lambda: hidden_snap)
    assert calls == [] and hidden_exec["execution_attempted"] is False, hidden_exec

    unseen = _cause_proposal("REROUTE", "U1")
    unseen_out = gate_decide(None, unseen, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert unseen_out["decision"] == "escalate" and unseen_out["reason"] == "outside catalog", unseen_out
    calls.clear()
    unseen_exec = apply_authorized_action(unseen_out, unseen, execute, healthy, lambda: healthy)
    assert calls == [] and unseen_exec["execution_attempted"] is False, unseen_exec

    long_expected = _cause_proposal("DO_NOTHING", "F1", expected="x" * (MAX_EXPECTED_LEN + 1))
    long_rejected = screen_proposal(long_expected)
    assert long_rejected and long_rejected["execution_error"] == "invalid proposal", long_rejected
    missing = screen_proposal({"target": "HostA_to_HostB", "reason": "no"})
    assert missing and missing["execution_error"] == "invalid proposal", missing

    backup_claim = _cause_proposal("REROUTE", "F12")
    backup_claim_out = gate_decide(None, backup_claim, healthy, [], mode="full", run_missing_tests=False, noise_id="N0")
    assert backup_claim_out["decision"] == "block" and backup_claim_out["reason"] == "proposal contradicted", backup_claim_out
    calls.clear()
    down_exec = apply_authorized_action(
        _gate_stand_in({"decision": "allow"}, hidden_snap, backup, []),
        backup,
        execute,
        hidden_snap,
        lambda: hidden_snap,
    )
    assert calls == [] and down_exec["execution_error"] == "backup path is down", down_exec

    withheld = []

    def no_exec(item):
        withheld.append(item)
        return {"executed": True, "action": item.get("action")}

    def refuse_gate(record):
        return {
            "decision": "block",
            "reason": "safety failed",
            "proposal_action": "REROUTE",
            "tests_run": [],
            "checks": {},
        }

    blocked_ep = run_episode(
        object(),
        {},
        noise_id="N0", fault_id="H0",
        run_oam=False,
        agent_fn=reroute_agent,
        gate_fn=refuse_gate,
        execute_fn=no_exec,
        _components=fakes,
    )
    assert blocked_ep["gate_decision"]["decision"] == "block", blocked_ep
    assert blocked_ep["execution_attempted"] is False and withheld == [], blocked_ep
    assert injected_between["gate_decision"]["decision"] == "allow"
    assert injected_between["execution_attempted"] is False
    assert injected_between["execution_error"] == "path qdisc changed since the gate decision"

    print("execution boundary: PASS")
    print("negative safety: PASS")


def _live_preflight(noise_ids=("N0",)):
    """Refuse live Mininet modes when calibration is unaccepted (H2)."""
    from bench.outcomes import refuse_live_if_unaccepted

    return refuse_live_if_unaccepted(list(noise_ids))


def live_oam_cost():
    """Full OAM battery on the diamond; checks measured cost fields.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-cost
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("live OAM cost check must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
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
    refused = _live_preflight()
    if refused:
        return refused
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
            noise_id="N0", fault_id="H0",
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


def live_model():
    """One live diamond snapshot through the configured LLM. No frozen writes.

    On the VM, as root, with the provider key in the same sudo environment:

        sudo --preserve-env=GROQ_API_KEY python3 bench/run_episode.py --live-model
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("live model check must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
    from agent.decision_agent import propose_with_predictions, provider_config

    cfg = provider_config()
    print(
        f"provider={cfg['provider']} model={cfg['model']} "
        f"effort={cfg['reasoning_effort']} {cfg['key_env']}="
        f"{'set' if cfg['api_key'] else 'missing'}"
    )
    if not cfg["api_key"]:
        print(
            f"set {cfg['key_env']} and rerun with sudo --preserve-env={cfg['key_env']}",
            file=sys.stderr,
        )
        return 2
    from bench.mininet_util import mn_cleanup
    from network.topology import create_network

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    try:
        snapshot = collect_current_state(net, links)
        decision = propose_with_predictions(snapshot)
    finally:
        net.stop()

    cost = decision.get("_cost") or {}
    names = {node.get("name") for node in (snapshot.get("nodes") or [])}
    metrics = snapshot.get("metrics") or {}
    print(
        f"live nodes={sorted(names)} "
        f"path={metrics.get('active_path')} "
        f"loss={metrics.get('packet_loss_percent')} "
        f"latency_ms={metrics.get('latency_ms')}"
    )
    print(
        f"latency_seconds={cost.get('latency_seconds')} "
        f"prompt_tokens={cost.get('prompt_tokens')} "
        f"completion_tokens={cost.get('completion_tokens')}"
    )
    print(
        f"action={decision.get('action')} "
        f"causes={[c.get('cause_id') for c in decision.get('causes') or []]}"
    )
    failures = []
    if not validate_record("telemetry", snapshot)[0]:
        failures.append("telemetry")
    if not {"hostA", "hostB", "r1", "r2", "r3", "r4"} <= names:
        failures.append("diamond nodes")
    if not validate_record("proposal", decision)[0]:
        failures.append("proposal")
    if failures:
        print("live model: FAIL " + "; ".join(failures))
        return 1
    print("live model: PASS")
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
    refused = _live_preflight()
    if refused:
        return refused
    from bench.mininet_util import mn_cleanup, run_in_node
    from control.actions import execute_gated
    from control.routes import RouteControl
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

    def controller(item, capability=None):
        executed.append(item)
        return execute_gated(net, capability)

    def one_sided(item):
        executed.append(item)
        run_in_node(net.get("r1"), "ip route replace 10.0.4.0/24 via 10.0.13.2 dev r1-eth2")
        return {"executed": True, "action": "REROUTE", "path": "backup"}

    def attempt(execute_fn=controller, approved=None):
        executed.clear()
        approved_snap = approved or read()
        return apply_authorized_action(
            _gate_stand_in(dict(allow), approved_snap, proposal, []),
            proposal,
            execute_fn,
            approved_snap,
            read,
            route_control=RouteControl(net),
        )

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
            "one-sided reroute is recorded as partial and compensated on r1 only",
            len(executed) == 1 and out["remediation_succeeded"] is False and out["execution_error"] == "partial remediation"
            and out["partial_application"] is True and (out["compensation"] or {}).get("verified") is True
            and (out["compensation"] or {}).get("routers") == ["r1"],
            str(out),
        )
        expect("exact routes restored by compensation", _routes_on_path(read(), "primary"))
        ctl.set_active_path(net, "primary")

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


def live_gate_inputs():
    """Frozen catalog/matrix plus a live diamond snapshot into decide(, noise_id="N0").

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-gate-inputs
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("gate inputs live check must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
    from bench.mininet_util import mn_cleanup
    from gate.decide import decide as gate_decide
    from gate.inputs import load_frozen_context, validate_snapshot
    from network.topology import create_network

    frozen_ok, frozen_errors, _frozen = load_frozen_context("N0")
    if not frozen_ok:
        print("gate inputs live: FAIL frozen " + "; ".join(frozen_errors))
        return 1

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    failures = []
    try:
        snap = collect_current_state(net, links)
        snap_errors = validate_snapshot(snap)
        if snap_errors:
            failures.append("snapshot " + "; ".join(snap_errors))
        if not validate_record("telemetry", snap)[0]:
            failures.append("telemetry schema")
        nothing = {
            "action": "DO_NOTHING",
            "target": "HostA_to_HostB",
            "reason": "stage9 live",
            "proposed_path": ["r1", "r2", "r4"],
        }
        allowed = gate_decide(net, nothing, snap, [], mode="full", run_missing_tests=False, noise_id="N0")
        if allowed.get("decision") != "allow":
            failures.append(f"do_nothing {allowed}")
        risky = {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "stage9 live",
            "proposed_path": ["r1", "r3", "r4"],
        }
        blocked = gate_decide(net, risky, snap, [], mode="full", run_missing_tests=False, noise_id="N0")
        if blocked.get("decision") == "allow" or blocked.get("reason") != "incomplete hypotheses":
            failures.append(f"reroute without hypotheses {blocked}")
        empty = gate_decide(net, dict(risky, causes=[{"cause_id": "F1", "predicted_tests": []}]), {}, [], mode="full", run_missing_tests=False, noise_id="N0")
        if empty.get("decision") != "escalate" or not str(empty.get("reason", "")).startswith("incomplete snapshot:"):
            failures.append(f"empty snapshot {empty}")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        net.stop()
        mn_cleanup()

    if failures:
        print("gate inputs live: FAIL " + "; ".join(failures))
        return 1
    print("gate inputs live: PASS")
    return 0


def live_gate():
    """Stage 10 algorithm on the live diamond: safety block, then F1 decide(, noise_id="N0").

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-gate
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("gate algorithm live check must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
    from bench.mininet_util import mn_cleanup
    from gate.decide import decide as gate_decide
    from gate.inputs import load_frozen_context
    from network.topology import create_network

    frozen_ok, frozen_errors, frozen = load_frozen_context("N0")
    if not frozen_ok:
        print("gate live: FAIL frozen " + "; ".join(frozen_errors))
        return 1
    cfg = frozen.get("gate") or {}
    if cfg.get("frozen") is not True or cfg.get("learned_grader") is not False:
        print(f"gate live: FAIL config {cfg}")
        return 1

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    failures = []
    try:
        inj.inject(net, "F12")
        snap12 = collect_current_state(net, links)
        unsafe = {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "stage10 live",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F12", "predicted_tests": []}],
        }
        blocked = gate_decide(net, unsafe, snap12, [], mode="full", run_missing_tests=False, noise_id="N0")
        if blocked.get("decision") != "block" or blocked.get("reason") not in {"safety failed", "proposal contradicted"}:
            failures.append(f"F12 reroute {blocked}")
        inj.reset_all(net, baseline=True)

        inj.inject(net, "F1")
        snap1 = collect_current_state(net, links)
        from agent.validate import MAX_EXPECTED_LEN
        from bench.contracts import TEST_IDS

        preds = []
        for row in frozen["matrix"]:
            if row.get("fault_id") != "F1":
                continue
            tid = row.get("test_id")
            expected = row.get("outcome")
            if tid not in TEST_IDS or not isinstance(expected, str) or not expected.strip():
                continue
            if len(expected) > MAX_EXPECTED_LEN:
                continue
            preds.append({"test_id": tid, "expected": expected})
        proposal = {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "stage10 live",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": preds}],
        }
        out = gate_decide(net, proposal, snap1, [], mode="full", run_missing_tests=True, noise_id="N0")
        if out.get("decision") not in {"allow", "block", "escalate"}:
            failures.append(f"F1 decide {out}")
        log = out.get("log") or {}
        if not log:
            failures.append(f"F1 no log decision={out.get('decision')} reason={out.get('reason')}")
        if "candidates" not in log or "F1" not in (log.get("candidates") or []):
            failures.append(f"F1 candidates {log}")
        used = (log.get("budget") or {}).get("tests_used_seconds")
        cap = (log.get("budget") or {}).get("budget_seconds")
        if used is None or cap is None or used > cap + 1e-9:
            failures.append(f"F1 budget {log.get('budget')}")
        if log.get("learned_grader") is not False:
            failures.append(f"learned grader {log.get('learned_grader')}")
        if "rules" not in log:
            failures.append("F1 rules missing")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        inj.reset_all(net, baseline=True)
        net.stop()
        mn_cleanup()

    if failures:
        print("gate live: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("gate live: PASS")
    return 0


def live_integrity():
    """Bind, replay, and revalidation on the live diamond. Does not reroute.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-integrity
    """
    import os
    import tempfile

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("verification integrity must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
    from bench.mininet_util import mn_cleanup
    from network.topology import create_network

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "live integrity",
        "proposed_path": ["r1", "r3", "r4"],
    }
    executed = []

    def read():
        return read_execution_state(net, links)

    def stub(item):
        executed.append(item)
        return {"executed": False, "action": "REROUTE"}

    failures = []

    def expect(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    try:
        approved = read()
        allow = {
            "decision": "allow",
            "reason": "integrity",
            "proposal_action": "REROUTE",
            "tests_run": [],
            "checks": {},
        }
        bound = _gate_stand_in(allow, approved, proposal, [])
        expect("gate-issued authorization", bool(bound.get("authorization", {}).get("signature")), str(bound.get("authorization")))

        executed.clear()
        out = apply_authorized_action(bound, proposal, stub, approved, read)
        expect(
            "bound allow executes once",
            executed == [proposal] and out.get("state_verified") is True and out.get("execution_attempted") is True,
            str(out),
        )

        executed.clear()
        replay = apply_authorized_action(bound, proposal, stub, approved, read)
        expect(
            "replay of a used decision is refused",
            not executed and replay.get("execution_error") == "decision already used",
            str(replay),
        )

        stale = _gate_stand_in(dict(allow), approved, proposal, [])
        other = dict(approved)
        other["routes"] = [
            dict(row, route=(row.get("route") or "") + " #stale")
            for row in (approved.get("routes") or [])
        ]
        executed.clear()
        mismatched = apply_authorized_action(stale, proposal, stub, other, read)
        expect(
            "decision bound to a different snapshot is refused",
            not executed and mismatched.get("execution_error") == "decision not bound to this snapshot",
            str(mismatched),
        )

        executed.clear()
        mutated = apply_authorized_action(stale, dict(proposal, action="DO_NOTHING"), stub, approved, read)
        expect(
            "proposal changed since the gate decision is refused",
            not executed and mutated.get("execution_error") == "proposal changed since the gate decision",
            str(mutated),
        )

        executed.clear()
        blocked = apply_authorized_action({"decision": "block"}, proposal, stub, approved, read)
        expect("block does not execute", not executed and blocked.get("execution_attempted") is False, str(blocked))

        record = {
            "episode_id": "live-integrity",
            "gate_decision": stale,
            "agent_proposal": proposal,
            "state_verified": False,
        }
        logged = save_episode(record, tempfile.mkdtemp())
        loaded = json.loads(Path(logged).read_text(encoding="utf-8"))
        expect(
            "auditable decision record",
            loaded.get("gate_decision", {}).get("authorization", {}).get("decision_id")
            == stale.get("authorization", {}).get("decision_id"),
            str(loaded.get("gate_decision")),
        )
        Path(logged).unlink()
        Path(logged).parent.rmdir()
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        net.stop()
        mn_cleanup()

    if failures:
        print("verification integrity: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("verification integrity: PASS")
    return 0


def live_safety():
    """Stage 12 negative and safety cases on the live diamond.

    On the VM, as root:

        sudo python3 bench/run_episode.py --live-safety
    """
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("negative safety must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    refused = _live_preflight()
    if refused:
        return refused
    from agent.validate import MAX_EXPECTED_LEN
    from bench.mininet_util import mn_cleanup, run_in_node
    from control.actions import execute_gated
    from control.routes import RouteControl
    from gate.decide import decide as gate_decide
    from network import controller as ctl
    from network.topology import create_network

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    executed = []
    failures = []

    def read():
        return read_execution_state(net, links)

    def stub(item):
        executed.append(item)
        return {"executed": False, "action": item.get("action")}

    def controller(item, capability=None):
        executed.append(item)
        return execute_gated(net, capability)

    def expect(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    def still_primary():
        return _routes_on_path(read(), "primary")

    try:
        snap = collect_current_state(net, links)
        approved = read()
        expect("healthy starts on primary", still_primary())

        fabricated = _cause_proposal("REROUTE", "H0")
        fabricated_out = gate_decide(net, fabricated, snap, [], mode="full", run_missing_tests=False, noise_id="N0")
        executed.clear()
        fabricated_exec = apply_authorized_action(fabricated_out, fabricated, stub, approved, read)
        expect(
            "healthy plus fabricated fault does not authorize reroute",
            fabricated_out.get("decision") != "allow"
            and fabricated_exec.get("execution_attempted") is False
            and not executed
            and still_primary(),
            f"decision={fabricated_out} exec={fabricated_exec}",
        )

        unseen = _cause_proposal("REROUTE", "U1")
        unseen_out = gate_decide(net, unseen, snap, [], mode="full", run_missing_tests=False, noise_id="N0")
        executed.clear()
        unseen_exec = apply_authorized_action(unseen_out, unseen, stub, approved, read)
        expect(
            "unseen fault falls back instead of authorizing",
            unseen_out.get("decision") == "escalate"
            and unseen_out.get("reason") == "outside catalog"
            and unseen_exec.get("execution_attempted") is False
            and not executed
            and still_primary(),
            f"decision={unseen_out} exec={unseen_exec}",
        )

        malformed = screen_proposal({"action": "reboot", "target": "HostA_to_HostB", "reason": "no"})
        long_pred = screen_proposal(_cause_proposal("DO_NOTHING", "F1", expected="x" * (MAX_EXPECTED_LEN + 1)))
        expect(
            "malformed AI output is rejected",
            malformed
            and malformed.get("execution_error") == "invalid proposal"
            and long_pred
            and long_pred.get("execution_error") == "invalid proposal"
            and still_primary(),
            f"malformed={malformed} long={long_pred}",
        )

        nothing = {
            "action": "DO_NOTHING",
            "target": "HostA_to_HostB",
            "reason": "stage12",
            "proposed_path": ["r1", "r2", "r4"],
        }
        allowed = gate_decide(net, nothing, snap, [], mode="full", run_missing_tests=False, noise_id="N0")
        executed.clear()
        # The gate's own authorization, verified against the snapshot it evaluated.
        nothing_exec = apply_authorized_action(
            allowed, nothing, controller, approved, read, gate_snapshot=snap, route_control=RouteControl(net)
        )
        expect(
            "gate allow and network outcome agree for do_nothing",
            allowed.get("decision") == "allow"
            and nothing_exec.get("execution_attempted") is True
            and still_primary(),
            f"decision={allowed} exec={nothing_exec}",
        )

        stale_approved = read()
        ctl.set_active_path(net, "backup")
        executed.clear()
        stale = apply_authorized_action(
            _gate_stand_in({"decision": "allow"}, stale_approved, _cause_proposal("REROUTE", "F1"), []),
            _cause_proposal("REROUTE", "F1"),
            stub,
            stale_approved,
            read,
        )
        expect(
            "changed state between verification and execution is refused",
            not executed
            and stale.get("execution_attempted") is False
            and stale.get("execution_error") == "route changed since the gate decision",
            str(stale),
        )
        ctl.set_active_path(net, "primary")
        expect("primary restored after stale-state case", still_primary())

        inj.inject(net, "F12")
        snap12 = collect_current_state(net, links)
        approved12 = read()
        hidden = _cause_proposal("REROUTE", "F1")
        hidden_out = gate_decide(net, hidden, snap12, [], mode="full", run_missing_tests=False, noise_id="N0")
        backup_claim = _cause_proposal("REROUTE", "F12")
        backup_out = gate_decide(net, backup_claim, snap12, [], mode="full", run_missing_tests=False, noise_id="N0")
        executed.clear()
        hidden_exec = apply_authorized_action(hidden_out, hidden, stub, approved12, read)
        backup_exec = apply_authorized_action(
            _gate_stand_in({"decision": "allow"}, approved12, backup_claim, []),
            backup_claim,
            stub,
            approved12,
            read,
        )
        loss = oam._parse_ping(run_in_node(net.get("hostA"), "ping -c 5 -i 0.2 -W 1 10.0.4.2"))["packet_loss_percent"]
        expect(
            "look-alike primary-down diagnosis is not allowed on backup-down",
            hidden_out.get("decision") != "allow"
            and hidden_exec.get("execution_attempted") is False,
            f"decision={hidden_out} exec={hidden_exec}",
        )
        expect(
            "hidden backup-down cause does not authorize reroute",
            hidden_out.get("decision") != "allow"
            and hidden_exec.get("execution_attempted") is False
            and not executed,
            f"decision={hidden_out} exec={hidden_exec}",
        )
        expect(
            "rerouting when the backup path is down is blocked",
            backup_out.get("decision") != "allow"
            and backup_exec.get("execution_error") == "backup path is down"
            and not executed
            and still_primary()
            and loss < 100.0,
            f"decision={backup_out} exec={backup_exec} loss={loss}",
        )
        inj.reset_all(net, baseline=True)
        expect("regression: primary restored after F12", still_primary())
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        inj.reset_all(net, baseline=True)
        net.stop()
        mn_cleanup()

    if failures:
        print("negative safety: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("negative safety: PASS")
    return 0


if __name__ == "__main__":
    if "--live-recheck" in sys.argv:
        sys.exit(live_recheck())
    if "--live-gate-inputs" in sys.argv:
        sys.exit(live_gate_inputs())
    if "--live-gate" in sys.argv:
        sys.exit(live_gate())
    if "--live-integrity" in sys.argv:
        sys.exit(live_integrity())
    if "--live-safety" in sys.argv:
        sys.exit(live_safety())
    if "--live-model" in sys.argv:
        sys.exit(live_model())
    if "--live" in sys.argv:
        sys.exit(live_diamond())
    if "--live-cost" in sys.argv:
        sys.exit(live_oam_cost())
    if "--aggregate-costs" in sys.argv:
        print(json.dumps(aggregate_episode_costs(), indent=2))
        sys.exit(0)
    _self_check()
