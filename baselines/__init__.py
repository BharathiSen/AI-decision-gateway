"""Experiment baselines B0–B5. One call shape for every method."""

from baselines.common import (
    DEFAULT_BUDGET_SECONDS,
    METHODS,
    PERMITTED_ACTIONS,
    attach_contract,
)

_DECISIONS = {"allow", "block", "escalate"}


def run_method(method, proposal, snapshot, test_results, net=None, budget_seconds=None, noise_id=None):
    """Run one method on the shared proposal, snapshot, and OAM results.

    The true fault id is not an input. net is used only when a method
    needs to run extra OAM tests. B0 does not change the network here.
    noise_id selects grading thresholds for B2–B5 (same levels as the gate).
    Callers must pass the episode's noise_id; there is no N0 default.
    """
    name = (method or "").upper()
    proposal = proposal or {}
    snapshot = snapshot or {}
    test_results = list(test_results or [])
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    if name not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if noise_id is None:
        raise ValueError(f"{name}: noise_id is required (pass the episode noise level)")
    from bench.outcomes import ThresholdsUnavailable

    try:
        result = _dispatch(name, proposal, snapshot, test_results, net, cap, noise_id)
    except ThresholdsUnavailable as exc:
        # Unaccepted calibration: no arm grades OAM. Fail closed, say why.
        result = {
            "method": name,
            "decision": "escalate",
            "action": proposal.get("action"),
            "tests_run": [],
            "reason": f"thresholds unavailable: {exc}",
            "threshold_status": exc.status,
        }
    except ValueError as exc:
        if "incomplete thresholds" in str(exc) or "noise_id" in str(exc):
            result = {
                "method": name,
                "decision": "escalate",
                "action": proposal.get("action"),
                "tests_run": [],
                "reason": str(exc),
            }
        else:
            raise

    result = attach_contract(result, proposal, budget_seconds=cap)
    if result.get("decision") not in _DECISIONS or "method" not in result or "action" not in result:
        raise ValueError(f"{name} returned an incomplete decision record: {result!r}")
    return result


def _dispatch(name, proposal, snapshot, test_results, net, cap, noise_id):
    # Episode noise level is required for grading. Callers must pass the
    # episode's N0/N1/N2; there is no silent cross-level fallback here.
    if noise_id is None:
        raise ValueError(f"{name}: noise_id is required (pass the episode noise level)")
    if name == "B0":
        from baselines.b0_ungated import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B1":
        from baselines.b1_fixed_rules import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B2":
        from baselines.b2_llm_checker import decide
        result = decide(
            proposal, snapshot, test_results, net=net, run_battery=net is not None, noise_id=noise_id
        )
    elif name == "B3":
        from baselines.b3_ml_classifier import decide
        result = decide(
            proposal,
            snapshot,
            test_results,
            net=net,
            budget_seconds=cap,
            noise_id=noise_id,
        )
    elif name == "B4":
        from baselines.b4_llm_checker import decide
        result = decide(
            proposal,
            snapshot,
            test_results,
            net=net,
            noise_id=noise_id,
            budget_seconds=cap,
        )
    elif name == "B5":
        from baselines.b5_adaptive import decide_b5
        from bench.outcomes import load_thresholds

        thresholds = load_thresholds(noise_id)
        result = decide_b5(
            net,
            proposal,
            snapshot,
            test_results,
            run_tests=net is not None,
            thresholds=thresholds,
            noise_id=noise_id,
            budget_seconds=cap,
        )
    else:
        raise ValueError(f"unknown method {name!r}")
    return result


from bench.outcomes import using_test_thresholds  # offline checks grade with test-only limits


@using_test_thresholds
def _self_check():
    from gate.selector import held_out_ids, _load_matrix
    from baselines.ablations import (
        gate_full,
        gate_no_discrimination,
        gate_no_grader,
        gate_no_predictions,
    )
    from baselines.b0_ungated import run as b0_run
    from baselines.common import budgeted_catalog, test_cost
    from bench.tests import DEFAULT_ORDER

    shared = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "check",
        "proposed_path": ["r1", "r3", "r4"],
    }
    healthy = {
        "links": [
            {"name": "r1-r2", "up": True},
            {"name": "r2-r4", "up": True},
            {"name": "r1-r3", "up": True},
            {"name": "r3-r4", "up": True},
        ],
        "metrics": {"packet_loss_percent": 0, "latency_ms": 10, "active_path": "primary"},
    }
    lossy = {
        "links": healthy["links"],
        "metrics": {"packet_loss_percent": 40, "latency_ms": 12, "active_path": "primary"},
    }
    backup_down = {
        "links": [
            {"name": "r1-r2", "up": True},
            {"name": "r2-r4", "up": True},
            {"name": "r1-r3", "up": False},
            {"name": "r3-r4", "up": True},
        ],
        "metrics": {"packet_loss_percent": 40, "latency_ms": 12, "active_path": "primary"},
    }

    for method in METHODS:
        out = run_method(method, shared, healthy, [], noise_id="N0")
        assert out["decision"] in _DECISIONS, out
        assert out["method"] == method
        assert "action" in out
        assert out["permitted_actions"] == list(PERMITTED_ACTIONS)
        assert "budget" in out and "tests_run" in out
        assert set(out["permitted_actions"]) <= set(PERMITTED_ACTIONS)

    try:
        run_method("B0", shared, healthy, [], fault_id="F1", noise_id="N0")
        raise AssertionError("fault_id was accepted")
    except TypeError:
        pass

    try:
        run_method("B6", shared, healthy, [], noise_id="N0")
        raise AssertionError("unknown method was accepted")
    except ValueError:
        pass

    try:
        run_method("B0", shared, healthy, [])
        raise AssertionError("missing noise_id was accepted")
    except ValueError as exc:
        assert "noise_id" in str(exc)

    b0 = run_method("B0", shared, healthy, [], noise_id="N0")
    assert b0["decision"] == "allow" and b0["tests_run"] == []

    b1_ok = run_method("B1", shared, healthy, [], noise_id="N0")
    assert b1_ok["action"] == "DO_NOTHING" and b1_ok["tests_run"] == []
    b1_loss = run_method("B1", shared, lossy, [], noise_id="N0")
    assert b1_loss["action"] == "REROUTE" and b1_loss["path"] == ["r1", "r3", "r4"]

    t1_ok = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 12, "connectivity": True}]
    t8_fail = [{"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": False}]
    # ACL rule still fires when required T1 is present and healthy.
    b2_acl = run_method("B2", shared, healthy, t1_ok + t8_fail, noise_id="N0")
    assert b2_acl["action"] == "ESCALATE", b2_acl
    assert "missing_evidence" not in b2_acl, b2_acl
    t1_loss = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 40, "latency_ms": 12, "connectivity": True}]
    b2_reroute = run_method("B2", shared, healthy, t1_loss, noise_id="N0")
    assert b2_reroute["action"] == "REROUTE", b2_reroute
    b2_f12 = run_method("B2", shared, backup_down, t1_loss, noise_id="N0")
    assert b2_f12["action"] == "ESCALATE", b2_f12
    # Replay (net is None) must not pretend a battery ran.
    assert b2_reroute["tests_run"] == ["T1_e2e_ping"]
    # Missing / ungradable / errored T1 must not fall through to DO_NOTHING.
    b2_ok = run_method("B2", shared, healthy, t1_ok, noise_id="N0")
    assert b2_ok["action"] == "DO_NOTHING" and b2_ok["decision"] == "allow", b2_ok
    b2_absent = run_method("B2", shared, healthy, [], noise_id="N0")
    assert b2_absent["decision"] == "escalate" and b2_absent["action"] == "ESCALATE", b2_absent
    assert b2_absent["missing_evidence"] == ["T1_e2e_ping"], b2_absent
    # T8-only battery without T1 escalates on missing required evidence.
    b2_t8_only = run_method("B2", shared, healthy, t8_fail, noise_id="N0")
    assert b2_t8_only["decision"] == "escalate" and b2_t8_only["missing_evidence"] == ["T1_e2e_ping"], b2_t8_only
    for bad in (None, float("nan"), float("inf")):
        t1_missing = [dict(t1_ok[0], packet_loss_percent=bad)]
        b2_missing = run_method("B2", shared, healthy, t1_missing, noise_id="N0")
        assert b2_missing["decision"] == "escalate" and b2_missing["action"] == "ESCALATE", (bad, b2_missing)
        assert b2_missing["missing_evidence"] == ["T1_e2e_ping"], (bad, b2_missing)
        assert b2_missing["path"] == [], (bad, b2_missing)
    t1_err = [dict(t1_ok[0], packet_loss_percent=None, error="TimeoutError: ping timed out")]
    b2_err = run_method("B2", shared, healthy, t1_err, noise_id="N0")
    assert b2_err["decision"] == "escalate" and b2_err["action"] == "ESCALATE", b2_err
    assert b2_err["missing_evidence"] == ["T1_e2e_ping"], b2_err
    assert "required evidence unavailable" in (b2_err.get("reason") or ""), b2_err

    t10 = {"test_id": "T10_speed", "mbit_per_sec": 0.1}
    b3 = run_method("B3", shared, healthy, t1_loss + [t10], budget_seconds=3.0, noise_id="N0")
    assert b3["tests_run"] == ["T1_e2e_ping"], b3
    assert "T10_speed" not in b3["tests_run"], b3
    assert b3["budget"]["within_budget"]
    assert b3["budget"]["tests_used_seconds"] <= 3.0 + 1e-9
    hidden = held_out_ids()
    assert b3.get("matched_fault") not in hidden
    assert all(row["fault_id"] not in hidden for row in _load_matrix())

    b4 = run_method("B4", shared, healthy, [], noise_id="N0")
    assert b4["decision"] == "escalate" and b4["reason"] == "no cause"
    caused = dict(shared)
    caused["causes"] = [{"cause_id": "F1", "predicted_tests": []}]
    b4_f1 = run_method("B4", caused, healthy, t1_loss, noise_id="N0")
    assert b4_f1["decision"] in _DECISIONS
    assert b4_f1["method"] == "B4"
    # B4 must never blind-truncate catalog T2 into an unparseable prefix.
    from agent.validate import MAX_EXPECTED_LEN
    from baselines.b4_llm_checker import _catalog_predictions, _compact_expected

    long_ok = "primary:hostA-r1=ok,r1-r2=ok,r2-r4=ok,r4-hostB=ok;backup:hostA-r1=ok,r1-r3=ok,r3-r4=ok,r4-hostB=ok"
    assert len(long_ok) > MAX_EXPECTED_LEN
    assert _compact_expected("T2_hop_ping", long_ok) is None
    assert _compact_expected("T1_e2e_ping", "x" * (MAX_EXPECTED_LEN + 1)) is None
    assert _compact_expected("T2_hop_ping", "not-a-t2-grade") is None
    assert _compact_expected("T2_hop_ping", "") is None
    assert _compact_expected("T2_hop_ping", None) is None
    # Incomplete / unparseable structured text must not become a truncated prefix.
    assert _compact_expected("T2_hop_ping", "primary:hostA-r1=ok,r1-r2=") is None
    oversized_prefix = long_ok[:MAX_EXPECTED_LEN]
    assert _compact_expected("T2_hop_ping", long_ok) != oversized_prefix
    f1_t2 = next(row["outcome"] for row in _load_matrix() if row["fault_id"] == "F1" and row["test_id"] == "T2_hop_ping")
    compact_f1 = _compact_expected("T2_hop_ping", f1_t2)
    assert compact_f1 is not None and len(compact_f1) <= MAX_EXPECTED_LEN, compact_f1
    assert "r2-r4=down" in compact_f1 and "=ok" not in compact_f1, compact_f1
    from bench.contracts import parse_t2_prediction

    assert parse_t2_prediction(compact_f1).get("ok") is True, compact_f1
    filled = _catalog_predictions("F1", ["T2_hop_ping", "T1_e2e_ping"], _load_matrix())
    assert any(p["test_id"] == "T2_hop_ping" and len(p["expected"]) <= MAX_EXPECTED_LEN for p in filled), filled
    assert all(len(p["expected"]) <= MAX_EXPECTED_LEN for p in filled), filled
    # H0 all-ok T2 is too long and has no non-ok hops: omit rather than truncate.
    h0_preds = _catalog_predictions("H0", ["T2_hop_ping"], _load_matrix())
    assert not any(p["test_id"] == "T2_hop_ping" for p in h0_preds), h0_preds

    b5 = run_method("B5", shared, healthy, [], noise_id="N0")
    assert b5["decision"] == "escalate" and b5["reason"] == "no cause"
    b5_bad = run_method("B5", {"action": "NOT_AN_ACTION"}, healthy, [], noise_id="N0")
    assert b5_bad["decision"] == "block"
    idle = {
        "action": "DO_NOTHING",
        "target": "HostA_to_HostB",
        "reason": "ok",
        "causes": [{"cause_id": "H0", "predicted_tests": []}],
    }
    noop_ok = [
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "connectivity": True, "error": None},
        {"test_id": "T4_mtu", "df_ping_ok": True, "error": None},
        {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None},
        {"test_id": "T10_speed", "mbit_per_sec": 50.0, "error": None},
    ]
    b5_idle = run_method("B5", idle, healthy, noop_ok, noise_id="N0")
    assert b5_idle["decision"] == "allow" and b5_idle["reason"] == "healthy no-op", b5_idle
    b5_idle_loss = run_method(
        "B5", idle, healthy, [{"test_id": "T1_e2e_ping", "packet_loss_percent": 30.0, "latency_ms": 10.0}], noise_id="N0"
    )
    assert b5_idle_loss["decision"] != "allow", b5_idle_loss
    b5_idle_incomplete = run_method(
        "B5", idle, healthy, [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0}], noise_id="N0"
    )
    assert b5_idle_incomplete["decision"] != "allow", b5_idle_incomplete
    b5_no_cause = run_method(
        "B5",
        {"action": "DO_NOTHING", "target": "HostA_to_HostB", "reason": "ok"},
        healthy,
        noop_ok,
        noise_id="N0",
    )
    assert b5_no_cause["decision"] == "escalate" and "missing cause" in b5_no_cause["reason"], b5_no_cause
    # B5 must evaluate every relevant alternative (not alts[0] only).
    from baselines.b5_adaptive import decide_b5
    import inspect as _inspect

    b5_src = _inspect.getsource(decide_b5)
    assert "alts[0]" not in b5_src
    assert "for alt in relevant" in b5_src
    # Multi-cause proposal expands candidates so ≥2 relevant alts need elimination.
    multi = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "multi-alt",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [
            {"cause_id": "F1", "predicted_tests": []},
            {"cause_id": "F10", "predicted_tests": []},
            {"cause_id": "F11", "predicted_tests": []},
        ],
    }
    b5_multi = decide_b5(None, multi, lossy, t1_loss, run_tests=False, noise_id="N0")
    assert len(b5_multi.get("relevant_alternatives") or []) >= 2, b5_multi
    # Incomplete evidence across the full relevant set must not allow.
    assert b5_multi["decision"] != "allow", b5_multi
    assert b5_multi["decision"] == "escalate", b5_multi
    assert set(b5_multi.get("remaining_alternatives") or []) == set(
        b5_multi.get("relevant_alternatives") or []
    ), b5_multi
    # H5: B3/B4/B5 receive the same noise_id thresholds / budget contract.
    b3_n1 = run_method("B3", shared, healthy, t1_loss, noise_id="N1")
    assert b3_n1.get("noise_id") == "N1"
    b4_sel = run_method("B4", caused, healthy, t1_loss, noise_id="N0")
    from gate.decide import decide as gate_decide

    vf = gate_decide(
        None,
        caused,
        healthy,
        t1_loss,
        mode="full",
        run_missing_tests=False,
        noise_id="N0",
    )
    assert list(b4_sel.get("selected_tests") or []) == list((vf.get("log") or {}).get("selected_tests") or []), (
        b4_sel.get("selected_tests"),
        (vf.get("log") or {}).get("selected_tests"),
    )

    # Ablations exist and share decide()'s return shape.
    empty_oam = []
    for fn in (gate_full, gate_no_predictions, gate_no_discrimination, gate_no_grader):
        audit = fn(None, caused, healthy, empty_oam, run_missing_tests=False)
        assert audit["decision"] in _DECISIONS, (fn.__name__, audit)

    # Cheapest-first budget never includes T10 at 8 s (T10 costs 8 s by itself,
    # and cheaper tests fill first).
    catalog = budgeted_catalog(8.0)
    assert "T10_speed" not in catalog
    assert test_cost("T1_e2e_ping") == 2.5
    assert sum(test_cost(tid) for tid in DEFAULT_ORDER) > DEFAULT_BUDGET_SECONDS

    # B0 live path: execute, measure T1 harm, restore the previous route.
    class _Node:
        def __init__(self, net, name):
            self.net = net
            self.name = name

        def IP(self):
            return "10.0.4.2" if self.name == "hostB" else "10.0.1.2"

        def cmd(self, line):
            backup = self.net.path == "backup"
            if "ip route show" in line:
                if self.name == "r4":
                    via, dev = ("10.0.34.1", "r4-eth1") if backup else ("10.0.24.1", "r4-eth0")
                    return f"10.0.1.0/24 via {via} dev {dev}"
                via, dev = ("10.0.13.2", "r1-eth2") if backup else ("10.0.12.2", "r1-eth1")
                return f"10.0.4.0/24 via {via} dev {dev}"
            if "ip route replace" in line:
                if "10.0.13.2" in line or "10.0.34.1" in line:
                    self.net.path = "backup"
                elif "10.0.12.2" in line or "10.0.24.1" in line:
                    self.net.path = "primary"
            return ""

    class _Net:
        def __init__(self):
            self.path = "primary"

        def get(self, name):
            return _Node(self, name)

    import bench.tests as oam_mod

    net = _Net()

    def fake_run(node, command, timeout=60.0):
        loss = 80 if net.path == "backup" else 0
        received = 10 - int(loss / 10)
        return f"10 packets transmitted, {received} received, {loss}% packet loss time=1.0 ms\n"

    saved = oam_mod.run_in_node
    oam_mod.run_in_node = fake_run
    try:
        # Explicit noise levels must reach measure_traffic_recovery independently.
        seen_noise = []

        def _spy_recovery(net_arg, noise_id="N0", ping_count=None, speed_seconds=None):
            seen_noise.append(noise_id)
            return {
                "results": [],
                "recovery_status": "measurement_unavailable",
                "traffic_recovered": None,
                "error": "spy",
            }

        import bench.run_episode as ep_mod

        saved_mtr = ep_mod.measure_traffic_recovery
        ep_mod.measure_traffic_recovery = _spy_recovery
        try:
            for level in ("N0", "N1", "N2"):
                b0_run(net, shared, measure=True, noise_id=level)
            try:
                b0_run(net, shared, measure=True)
                raise AssertionError("B0 run accepted missing noise_id")
            except ValueError as exc:
                assert "noise_id" in str(exc)
        finally:
            ep_mod.measure_traffic_recovery = saved_mtr
        assert seen_noise == ["N0", "N1", "N2"], seen_noise

        live = b0_run(net, shared, noise_id="N0")
    finally:
        oam_mod.run_in_node = saved
    assert live["decision"] == "allow"
    assert live["execution"].get("executed") is True
    assert live["harm"] is True, live
    assert live["rolled_back"] is True
    assert net.path == "primary", net.path
    assert live["before_path"] == "primary"
    # C4: B0 gets the same post-action recovery battery opportunity; command
    # success is not traffic recovery.
    assert live.get("post_oam") is not None, live
    assert live.get("traffic_recovered") is not True, live
    assert live.get("recovery_status") in (
        "failed",
        "partial",
        "measurement_unavailable",
        "recovered",
    ), live

    idle = {"action": "DO_NOTHING", "target": "HostA_to_HostB", "reason": "ok"}
    oam_mod.run_in_node = fake_run
    try:
        skipped = b0_run(net, idle, noise_id="N0")
    finally:
        oam_mod.run_in_node = saved
    assert skipped["execution"].get("executed") is False
    assert skipped["harm"] is False
    assert skipped["rolled_back"] is False
    assert skipped.get("post_oam") is not None, "DO_NOTHING still gets post-action measurement"
    assert "recovery_status" in skipped
    assert skipped["execution_error"] is None and live["execution_error"] is None
    assert skipped["execution_refused"] is False and live["execution_refused"] is False

    # A reroute the controller refuses was never applied: harm is unknown,
    # reported as an execution error, and never scored as "no harm".
    for refused_path in (["hostA", "r1", "r3", "r4", "hostB"], [], ["r1", "r3", "r2", "r4"]):
        refused_proposal = dict(shared, proposed_path=refused_path)
        net.path = "primary"
        oam_mod.run_in_node = fake_run
        try:
            refused = b0_run(net, refused_proposal, noise_id="N0")
        finally:
            oam_mod.run_in_node = saved
        assert refused["execution"].get("executed") is False, refused["execution"]
        assert refused["execution_error"] and "ValueError" in refused["execution_error"], refused
        assert refused["execution_refused"] is True, refused
        assert refused["harm"] is None, refused["harm"]
        assert refused["rolled_back"] is False and net.path == "primary", (refused["rolled_back"], net.path)
        assert refused.get("post_oam") is not None, "a refused action still gets post-action measurement"

    # An unparseable action is also refused before any change.
    net.path = "primary"
    oam_mod.run_in_node = fake_run
    try:
        unparsed = b0_run(net, dict(shared, action="reboot_everything"), noise_id="N0")
    finally:
        oam_mod.run_in_node = saved
    assert unparsed["execution_refused"] is True and unparsed["harm"] is None, unparsed
    assert net.path == "primary"

    # Verified restore. A fake with one route per router, so a half-applied
    # reroute and a faulted, asymmetric start are representable, and each
    # router's route command can succeed, raise, or fail silently.
    from bench.run_episode import TopologyContaminated

    class _RouteNode:
        VIA = {
            "r1": {"primary": "10.0.12.2", "backup": "10.0.13.2"},
            "r4": {"primary": "10.0.24.1", "backup": "10.0.34.1"},
        }

        def __init__(self, net, name):
            self.net = net
            self.name = name

        def IP(self):
            return "10.0.4.2" if self.name == "hostB" else "10.0.1.2"

        def cmd(self, line):
            if "ip route show" in line:
                index = self.net.reads
                self.net.reads += 1
                if self.name in self.net.unreadable or index in self.net.fail_reads:
                    raise RuntimeError(f"{self.name} shell stuck")
                stale = index in self.net.stale_reads
                texts = self.net.frozen_texts if stale else self.net.texts
                if self.name in texts:
                    return texts[self.name]
                routes = self.net.frozen if stale else self.net.routes
                path = routes.get(self.name)
                if path not in ("primary", "backup"):
                    return ""
                dest = "10.0.4.0/24" if self.name == "r1" else "10.0.1.0/24"
                return f"{dest} via {self.VIA[self.name][path]} dev {self.name}-ethX"
            if "ip route replace" in line or "ip route del" in line:
                if "ip route del" in line:
                    wanted = "none"
                else:
                    wanted = next(path for path, via in self.VIA[self.name].items() if via in line)
                self.net.commands.append((self.name, wanted))
                outcome = self.net.plan.pop(0) if self.net.plan else "ok"
                if outcome == "raise":
                    raise RuntimeError(f"{self.name} unreachable")
                if outcome == "silent":
                    return "RTNETLINK answers: Network is unreachable"
                self.net.routes[self.name] = wanted
                if self.name in self.net.texts:
                    spec = line.split("ip route replace ", 1)[1].strip() if "ip route replace" in line else ""
                    if outcome == "strip":
                        # Lands on the right path but loses part of the route.
                        spec = spec.replace(" proto static", "")
                    self.net.texts[self.name] = spec
            return ""

    class _RouteNet:
        def __init__(self, plan=(), routes=None, unreadable=(), stale_reads=(), fail_reads=(), texts=None):
            self.routes = dict(routes or {"r1": "primary", "r4": "primary"})
            self.frozen = dict(self.routes)
            self.texts = dict(texts or {})
            self.frozen_texts = dict(self.texts)
            self.plan = list(plan)
            self.commands = []
            self.unreadable = set(unreadable)
            self.reads = 0
            self.stale_reads = set(stale_reads)
            self.fail_reads = set(fail_reads)

        def get(self, name):
            return _RouteNode(self, name)

    def _route_run(route_net, proposal_arg=None):
        def lossy(node, command, timeout=60.0):
            routes = route_net.routes
            loss = 0 if routes.get("r1") == routes.get("r4") == "primary" else 80
            received = 10 - int(loss / 10)
            return f"10 packets transmitted, {received} received, {loss}% packet loss time=1.0 ms\n"

        oam_mod.run_in_node = lossy
        try:
            return b0_run(route_net, proposal_arg or shared, noise_id="N0")
        finally:
            oam_mod.run_in_node = saved

    def _stopped(route_net, proposal_arg=None):
        try:
            _route_run(route_net, proposal_arg)
        except TopologyContaminated as exc:
            return exc.record
        raise AssertionError("the run was expected to stop with TopologyContaminated")

    both_primary = {"r1": "primary", "r4": "primary"}
    both_backup = {"r1": "backup", "r4": "backup"}

    # 1. Successful reroute: applied on both routers, harm measured before the
    #    restore, each router's exact route put back and read back.
    ok_net = _RouteNet()
    ok = _route_run(ok_net)
    assert ok["execution"].get("executed") is True and ok["execution_error"] is None, ok
    assert ok["partial_application"] is False and ok["execution_refused"] is False
    assert ok["routes_before"] == both_primary and ok["routes_after"] == both_backup, ok
    assert ok["harm"] is True, "harm is read while the reroute is still in place"
    assert ok["rollback_attempted"] is True and ok["rollback_verified"] is True and ok["rolled_back"] is True
    assert ok["routes_restored"] == both_primary and ok_net.routes == both_primary
    assert ok_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "primary"), ("r4", "primary")], ok_net.commands
    assert ok["rollback_error"] is None
    assert ok["route_lines_before"] == {
        "r1": "10.0.4.0/24 via 10.0.12.2 dev r1-ethX",
        "r4": "10.0.1.0/24 via 10.0.24.1 dev r4-ethX",
    }, ok["route_lines_before"]

    # 2. First-router failure: nothing changed, so nothing to restore. It is an
    #    action failure, not a refusal, and harm is still evaluated.
    first_net = _RouteNet(plan=["raise"])
    first = _route_run(first_net)
    assert first["execution"].get("executed") is False and "RuntimeError" in first["execution_error"], first
    assert first["execution_refused"] is False and first["partial_application"] is False
    assert first["routes_after"] == both_primary
    assert first["rollback_attempted"] is False and first["rollback_verified"] is None and first["rolled_back"] is False
    assert first["harm"] is False, first["harm"]
    assert first_net.commands == [("r1", "backup")], first_net.commands

    # 3. Second-router failure: r1 moved, r4 raised. Harm is measured on the
    #    half-applied network. Only the router that changed is restored.
    second_net = _RouteNet(plan=["ok", "raise"])
    second = _route_run(second_net)
    assert second["execution"].get("executed") is False and "RuntimeError" in second["execution_error"], second
    assert second["execution_refused"] is False and second["partial_application"] is True
    assert second["routes_after"] == {"r1": "backup", "r4": "primary"}, second["routes_after"]
    assert second["harm"] is True, second["harm"]
    assert second["rollback_attempted"] is True and second["rollback_verified"] is True and second["rolled_back"] is True
    assert second["routes_restored"] == both_primary and second_net.routes == both_primary
    assert second_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "primary")], second_net.commands

    # 4. Second router fails without raising: the controller reports success.
    #    Only the route read-back shows the reroute is half-applied.
    silent_net = _RouteNet(plan=["ok", "silent"])
    silent = _route_run(silent_net)
    assert silent["execution"].get("executed") is True, silent["execution"]
    assert silent["execution_error"] and "controller reported success" in silent["execution_error"], silent
    assert silent["partial_application"] is True and silent["harm"] is True
    assert silent["rollback_verified"] is True and silent_net.routes == both_primary

    # 5. Failed restore: the restore command raises on r1. The run stops with
    #    TopologyContaminated and the result attached. One attempt, no retry.
    stuck_net = _RouteNet(plan=["ok", "raise", "raise"])
    stuck = _stopped(stuck_net)
    assert stuck["partial_application"] is True and stuck["rollback_attempted"] is True
    assert stuck["rollback_verified"] is False and stuck["rolled_back"] is False
    assert "RuntimeError" in stuck["rollback_error"], stuck["rollback_error"]
    assert stuck["harm"] is True and stuck["execution_error"], "the failed action and its harm are still recorded"
    assert stuck_net.routes == {"r1": "backup", "r4": "primary"}
    assert stuck_net.commands[2:] == [("r1", "primary")], f"exactly one restore attempt, got {stuck_net.commands[2:]}"

    # 6. Restore that returns without error but leaves the router on the wrong
    #    path is not verified: same stop, still no retry.
    quiet_net = _RouteNet(plan=["ok", "raise", "silent"])
    quiet = _stopped(quiet_net)
    assert quiet["rollback_attempted"] is True and quiet["rollback_verified"] is False
    assert quiet["routes_restored"] == {"r1": "backup", "r4": "primary"}, quiet["routes_restored"]
    assert "after restore" in quiet["rollback_error"], quiet["rollback_error"]
    assert quiet_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "primary")], quiet_net.commands

    # 7. Restore after a fully applied reroute is verified on both routers.
    lost_net = _RouteNet(plan=["ok", "ok", "ok", "silent"])
    lost = _stopped(lost_net)
    assert lost["partial_application"] is False and lost["execution_error"] is None
    assert lost["rollback_verified"] is False and lost["routes_restored"] == {"r1": "primary", "r4": "backup"}
    assert len(lost_net.commands) == 4, lost_net.commands

    # 8. A faulted, asymmetric start is not contamination. B0 still acts, and
    #    puts back exactly what was there.
    #    F2 shape: r1 has lost its route to the far LAN.
    f2_start = {"r1": "none", "r4": "primary"}
    f2_net = _RouteNet(routes=f2_start)
    f2 = _route_run(f2_net)
    assert f2["routes_before"] == f2_start and f2["route_lines_before"]["r1"] == "", f2
    assert f2["before_path"] == "none", f2["before_path"]
    assert f2["routes_after"] == both_backup and f2["execution_error"] is None and f2["partial_application"] is False
    assert f2["rollback_attempted"] is True and f2["rollback_verified"] is True
    assert f2_net.routes == f2_start, f2_net.routes
    assert f2_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "none"), ("r4", "primary")], f2_net.commands
    assert f2["execution_refused"] is False and f2["harm"] is not None
    #    U5 shape: r4 already points at the backup next hop.
    u5_start = {"r1": "primary", "r4": "backup"}
    u5_net = _RouteNet(routes=u5_start)
    u5 = _route_run(u5_net)
    assert u5["routes_after"] == both_backup and u5["execution_error"] is None, u5
    assert u5["before_path"] == "primary", u5["before_path"]
    assert u5["rollback_verified"] is True and u5_net.routes == u5_start
    assert u5_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "primary")], u5_net.commands
    #    A failed restore from a faulted start still stops the run.
    f2_stuck_net = _RouteNet(routes=f2_start, plan=["ok", "ok", "silent"])
    f2_stuck = _stopped(f2_stuck_net)
    assert f2_stuck["rollback_verified"] is False and f2_stuck["routes_restored"] == {"r1": "backup", "r4": "primary"}
    #    Routes that cannot be read before the action cannot be restored:
    #    nothing is executed.
    blind_start = _RouteNet(unreadable=["r4"])
    assert _stopped(blind_start)["execution"] is None
    assert blind_start.commands == [], blind_start.commands
    #    ...but a no-op is still measured on an asymmetric start.
    asym_idle = _route_run(_RouteNet(routes={"r1": "backup", "r4": "primary"}), idle)
    assert asym_idle["rollback_attempted"] is False and asym_idle["execution_refused"] is False
    assert asym_idle["harm"] is False

    # 9. Refused before execution stays out of the harm reading, with no
    #    route command and no restore.
    refused_net = _RouteNet()
    refused_again = _route_run(refused_net, dict(shared, proposed_path=["r1", "r3", "r2", "r4"]))
    assert refused_again["execution_refused"] is True and refused_again["harm"] is None
    assert refused_again["partial_application"] is False and refused_again["rollback_attempted"] is False
    assert refused_again["rollback_verified"] is None and refused_net.commands == []

    # 10. Rerouting to the path already in use changes nothing: no restore.
    same_net = _RouteNet(routes=both_backup)
    same = _route_run(same_net)
    assert same["execution_error"] is None and same["partial_application"] is False
    assert same["rollback_attempted"] is False and same["rolled_back"] is False
    assert same_net.routes == both_backup

    # 11. REBALANCE toggles, is read back and restored like a reroute.
    rebalance = dict(shared, action="REBALANCE")
    for start, toggled in (("primary", "backup"), ("backup", "primary")):
        rb_net = _RouteNet(routes={"r1": start, "r4": start})
        rb = _route_run(rb_net, rebalance)
        assert rb["execution_error"] is None and rb["partial_application"] is False, rb
        assert rb["routes_after"] == {"r1": toggled, "r4": toggled}, (start, rb["routes_after"])
        assert rb["rollback_attempted"] is True and rb["rollback_verified"] is True
        assert rb_net.routes == {"r1": start, "r4": start}, (start, rb_net.routes)
        assert rb["execution_refused"] is False
    rb_half_net = _RouteNet(plan=["ok", "raise"])
    rb_half = _route_run(rb_half_net, rebalance)
    assert rb_half["partial_application"] is True and rb_half["execution_refused"] is False, rb_half
    assert rb_half["harm"] is True and rb_half["rollback_verified"] is True
    assert rb_half_net.routes == both_primary
    #     From an asymmetric start it toggles away from r1's path and is restored.
    rb_asym_start = {"r1": "backup", "r4": "primary"}
    rb_asym_net = _RouteNet(routes=rb_asym_start)
    rb_asym = _route_run(rb_asym_net, rebalance)
    assert rb_asym["routes_after"] == both_primary and rb_asym["execution_error"] is None, rb_asym
    assert rb_asym["rollback_verified"] is True and rb_asym_net.routes == rb_asym_start
    assert rb_asym_net.commands == [("r1", "primary"), ("r4", "primary"), ("r1", "backup")], rb_asym_net.commands

    # 12. Stale after-read: both routers still show the old routes to the
    #     first read after the action. The confirming read catches it and the
    #     routes are restored. Reads 0-1 are before the action, 2-3 right after.
    stale_net = _RouteNet(stale_reads={2, 3})
    stale = _route_run(stale_net)
    assert stale["routes_after"] == both_backup, stale["routes_after"]
    assert stale["execution_error"] is None and stale["partial_application"] is False, stale
    assert stale["rollback_attempted"] is True and stale["rollback_verified"] is True
    assert stale_net.routes == both_primary
    #     A genuine no-change is confirmed and left alone.
    assert first_net.reads == 6, f"first-router failure: 2 before, 2 after, 2 confirming; got {first_net.reads}"

    # 13. Unreadable after the action: an infrastructure gap, not an action
    #     failure. State is unknown, so one restore pass and a verified read-back.
    blind_net = _RouteNet(fail_reads={3})
    blind = _route_run(blind_net)
    assert blind["route_state_unreadable"] is True and blind["partial_application"] is None, blind
    assert blind["execution_error"] is None, blind["execution_error"]
    assert blind["harm"] is True and blind["rollback_attempted"] is True and blind["rollback_verified"] is True
    assert blind_net.routes == both_primary
    assert ok["route_state_unreadable"] is False and second["route_state_unreadable"] is False
    #     Unreadable at the read-back after the restore is not verified.
    blind_restore = _stopped(_RouteNet(fail_reads={5}))
    assert blind_restore["rollback_attempted"] is True and blind_restore["rollback_verified"] is False

    #     Both routers unreadable after the action: state unknown on both,
    #     so both get their captured route back, then a verified read-back.
    blind_both_net = _RouteNet(fail_reads={2, 3})
    blind_both = _route_run(blind_both_net)
    assert blind_both["route_state_unreadable"] is True and blind_both["partial_application"] is None, blind_both
    assert blind_both["routes_after"] == {"r1": "unreadable", "r4": "unreadable"}, blind_both["routes_after"]
    assert blind_both["rollback_attempted"] is True and blind_both["rollback_verified"] is True
    assert blind_both_net.commands == [("r1", "backup"), ("r4", "backup"), ("r1", "primary"), ("r4", "primary")], blind_both_net.commands
    assert blind_both_net.routes == both_primary
    #     An unreadable start stops only actions that change routes: a no-op
    #     is still measured and returns normally.
    blind_idle_net = _RouteNet(unreadable=["r4"])
    blind_idle = _route_run(blind_idle_net, idle)
    assert blind_idle["execution_refused"] is False and blind_idle["rollback_attempted"] is False, blind_idle
    assert blind_idle["harm"] is False and blind_idle["routes_before"]["r4"] == "unreadable"
    assert blind_idle_net.commands == []

    # 13b. Exact routes, not path labels. The captured route carries text the
    #      path label ignores ("proto static").
    exact_start = {
        "r1": "10.0.4.0/24 via 10.0.12.2 dev r1-eth1 proto static",
        "r4": "10.0.1.0/24 via 10.0.24.1 dev r4-eth0 proto static",
    }
    #      Reroute and back: the exact text is put back, not just the path.
    exact_net = _RouteNet(texts=exact_start)
    exact = _route_run(exact_net)
    assert exact["route_lines_before"] == exact_start, exact["route_lines_before"]
    assert exact["routes_after"] == both_backup and exact["rollback_verified"] is True, exact
    assert exact_net.texts == exact_start, exact_net.texts
    #      A restore that lands on the right path but not the captured route is
    #      NOT verified: the run stops.
    strip_net = _RouteNet(texts=exact_start, plan=["ok", "ok", "strip", "ok"])
    strip = _stopped(strip_net)
    assert strip["routes_restored"] == both_primary, "same path label on both routers"
    assert strip["rollback_verified"] is False and strip["rolled_back"] is False, strip
    assert "after restore" in strip["rollback_error"], strip["rollback_error"]
    assert strip_net.texts["r1"] == "10.0.4.0/24 via 10.0.12.2 dev r1-eth1"
    #      An action that keeps the path label but changes the route is a
    #      change: it is restored. (REROUTE to the path already in use, where
    #      the controller's command drops "proto static" on r1.)
    relabel_net = _RouteNet(texts={"r1": exact_start["r1"]})
    relabel = _route_run(relabel_net, dict(shared, proposed_path=["r1", "r2", "r4"]))
    assert relabel["routes_after"] == both_primary and relabel["partial_application"] is False, relabel
    assert relabel["rollback_attempted"] is True and relabel["rollback_verified"] is True, relabel
    assert relabel_net.texts["r1"] == exact_start["r1"], relabel_net.texts
    assert relabel_net.commands == [("r1", "primary"), ("r4", "primary"), ("r1", "primary")], relabel_net.commands
    assert relabel_net.reads == 6, f"2 before, 2 after, 2 read-back; no confirming read needed: {relabel_net.reads}"
    #      Same, but the first read after the action is stale and still shows
    #      the captured routes. The confirming read compares exact routes, so
    #      the label-preserving change is still found and restored.
    relabel_stale_net = _RouteNet(texts={"r1": exact_start["r1"]}, stale_reads={2, 3})
    relabel_stale = _route_run(relabel_stale_net, dict(shared, proposed_path=["r1", "r2", "r4"]))
    assert relabel_stale["rollback_attempted"] is True and relabel_stale["rollback_verified"] is True, relabel_stale
    assert relabel_stale_net.texts["r1"] == exact_start["r1"], relabel_stale_net.texts

    # 14. Route reads use their own process (pexec) when the node has one,
    #     never the shared shell. A failed read is unreadable, never "no route".
    from baselines.b0_ungated import _route_state, _router_paths

    class _PexecNode:
        def __init__(self, name, code=0, extra=""):
            self.name = name
            self.code = code
            self.extra = extra

        def cmd(self, line):
            return "stale shell output via 10.0.13.2 via 10.0.34.1 "

        def pexec(self, args):
            assert args[:2] == ["bash", "-c"] and args[2].startswith("ip route show "), args
            via = "10.0.12.2" if self.name == "r1" else "10.0.24.1"
            dest = args[2].split()[-1]
            return f"{dest} via {via} dev {self.name}-eth0 linkdown \n{self.extra}", "", self.code

    class _PexecNet:
        def __init__(self, code=0, extra=""):
            self.code = code
            self.extra = extra

        def get(self, name):
            return _PexecNode(name, self.code, self.extra)

    assert _router_paths(_PexecNet()) == both_primary
    # Kernel state flags are not part of the route that gets put back.
    assert _route_state(_PexecNet())[0]["r1"] == "10.0.4.0/24 via 10.0.12.2 dev r1-eth0"
    assert _router_paths(_PexecNet(code=2)) == {"r1": "unreadable", "r4": "unreadable"}
    assert _route_state(_PexecNet(code=2))[0] == {"r1": None, "r4": None}
    assert _router_paths(_PexecNet(extra="10.0.4.0/24 via 10.0.99.9 dev x\n")) == {"r1": "unreadable", "r4": "unreadable"}

    # 15. A policy that cannot be read is an execution error, never an
    #     exception out of run(), and nothing is executed.
    import gate.checks as _gate_checks

    real_choice = _gate_checks.reroute_choice

    def _broken_choice(_proposal):
        raise OSError("policy.yaml unreadable")

    _gate_checks.reroute_choice = _broken_choice
    try:
        policy_net = _RouteNet()
        policy_fail = _route_run(policy_net)
    finally:
        _gate_checks.reroute_choice = real_choice
    assert policy_fail["execution_error"].startswith("policy check: OSError"), policy_fail["execution_error"]
    assert policy_fail["execution"].get("executed") is False and policy_net.commands == []
    assert policy_fail["rollback_attempted"] is False
    # Current scoring, pending owner decision R-2: not a refusal, so it has a
    # harm reading ("no harm" here) and enters the B0 harm rate.
    assert policy_fail["execution_refused"] is False and policy_fail["harm"] is False, policy_fail

    # The three outcomes are separate fields and never implied by one another.
    assert second["execution_error"] and second["rollback_verified"] is True, "action failed, restore succeeded"
    assert second["traffic_recovered"] is not True, "a verified restore is not traffic recovery"
    assert "traffic_recovered" in ok and ok["post_oam"] is not None

    # Metrics read the same harm for a half-applied action as before, and a
    # refusal is still left out of the harm rate.
    from bench.metrics import b0_harm_from_result

    assert b0_harm_from_result(second)["evaluated"] is True and b0_harm_from_result(second)["harm"] is True
    assert b0_harm_from_result(first)["evaluated"] is True and b0_harm_from_result(first)["harm"] is False
    assert b0_harm_from_result(refused_again)["evaluated"] is False
    assert b0_harm_from_result(refused_again)["execution_refused"] is True
    # Zero-change failures are counted beside the harm rate (reporting only),
    # straight from what run() records: first-router failure and policy failure
    # yes; a half-applied action, a clean reroute and a refusal no.
    assert b0_harm_from_result(first)["zero_change_failure"] is True
    assert b0_harm_from_result(policy_fail)["zero_change_failure"] is True
    assert b0_harm_from_result(second)["zero_change_failure"] is False
    assert b0_harm_from_result(ok)["zero_change_failure"] is False
    assert "zero_change_failure" not in b0_harm_from_result(refused_again)


if __name__ == "__main__":
    _self_check()
    print("baselines: PASS")
