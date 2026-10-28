"""Experiment baselines B0–B5. One call shape for every method."""

from baselines.common import (
    DEFAULT_BUDGET_SECONDS,
    METHODS,
    PERMITTED_ACTIONS,
    attach_contract,
)

_DECISIONS = {"allow", "block", "escalate"}


def run_method(method, proposal, snapshot, test_results, net=None, budget_seconds=None):
    """Run one method on the shared proposal, snapshot, and OAM results.

    The true fault id is not an input. net is used only when a method
    needs to run extra OAM tests. B0 does not change the network here.
    """
    name = (method or "").upper()
    proposal = proposal or {}
    snapshot = snapshot or {}
    test_results = list(test_results or [])
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else budget_seconds

    if name == "B0":
        from baselines.b0_ungated import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B1":
        from baselines.b1_fixed_rules import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B2":
        from baselines.b2_llm_checker import decide
        result = decide(proposal, snapshot, test_results, net=net, run_battery=net is not None)
    elif name == "B3":
        from baselines.b3_ml_classifier import decide
        result = decide(proposal, snapshot, test_results, net=net, budget_seconds=cap)
    elif name == "B4":
        from baselines.b4_llm_checker import decide
        result = decide(proposal, snapshot, test_results, net=net)
    elif name == "B5":
        from baselines.b5_adaptive import decide_b5
        result = decide_b5(net, proposal, snapshot, test_results, run_tests=net is not None)
    else:
        raise ValueError(f"unknown method {method!r}")

    result = attach_contract(result, proposal, budget_seconds=cap)
    if result.get("decision") not in _DECISIONS or "method" not in result or "action" not in result:
        raise ValueError(f"{name} returned an incomplete decision record: {result!r}")
    return result


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
        out = run_method(method, shared, healthy, [])
        assert out["decision"] in _DECISIONS, out
        assert out["method"] == method
        assert "action" in out
        assert out["permitted_actions"] == list(PERMITTED_ACTIONS)
        assert "budget" in out and "tests_run" in out
        assert set(out["permitted_actions"]) <= set(PERMITTED_ACTIONS)

    try:
        run_method("B0", shared, healthy, [], fault_id="F1")
        raise AssertionError("fault_id was accepted")
    except TypeError:
        pass

    try:
        run_method("B6", shared, healthy, [])
        raise AssertionError("unknown method was accepted")
    except ValueError:
        pass

    b0 = run_method("B0", shared, healthy, [])
    assert b0["decision"] == "allow" and b0["tests_run"] == []

    b1_ok = run_method("B1", shared, healthy, [])
    assert b1_ok["action"] == "DO_NOTHING" and b1_ok["tests_run"] == []
    b1_loss = run_method("B1", shared, lossy, [])
    assert b1_loss["action"] == "REROUTE" and b1_loss["path"] == ["r1", "r3", "r4"]

    t8_fail = [{"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": False}]
    b2_acl = run_method("B2", shared, healthy, t8_fail)
    assert b2_acl["action"] == "ESCALATE", b2_acl
    t1_loss = [{"test_id": "T1_e2e_ping", "packet_loss_percent": 40, "latency_ms": 12, "connectivity": True}]
    b2_reroute = run_method("B2", shared, healthy, t1_loss)
    assert b2_reroute["action"] == "REROUTE", b2_reroute
    b2_f12 = run_method("B2", shared, backup_down, t1_loss)
    assert b2_f12["action"] == "ESCALATE", b2_f12
    # Replay (net is None) must not pretend a battery ran.
    assert b2_reroute["tests_run"] == ["T1_e2e_ping"]

    t10 = {"test_id": "T10_speed", "mbit_per_sec": 0.1}
    b3 = run_method("B3", shared, healthy, t1_loss + [t10], budget_seconds=3.0)
    assert b3["tests_run"] == ["T1_e2e_ping"], b3
    assert "T10_speed" not in b3["tests_run"], b3
    assert b3["budget"]["within_budget"]
    assert b3["budget"]["tests_used_seconds"] <= 3.0 + 1e-9
    hidden = held_out_ids()
    assert b3.get("matched_fault") not in hidden
    assert all(row["fault_id"] not in hidden for row in _load_matrix())

    b4 = run_method("B4", shared, healthy, [])
    assert b4["decision"] == "escalate" and b4["reason"] == "no cause"
    caused = dict(shared)
    caused["causes"] = [{"cause_id": "F1", "predicted_tests": []}]
    b4_f1 = run_method("B4", caused, healthy, t1_loss)
    assert b4_f1["decision"] in _DECISIONS
    assert b4_f1["method"] == "B4"

    b5 = run_method("B5", shared, healthy, [])
    assert b5["decision"] == "escalate" and b5["reason"] == "no cause"
    b5_bad = run_method("B5", {"action": "NOT_AN_ACTION"}, healthy, [])
    assert b5_bad["decision"] == "block"

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
        live = b0_run(net, shared)
    finally:
        oam_mod.run_in_node = saved
    assert live["decision"] == "allow"
    assert live["execution"].get("executed") is True
    assert live["harm"] is True, live
    assert live["rolled_back"] is True
    assert net.path == "primary", net.path
    assert live["before_path"] == "primary"

    idle = {"action": "DO_NOTHING", "target": "HostA_to_HostB", "reason": "ok"}
    oam_mod.run_in_node = fake_run
    try:
        skipped = b0_run(net, idle)
    finally:
        oam_mod.run_in_node = saved
    assert skipped["execution"].get("executed") is False
    assert skipped["harm"] is False
    assert skipped["rolled_back"] is False


if __name__ == "__main__":
    _self_check()
    print("baselines: PASS")
