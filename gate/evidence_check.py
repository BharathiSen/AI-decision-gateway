"""Evidence-semantics regressions for the gate and B5. Offline, no network.

python -m gate.evidence_check

Positive evidence: a measured row graded healthy or matching a prediction.
Negative evidence: a measured row graded degraded or contradicting a prediction.
Missing evidence: a test that was not run, a row with `error` (tool or
infrastructure failure), or a missing/non-finite loss or throughput
measurement (None grade — never inferred as down/slow). Missing evidence
never eliminates an alternative, never passes a consistency check, and
never justifies a no-op.
Contradictory evidence: healthy no-op tests next to a snapshot that reports
loss or no connectivity. A no-op is not justified on contradictory evidence.
"""

import copy


def _snapshot(loss=0.0, connectivity=True):
    links = [
        {"name": name, "up": True, "endpoints": ["a", "b"], "latency_ms": 10.0, "packet_loss_percent": 0.0}
        for name in ("r1-r2", "r2-r4", "r1-r3", "r3-r4")
    ]
    return {
        "timestamp": "2026-11-06T00:00:00+00:00",
        "links": links,
        "routes": [
            {"node": "r1", "prefix": "10.0.4.0/24", "route": "10.0.4.0/24 via 10.0.12.2 dev r1-eth1"},
            {"node": "r4", "prefix": "10.0.1.0/24", "route": "10.0.1.0/24 via 10.0.24.1 dev r4-eth1"},
        ],
        "metrics": {
            "packet_loss_percent": loss,
            "latency_ms": 40.0,
            "connectivity": connectivity,
            "active_path": "primary",
        },
    }


T1_OK = {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 40.0, "connectivity": True, "error": None}
T4_OK = {"test_id": "T4_mtu", "df_ping_ok": True, "error": None}
T8_OK = {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True, "error": None}
T10_OK = {"test_id": "T10_speed", "mbit_per_sec": 50.0, "error": None}
NOOP_HEALTHY = [T1_OK, T4_OK, T8_OK, T10_OK]


def _errored(row, text="ping command_error"):
    out = dict(row)
    out["error"] = text
    if out["test_id"] == "T1_e2e_ping":
        out.update(packet_loss_percent=100.0, connectivity=False, latency_ms=None)
    if out["test_id"] == "T4_mtu":
        out["df_ping_ok"] = False
    if out["test_id"] == "T10_speed":
        out["mbit_per_sec"] = None
    return out


def _noop_proposal(cause_id="H0"):
    return {
        "action": "DO_NOTHING",
        "target": "HostA_to_HostB",
        "reason": "healthy",
        "proposed_path": [],
        "causes": [{"cause_id": cause_id, "predicted_tests": []}],
    }


from bench.outcomes import using_test_thresholds  # offline checks grade with test-only limits


@using_test_thresholds
def main():
    from agent.test_validate import good_proposal
    from baselines.b5_adaptive import decide_b5
    from bench.tests import DEFAULT_ORDER
    from gate.checks import NOOP_EVIDENCE_TESTS, noop_justified
    from gate.decide import _candidate_set, _symptoms_from_snapshot, decide
    from gate.inputs import load_frozen_context
    from gate.selector import covering_separating_tests

    noop = _noop_proposal("H0")
    healthy = list(NOOP_HEALTHY)
    assert NOOP_EVIDENCE_TESTS == ("T1_e2e_ping", "T4_mtu", "T8_tcp_vs_ping", "T10_speed")

    # Control: complete positive evidence + permitted cause justifies a no-op.
    assert noop_justified(_snapshot(), healthy, noise_id="N0")
    gate_ok = decide(None, noop, _snapshot(), healthy, run_missing_tests=False, noise_id="N0")
    assert gate_ok["decision"] == "allow", gate_ok
    assert decide_b5(None, noop, _snapshot(), healthy, run_tests=False, noise_id="N0")["decision"] == "allow"

    # Incomplete battery (missing T10) never justifies a no-op.
    incomplete = [T1_OK, T4_OK, T8_OK]
    assert not noop_justified(_snapshot(), incomplete, noise_id="N0")
    assert decide(None, noop, _snapshot(), incomplete, run_missing_tests=False, noise_id="N0")["decision"] == "escalate"
    assert decide(None, noop, _snapshot(), [], run_missing_tests=False, noise_id="N0")["decision"] == "escalate"

    # Degraded T10 refuses DO_NOTHING.
    slow = [T1_OK, T4_OK, T8_OK, {"test_id": "T10_speed", "mbit_per_sec": 0.1, "error": None}]
    assert not noop_justified(_snapshot(), slow, noise_id="N0")
    assert decide(None, noop, _snapshot(), slow, run_missing_tests=False, noise_id="N0")["decision"] == "escalate"

    # Missing cause or cause that forbids DO_NOTHING escalates even with healthy OAM.
    bare = {"action": "DO_NOTHING", "target": "HostA_to_HostB", "reason": "healthy", "proposed_path": []}
    bare_out = decide(None, bare, _snapshot(), healthy, run_missing_tests=False, noise_id="N0")
    assert bare_out["decision"] == "escalate" and "missing cause" in bare_out["reason"], bare_out
    bad_cause = decide(None, _noop_proposal("F3"), _snapshot(), healthy, run_missing_tests=False, noise_id="N0")
    assert bad_cause["decision"] == "escalate" and "not permitted" in bad_cause["reason"], bad_cause

    # Missing evidence: an errored no-op test refuses the no-op.
    for broken in (
        _errored(T1_OK),
        _errored(T4_OK, "ping dns"),
        _errored(T8_OK, "RuntimeError: nc missing"),
        _errored(T10_OK, "iperf timeout"),
        _errored(T10_OK, "timeout"),
        _errored(T10_OK, "empty_output"),
    ):
        rows = [broken if row["test_id"] == broken["test_id"] else row for row in healthy]
        assert not noop_justified(_snapshot(), rows, noise_id="N0"), broken
        gate_out = decide(None, noop, _snapshot(), rows, run_missing_tests=False, noise_id="N0")
        assert gate_out["decision"] == "escalate" and "test unavailable" in gate_out["reason"], gate_out
        assert decide_b5(None, noop, _snapshot(), rows, run_tests=False, noise_id="N0")["decision"] == "escalate"

    # Contradictory evidence: healthy OAM next to a broken snapshot.
    for snap in (_snapshot(loss=40.0), _snapshot(loss=100.0, connectivity=False)):
        assert not noop_justified(snap, healthy, noise_id="N0")
        assert decide(None, noop, snap, healthy, run_missing_tests=False, noise_id="N0")["decision"] == "escalate"
        assert decide_b5(None, noop, snap, healthy, run_tests=False, noise_id="N0")["decision"] == "escalate"
    # Snapshot without a loss number is not evidence of health by itself.
    blank = _snapshot()
    blank["metrics"].pop("packet_loss_percent")
    assert not noop_justified(blank, [], noise_id="N0")

    # Risky action: every test errored. No elimination, no allow, errors logged.
    proposal = good_proposal()
    lossy = _snapshot(loss=20.0)
    all_errored = [{"test_id": tid, "error": "RuntimeError: tc failed"} for tid in DEFAULT_ORDER]
    out = decide(None, copy.deepcopy(proposal), lossy, all_errored, run_missing_tests=False, noise_id="N0")
    # Missing evidence escalates: it neither confirms nor contradicts.
    assert out["decision"] == "escalate" and "insufficient evidence" in out["reason"], out
    log = out["log"]
    assert not log.get("eliminated"), log
    assert sorted(log.get("errored_tests") or []) == sorted(DEFAULT_ORDER), log
    assert not any(rule.startswith("eliminate") for rule in log.get("rules") or []), log

    assert out["tests_run"] == [], "errored rows must not appear as evidence"

    # Consistency scope = discriminating (config/gate.yaml, protocol §5).
    assert out["checks"]["consistency"]["scope"] == "discriminating"
    disc = out["checks"]["consistency"]["discriminating_tests"]
    assert disc, "F3 on a lossy snapshot must have a discriminating test"
    assert out["checks"]["discrimination"].get("vacuous") is not True
    tid = disc[0]
    from bench.outcomes import grade_result
    from bench.tests import HOPS, HOPS_BACKUP

    def _hops(spec, lossy_hop=None):
        return [
            {"src": src, "dst_ip": ip, "packet_loss_percent": 30.0 if src == lossy_hop else 0.0, "latency_ms": 10.0}
            for src, _node, ip in spec
        ]

    realistic = {
        "T2_hop_ping": {"test_id": "T2_hop_ping", "error": None, "path": "primary",
                        "hops_primary": _hops(HOPS, lossy_hop="r4"), "hops_backup": _hops(HOPS_BACKUP)},
        "T1_e2e_ping": dict(T1_OK, packet_loss_percent=20.0),
        "T4_mtu": dict(T4_OK),
    }
    assert tid in realistic, f"add a realistic row for discriminating test {tid}"
    observed = realistic[tid]
    realistic["T2_hop_ping"]["hops"] = realistic["T2_hop_ping"]["hops_primary"]
    # Partial T2 predictions (named hops only) fit the 80-character contract;
    # a full two-path T2 string does not.
    observed_token = {"T2_hop_ping": "primary:r4-hostB=loss", "T1_e2e_ping": grade_result(observed, None),
                      "T4_mtu": "df_ok"}[tid]
    alt_token = {"T2_hop_ping": "primary:r4-hostB=ok", "T1_e2e_ping": "ok", "T4_mtu": "mtu_fail"}[tid]
    assert observed_token != alt_token

    def with_predictions(preds):
        p = copy.deepcopy(proposal)
        p["causes"][0]["predicted_tests"] = preds
        return p

    # (1) A wrong side-prediction (not a discriminating test) cannot block.
    side = with_predictions([{"test_id": "T1_e2e_ping", "expected": "latency"}])
    side_out = decide(None, side, lossy, [T1_OK, observed], run_missing_tests=False, noise_id="N0")
    assert side_out["decision"] == "escalate", side_out
    assert "no prediction for discriminating test" in side_out["reason"], side_out
    # (2) No prediction for the discriminating test: missing evidence, escalate.
    assert tid in side_out["checks"]["consistency"]["missing_predictions"]
    # (2b) Schema-invalid prediction (missing expected key) → BLOCK, never authorize.
    # Distinct from (2)/(4): well-formed proposals missing selected evidence → ESCALATE.
    malformed = with_predictions([{"test_id": tid}])
    malformed_out = decide(None, malformed, lossy, [observed], run_missing_tests=False, noise_id="N0")
    assert malformed_out["decision"] == "block", malformed_out
    assert "invalid proposal" in malformed_out["reason"] and "missing expected" in malformed_out["reason"], malformed_out
    # (3) A contradicted prediction on the discriminating test: negative evidence, block.
    wrong = with_predictions([{"test_id": tid, "expected": alt_token}])
    wrong_out = decide(None, wrong, lossy, [observed], run_missing_tests=False, noise_id="N0")
    assert wrong_out["checks"]["consistency"]["predictions_checked"] == 1, wrong_out
    assert wrong_out["decision"] == "block" and wrong_out["reason"] == "consistency failed", wrong_out
    # (3b) The same test predicted correctly: consistency passes; the verdict
    # then rests on discrimination, never on side-predictions.
    right = with_predictions([{"test_id": tid, "expected": observed_token}, {"test_id": "T1_e2e_ping", "expected": "latency"}])
    right_out = decide(None, right, lossy, [observed, T1_OK], run_missing_tests=False, noise_id="N0")
    assert right_out["checks"]["consistency"]["passed"] is True, right_out
    assert right_out["decision"] != "block", right_out
    # (4) Discriminating prediction present but the test was not observed: escalate.
    unseen = with_predictions([{"test_id": tid, "expected": observed_token}])
    unseen_out = decide(None, unseen, lossy, [], run_missing_tests=False, noise_id="N0")
    assert unseen_out["decision"] == "escalate" and "not observed" in unseen_out["reason"], unseen_out
    # (5) The ablation scope=all keeps the old rule: a wrong side-prediction blocks.
    import importlib

    gd = importlib.import_module("gate.decide")
    real_load = gd.load_frozen_context

    def scope_all(noise_id="N0"):
        ok, errs, ctx = real_load(noise_id)
        return ok, errs, dict(ctx, gate=dict(ctx.get("gate") or {}, consistency_scope="all"))

    gd.load_frozen_context = scope_all
    try:
        legacy = decide(None, copy.deepcopy(proposal), lossy,
                        [T1_OK, {"test_id": "T7_route_check", "active": "primary", "error": None}],
                        run_missing_tests=False, noise_id="N0")
    finally:
        gd.load_frozen_context = real_load
    assert legacy["decision"] == "block" and legacy["reason"] == "consistency failed", legacy
    assert legacy["checks"]["consistency"]["scope"] == "all"
    # (6) No discriminating test at all (no relevant alternative): fail-safe escalate.
    healthy_snap = _snapshot()
    vacuous = decide(None, copy.deepcopy(proposal), healthy_snap, [T1_OK], run_missing_tests=False, noise_id="N0")
    assert vacuous["decision"] == "escalate" and "no discriminating test" in vacuous["reason"], vacuous
    assert vacuous["checks"]["discrimination"].get("vacuous") is True
    assert vacuous["checks"]["discrimination"].get("passed") is False
    # (6b) ALLOW stays reachable: a correct discriminating prediction plus
    # evidence that eliminates the alternative allows; contradicting evidence blocks.
    if tid == "T2_hop_ping":
        def _t2(lossy_hop, backup_hop=None):
            row = {"test_id": "T2_hop_ping", "error": None, "path": "primary",
                   "hops_primary": _hops(HOPS, lossy_hop=lossy_hop), "hops_backup": _hops(HOPS_BACKUP, lossy_hop=backup_hop)}
            row["hops"] = row["hops_primary"]
            return row

        f3_pred = with_predictions([{"test_id": "T2_hop_ping", "expected": "primary:r2-r4=loss"}])
        allowed = decide(None, f3_pred, lossy, [_t2("r2"), T1_OK], run_missing_tests=False, noise_id="N0")
        assert allowed["decision"] == "allow" and allowed["log"]["eliminated"], allowed
        refuted = decide(None, copy.deepcopy(f3_pred), lossy, [_t2("r4", backup_hop="r4"), T1_OK], run_missing_tests=False, noise_id="N0")
        assert refuted["decision"] == "block" and refuted["reason"] == "consistency failed", refuted
        # H6: a non-match with the alternative alone must not eliminate when the
        # observation does not support the primary hypothesis.
        unsupported = with_predictions([{"test_id": "T2_hop_ping", "expected": "primary:r1-r2=loss"}])
        odd = decide(None, unsupported, lossy, [_t2("r2"), T1_OK], run_missing_tests=False, noise_id="N0")
        assert not (odd.get("log") or {}).get("eliminated"), odd.get("log")
        comps = (odd.get("log") or {}).get("comparisons") or []
        assert comps and all(row.get("supports_primary") is False for row in comps), comps

    # (7) Candidate set + covering are independent of which cause is primary.
    # Swap primary with only that id listed as an AI cause (not both pre-listed).
    ok, _errs, frozen = load_frozen_context("N0")
    assert ok
    symptoms = _symptoms_from_snapshot(lossy)
    cands_a = _candidate_set("F3", ["F3"], symptoms, frozen["held_out"] or set())
    cands_b = _candidate_set("F11", ["F11"], symptoms, frozen["held_out"] or set())
    assert "F3" in cands_a and "F11" in cands_a
    assert "F3" in cands_b and "F11" in cands_b
    assert set(cands_a) == set(cands_b), (cands_a, cands_b)
    cover_a = covering_separating_tests(cands_a, fixes=frozen["fixes"], matrix=frozen["matrix"])
    cover_b = covering_separating_tests(cands_b, fixes=frozen["fixes"], matrix=frozen["matrix"])
    assert cover_a == cover_b and cover_a, (cover_a, cover_b)

    # (7b) B4 parity: same selected tests as VeriFix; only prediction source differs.
    from baselines.b4_llm_checker import decide as decide_b4

    verifix = decide(None, copy.deepcopy(proposal), lossy, [observed, T1_OK], run_missing_tests=False, noise_id="N0")
    b4 = decide_b4(copy.deepcopy(proposal), lossy, [observed, T1_OK], net=None, run_missing_tests=False, noise_id="N0")
    assert list(verifix.get("log", {}).get("selected_tests") or []) == list(b4.get("selected_tests") or []), (
        verifix.get("log", {}).get("selected_tests"),
        b4.get("selected_tests"),
    )

    # (8) The gate config must name a scope; a missing key fails closed.
    from gate import inputs as gin

    real_path = gin._GATE
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "gate.yaml"
        bad.write_text("frozen: true\nlearned_grader: false\nbudget_seconds: 8.0\n", encoding="utf-8")
        gin._GATE = bad
        try:
            _cfg, errs = gin.load_gate_config()
        finally:
            gin._GATE = real_path
        assert any("consistency_scope" in e for e in errs), errs
    b5 = decide_b5(None, copy.deepcopy(proposal), lossy, all_errored, run_tests=False, noise_id="N0")
    assert b5["decision"] != "allow", b5
    assert sorted(b5.get("errored_tests") or []) == sorted(DEFAULT_ORDER), b5

    # The same proposal with the errored rows removed reaches the same verdict:
    # an error row adds nothing that an absent test would not.
    absent = decide(None, copy.deepcopy(proposal), lossy, [], run_missing_tests=False, noise_id="N0")
    assert absent["decision"] == out["decision"], (absent["decision"], out["decision"])

    # Non-finite measurements are unavailable evidence, never mismatches/blocks.
    import math

    nan_rows = [
        {"test_id": "T1_e2e_ping", "packet_loss_percent": math.nan, "latency_ms": 10.0, "error": None},
    ]
    from gate.checks import check_consistency

    nan_cons = check_consistency(
        {"causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "ok"}]}]},
        nan_rows,
    )
    assert nan_cons["predictions_mismatched"] == 0, nan_cons
    assert nan_cons["predictions_unchecked"] == 1, nan_cons
    assert "unavailable" in (nan_cons["predictions"][0].get("reason") or ""), nan_cons

    # E2: errored observations are unchecked (unavailable), never mismatches.
    from gate.checks import check_consistency

    e2_proposal = with_predictions([{"test_id": "T1_e2e_ping", "expected": "ok"}])
    e2_cons = check_consistency(
        e2_proposal,
        [{"test_id": "T1_e2e_ping", "error": "timeout", "packet_loss_percent": 100.0}],
    )
    assert e2_cons["predictions_checked"] == 0 and e2_cons["predictions_mismatched"] == 0, e2_cons
    assert e2_cons["predictions_unchecked"] == 1 and e2_cons["passed"] is False, e2_cons
    assert e2_cons["predictions"][0]["status"] == "unchecked"
    assert "unavailable" in (e2_cons["predictions"][0].get("reason") or "")

    # E2: freshly executed errored tests escalate as unavailable, not block.
    from bench import tests as oam_mod

    saved_run = oam_mod.run_test
    fresh_calls = []

    def _fresh_error(net, tid, **kwargs):
        fresh_calls.append(tid)
        return {"test_id": tid, "error": "timeout"}

    oam_mod.run_test = _fresh_error
    try:
        fresh_out = decide(
            object(),
            copy.deepcopy(proposal),
            lossy,
            [],
            mode="full",
            run_missing_tests=True, noise_id="N0")
    finally:
        oam_mod.run_test = saved_run
    assert fresh_calls, "expected at least one fresh OAM run"
    assert fresh_out["decision"] == "escalate", fresh_out
    assert fresh_out["decision"] != "block"
    reason = fresh_out.get("reason") or ""
    assert "unavailable" in reason or "insufficient evidence" in reason, reason
    assert "consistency failed" not in reason
    log = fresh_out.get("log") or {}
    assert not any(
        row.get("status") == "mismatched"
        for row in ((fresh_out.get("checks") or {}).get("consistency") or {}).get("predictions") or []
    )

    # Missing loss/throughput is unavailable (not down/slow): no mismatch, no elimination.
    from bench.outcomes import grade_result, load_test_thresholds
    from bench.tests import HOPS, HOPS_BACKUP

    limits = load_test_thresholds("N0")
    assert (
        grade_result(
            {"test_id": "T1_e2e_ping", "packet_loss_percent": None, "latency_ms": 10.0, "error": None},
            limits,
        )
        is None
    )
    assert (
        grade_result({"test_id": "T10_speed", "mbit_per_sec": None, "error": None}, limits) is None
    )
    none_hop = {
        "test_id": "T2_hop_ping",
        "error": None,
        "path": "primary",
        "hops_primary": [
            {"src": "r1", "dst_ip": "10.0.12.2", "packet_loss_percent": None, "latency_ms": 10.0},
        ],
        "hops_backup": [
            {"src": "r1", "dst_ip": "10.0.13.2", "packet_loss_percent": 0.0, "latency_ms": 10.0},
        ],
    }
    assert grade_result(none_hop, limits) is None
    # Finite measurements and structured errors keep prior semantics.
    assert grade_result(
        {"test_id": "T1_e2e_ping", "packet_loss_percent": 100.0, "latency_ms": None, "error": None},
        limits,
    ) == "down"
    assert grade_result({"test_id": "T10_speed", "mbit_per_sec": 0.05, "error": None}, limits) == "slow"
    assert (
        grade_result(
            {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "error": "timeout"},
            limits,
        )
        is None
    )

    none_t1_cons = check_consistency(
        {"causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "ok"}]}]},
        [{"test_id": "T1_e2e_ping", "packet_loss_percent": None, "latency_ms": 10.0, "error": None}],
    )
    assert none_t1_cons["predictions_mismatched"] == 0, none_t1_cons
    assert none_t1_cons["predictions_unchecked"] == 1, none_t1_cons
    assert "unavailable" in (none_t1_cons["predictions"][0].get("reason") or ""), none_t1_cons

    # F1/F3: None-grade separators must not eliminate alternatives (mutation: None→eliminate).
    f1_none = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "f1",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "loss"}]}],
    }
    f1_none_out = decide(
        None,
        f1_none,
        lossy,
        [{"test_id": "T1_e2e_ping", "packet_loss_percent": None, "latency_ms": 12.0, "error": None}],
        run_missing_tests=False,
        noise_id="N0",
    )
    assert not (f1_none_out.get("log") or {}).get("eliminated"), f1_none_out.get("log")
    assert f1_none_out["decision"] != "allow", f1_none_out

    if tid == "T2_hop_ping":
        def _t2_none_loss():
            return {
                "test_id": "T2_hop_ping",
                "error": None,
                "path": "primary",
                "hops_primary": [
                    {
                        "src": src,
                        "dst_ip": ip,
                        "packet_loss_percent": None,
                        "latency_ms": 10.0,
                    }
                    for src, _node, ip in HOPS
                ],
                "hops_backup": [
                    {
                        "src": src,
                        "dst_ip": ip,
                        "packet_loss_percent": 0.0,
                        "latency_ms": 10.0,
                    }
                    for src, _node, ip in HOPS_BACKUP
                ],
            }

        f3_none = with_predictions([{"test_id": "T2_hop_ping", "expected": "primary:r2-r4=loss"}])
        f3_none_out = decide(
            None, f3_none, lossy, [_t2_none_loss(), T1_OK], run_missing_tests=False, noise_id="N0"
        )
        assert not (f3_none_out.get("log") or {}).get("eliminated"), f3_none_out.get("log")
        unavail = (f3_none_out.get("log") or {}).get("unavailable_tests") or []
        assert unavail or f3_none_out["decision"] in ("escalate", "block"), f3_none_out

    # Explicit noise_id: missing refuses; N0/N1/N2 load independently.
    missing_noise = decide(None, noop, _snapshot(), healthy, run_missing_tests=False)
    assert missing_noise["decision"] == "escalate" and "noise_id required" in missing_noise["reason"], missing_noise
    ok_none, errs_none, _ = load_frozen_context()
    assert ok_none is False and any("noise_id required" in e for e in errs_none), errs_none
    for level in ("N0", "N1", "N2"):
        ok_lvl, errs_lvl, frozen_lvl = load_frozen_context(level)
        assert ok_lvl, (level, errs_lvl)
        assert frozen_lvl["noise_id"] == level, frozen_lvl
        graded = decide(
            None, noop, _snapshot(), healthy, run_missing_tests=False, noise_id=level
        )
        assert graded["decision"] == "allow", (level, graded)

    print("evidence semantics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
