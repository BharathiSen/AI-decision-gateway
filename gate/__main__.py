from bench.outcomes import test_thresholds
from gate.inputs import load_frozen_context, pipeline_interfaces, validate_proposal_inputs, validate_snapshot

# Offline module check grades with test-only limits; research mode refuses
# unaccepted calibration and is not used here.
with test_thresholds():
    ok, errors, context = load_frozen_context("N0")
    if not ok:
        raise SystemExit("frozen inputs: " + "; ".join(errors))
    stages = pipeline_interfaces()
    assert set(stages) >= {
        "candidate_generation",
        "test_selection",
        "outcome_grading",
        "safety_checking",
        "consistency_checking",
        "discrimination_checking",
    }
    assert validate_snapshot({})
    assert validate_proposal_inputs(["not", "a", "proposal"])
    assert context["matrix"] and context["catalog"] and context["costs"] and context["thresholds"]
    cfg = context.get("gate") or {}
    assert cfg.get("frozen") is True and cfg.get("learned_grader") is False
    assert cfg.get("budget_seconds") == 8.0
    # Test cost lookup: exact ids, whole T<n> numbers, independent of yaml order.
    from gate import selector as _selector

    _costs = context["costs"]
    for _tid, _entry in _costs.items():
        assert _selector.test_cost(_tid, _costs) == _entry["time_seconds"], _tid
    _reversed = dict(reversed(list(_costs.items())))
    _t1, _t10 = _costs["T1_e2e_ping"]["time_seconds"], _costs["T10_speed"]["time_seconds"]
    assert _t1 != _t10
    assert _selector.test_cost("T1_e2e_ping", _reversed) == _t1 and _selector.test_cost("T1", _reversed) == _t1
    assert _selector.test_cost("T10", _costs) == _t10
    assert _selector.test_cost("T3_unknown", _costs) == 999
print("gate inputs: PASS")
