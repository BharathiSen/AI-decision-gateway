from gate.inputs import load_frozen_context, pipeline_interfaces, validate_proposal_inputs, validate_snapshot

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
print("gate inputs: PASS")
