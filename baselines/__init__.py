"""Experiment baselines B0–B5. One call shape for every method."""

_DECISIONS = {"allow", "block", "escalate"}


def _with_action(result, proposal):
    if "action" not in result:
        result = dict(result)
        result["action"] = (proposal or {}).get("action")
    return result


def run_method(method, proposal, snapshot, test_results, net=None):
    """Run one method on the shared proposal, snapshot, and OAM results.

    The true fault id is not an input. net is used only when B5 runs one
    more test. B0 does not change the network here.
    """
    name = (method or "").upper()
    proposal = proposal or {}
    snapshot = snapshot or {}
    test_results = list(test_results or [])

    if name == "B0":
        from baselines.b0_ungated import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B1":
        from baselines.b1_fixed_rules import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B2":
        from baselines.b2_llm_checker import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B3":
        from baselines.b3_ml_classifier import decide
        result = decide(proposal, snapshot, test_results)
    elif name == "B5":
        from baselines.b5_adaptive import decide_b5
        result = decide_b5(net, proposal, snapshot, test_results, run_tests=net is not None)
    else:
        raise ValueError(f"unknown method {method!r}")

    result = _with_action(result, proposal)
    if result.get("decision") not in _DECISIONS or "method" not in result or "action" not in result:
        raise ValueError(f"{name} returned an incomplete decision record: {result!r}")
    return result
