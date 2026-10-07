"""Gate ablations: no predictions, no discrimination, no grader."""

from gate.decide import decide


def gate_no_predictions(net, proposal, snapshot, test_results, **kw):
    p = dict(proposal)
    p["causes"] = []
    return decide(net, p, snapshot, test_results, mode="pilot_v0", **kw)


def gate_no_discrimination(net, proposal, snapshot, test_results, **kw):
    return decide(net, proposal, snapshot, test_results, mode="pilot_v0", **kw)


def gate_full(net, proposal, snapshot, test_results, **kw):
    return decide(net, proposal, snapshot, test_results, mode="full", **kw)
