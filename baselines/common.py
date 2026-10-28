"""Shared baseline contract: permitted actions and comparable test budgets."""

from bench.tests import DEFAULT_ORDER
from gate.selector import _load_costs

METHODS = ("B0", "B1", "B2", "B3", "B4", "B5")
PERMITTED_ACTIONS = ("REROUTE", "DO_NOTHING", "ESCALATE")
# B3 information cap. Typical VeriFix selected cost is T5+T6+T7+T4+T1 = 6.5 s.
# B2 is the uncapped full battery (~26.5 s). B4/B5 use the VeriFix selector.
DEFAULT_BUDGET_SECONDS = 8.0
BACKUP_PATH = ["r1", "r3", "r4"]


def test_cost(test_id):
    costs = _load_costs()
    if test_id in costs:
        return float(costs[test_id].get("time_seconds", 0) or 0)
    prefix = (test_id or "").split("_")[0]
    for key, spec in costs.items():
        if key.startswith(prefix):
            return float(spec.get("time_seconds", 0) or 0)
    return 0.0


def used_seconds(test_ids):
    return sum(test_cost(tid) for tid in (test_ids or []) if tid)


def cheapest_within_budget(test_ids, budget_seconds=None):
    """Greedy cheapest-first fill that never exceeds the remaining budget."""
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    ordered = sorted((tid for tid in (test_ids or []) if tid), key=lambda tid: (test_cost(tid), tid))
    chosen = []
    used = 0.0
    for tid in ordered:
        cost = test_cost(tid)
        if used + cost <= cap + 1e-9:
            chosen.append(tid)
            used += cost
    return chosen, used


def budgeted_catalog(budget_seconds=None):
    """Permitted OAM tests that fit in the B3 information budget, cheapest first."""
    chosen, _used = cheapest_within_budget(DEFAULT_ORDER, budget_seconds)
    return chosen


def attach_contract(result, proposal, budget_seconds=None):
    """Every method record carries the same action, tests_run, and budget fields."""
    cap = DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    result = dict(result)
    if "action" not in result:
        result["action"] = (proposal or {}).get("action")
    ids = result.get("tests_run")
    if ids is None:
        ids = []
    elif isinstance(ids, set):
        ids = list(ids)
    result["tests_run"] = [tid for tid in ids if tid]
    used = used_seconds(result["tests_run"])
    result["permitted_actions"] = list(PERMITTED_ACTIONS)
    result["budget"] = {
        "budget_seconds": cap,
        "tests_used_seconds": used,
        "within_budget": used <= cap + 1e-9,
    }
    return result
