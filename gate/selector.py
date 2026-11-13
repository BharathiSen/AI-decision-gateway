"""Pick cheapest OAM test that separates candidate causes."""

import csv
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_MATRIX = _ROOT / "results" / "observability_matrix.csv"
_COSTS = _ROOT / "config" / "costs.yaml"
_FIXES = _ROOT / "config" / "correct_fixes.yaml"


def held_out_ids():
    """Faults and compounds marked split: held_out. The gate never sees their labels or matrix rows."""
    out = set()
    for name, key in (("faults.yaml", "faults"), ("compounds.yaml", "compounds")):
        with (_ROOT / "config" / name).open() as f:
            out |= {item["id"] for item in yaml.safe_load(f)[key] if item.get("split") == "held_out"}
    return out


def _load_costs():
    with _COSTS.open() as f:
        return yaml.safe_load(f)["tests"]


def _load_fixes():
    with _FIXES.open() as f:
        doc = yaml.safe_load(f)
    out = {}
    out.update(doc.get("faults", {}))
    out.update(doc.get("compounds", {}))
    hidden = held_out_ids()
    return {k: v for k, v in out.items() if k not in hidden}


def _load_matrix():
    if not _MATRIX.exists():
        return []
    hidden = held_out_ids()
    with _MATRIX.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("fault_id") not in hidden]


def needs_different_fixes(fault_a, fault_b, fixes=None):
    fixes = fixes or _load_fixes()
    fa = set(fixes.get(fault_a, {}).get("correct", []))
    fb = set(fixes.get(fault_b, {}).get("correct", []))
    return fa != fb and bool(fa) and bool(fb)


def separating_tests(fault_a, fault_b, matrix=None):
    matrix = matrix or _load_matrix()
    tests = []
    for row in matrix:
        if row["fault_id"] == fault_a:
            ta = row["test_id"]
            oa = row["outcome"]
            ob = next(
                (r["outcome"] for r in matrix if r["fault_id"] == fault_b and r["test_id"] == ta),
                None,
            )
            if ob is not None and oa != ob:
                tests.append(ta)
    return tests


def pick_cheapest_separating_test(fault_a, fault_b, already_ran=None, matrix=None):
    """Return test_id or None if no affordable separator exists."""
    already_ran = set(already_ran or [])
    costs = _load_costs()
    candidates = [t for t in separating_tests(fault_a, fault_b, matrix=matrix) if t not in already_ran]
    if not candidates:
        return None

    candidates.sort(key=lambda tid: test_cost(tid, costs))
    return candidates[0]


def test_cost(tid, costs):
    """time_seconds for a test id: exact key, else the key with the same T<n> number, else 999.

    The number is compared whole, so T1 never takes T10's cost (the earlier
    prefix match depended on T1 being listed before T10 in costs.yaml).
    """
    entry = costs.get(tid)
    if entry is None:
        number = tid.split("_")[0]
        entry = next((v for k, v in costs.items() if k.split("_")[0] == number), None)
    if entry is None:
        return 999
    return entry.get("time_seconds", 999)


def covering_separating_tests(candidates, fixes=None, matrix=None):
    """Cheapest matrix separators covering all candidate pairs with different fixes.

    Selection is model-independent: the agent cause id is only a set member.
    Predictions must not be consulted before this set is frozen.
    """
    fixes = fixes or _load_fixes()
    ids = sorted({cid for cid in (candidates or []) if isinstance(cid, str) and cid})
    chosen = []
    pairs = []
    for i, fault_a in enumerate(ids):
        for fault_b in ids[i + 1 :]:
            if needs_different_fixes(fault_a, fault_b, fixes=fixes):
                pairs.append((fault_a, fault_b))
    for fault_a, fault_b in pairs:
        tid = pick_cheapest_separating_test(fault_a, fault_b, already_ran=chosen, matrix=matrix)
        if tid and tid not in chosen:
            chosen.append(tid)
    return chosen
