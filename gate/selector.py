"""Pick cheapest OAM test that separates candidate causes."""

import csv
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_MATRIX = _ROOT / "results" / "observability_matrix.csv"
_COSTS = _ROOT / "config" / "costs.yaml"
_FIXES = _ROOT / "config" / "correct_fixes.yaml"


def _load_costs():
    with _COSTS.open() as f:
        return yaml.safe_load(f)["tests"]


def _load_fixes():
    with _FIXES.open() as f:
        doc = yaml.safe_load(f)
    out = {}
    out.update(doc.get("faults", {}))
    out.update(doc.get("compounds", {}))
    return out


def _load_matrix():
    if not _MATRIX.exists():
        return []
    with _MATRIX.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def pick_cheapest_separating_test(fault_a, fault_b, already_ran=None):
    """Return test_id or None if no affordable separator exists."""
    already_ran = set(already_ran or [])
    costs = _load_costs()
    candidates = [t for t in separating_tests(fault_a, fault_b) if t not in already_ran]
    if not candidates:
        return None

    def cost(tid):
        key = tid if tid in costs else tid.replace("_", "_", 1)
        for k, v in costs.items():
            if k.startswith(tid.split("_")[0]):
                return v.get("time_seconds", 999)
        return 999

    candidates.sort(key=cost)
    return candidates[0]
