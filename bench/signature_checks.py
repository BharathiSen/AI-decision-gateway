"""Compare a fault × test matrix with config/expected_signatures.yaml."""

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "expected_signatures.yaml"


def load_expected(path=None):
    with Path(path or _CONFIG).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def outcome_map(rows):
    """rows: list of {fault_id, test_id, outcome} -> {fault_id: {test_id: outcome}}."""
    by_fault = {}
    for row in rows:
        by_fault.setdefault(row["fault_id"], {})[row["test_id"]] = row["outcome"]
    return by_fault


def separation_errors(rows, spec=None):
    """Return human-readable failures for required fault differences.

    A fault with required=true must differ from H0 on at least one test in
    differs_from_healthy. Each pair must differ on at least one separating test.
    """
    spec = spec or load_expected()
    by_fault = outcome_map(rows)
    healthy = by_fault.get("H0", {})
    errors = []

    for fault_id, info in (spec.get("faults") or {}).items():
        if not info or not info.get("required"):
            continue
        if fault_id not in by_fault:
            errors.append(f"{fault_id}: missing from matrix")
            continue
        wanted = info.get("differs_from_healthy") or []
        if not wanted:
            continue
        changed = [
            tid
            for tid in wanted
            if by_fault[fault_id].get(tid) != healthy.get(tid)
            and by_fault[fault_id].get(tid) is not None
            and healthy.get(tid) is not None
        ]
        if not changed:
            errors.append(
                f"{fault_id}: expected a difference from H0 on {wanted}, "
                f"got { {t: by_fault[fault_id].get(t) for t in wanted} }"
            )

    for pair in spec.get("pairs") or []:
        a, b = pair["fault_a"], pair["fault_b"]
        tests = pair.get("separating_tests") or []
        if a not in by_fault or b not in by_fault:
            errors.append(f"{a} vs {b}: missing fault in matrix")
            continue
        separated = [
            tid
            for tid in tests
            if by_fault[a].get(tid) is not None
            and by_fault[b].get(tid) is not None
            and by_fault[a][tid] != by_fault[b][tid]
        ]
        if not separated:
            errors.append(
                f"{a} vs {b}: not separated by {tests} "
                f"({a}={ {t: by_fault[a].get(t) for t in tests} }, "
                f"{b}={ {t: by_fault[b].get(t) for t in tests} })"
            )
    return errors
