"""Compare a fault × test matrix with config/expected_signatures.yaml."""

from pathlib import Path

import yaml

from bench.outcomes import healthy_outcome

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "expected_signatures.yaml"

# Traffic counts as recovered when these all read healthy: ping loss and
# latency, a full-size DF packet, TCP next to ping, and throughput.
RECOVERY_TESTS = ("T1_e2e_ping", "T4_mtu", "T8_tcp_vs_ping", "T10_speed")
REMEDIATIONS = ("REROUTE", "REBALANCE")


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


def degraded_tests(outcomes, tests=RECOVERY_TESTS):
    """Recovery tests that did not read healthy. A missing test counts as degraded."""
    return [tid for tid in tests if outcomes.get(tid) != healthy_outcome(tid)]


def valid_fix_set(actions):
    """Measured valid fixes for one scenario, or None if a needed measurement failed.

    actions maps DO_NOTHING and each remediation to {"recovered": bool|None}.
    If traffic is fine without a change, only DO_NOTHING is valid, because
    any change is unnecessary. Otherwise the valid fixes are the remediations
    that restored traffic, and ESCALATE if none did.
    """
    baseline = (actions.get("DO_NOTHING") or {}).get("recovered")
    if baseline is None:
        return None
    if baseline:
        return ["DO_NOTHING"]
    measured = [(name, (actions.get(name) or {}).get("recovered")) for name in REMEDIATIONS]
    if any(ok is None for _, ok in measured):
        return None
    return [name for name, ok in measured if ok] or ["ESCALATE"]


def fix_pairs(by_fault, fixes):
    """Scenario pairs with no fix in common, and the tests that tell them apart.

    by_fault is {scenario: {test_id: outcome}}. A pair with no separating
    test is inseparable: the evidence cannot pick a fix, so ESCALATE is the
    expected safe action.
    """
    ids = [sid for sid in by_fault if fixes.get(sid)]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if set(fixes[a]) & set(fixes[b]):
                continue
            separating = [
                tid for tid, outcome in by_fault[a].items()
                if tid in by_fault[b] and by_fault[b][tid] != outcome
            ]
            pairs.append({
                "fault_a": a,
                "fault_b": b,
                "fixes_a": list(fixes[a]),
                "fixes_b": list(fixes[b]),
                "different_fixes": True,
                "separating_tests": separating,
                "inseparable": not separating,
                "expected_action": None if separating else "ESCALATE",
            })
    return pairs


def expected_actions(fixes, pairs):
    """Measured fixes per scenario, with ESCALATE for any scenario in an inseparable pair."""
    expected = {sid: list(v) for sid, v in fixes.items() if v}
    for pair in pairs:
        if pair["inseparable"]:
            expected[pair["fault_a"]] = ["ESCALATE"]
            expected[pair["fault_b"]] = ["ESCALATE"]
    return expected
