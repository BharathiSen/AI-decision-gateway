"""Fix-aware and baseline OAM selection policies for offline replay (research T1).

Does not modify gate/selector.py. FA-current calls the production cover so
historical gate behaviour remains an evaluable arm.

    python bench/selection_policies.py
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.selector import (
    covering_separating_tests,
    needs_different_fixes,
    separating_tests,
    test_cost,
    _load_costs,
    _load_fixes,
    _load_matrix,
)

ARM_NAMES = (
    "FA-D(a)",
    "FA-D'",
    "FA-current",
    "ALL-PAIRS",
    "INFO-GAIN",
    "FULL",
    "RAND-k",
    "CHEAP-k",
    "NONE",
)

FULL_TEST_IDS = (
    "T1_e2e_ping",
    "T2_hop_ping",
    "T4_mtu",
    "T5_if_errors",
    "T6_queue_stats",
    "T7_route_check",
    "T8_tcp_vs_ping",
    "T9_oneway_loss",
    "T10_speed",
)


def fix_set(fault_id, fixes=None):
    fixes = fixes or _load_fixes()
    return set((fixes.get(fault_id) or {}).get("correct") or [])


def action_wrong_if(fault_id, action, fixes=None):
    """True when executing `action` is wrong if `fault_id` is the true cause."""
    allowed = fix_set(fault_id, fixes=fixes)
    if not allowed:
        return True
    return action not in allowed


def d_of_a(candidates, action, fixes=None):
    """D(a) = { c in K : a not in F(c) }."""
    return [c for c in candidates if action_wrong_if(c, action, fixes=fixes)]


def d_prime(candidates, primary, fixes=None):
    """D' = { c in K : F(c) != F(primary) } (production relevance)."""
    return [
        c
        for c in candidates
        if c != primary and needs_different_fixes(primary, c, fixes=fixes)
    ]


def correct_greedy_cover(pairs, matrix=None, costs=None):
    """Cheapest cover that reuses already-chosen separators (fixes the FA-current defect)."""
    matrix = matrix if matrix is not None else _load_matrix()
    costs = costs if costs is not None else _load_costs()
    chosen = []
    for fault_a, fault_b in pairs:
        seps = separating_tests(fault_a, fault_b, matrix=matrix)
        if any(tid in chosen for tid in seps):
            continue
        if not seps:
            continue
        seps = sorted(seps, key=lambda tid: test_cost(tid, costs))
        chosen.append(seps[0])
    return chosen


def pairs_vs_primary(primary, alternatives):
    return [(primary, alt) for alt in alternatives if alt != primary]


def all_pairs(candidates):
    ids = sorted({c for c in candidates if isinstance(c, str) and c})
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            out.append((a, b))
    return out


def select_tests(arm, *, candidates, primary, action, matrix=None, fixes=None, costs=None, k=None, rng=None):
    """Return the ordered test id list for one selection arm."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm}")
    matrix = matrix if matrix is not None else _load_matrix()
    fixes = fixes if fixes is not None else _load_fixes()
    costs = costs if costs is not None else _load_costs()
    cands = [c for c in (candidates or []) if isinstance(c, str) and c]
    if arm == "NONE":
        return []
    if arm == "FULL":
        return list(FULL_TEST_IDS)
    if arm == "FA-current":
        return covering_separating_tests(cands, fixes=fixes, matrix=matrix)
    if arm == "FA-D(a)":
        alts = d_of_a(cands, action, fixes=fixes)
        return correct_greedy_cover(pairs_vs_primary(primary, alts), matrix=matrix, costs=costs)
    if arm == "FA-D'":
        alts = d_prime(cands, primary, fixes=fixes)
        return correct_greedy_cover(pairs_vs_primary(primary, alts), matrix=matrix, costs=costs)
    if arm == "ALL-PAIRS":
        return correct_greedy_cover(all_pairs(cands), matrix=matrix, costs=costs)
    if arm == "INFO-GAIN":
        return _info_gain(cands, matrix=matrix, costs=costs, budget=k)
    if arm in ("RAND-k", "CHEAP-k"):
        if k is None:
            k = len(
                select_tests(
                    "FA-D(a)",
                    candidates=cands,
                    primary=primary,
                    action=action,
                    matrix=matrix,
                    fixes=fixes,
                    costs=costs,
                )
            )
        pool = list(FULL_TEST_IDS)
        if arm == "CHEAP-k":
            pool.sort(key=lambda tid: test_cost(tid, costs))
            return pool[: max(0, int(k))]
        rng = rng or random.Random(0)
        if k <= 0:
            return []
        if k >= len(pool):
            return list(pool)
        return rng.sample(pool, int(k))
    raise ValueError(arm)


def _outcomes_by_fault(matrix):
    by = defaultdict(dict)
    for row in matrix:
        by[row["fault_id"]][row["test_id"]] = row["outcome"]
    return by


def _info_gain(candidates, matrix=None, costs=None, budget=None):
    """Sequential max expected entropy reduction over uniform prior on candidates."""
    matrix = matrix if matrix is not None else _load_matrix()
    costs = costs if costs is not None else _load_costs()
    remaining = [c for c in candidates if isinstance(c, str) and c]
    if len(remaining) <= 1:
        return []
    by = _outcomes_by_fault(matrix)
    chosen = []
    available = [t for t in FULL_TEST_IDS if t]
    limit = len(FULL_TEST_IDS) if budget is None else max(0, int(budget))
    while len(remaining) > 1 and len(chosen) < limit and available:
        best = None
        best_score = -1.0
        for tid in available:
            buckets = defaultdict(list)
            for c in remaining:
                buckets[by.get(c, {}).get(tid, "?")].append(c)
            # expected posterior entropy
            n = float(len(remaining))
            ent = 0.0
            for group in buckets.values():
                p = len(group) / n
                if p > 0:
                    ent -= p * math.log(max(len(group), 1) / n)
            # prefer lower residual entropy; break ties by cost
            score = -ent - 1e-6 * test_cost(tid, costs)
            if score > best_score:
                best_score = score
                best = tid
        if best is None:
            break
        chosen.append(best)
        available.remove(best)
        # collapse to one outcome bucket under a uniform random observation model:
        # keep all hypotheses (selection only); INFO-GAIN arm still returns the sequence.
        # Decision-time elimination uses real observations in replay.
        # Shrink remaining only when a test partitions and we pretend modal outcome.
        buckets = defaultdict(list)
        for c in remaining:
            buckets[by.get(c, {}).get(best, "?")].append(c)
        # For selection planning without an observation, stop after k steps matched to FA.
        if budget is not None and len(chosen) >= limit:
            break
        # Without live observation, do not shrink remaining (pure planning sequence).
        if budget is None and len(chosen) >= max(1, len(remaining) - 1):
            break
    return chosen


def measured_seconds(test_ids, oam_results):
    """Sum measured duration_seconds for selected tests when present."""
    by_id = {
        row.get("test_id"): row
        for row in (oam_results or [])
        if isinstance(row, dict) and row.get("test_id")
    }
    total = 0.0
    missing = 0
    for tid in test_ids:
        row = by_id.get(tid) or {}
        dur = row.get("duration_seconds")
        if isinstance(dur, (int, float)):
            total += float(dur)
        else:
            missing += 1
            total += float(test_cost(tid, _load_costs()))
    return {"seconds": total, "fallback_cost_rows": missing, "test_count": len(test_ids)}


def _self_check():
    fixes = _load_fixes()
    matrix = _load_matrix()
    assert matrix, "observability matrix required"
    cands = ["F1", "F3", "F12", "H0"]
    # D(a) for REROUTE excludes causes that allow REROUTE; includes F12/H0.
    d_a = d_of_a(cands, "REROUTE", fixes=fixes)
    assert "F12" in d_a and "H0" in d_a
    assert "F1" not in d_a and "F3" not in d_a
    # Correct cover must not exceed FA-current length on the same D' pairs.
    primary = "F3"
    alts = d_prime(cands, primary, fixes=fixes)
    pairs = pairs_vs_primary(primary, alts)
    fixed = correct_greedy_cover(pairs, matrix=matrix)
    current = covering_separating_tests([primary] + alts, fixes=fixes, matrix=matrix)
    assert len(fixed) <= len(current), (fixed, current)
    # Defect illustration: already-chosen tests should be reusable.
    # If current picked something, fixed should still separate every pair.
    for a, b in pairs:
        seps = set(separating_tests(a, b, matrix=matrix))
        assert not seps or any(t in fixed for t in seps) or not seps, (a, b, fixed, seps)
    for arm in ARM_NAMES:
        selected = select_tests(
            arm, candidates=cands, primary="F3", action="REROUTE", matrix=matrix, fixes=fixes, k=2
        )
        assert isinstance(selected, list), arm
        assert all(isinstance(t, str) for t in selected), (arm, selected)
    print("selection policies: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
