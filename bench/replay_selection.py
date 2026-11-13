"""Offline paired replay of selection arms over saved episodes (research T2/T4).

Smoke only under test thresholds. Not confirmatory evidence: trusted labels and
accepted research thresholds are still required before paper numbers.

    python bench/replay_selection.py --self-check
    python bench/replay_selection.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.contracts import refuse_protected_path
from bench.outcomes import using_test_thresholds
from bench.selection_policies import (
    ARM_NAMES,
    measured_seconds,
    select_tests,
)
from gate import selector
from gate.decide import decide

ROOT = Path(__file__).resolve().parent.parent
SMOKE_OUT = ROOT / "results" / "acceptance" / "fix_aware_smoke.json"
DEFAULT_EPISODE_DIRS = (
    ROOT / "results" / "pilot" / "v4_episodes",
    ROOT / "results" / "pilot" / "v3b_episodes",
    ROOT / "results" / "pilot" / "v3_episodes",
    ROOT / "results" / "episodes",
)

# Arms that share production D' relevance; only the selected test set differs.
DECISION_ARMS = ("FA-D'", "FA-current", "ALL-PAIRS", "FULL", "NONE")


def load_episodes(dirs=None):
    dirs = dirs or DEFAULT_EPISODE_DIRS
    rows = []
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if not data.get("agent_proposal") or not data.get("network_snapshot"):
                continue
            data["_path"] = str(path)
            rows.append(data)
    return rows


def _primary_and_action(proposal):
    proposal = proposal or {}
    causes = proposal.get("causes") or []
    primary = None
    if causes and isinstance(causes[0], dict):
        primary = causes[0].get("cause_id")
    return primary, proposal.get("action")


def _candidates(proposal, fault_id=None):
    ids = []
    for cause in (proposal or {}).get("causes") or []:
        cid = cause.get("cause_id") if isinstance(cause, dict) else None
        if isinstance(cid, str) and cid:
            ids.append(cid)
    if fault_id and fault_id not in ids:
        ids.append(fault_id)
    # Keep stable unique order
    out = []
    for cid in ids:
        if cid not in out:
            out.append(cid)
    return out


def filter_oam(oam_results, selected_ids):
    want = set(selected_ids or [])
    return [row for row in (oam_results or []) if isinstance(row, dict) and row.get("test_id") in want]


def decide_with_selection(net, proposal, snapshot, oam_results, selected_ids, noise_id="N0"):
    """Run production decide() but freeze covering separators to `selected_ids`."""
    selected_ids = list(selected_ids or [])
    real = selector.covering_separating_tests

    def _forced(candidates, fixes=None, matrix=None):
        return list(selected_ids)

    selector.covering_separating_tests = _forced
    try:
        return decide(
            net,
            proposal,
            snapshot,
            oam_results,
            mode="full",
            run_missing_tests=False,
            noise_id=noise_id,
        )
    finally:
        selector.covering_separating_tests = real


def replay_episode(record, arms=None, decision_arms=None):
    proposal = record.get("agent_proposal") or {}
    snapshot = record.get("network_snapshot") or {}
    oam = ((record.get("oam") or {}).get("results")) or []
    noise_id = record.get("noise_id") or "N0"
    fault_id = record.get("compound_id") or record.get("fault_id")
    primary, action = _primary_and_action(proposal)
    cands = _candidates(proposal, fault_id=fault_id)
    arms = arms or ARM_NAMES
    decision_arms = set(decision_arms or DECISION_ARMS)
    out = {
        "episode_id": record.get("episode_id"),
        "fault_id": fault_id,
        "noise_id": noise_id,
        "primary": primary,
        "action": action,
        "arms": {},
    }
    if not primary or not action:
        out["skip"] = "missing primary or action"
        return out
    fa_k = len(
        select_tests(
            "FA-D(a)",
            candidates=cands,
            primary=primary,
            action=action,
        )
    )
    for arm in arms:
        selected = select_tests(
            arm,
            candidates=cands,
            primary=primary,
            action=action,
            k=fa_k,
        )
        cost = measured_seconds(selected, oam)
        row = {
            "selected_tests": selected,
            "test_count": cost["test_count"],
            "seconds": cost["seconds"],
            "fallback_cost_rows": cost["fallback_cost_rows"],
        }
        if arm in decision_arms:
            filtered = filter_oam(oam, selected)
            try:
                decision = decide_with_selection(
                    None, proposal, snapshot, filtered, selected, noise_id=noise_id
                )
                row["decision"] = decision.get("decision")
                row["reason"] = decision.get("reason")
            except Exception as exc:  # smoke must not abort the batch
                row["decision"] = "error"
                row["reason"] = f"{type(exc).__name__}: {exc}"
        else:
            row["decision"] = None
            row["reason"] = "selection-cost only (D(a) decision rule not wired into production decide)"
        out["arms"][arm] = row
    return out


def summarize(replays):
    by_arm = defaultdict(lambda: {"n": 0, "tests": 0.0, "seconds": 0.0, "decisions": defaultdict(int)})
    for rep in replays:
        if rep.get("skip"):
            continue
        for arm, row in (rep.get("arms") or {}).items():
            bucket = by_arm[arm]
            bucket["n"] += 1
            bucket["tests"] += float(row.get("test_count") or 0)
            bucket["seconds"] += float(row.get("seconds") or 0)
            dec = row.get("decision")
            if dec:
                bucket["decisions"][dec] += 1
    summary = {}
    for arm, bucket in by_arm.items():
        n = max(bucket["n"], 1)
        summary[arm] = {
            "episodes": bucket["n"],
            "mean_tests": bucket["tests"] / n,
            "mean_seconds": bucket["seconds"] / n,
            "decisions": dict(bucket["decisions"]),
        }
    return summary


@using_test_thresholds
def run_smoke(dirs=None, out_path=None):
    refuse_protected_path(out_path or SMOKE_OUT)
    episodes = load_episodes(dirs)
    replays = [replay_episode(ep) for ep in episodes]
    summary = summarize(replays)
    payload = {
        "status": "smoke_only_not_evidence",
        "disclaimer": (
            "Test thresholds only. Trusted labels are incomplete and "
            "accepted_by_noise_level is empty. Do not cite these rates as results."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episodes),
        "decision_arms": list(DECISION_ARMS),
        "summary": summary,
        "replays": replays,
    }
    path = Path(out_path or SMOKE_OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return payload


def _self_check():
    from bench.selection_policies import _self_check as policies_check

    assert policies_check() == 0
    episodes = load_episodes()
    assert episodes, "need saved episodes for smoke"
    sample = episodes[0]
    rep = replay_episode(sample, arms=("FA-D(a)", "FA-current", "ALL-PAIRS", "NONE"))
    assert "arms" in rep and "FA-D(a)" in rep["arms"]
    fa = rep["arms"]["FA-D(a)"]["test_count"]
    allp = rep["arms"]["ALL-PAIRS"]["test_count"]
    none = rep["arms"]["NONE"]["test_count"]
    assert none == 0
    assert fa <= allp or allp == 0
    payload = run_smoke()
    assert payload["episode_count"] >= 1
    assert Path(SMOKE_OUT).exists()
    print(f"episodes: {payload['episode_count']}")
    for arm, row in sorted((payload.get("summary") or {}).items()):
        print(f"  {arm}: mean_tests={row['mean_tests']:.2f} decisions={row.get('decisions')}")
    print("replay selection: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.smoke:
        payload = run_smoke()
        print(f"wrote {SMOKE_OUT} episodes={payload['episode_count']}")
        print("status: smoke_only_not_evidence")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
