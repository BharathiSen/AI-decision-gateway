"""Write results/paper/MANAGER_RESULTS_v1.md from command-emitted JSON only.

Interim Path B note: confirmatory rates stay unavailable until label runs.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERIM_BANNER = (
    "> **Interim - Path B - pending label runs.** Confirmatory rates are unavailable "
    "until WSL multi-noise label repeats + `label_audit --write`. "
    "Do not treat this note as protocol-trusted."
)


def rate(obj, key):
    x = obj[key]
    return f"{x['count']}/{x['denom']} = {x['rate']}"


def main() -> int:
    summary = json.loads((ROOT / "results/experiment_summary_paper_v1_rescored.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "results/paper/paper_v1_rescored_metrics.json").read_text(encoding="utf-8"))
    pilot = json.loads((ROOT / "results/pilot/paper_v1.json").read_text(encoding="utf-8"))
    blocks = json.loads(
        (ROOT / "results/paper/paper_v1_analysis_rescored/block_reasons.json").read_text(encoding="utf-8")
    )
    prov_path = ROOT / "results/paper/paper_v1_rescore_provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}

    v = summary["paper_primary"]["verifix"]
    vf = (metrics.get("slices") or {}).get("all", {}).get("verifix") or {}
    blocked = summary.get("paper_confirmatory_blocked") or {}
    assess = blocks.get("assessment") or {}
    labels = blocks.get("labels") or {}

    counts: Counter[str] = Counter()
    for path in (ROOT / "results/pilot/paper_v1_episodes").glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        ap = rec.get("agent_proposal") or {}
        counts[str(ap.get("_model") or ap.get("model") or "unknown")] += 1
    mix = ", ".join(f"{k}={n}" for k, n in counts.most_common())

    md = f"""# VeriFix manager results (paper_v1)

{INTERIM_BANNER}

Generated: {datetime.now(timezone.utc).isoformat()}

Sources only (no hand-typed rates):

- `results/experiment_summary_paper_v1_rescored.json`
- `results/paper/paper_v1_rescored_metrics.json`
- `results/paper/paper_v1_analysis_rescored/block_reasons.json`
- `results/paper/paper_v1_rescore_provenance.json`
- `results/pilot/paper_v1.json`

## Setup

- Topology: Mininet diamond (primary `r1-r2-r4`, backup `r1-r3-r4`) on **WSL Ubuntu-22.04** (not VirtualBox).
- Scenarios: 23 measured labels in `results/ground_truth.json` (sha256 `{prov.get('ground_truth_sha256') or (blocked.get('label_provenance') or {}).get('sha256')}`).
- Live batch: 300 episodes under `results/pilot/paper_v1_episodes/`.
- Trust path: **Path B** — {prov.get('note') or 'pending label runs; VERIFIX_LABELS unset; confirmatory denoms 0.'}

## Label trust (confirmatory vs descriptive)

- Metrics `slices.all.verifix`: `n_confirmatory={vf.get('n_confirmatory')}`, `excluded_untrusted={vf.get('excluded_untrusted')}`, `n={vf.get('n')}`
- Provenance: `verifix_n_confirmatory={prov.get('verifix_n_confirmatory')}`, `metrics_scored={prov.get('metrics_scored')}`
- Summary `paper_confirmatory_blocked`: `n_trusted_scenarios={blocked.get('n_trusted_scenarios')}`, `n_untrusted_scenarios={blocked.get('n_untrusted_scenarios')}`
- Block-reason labels: trusted=`{len(labels.get('trusted_scenarios') or [])}`, untrusted=`{len(labels.get('untrusted_scenarios') or [])}`
- Paper primary VeriFix rates tagged `{v.get('label')}` — **not confirmatory**
- `paper_label_caveat`: {summary.get('paper_label_caveat')}

## Main table (VeriFix descriptive measured-label rates)

From `paper_primary.verifix` in `results/experiment_summary_paper_v1_rescored.json`:

| Metric | Value |
|--------|-------|
| n | {v['n']} |
| unsafe_allow | {rate(v, 'unsafe_allow')} |
| false_allow | {rate(v, 'false_allow')} |
| correct_allow | {rate(v, 'correct_allow')} |
| unnecessary_escalation | {rate(v, 'unnecessary_escalation')} |
| allow_rate_raw | {v['allow_rate_raw']} |
| block_rate_raw | {v['block_rate_raw']} |
| escalate_rate_raw | {v['escalate_rate_raw']} |
| test_cost_seconds_mean | {v['test_cost_seconds_mean']} |
| mean_tests_run | {v['mean_tests_run']} |
| recovery_seconds_mean | {v['recovery_seconds_mean']} |
| n_correct_proposal | {v['n_correct_proposal']} |
| label | {v.get('label')} |

Caveat (`paper_primary.verifix.note`): {v.get('note')}

## Why ~34% blocks (blocked-but-correct)

From `block_reasons.json` (measured `correct_actions`, not catalog):

- `n_episodes_scanned={blocks.get('n_episodes_scanned')}`
- `n_verifix_block={blocks['n_verifix_block']}`
- `n_blocked_correct_measured_label={blocks['n_blocked_correct_measured_label']}`
- `reason_counts_blocked_correct`: `{json.dumps(blocks.get('reason_counts_blocked_correct'))}`
- `consistency_mismatched_tests_blocked_correct`: `{json.dumps(blocks.get('consistency_mismatched_tests_blocked_correct'))}`
- Assessment: consistency=`{assess.get('consistency_failed')}`; safety=`{assess.get('safety_failed_host_inclusive_path')}`; `fix_applied={assess.get('fix_applied')}`

## What works

- Offline demo: `python bench/accept.py --demo` → allow / block / escalate (`results/acceptance/demo_offline/`).
- Gate does not execute on block/escalate; execution only after recheck.

## Known limits (honesty)

- **Path B / untrusted labels:** confirmatory safety stays null until WSL label repeats (≥3×N0/N1/N2) + `label_audit --write`.
- **Pilot:** episodes_attempted={pilot.get('episodes_attempted')}, episodes_scored={pilot.get('episodes_scored')}, invalid_output={pilot.get('invalid_output')}, model_errors={pilot.get('model_errors')} (`results/pilot/paper_v1.json`).
- **Mid-batch model mix** (episode `_model` recount): {mix}. Freeze agent field: `{pilot.get('agent')}`.
- **Live accept:** `sudo python3 bench/accept.py --live-accept` needs interactive sudo in WSL.
- **Rescore provenance note:** {prov.get('note')}

## How to recompute

```powershell
python bench/accept.py --demo
python scripts/write_manager_results.py
```

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Users/Admin/Projects/AI-decision-gateway
python3 -m bench.label_audit results/ground_truth.json
```
"""
    out = ROOT / "results/paper/MANAGER_RESULTS_v1.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
