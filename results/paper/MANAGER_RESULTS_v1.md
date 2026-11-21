# VeriFix manager results (paper_v1)

> **Interim - Path B - pending label runs.** Confirmatory rates are unavailable until WSL multi-noise label repeats + `label_audit --write`. Do not treat this note as protocol-trusted.

Generated: 2026-11-21T15:15:43.826676+00:00

Sources only (no hand-typed rates):

- `results/experiment_summary_paper_v1_rescored.json`
- `results/paper/paper_v1_rescored_metrics.json`
- `results/paper/paper_v1_analysis_rescored/block_reasons.json`
- `results/paper/paper_v1_rescore_provenance.json`
- `results/pilot/paper_v1.json`

## Setup

- Topology: Mininet diamond (primary `r1-r2-r4`, backup `r1-r3-r4`) on **WSL Ubuntu-22.04** (not VirtualBox).
- Scenarios: 23 measured labels in `results/ground_truth.json` (sha256 `b19e33d2a09d01e498c94d452f28206c148034662b08489254c3e7f2f558e434`).
- Live batch: 300 episodes under `results/pilot/paper_v1_episodes/`.
- Trust path: **Path B** — Path B: VERIFIX_LABELS unset; decisions filled offline via compare_episode into /tmp cache; confirmatory denoms stay 0.

## Label trust (confirmatory vs descriptive)

- Metrics `slices.all.verifix`: `n_confirmatory=0`, `excluded_untrusted=300`, `n=300`
- Provenance: `verifix_n_confirmatory=0`, `metrics_scored=300`
- Summary `paper_confirmatory_blocked`: `n_trusted_scenarios=0`, `n_untrusted_scenarios=23`
- Block-reason labels: trusted=`0`, untrusted=`23`
- Paper primary VeriFix rates tagged `descriptive_measured_labels_untrusted` — **not confirmatory**
- `paper_label_caveat`: Confirmatory rates unavailable: results/ground_truth.json has no trusted=true scenarios (label_trust False for all). score_arm blanks GT flags when untrusted. These rates apply the measured expected_actions formula descriptively over all scored episodes; they are NOT confirmatory.

## Main table (VeriFix descriptive measured-label rates)

From `paper_primary.verifix` in `results/experiment_summary_paper_v1_rescored.json`:

| Metric | Value |
|--------|-------|
| n | 300 |
| unsafe_allow | 10/300 = 0.03333333333333333 |
| false_allow | 10/300 = 0.03333333333333333 |
| correct_allow | 58/152 = 0.3815789473684211 |
| unnecessary_escalation | 29/152 = 0.19078947368421054 |
| allow_rate_raw | 0.22666666666666666 |
| block_rate_raw | 0.3433333333333333 |
| escalate_rate_raw | 0.43 |
| test_cost_seconds_mean | 16.826133333333342 |
| mean_tests_run | 6.98 |
| recovery_seconds_mean | 7.739599518518518 |
| n_correct_proposal | 152 |
| label | descriptive_measured_labels_untrusted |

Caveat (`paper_primary.verifix.note`): Confirmatory rates unavailable: results/ground_truth.json has no trusted=true scenarios (label_trust False for all). score_arm blanks GT flags when untrusted. These rates apply the measured expected_actions formula descriptively over all scored episodes; they are NOT confirmatory.

## Why ~34% blocks (blocked-but-correct)

From `block_reasons.json` (measured `correct_actions`, not catalog):

- `n_episodes_scanned=300`
- `n_verifix_block=103`
- `n_blocked_correct_measured_label=65`
- `reason_counts_blocked_correct`: `{"consistency failed": 46, "safety failed": 19}`
- `consistency_mismatched_tests_blocked_correct`: `{"T2_hop_ping": 35, "T1_e2e_ping": 14}`
- Assessment: consistency=`intended when agent predictions disagree with OAM (gate doing its job)`; safety=`agent often proposes host-inclusive paths; policy allows only [r1,r2,r4] or [r1,r3,r4]. Count as product gap (path shape), not invented proof.`; `fix_applied=False`

## What works

- Offline demo: `python bench/accept.py --demo` → allow / block / escalate (`results/acceptance/demo_offline/`).
- Gate does not execute on block/escalate; execution only after recheck.

## Known limits (honesty)

- **Path B / untrusted labels:** confirmatory safety stays null until WSL label repeats (≥3×N0/N1/N2) + `label_audit --write`.
- **Pilot:** episodes_attempted=300, episodes_scored=252, invalid_output=44, model_errors=4 (`results/pilot/paper_v1.json`).
- **Mid-batch model mix** (episode `_model` recount): gemini-3.5-flash-lite=235, openai/gpt-oss-20b=42, gemini-3.1-flash-lite=7, gemini-3.6-flash=7, unknown=7, gemini-3.8-flash=2. Freeze agent field: `model:gemini/gemini-3.8-flash`.
- **Live accept:** `sudo python3 bench/accept.py --live-accept` needs interactive sudo in WSL.
- **Rescore provenance note:** Path B: VERIFIX_LABELS unset; decisions filled offline via compare_episode into /tmp cache; confirmatory denoms stay 0.

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
