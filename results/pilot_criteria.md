# Phase 9 pilot — frozen go / no-go criteria

**Frozen before pilot runs.** Do not change thresholds after experiments begin.

## Scope

- ~500 episodes: normal mode, stress modes, ≥30% healthy (H0)
- Methods compared: **Gate v0** (consistency + safety) vs **B5** (adaptive testing, no AI predictions)
- Same fault catalog, test catalog, and time budget per episode

## Metrics

| Metric | Definition |
|--------|------------|
| Wrong fix rate | Proposals where action ∉ `correct_fixes.yaml` for true fault |
| Unsafe allow | Gate allows a fix that is not in correct set for true fault |
| Prediction accuracy | When diagnosis correct, fraction of predicted tests matching observed |
| Test cost | Sum of `config/costs.yaml` time_seconds for tests executed |
| Escalation rate | Fraction of risky proposals escalated |

## Go / no-go

**Continue main track (full gate + paper RQ3)** if ALL:

1. Wrong or unnecessary LLM fixes in **≥15%** of non-H0 normal episodes, **and**
2. Gate v0 beats B5 on **unsafe allow rate** OR **test cost** (paired, same episodes, p<0.05 bootstrap or CI non-overlap)

**Pivot to measurement study (RQ1/RQ2 lead)** if:

- Gate v0 unsafe allow ≈ B5 (difference <2 percentage points) on ≥400 paired episodes

**Stress-led track** if:

- Normal-mode wrong fix rate <5% but stress-mode wrong fix rate ≥25%

## Recording

Results written to `results/pilot/summary.json` by `bench/run_pilot.py`.
