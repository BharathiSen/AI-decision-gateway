# VeriFix experiment protocol (v1 — research freeze)

**Status: frozen for campaign registration.** This document, together with
`config/gate.yaml`, the measured consensus label file (when produced), and the
git commit / worktree tree id recorded in every run manifest, define the
pre-registered study. Offline remediation (gate safety, measurement taxonomy,
calibration integrity, recovery measurement, experiment integrity) is complete
on the host. **No claim is made that the hypothesis is proven, that the study
is publication-ready, or that offline simulation substitutes for Mininet
campaigns.**

Freeze means: do not change decision semantics, label methodology, splits,
primary metrics, or baseline contracts after held-out episode collection
begins. Any such change requires a new protocol version and a new `--out`.

Companion readiness index: not present in the tracked tree (do not invent a
new document). Calibration acceptance and campaign gates are defined in §8
of this protocol.

---

## 1. Frozen hypothesis and research questions

### Core hypothesis (H1)

AI-predicted OAM expectations, used by the VeriFix gate to choose and
interpret discriminating tests, improve the **safety–verification-cost
tradeoff** relative to a **non-AI adaptive OAM baseline** (B5), and relative
to the **same gate with catalog expectations** (B4), on held-out scenarios.

Primary isolation: **VeriFix vs B4** (same gate, same covering separators,
same thresholds, budget, evidence rules, and stopping rules; only the source
of expected outcomes differs — AI predictions vs development matrix).

Second primary comparison: **VeriFix vs B5** (non-AI adaptive).

### Null (H0)

On held-out scenarios under the pre-registered tests in §7, VeriFix does not
reduce unsafe authorization (false ALLOW) relative to B4 or B5 at equal or
lower verification cost, or any safety gain appears only with higher cost.

### Research questions

| ID | Question | Primary evidence |
|---|---|---|
| RQ1 | Do AI predictions reduce unsafe ALLOW vs B4 at equal/lower cost? | Held-out McNemar + cost |
| RQ2 | Do AI predictions improve the safety–cost tradeoff vs B5? | Held-out McNemar + cost |
| RQ3 | What is the contribution of consistency / discrimination / grader? | Ablations (descriptive) |
| RQ4 | How often do prediction mismatches and infra failures drive ESCALATE/BLOCK? | Prediction + infra rates |

Do not treat development-split exploratory numbers as confirmatory.

---

## 2. Baseline definitions (as implemented)

Every arm sees the same proposal, snapshot, and shared OAM battery for an
episode (paired design). **No arm receives the fault id or evaluation label.**
Action space for scoring: `{REROUTE, DO_NOTHING, ESCALATE}`. Shared information
budget for newly executed OAM: `DEFAULT_BUDGET_SECONDS = 8.0`
(`baselines/common.py`, `config/gate.yaml`).

| Arm | Implementation | Role |
|---|---|---|
| **VeriFix** (`gate.decide` / `gate_full`) | Gate with AI `predicted_tests`, covering separators, safety, discrimination, discriminating-scope consistency | Treatment |
| **B4** | Same `gate.decide`; primary expected outcomes filled from the development observability matrix. **Not an LLM.** | Primary control (isolates prediction source) |
| **B5** | Non-AI adaptive: covering separators when available; positive-support separation; same noise thresholds and DO_NOTHING cause-permit rule as VeriFix | Primary control (non-AI adaptive) |
| **B3** | Catalog nearest-neighbour on cheapest-first tests within the 8 s budget; episode noise thresholds. **Not a trained ML model.** | Non-AI reference |
| **B2** | Full OAM battery then fixed rules (may exceed 8 s live) | Full-OAM cost upper bound |
| **B1** | Fixed loss-then-reroute rules on telemetry; no tests | Minimal-OAM reference |
| **B0** | Ungated execute of the proposal; harm + **independent post-action recovery battery** (same T1/T4/T8/T10 recovery rule as VeriFix) | Safety lower bound |
| Ablations | `gate_no_predictions`, `gate_no_discrimination`, `gate_no_grader`, `gate_v0` | Component contribution |

**Parity requirements (H5):** equivalent noise thresholds (`noise_id`),
equivalent evidence availability on the shared battery, equivalent recovery
measurement opportunity after authorized execution (VeriFix episode path and
B0 live path), equivalent scoring labels (`VERIFIX_LABELS` / consensus),
equivalent action semantics, and the documented 8 s budget (B2 exempt as the
full-battery reference).

---

## 3. Episode design and experimental controls

### Episode

One episode = `(scenario, noise_id, seed, repeat)`. Live path: reset → inject →
telemetry → full OAM battery → agent proposal → gate → authorized execute only
on bound ALLOW after recheck → **independent post-remediation OAM** → teardown.
Failed reset/cleanup **halts** the campaign (`TopologyContaminated`); partial
records are preserved; the topology is not reused.

### Independent variables

| Factor | Levels |
|---|---|
| Strategy | VeriFix, B0–B5, ablations |
| Scenario | Single fault / compound / H0 |
| Split | development / held-out |
| Noise | N0, N1, N2 (`config/noise.yaml`) |
| Seed | Inject/agent sampling seeds recorded in the manifest |

### Noise

| Level | Intent |
|---|---|
| N0 | No intentional background noise |
| N1 | Moderate background traffic (2 Mbit) + jitter |
| N2 | High background traffic (5 Mbit) + larger jitter |

Thresholds are noise-specific. Research grading uses only
`accepted_by_noise_level` in `config/thresholds.yaml` after
`--write-thresholds`. `grading_from_report` is **false**: withdrawn values in
`results/calibration_report.json` never grade.

**T8 policy (fixed, not fitted):** `tcp_must_succeed: true` at N0, N1, and N2
(`config/thresholds.yaml` `policy_by_noise_level`). A TCP connect failure is a
fault signal at every noise level, including N2 background traffic.

### Model / decoding (per campaign)

Record in the run manifest from `config/models.yaml` and the live agent
config: provider, model id, temperature, `max_tokens`, any seed, prompt
version / SHA-256 (`prompt_version`, `prompt_sha256` on episodes). One
provider/model/effort per campaign. Changing the model starts a new campaign
`--out`.

### Exclusion rules

- Held-out ids (U1–U5, C4, C5) never enter tuning, candidate training, grader
  training, or threshold writes.
- Errored OAM rows are **missing evidence**, never healthy grades.
- Untrusted labels are reported separately and excluded from primary rates.
- Invalid proposals score as non-executable (cannot be false ALLOW; remain in
  attempt denominators).
- If calibration acceptance fails for a noise level (§8), that noise level is
  refused entirely for research grading and new campaigns (whole-level
  fail-closed). There is no T10-only exclusion carve-out.

---

## 4. Development and held-out scenarios

Defined in `config/faults.yaml` and `config/compounds.yaml`.

| Split | Scenarios |
|---|---|
| Baseline / healthy | H0 |
| Development | F1–F12, C1–C3 |
| Held-out | U1–U5, C4, C5 |

Held-out compounds share no component pair with C1–C3. Held-out faults never
enter the gate candidate set or B3 signature table.

---

## 5. Ground-truth generation methodology

Labels are **measured recovery**, independent of the gate catalog and of the
agent’s proposed action.

1. `bench/verify_fault_observability.py` injects each scenario, runs
   DO_NOTHING / REROUTE / REBALANCE, and records recovery with T1, T4, T8
   healthy and T10 ≥ 4 Mbit (`bench/signature_checks.py`).
2. Repeat at N0, N1, N2 with distinct seeds; combine with
   `python -m bench.label_audit`.
3. A label is **trusted** only with ≥ 3 agreeing clean runs at each noise
   level. Disagreement → ESCALATE fallback. Signature equal to H0 → NO_EFFECT
   (failed injection), not a trusted DO_NOTHING label.
4. Label action space: `{REROUTE, DO_NOTHING, ESCALATE}`. REBALANCE-only
   measured fixes map to ESCALATE for evaluation.
5. `config/correct_fixes.yaml` is **gate knowledge only**, never the evaluation
   ground truth. Point campaigns at consensus via `VERIFIX_LABELS`.

Command success is never recovery: episode `recovery_status` /
`traffic_recovered` come from post-action OAM (`bench/run_episode.py`).

---

## 6. Sample-size and repeated-trial rationale

| Split | Design (floors) | Count |
|---|---|---|
| Development | 16 scenarios × 3 noise × 3 seeds | 144 episodes |
| Held-out | 7 scenarios × 3 noise × 5 seeds | 105 episodes |

These are **minimum collection floors**, not a completed power analysis. Exact
two-sided McNemar cannot reach p < 0.05 with fewer than 6 discordant
configurations; Holm adjustment is over the two primary pairs (vs B4, vs B5).
Repeats of one `(scenario, noise, seed)` are clustered to one configuration
outcome for paired tests. Additional repeats may be added under the same
protocol version without changing metrics or splits; they cannot rescue a
post-hoc metric change.

---

## 7. Metrics and statistical analysis plan

Denominators are explicit. A zero denominator is reported as undefined
(`null`), never as 0. Implementation: `bench/metrics.py`, recovery fields on
episode records, agent quality in `score_agent_quality`.

### Required metrics

| Metric | Definition | Implementation notes |
|---|---|---|
| Unsafe authorization rate | ALLOW ∧ evaluated action ∉ trusted label | `false_allow` / `unsafe_action`; denom = episodes attempted |
| False-block rate | BLOCK ∧ action ∈ label | `correct_blocked`; denom = episodes with correct proposal |
| Escalation rate | ESCALATE (and unnecessary escalation when action ∈ label) | attempted / correct-proposal denoms |
| Verification cost | Σ catalog (and measured) seconds of **selected** tests | `test_cost_seconds`; mean and p95 |
| OAM test count | Number of selected tests | per episode |
| Additional bytes | Sum of probe `bytes` when reported | `oam_cost` / probe totals; null if unknown |
| Decision latency | Gate wall time; agent latency separate | `gate_cost`, `agent_cost` |
| Actual recovery success | Post-action traffic recovered | `traffic_recovered`, `recovery_status` (`recovered` / `partial` / `failed` / `measurement_unavailable` / `not_attempted`) |
| Recovery time | Remediation start → post-OAM completion when status is `recovered` | `recovery_time_seconds` |
| Prediction consistency | Matched vs checked primary-cause predictions on selected tests | `prediction_accuracy`, mismatch → consistency BLOCK |
| Infrastructure failure rate | Episodes with infra/measurement failure classes | `infra_error` episodes; OAM rows with `error`; calibration inclusive FAR |

Also report: correct-ALLOW rate; B0 harm-before-rollback; teardown/contamination
halts.

### Statistics

- Per arm and slice: rate with **Wilson 95% CI**.
- Primary test: exact two-sided **McNemar** on discordant configurations for
  false-ALLOW, VeriFix vs B4 and VeriFix vs B5, **Holm-adjusted**, α = 0.05.
- All other comparisons are **descriptive** (no confirmatory p-values).
- Cost: paired mean difference with percentile bootstrap CI over
  configurations (implement before claiming cost superiority; do not invent
  numbers).
- Slices: known / held-out / compound / noise / stress; report null and
  negative results.

---

## 8. Provenance, calibration order, and acceptance

### Calibration order (pre-registered)

1. Collect healthy H0 samples at N0/N1/N2 into a **new** episode-aligned
   samples path (`results/calibration_samples_episode.jsonl` or a dated
   sibling). Fault ids are never injected for calibration.
2. Fit per noise level (`bench/calibrate_healthy.py`): data-only quantiles;
   copy T8 policy from `policy_by_noise_level`; no withdrawn floors.
3. Score a held-out healthy eval set; compute
   `eval_error_inclusive_false_alarm_rate` and first-try T10 completeness.
4. Write a **new** `--report` path (the committed
   `results/calibration_report.json` is protected and refused).
5. Only when `ready_to_write` passes: `--write-thresholds` into
   `accepted_by_noise_level`. Live runners preflight-refuse unaccepted levels.

### Fitting constants

| Constant | Value | Role |
|---|---|---|
| Fit quantile ladder | p95/p05 → p99/p01 → p995/p005 → max/min | smallest upper quantile meeting FAR target on the fit set |
| Latency/loss multiplier | × 1.1 on fitted upper quantile | slack above empirical healthy |
| Loss resolution floor | one ping of the episode battery | never finer than measurement grain |
| `false_alarm_rate_target` | 0.05 | write-decision ceiling |
| `FIRST_TRY_T10_TARGET` | 0.95 | first-try T10 completeness; backfill does not count |
| `MIN_CALIBRATION_RUNS` / `MIN_EVAL_RUNS` | 200 / 50 | acceptance floors |
| T8 | `tcp_must_succeed: true` at N0–N2 | fixed policy, not fitted |

### Acceptance rules

Per noise level, `acceptance_errors` / `ready_to_write` require: exact
`noise_id`; full episode-aligned battery; calibration/eval run floors;
`used_evaluation_faults` false; write-decision metric
`eval_error_inclusive_false_alarm_rate` ≤ 0.05 (finite); first-try T10
completeness ≥ 0.95; complete finite T1/T2/T8/T10 threshold set. Partial or
NaN threshold dictionaries refuse grading (never fall open to DO_NOTHING).

### Provenance hashes

Every campaign output must record:

- `git_commit` and clean worktree tree id (runners refuse dirty/untracked
  source for live provenance paths).
- Config fingerprint (`bench/run_batch.provenance` / pilot manifest).
- Protocol version string (`v1`) and path to this file.
- `prompt_version` / `prompt_sha256`.
- Model provider, model id, temperature, max tokens, decoding seed if any.
- `noise_id`, scenario ids, inject seeds, repeat index.
- Label file path and SHA-256 (`labels` block in summary).
- Gate config: `consistency_scope`, `budget_seconds`, `learned_grader`,
  `min_grade_confidence`.
- Calibration acceptance binding: `acceptance_sha256` over exact `noise_id`,
  acceptance evidence fields, `report_sha256`, `samples_sha256`, and the
  written threshold set (`bench.outcomes.acceptance_binding_hash`).
- Halt artifacts (`*.halt.json`) on contamination or fatal agent errors.

### Test-only thresholds

`bench/test_thresholds.yaml` is an offline fixture only. It is reachable solely
inside `test_thresholds()` / `@using_test_thresholds` (context-local depth).
Research grading, live preflight, and `--write-thresholds` never read it.
Test mode must restore on exit; a fresh process without the fixture refuses
research grading when no accepted level exists.

Protected paths (`results/calibration_report.json`,
`results/pilot/summary.json`, `results/observability_matrix.csv`, etc.) must
not be overwritten from the host; new campaigns use new `--out` directories.
Resume refuses identity mismatch after code/prompt/label/gate changes.

---

## 9. Reproducibility instructions

### Offline (Windows host — no Mininet, no API required for these checks)

```powershell
python -m gate.evidence_check
python bench/run_episode.py
python bench/tests.py
python bench/calibrate_healthy.py
python -c "from baselines import _self_check; _self_check(); print('baselines: PASS')"
```

### Live (Ubuntu Mininet VM, root, clean commit)

| Step | Command | Output |
|---|---|---|
| 0 offline | `sudo python3 bench/run_pilot.py --self-check` (+ host offline suite) | PASS required |
| 1 fault validity | `sudo python3 bench/validate_faults.py --out results/fault_validation/<run-id>.json` | inject/reset cycles × N0/N1/N2 |
| 1b calibration | follow §8 order: collect → fit → new `--report` → `--write-thresholds` only if `ready_to_write` | accepted_by_noise_level + binding hashes |
| 2 labels | `sudo python3 bench/verify_fault_observability.py --out-dir results/labels/<N>-s<seed> --noise <N> --seed <seed>` | matrix + ground_truth per run |
| 3 consensus | `python3 -m bench.label_audit results/labels/* --write results/labels/consensus.json` | trusted/untrusted |
| 4 development | `VERIFIX_LABELS=… sudo --preserve-env=GROQ_API_KEY,VERIFIX_LABELS python3 bench/run_pilot.py --require-model --save-episodes … --out results/campaign/dev-<run-id>.json` | episodes + summary |
| 5 freeze commit | commit protocol + gate config + labels; record commit | — |
| 6 held-out | as step 4 with held-out ids → `heldout-<run-id>.json` | — |
| 7 arms | `python3 bench/run_experiments.py --episodes-dir … --out …_arms.json` | paired decisions |
| 8 metrics | `python3 bench/metrics.py …`; `python3 bench/analyze.py …` | CIs, McNemar, tables |

Exit codes: 2 = refused (provenance / protected path / unaccepted calibration);
3 = fatal agent error (fix, then `--resume`); 4 = reset/cleanup failed
(`sudo mn -c`, then `--resume`). Never edit state files by hand.

---

## 10. Decision semantics (regression: `python -m gate.evidence_check`)

| Evidence | Meaning | Gate outcome |
|---|---|---|
| Positive | Measured row supports the primary prediction | can support ALLOW / elimination |
| Negative | Measured row contradicts a checked prediction, or safety fails | BLOCK |
| Missing | Not run, unavailable, or row `error` (infra/tool) | ESCALATE; never eliminates |
| Contradictory | Healthy no-op tests vs broken snapshot | ESCALATE |
| Outside catalog / inseparable / unaffordable | — | ESCALATE |

**Alternative elimination (H6):** eliminate an alternative only when the
observation **supports the primary** and **contradicts the alternative**. A
non-match with the alternative alone is not proof. Grader confidence without
a trained reliability model is 1.0 on exact support else 0.0.

**Consistency scope:** `discriminating` (`config/gate.yaml`). Ablation
`consistency_scope: all` and `pilot_v0` remain available but are not primary.

ALLOW requires: valid proposal, safety pass, action in the catalog fix set for
the primary cause, every relevant alternative eliminated (full mode), and
consistency rules. Execution only after binding + state recheck
(`apply_authorized_action`).

---

## 11. Known limitations and threats to validity

1. Primary topology is one small diamond; larger generated grids are not the
   episode runner’s default.
2. One LLM provider/model per campaign; no cross-model generalization claim.
3. Verification cost for paired arms is largely counterfactual from a shared
   battery (stationarity assumption over the battery duration).
4. Labels and recovery use emulated OAM (T1/T4/T8/T10), not production user
   traffic.
5. The gate’s frozen observability matrix is development data; matrix error
   affects VeriFix, B4, and B5 similarly but still biases absolute rates.
6. Healthy calibration on the VM has historically failed FAR / T10
   completeness targets; thresholds must not be written until acceptance
   passes — offline logic fixes do not invent VM numbers.
7. Post-remediation recovery may race link convergence (no mandatory settle
   equal to the matrix’s settle in the episode path); infra failures yield
   `measurement_unavailable`, not success.
8. Offline host checks are not Mininet proof. Publication claims require
   completed live campaigns under this freeze.
9. Prior Phase 9 go/no-go text in `results/pilot_criteria.md` is historical;
   this protocol governs the main pre-registered campaign.

---

## 12. What this freeze does *not* claim

- That H1 is confirmed or rejected.
- That offline PASS implies external validity.
- That historical pilot or calibration JSON/CSV files are the confirmatory
  dataset (they remain protected artifacts; new campaigns write new paths).
