# AI Decision Gateway

Research prototype for **verification-gated** LLM-proposed network fixes. An AI agent **proposes** remediation; an independent **gate** verifies proposals with operations/administration/maintenance (OAM) tests and structured predictions; **control** applies allowed changes only on an emulated network.

Paper track: *Fix-Aware Evidence Selection for Authorizing AI-Proposed Network Remediation* (see private `docs/` research package).

## Linux host (Mininet)

Live network work runs on **WSL Ubuntu-22.04** with Mininet (system Python, as root). This project does **not** use VirtualBox or a separate VM image for Linux.

Typical layout on this machine:

- Windows checkout: `C:\Users\Admin\Projects\AI-decision-gateway`
- Same tree in WSL: `/mnt/c/Users/Admin/Projects/AI-decision-gateway` (and/or `~/AI-decision-gateway`)
- Enter WSL: `wsl -d Ubuntu-22.04`, then `cd` to the repo and run `sudo python3 …`

Offline self-checks and `python bench/accept.py --demo` run on Windows without Mininet. Threshold acceptance hashes that point at `/mnt/c/...` paths are valid when graded **inside WSL**, not from native Windows Python.

## Architecture

```text
Telemetry ──► LLM agent (propose only) ──► Gate (verify) ──► Control (execute)
     ▲                                              │
     └──────────── Mininet diamond topology ◄───────┘
```

- **Topology:** `hostA → r1 → r2 → r4 → hostB` (primary) and `hostA → r1 → r3 → r4 → hostB` (backup).
- **Separation of powers:** `agent/` cannot import `network.controller` or fault injectors; only `control/` executes fixes.

## Repository layout

| Path | Purpose |
|------|---------|
| [`network/`](network/) | Mininet topology, link control, interactive fault injection |
| [`telemetry/`](telemetry/) | Live metrics/topology JSON for agent and gate |
| [`agent/`](agent/) | Groq LLM proposer (xAI Grok / OpenRouter / Gemini optional) + JSON schema validation |
| [`gate/`](gate/) | Consistency, discrimination, safety; adaptive test selection |
| [`control/`](control/) | Canonical actions (`REROUTE`, `DO_NOTHING`, `ESCALATE`, …) |
| [`bench/`](bench/) | OAM tests T1–T10, episodes, pilot, calibration, observability matrix |
| [`baselines/`](baselines/) | B0–B5 comparators and gate ablations |
| [`config/`](config/) | Fault catalog, costs, thresholds, ground-truth correct fixes |
| [`policy/`](policy/) | Operator policy context for the LLM |
| [`results/`](results/) | Pilot metrics, episodes, calibration, observability matrix |
| [`paper/`](paper/) | Figure generation from saved results |

## Requirements

### Everywhere (agent + replay)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...     # optional; stub agent used if unset
python agent/decision_agent.py  # one live Groq call: prints latency + tokens
```

### WSL Ubuntu + Mininet (live network + benchmarks)

Install [Mininet](http://mininet.org/) inside **WSL Ubuntu-22.04** (system Python, not a venv). Run live scripts as **root** in that distro — not VirtualBox:

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Users/Admin/Projects/AI-decision-gateway
sudo python3 main.py                    # interactive CLI
sudo python3 bench/run_pilot.py --save-episodes --episodes 500 --out results/pilot/<run-name>.json
```

Windows can develop offline; live / label / calibration numbers require WSL + Mininet + sudo.

## Quick start (results path)

### 1. Offline product demo (Windows, no Mininet)

Three scenarios (authorized allow, unsafe block, insufficient-evidence escalate).
Uses test thresholds only — labeled `test_mode_simulation`, not live traffic proof:

```powershell
python bench/accept.py --demo
```

Expect `demo offline: PASS`. Records: `results/acceptance/demo_offline/`.

### 1b. paper_v1 results note

Quoted only from command-emitted JSON (Path B = untrusted labels, confirmatory denoms 0):
[`results/paper/paper_v1_results_note.md`](results/paper/paper_v1_results_note.md)
(regenerate with `python scripts/write_results_note.py`).

Re-score frozen episodes inside **WSL** (thresholds hash `/mnt/c/...` paths; do not invent rates):

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Users/Admin/Projects/AI-decision-gateway
unset VERIFIX_LABELS
python3 bench/metrics.py --episodes-dir results/pilot/paper_v1_episodes --out results/paper/paper_v1_rescored_metrics.json
python3 bench/analyze.py --from results/pilot/paper_v1_episodes --out-dir results/paper/paper_v1_analysis_rescored
python3 bench/run_experiments.py --episodes-dir results/pilot/paper_v1_episodes --out results/experiment_summary_paper_v1_rescored.json
python3 bench/labels.py && python3 bench/metrics.py --self-check && python3 bench/analyze.py --self-check
```

### 1c. Live accept (WSL + sudo + Mininet)

```bash
sudo python3 bench/accept.py --live-accept
```

Requires interactive sudo and accepted research thresholds. Do not fake PASS if sudo is unavailable.

### 2. Interactive demo (WSL + Mininet)

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Users/Admin/Projects/AI-decision-gateway
sudo python3 main.py
```

Useful CLI commands: `status`, `fail_primary`, `restore all`, `reroute backup`, `propose`.

### 3. Populate results (one command)

**Offline simulation** (no Mininet — dev/CI; uses stub agent + observability matrix):

```bash
python scripts/finish_evaluation.py --mode offline --episodes 500
```

**Live pilot** (paper-grade; WSL Ubuntu-22.04 + Mininet + root — not VirtualBox):

```bash
wsl -d Ubuntu-22.04
cd /mnt/c/Users/Admin/Projects/AI-decision-gateway
sudo --preserve-env=GROQ_API_KEY,GEMINI_API_KEY,LLM_PROVIDER python3 scripts/finish_evaluation.py --mode live --run-name v5 --episodes 500
```

Live mode requires root and `--run-name`. Outputs go to new, run-specific paths; preserved evidence
(`results/pilot/summary.json`, `results/experiment_summary.json`, `results/episodes/`, calibration, matrix,
ground truth) is never overwritten (`bench.contracts.refuse_protected_path`):

- live: `results/pilot/<run-name>.json`, `results/pilot/<run-name>_episodes/`, `results/experiment_summary_<run-name>.json`
- offline: `results/offline_episodes/`, `results/offline_experiment_summary.json` (simulation, not live evidence)
- `paper/figures/` — regenerated from results (skip with `--skip-figures`)

Pilot success criteria are frozen in [`results/pilot_criteria.md`](results/pilot_criteria.md).

### 3. Tests without Mininet

Offline self-checks (no root, no network, no model calls). Each exits non-zero on failure. They do not
write preserved results. Passing them is not live validation: route, OAM and rollback behaviour on real
kernels is only checked under WSL Ubuntu-22.04 with Mininet (not VirtualBox).

```bash
python agent/test_validate.py
python bench/tests.py
python bench/run_episode.py
python bench/calibrate_healthy.py
python bench/test_inject.py
python -m gate
python -m baselines
python config/validate.py
python bench/test_outcomes.py
python bench/labels.py
python bench/label_audit.py
python bench/cleanup_check.py
python gate/evidence_check.py
python -m control
python telemetry/collector.py
python bench/scenarios.py --self-check
python bench/metrics.py --self-check
python bench/run_batch.py --self-check
python bench/analyze.py --self-check
python bench/validate_live.py --self-check
python bench/run_experiments.py --self-check
python bench/verify_fault_observability.py --self-check
python bench/run_pilot.py --self-check
python bench/accept.py --self-check
```

`bench/run_experiments.py` without `--self-check` refuses to overwrite `results/experiment_summary.json`;
pass `--out` with a new path to aggregate saved episodes.

## Key scripts

| Command | Description |
|---------|-------------|
| `scripts/finish_evaluation.py` | Offline or live pilot + experiment summary + figures |
| `bench/run_pilot.py` | Phase 9 pilot (gate v0 vs B5) |
| `bench/run_batch.py` | Resumable episode batch to `results/episodes/` |
| `bench/run_experiments.py` | Replay episodes into method comparison table |
| `bench/build_observability_matrix.py` | Build fault × test matrix (root, Mininet) |
| `bench/calibrate_healthy.py` | H0 threshold calibration (root, Mininet) |
| `paper/generate_figures.py` | Figures from `results/*.json` |

## Agent contract

The model prompt is `SYSTEM_PROMPT` in `agent/decision_agent.py`. It returns one action plus optional cause-level test predictions; the gate chooses which OAM tests to run, not the agent.

## Environment variables

| Variable | Used by |
|----------|---------|
| `GROQ_API_KEY` (`gsk_...`) | `agent/decision_agent.py`, live pilot when set (default provider) |
| `XAI_API_KEY` | xAI Grok, with `LLM_PROVIDER=grok` |
| `LLM_PROVIDER` | `groq` (default), `grok`, `openrouter`, or `gemini` |
| `LLM_MODEL` / `LLM_REASONING_EFFORT` | Override model (Groq default `openai/gpt-oss-120b`) / effort (default `low`) |
| `GROQ_LOG=info` | Log each HTTP request, retry and call latency |
| `VERIFIX_GATE_SIGNING_KEY` | Gate service only: Ed25519 private key (PKCS#8 PEM, mode 0600). Unset in one process: an ephemeral per-process key (integrity only, not a security boundary) |
| `VERIFIX_GATE_PUBLIC_KEY` | Verifier side: the gate's Ed25519 public key. The executor service takes it as `--public-key` |
| `VERIFIX_LEDGER_PATH` | In-process boundary: durable replay ledger (JSON lines). Unset: per-process ledger |
| `VERIFIX_LABELS` | Label file for metrics (default `results/ground_truth.json`); see `bench/labels.py` |

## License / release

Intended for reproducibility alongside the paper submission (code, configs, saved episodes).
