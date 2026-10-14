# AI Decision Gateway

Research prototype for **prediction-informed OAM gating** of LLM-proposed network fixes. An AI agent **proposes** remediation; an independent **gate** verifies proposals with operations/administration/maintenance (OAM) tests and structured predictions; **control** applies allowed changes only on an emulated network.

Paper track: *Prediction-Informed OAM Gates for LLM-Proposed Network Remediation*.

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
| [`agent/`](agent/) | OpenRouter LLM proposer + JSON schema validation |
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
export OPENROUTER_API_KEY=...   # optional; stub agent used if unset
```

### Mininet VM (live network + benchmarks)

Install [Mininet](http://mininet.org/) on Ubuntu (system Python). Run benchmark scripts as **root**:

```bash
sudo python3 main.py                    # interactive CLI
sudo python3 bench/run_pilot.py --save-episodes --episodes 500
```

Windows/macOS hosts can develop offline; live numbers require the VM.

## Quick start

### 1. Interactive demo (Mininet VM)

```bash
sudo python3 main.py
```

Useful CLI commands: `status`, `fail_primary`, `restore all`, `reroute backup`, `propose`.

### 2. Populate results (one command)

**Offline simulation** (no Mininet — dev/CI; uses stub agent + observability matrix):

```bash
python scripts/finish_evaluation.py --mode offline --episodes 500
```

**Live pilot** (paper-grade; Mininet + root):

```bash
sudo --preserve-env=OPENROUTER_API_KEY python3 scripts/finish_evaluation.py --mode live --episodes 500
```

Outputs:

- `results/pilot/summary.json` — wrong-fix rate, gate v0 vs B5 unsafe allow, test cost, go/no-go hint
- `results/episodes/*.json` — saved episodes for replay
- `results/experiment_summary.json` — aggregated false-allow rates by method
- `paper/figures/` — regenerated from results

Pilot success criteria are frozen in [`results/pilot_criteria.md`](results/pilot_criteria.md).

### 3. Tests without Mininet

```bash
python agent/test_validate.py
python bench/test_inject.py
python bench/run_experiments.py --episodes-dir results/episodes
```

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
| `OPENROUTER_API_KEY` | `agent/decision_agent.py`, live pilot when set |

## License / release

Intended for reproducibility alongside the paper submission (code, configs, saved episodes).
