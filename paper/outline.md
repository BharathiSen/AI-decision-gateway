# Paper outline (6–8 pages, CNSM / NOMS / NetSoft)

## Title (working)

Prediction-Informed OAM Gates for LLM-Proposed Network Remediation

## 1. Introduction

- LLM agents propose fixes; wrong fixes harm production networks
- Separation of powers: propose vs verify vs execute
- Contribution: gate using AI structured test predictions + discriminating OAM

## 2. Related work

- NIKA, SADE, FaulT-Bench, NetConfBench, SpecRCA
- Adaptive test selection; LLM for netops (ungated risks)

## 3. How often are AI fixes wrong? (RQ1)

- Diamond topology, fault catalog, stress modes
- Wrong / unnecessary / unsafe rates

## 4. System

- Telemetry, fault injection, OAM T1–T10
- Agent schema with cause-level predictions
- Gate: consistency, discrimination, safety
- Baselines B0–B5

## 5. Evaluation (RQ2–RQ4)

- Pilot criteria (frozen)
- Replay + live cost
- Held-out faults U1–U5, compounds C4–C5
- Topology 2 (16-router grid)
- Metrics: false-allow, unsafe-action, blocked correct fixes, escalations

## 6. Limitations

- Catalog-bound discrimination; emulated Mininet
- Model/version sensitivity

## 7. Conclusion

- When prediction-informed gating beats adaptive testing without predictions

## Release

- Code, configs, `docs/fault_sources.md`, saved episodes, arXiv alongside submission
