#!/usr/bin/env python3
"""
Run the pilot, aggregate experiments and draw figures into run-specific outputs.

Preserved evidence (results/pilot/summary.json, results/experiment_summary.json,
results/episodes/, v3/v3b/v4 episodes, calibration) is never written. Every
output path passes bench.contracts.refuse_protected_path first.

Offline (no Mininet, e.g. Windows dev / CI smoke):
    python scripts/finish_evaluation.py --mode offline --episodes 500

Live (Mininet VM, root):
    sudo --preserve-env=GROQ_API_KEY python3 scripts/finish_evaluation.py --mode live --run-name v5 --episodes 500

Paper batch (VirtualBox, ≥300 paired episodes, stratified):
    sudo --preserve-env=GROQ_API_KEY python3 scripts/finish_evaluation.py \\
        --mode live --run-name paper_v1 --episodes 300 --paper-stratified --write-freeze
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.contracts import atomic_write_text, refuse_protected_path  # noqa: E402

OFFLINE_SUMMARY = ROOT / "results" / "offline_experiment_summary.json"
PAPER_ARMS = ("verifix", "B6", "B2", "B0")
PAPER_FREEZE = ROOT / "results" / "paper" / "paper_v1_manifest.json"


def _run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def _sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_paths(run_name):
    pilot_out = ROOT / "results" / "pilot" / f"{run_name}.json"
    episodes_dir = pilot_out.with_name(pilot_out.stem + "_episodes")
    experiment_out = ROOT / "results" / f"experiment_summary_{run_name}.json"
    for path in (pilot_out, episodes_dir, experiment_out):
        refuse_protected_path(path)
    return pilot_out, episodes_dir, experiment_out


def write_paper_freeze(run_name="paper_v1", seed=1, episodes=300):
    """Hash freeze inputs for the paper batch. No further logic changes after this."""
    from agent.decision_agent import PROMPT_SHA256, PROMPT_VERSION
    from baselines.common import METHODS
    from bench.run_batch import config_fingerprint, software_version
    from bench.run_pilot import PAPER_COMPARE_ARMS, PAPER_NOISE_LEVELS

    refuse_protected_path(PAPER_FREEZE)
    PAPER_FREEZE.parent.mkdir(parents=True, exist_ok=True)

    costs_path = ROOT / "results" / "live_oam_cost.json"
    costs = None
    if costs_path.exists():
        costs = json.loads(costs_path.read_text(encoding="utf-8"))

    model_id = None
    models_path = ROOT / "config" / "models.yaml"
    try:
        import yaml

        models = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        agents = models.get("agents") or []
        if agents:
            model_id = agents[0].get("model")
    except Exception:
        model_id = None

    manifest = {
        "freeze_id": run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "No further gate/baseline logic changes before or during the live paper batch.",
        "git": {
            "head": software_version(),
            "tag": None,
        },
        "seed": seed,
        "target_episodes": episodes,
        "paper_arms": list(PAPER_COMPARE_ARMS),
        "full_battery": "B2",
        "cause_aware_control": "B6",
        "fix_aware": "verifix",
        "ungated_floor": "B0",
        "all_methods": list(METHODS) + ["verifix"],
        "noise_levels": list(PAPER_NOISE_LEVELS),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "model_id": model_id,
        "config_fingerprint": config_fingerprint(),
        "artifacts": {
            "thresholds.yaml": {
                "path": "config/thresholds.yaml",
                "sha256": _sha256(ROOT / "config" / "thresholds.yaml"),
            },
            "correct_fixes.yaml": {
                "path": "config/correct_fixes.yaml",
                "sha256": _sha256(ROOT / "config" / "correct_fixes.yaml"),
            },
            "faults.yaml": {
                "path": "config/faults.yaml",
                "sha256": _sha256(ROOT / "config" / "faults.yaml"),
            },
            "compounds.yaml": {
                "path": "config/compounds.yaml",
                "sha256": _sha256(ROOT / "config" / "compounds.yaml"),
            },
            "observability_matrix.csv": {
                "path": "results/observability_matrix.csv",
                "sha256": _sha256(ROOT / "results" / "observability_matrix.csv"),
            },
            "inseparable_pairs.json": {
                "path": "results/inseparable_pairs.json",
                "sha256": _sha256(ROOT / "results" / "inseparable_pairs.json"),
            },
            "inseparable_pairs.meta.json": {
                "path": "results/inseparable_pairs.meta.json",
                "sha256": _sha256(ROOT / "results" / "inseparable_pairs.meta.json"),
            },
            "live_oam_cost.json": {
                "path": "results/live_oam_cost.json",
                "sha256": _sha256(costs_path),
                "summary": {
                    "tests": len(((costs or {}).get("cost") or {}).get("tests") or []),
                    "duration_seconds": ((costs or {}).get("cost") or {}).get("duration_seconds"),
                    "additional_bytes": ((costs or {}).get("cost") or {}).get("additional_bytes"),
                }
                if costs
                else None,
            },
        },
    }
    # Optional annotated tag if present.
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if tag.returncode == 0 and tag.stdout.strip():
            manifest["git"]["tag"] = tag.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    atomic_write_text(PAPER_FREEZE, json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote paper freeze {PAPER_FREEZE}", flush=True)
    return manifest


def _filter_paper_arms(summary):
    """Keep full summary; attach a paper_primary view over verifix/B6/B2/B0."""
    methods = (summary or {}).get("methods") or {}
    summary = dict(summary or {})
    summary["paper_primary_arms"] = list(PAPER_ARMS)
    summary["paper_primary"] = {arm: methods[arm] for arm in PAPER_ARMS if arm in methods}
    summary["full_battery"] = "B2"
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run pilot + experiment aggregation + figures")
    parser.add_argument("--mode", choices=["offline", "live"], default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", default=None, help="Live run label; outputs go to results/pilot/<name>*.json")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--calibration-report",
        default=None,
        help="Opt-in: recalibrate into this new report path (the frozen calibration report is refused)",
    )
    parser.add_argument(
        "--paper-stratified",
        action="store_true",
        help="Stratify live episodes by fault/compound/held-out and N0/N1/N2",
    )
    parser.add_argument(
        "--write-freeze",
        action="store_true",
        help="Write results/paper/paper_v1_manifest.json before the live batch",
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Only write the paper freeze manifest, then exit",
    )
    args = parser.parse_args()

    if args.freeze_only or args.write_freeze:
        write_paper_freeze(run_name=args.run_name or "paper_v1", seed=args.seed, episodes=args.episodes)
        if args.freeze_only:
            return

    mode = args.mode
    if mode is None:
        mode = "live" if os.name != "nt" and os.geteuid() == 0 else "offline"
        print(f"Auto-selected mode: {mode}", flush=True)

    if mode == "live":
        if os.geteuid() != 0:
            print("Live mode requires root (Mininet VM). Use --mode offline otherwise.", file=sys.stderr)
            sys.exit(2)
        if not args.run_name:
            print("Live mode requires --run-name so outputs never mix with earlier runs.", file=sys.stderr)
            sys.exit(2)
        pilot_out, episodes_dir, experiment_out = live_paths(args.run_name)
        if args.calibration_report:
            refuse_protected_path(args.calibration_report)
        pilot_cmd = [
            sys.executable,
            "bench/run_pilot.py",
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--require-model",
            "--resume",
            "--save-episodes",
            "--out",
            str(pilot_out),
        ]
        if args.paper_stratified:
            pilot_cmd.append("--paper-stratified")
        _run(pilot_cmd, env=os.environ.copy())
        if args.calibration_report:
            _run([
                sys.executable, "bench/calibrate_healthy.py", "--runs", "200",
                "--report", str(args.calibration_report),
            ])
        _run([
            sys.executable, "bench/run_experiments.py",
            "--episodes-dir", str(episodes_dir),
            "--out", str(experiment_out),
        ])
        if experiment_out.exists():
            summary = json.loads(experiment_out.read_text(encoding="utf-8"))
            summary = _filter_paper_arms(summary)
            atomic_write_text(experiment_out, json.dumps(summary, indent=2) + "\n")
        done = f"{pilot_out} and {experiment_out}"
    else:
        from bench.offline_evaluation import OFFLINE_EPISODES, run_offline_pilot
        from bench.run_experiments import aggregate

        refuse_protected_path(OFFLINE_SUMMARY)
        run_offline_pilot(episodes=args.episodes, seed=args.seed, save_episodes=True)
        summary = aggregate(OFFLINE_EPISODES)
        summary["environment"] = "offline_simulation"
        summary["episode_count"] = args.episodes
        summary = _filter_paper_arms(summary)
        atomic_write_text(OFFLINE_SUMMARY, json.dumps(summary, indent=2))
        done = str(OFFLINE_SUMMARY)

    if not args.skip_figures:
        _run([sys.executable, "paper/generate_figures.py"])

    if mode == "live" and args.run_name:
        _print_paper_report(live_paths(args.run_name)[2])

    print(f"Done. See {done}", flush=True)


def _print_paper_report(experiment_out):
    """Stage G console summary from saved experiment results only."""
    path = Path(experiment_out)
    if not path.exists():
        print(f"paper report: missing {path}", flush=True)
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    primary = summary.get("paper_primary") or {
        arm: (summary.get("methods") or {}).get(arm) for arm in PAPER_ARMS
    }
    paired = (summary.get("paired_tests") or summary.get("mcnemar") or {})
    print("\n=== Paper primary arms (saved results) ===", flush=True)
    for arm in PAPER_ARMS:
        row = primary.get(arm) or {}
        print(
            f"  {arm}: unsafe_allow={row.get('unsafe_action_rate', row.get('unsafe_action'))} "
            f"correct_allow={row.get('correct_allow_rate', row.get('correct_allow'))} "
            f"escalate={row.get('escalate_rate', row.get('escalate'))} "
            f"tests={row.get('mean_tests', row.get('tests_run_mean'))} "
            f"recovery_s={row.get('mean_recovery_seconds')}",
            flush=True,
        )
    print(f"  full_battery label: B2 | paired keys: {sorted(paired)[:8]}", flush=True)
    slices = summary.get("slices") or {}
    if "held_out" in slices:
        print(f"  held_out slice: {slices.get('held_out')}", flush=True)


if __name__ == "__main__":
    main()
