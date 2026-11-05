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
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.contracts import refuse_protected_path  # noqa: E402

OFFLINE_SUMMARY = ROOT / "results" / "offline_experiment_summary.json"


def _run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def live_paths(run_name):
    pilot_out = ROOT / "results" / "pilot" / f"{run_name}.json"
    episodes_dir = pilot_out.with_name(pilot_out.stem + "_episodes")
    experiment_out = ROOT / "results" / f"experiment_summary_{run_name}.json"
    for path in (pilot_out, episodes_dir, experiment_out):
        refuse_protected_path(path)
    return pilot_out, episodes_dir, experiment_out


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
    args = parser.parse_args()

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
        _run(
            [
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
            ],
            env=os.environ.copy(),
        )
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
        done = f"{pilot_out} and {experiment_out}"
    else:
        from bench.offline_evaluation import OFFLINE_EPISODES, run_offline_pilot
        from bench.run_experiments import aggregate

        refuse_protected_path(OFFLINE_SUMMARY)
        run_offline_pilot(episodes=args.episodes, seed=args.seed, save_episodes=True)
        summary = aggregate(OFFLINE_EPISODES)
        summary["environment"] = "offline_simulation"
        summary["episode_count"] = args.episodes
        OFFLINE_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        done = str(OFFLINE_SUMMARY)

    if not args.skip_figures:
        _run([sys.executable, "paper/generate_figures.py"])

    print(f"Done. See {done}", flush=True)


if __name__ == "__main__":
    main()
