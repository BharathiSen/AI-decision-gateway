#!/usr/bin/env python3
"""
Populate results/pilot/summary.json, results/experiment_summary.json, and figures.

Offline (no Mininet, e.g. Windows dev / CI smoke):
    python scripts/finish_evaluation.py --mode offline --episodes 500

Live (Mininet VM, root):
    sudo --preserve-env=OPENROUTER_API_KEY python3 scripts/finish_evaluation.py --mode live --episodes 500
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run pilot + experiment aggregation + figures")
    parser.add_argument("--mode", choices=["offline", "live"], default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "live" if os.name != "nt" and os.geteuid() == 0 else "offline"
        print(f"Auto-selected mode: {mode}", flush=True)

    if mode == "live":
        if os.geteuid() != 0:
            print("Live mode requires root (Mininet VM). Use --mode offline otherwise.", file=sys.stderr)
            sys.exit(2)
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
            ],
            env=os.environ.copy(),
        )
        if not args.skip_calibration:
            _run([sys.executable, "bench/calibrate_healthy.py", "--runs", "200"])
        _run([sys.executable, "bench/run_experiments.py"])
    else:
        from bench.offline_evaluation import run_offline_pilot
        from bench.run_experiments import aggregate

        run_offline_pilot(episodes=args.episodes, seed=args.seed, save_episodes=True)
        summary = aggregate(ROOT / "results" / "episodes")
        summary["environment"] = "offline_simulation"
        summary["episode_count"] = args.episodes
        (ROOT / "results" / "experiment_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if not args.skip_figures:
        _run([sys.executable, "paper/generate_figures.py"])

    print("Done. See results/pilot/summary.json and results/experiment_summary.json", flush=True)


if __name__ == "__main__":
    main()
