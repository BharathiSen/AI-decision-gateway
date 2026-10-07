"""
Calibrate thresholds from healthy (H0) runs and estimate false-alarm rate.

    sudo python3 bench/calibrate_healthy.py --runs 200 --noise N0
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench import inject as inj
from bench import tests as oam
from bench.traffic import apply_noise, clear_noise


def _percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def calibrate(noise_id, runs, seed_base=0):
    state = create_network()
    net = state["net"]
    noise_applied = apply_noise(net, noise_id, state["links"])
    latencies = []
    losses = []
    speeds = []
    false_alarms = 0

    try:
        for i in range(runs):
            inj.reset_all(net)
            bundle = oam.run_all(net)
            for r in bundle["results"]:
                if r["test_id"] == "T1_e2e_ping":
                    if r.get("latency_ms") is not None:
                        latencies.append(r["latency_ms"])
                    losses.append(r.get("packet_loss_percent", 100))
                if r["test_id"] == "T10_speed" and r.get("mbit_per_sec"):
                    speeds.append(r["mbit_per_sec"])
            alarms = oam.evaluate_against_thresholds(bundle["results"], noise_id)
            if alarms:
                false_alarms += 1
    finally:
        clear_noise(net, noise_applied, state["links"])
        inj.reset_all(net)
        net.stop()

    report = {
        "noise_id": noise_id,
        "runs": runs,
        "false_alarm_rate": false_alarms / runs if runs else 0,
        "T1_latency_p95": _percentile(latencies, 95),
        "T1_loss_p95": _percentile(losses, 95),
        "T10_speed_p05": _percentile(speeds, 5),
    }
    return report


def update_thresholds_yaml(noise_id, report):
    path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
    with path.open() as f:
        doc = yaml.safe_load(f)
    level = doc["by_noise_level"].setdefault(noise_id, {})
    level["T1_e2e_ping"] = {
        "max_loss_percent": max(5.0, (report.get("T1_loss_p95") or 5) * 1.1),
        "max_latency_ms": max(50.0, (report.get("T1_latency_p95") or 50) * 1.1),
    }
    level["T10_speed"] = {
        "min_mbit": max(0.1, (report.get("T10_speed_p05") or 1.0) * 0.9),
    }
    doc.setdefault("calibration", {})["last_run"] = report
    with path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--noise", default="N0")
    parser.add_argument("--write-thresholds", action="store_true")
    parser.add_argument("--report", default="results/calibration_report.json")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("calibrate_healthy.py requires root on the Mininet VM.", file=sys.stderr)
        sys.exit(2)

    report = calibrate(args.noise, args.runs)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.write_thresholds:
        update_thresholds_yaml(args.noise, report)


if __name__ == "__main__":
    main()
