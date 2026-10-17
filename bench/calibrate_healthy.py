"""
Calibrate thresholds from healthy (H0) runs and estimate false-alarm rate.

    sudo python3 bench/calibrate_healthy.py --noise N0
    sudo python3 bench/calibrate_healthy.py --runs 200 --full --noise N0
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
from bench.mininet_util import mn_cleanup, ensure_iperf3_server, recover_net


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


# The report only stores T1 and T10. Hop pings dominate a full battery
# and are not used for these thresholds.
_THRESHOLD_TESTS = ["T1_e2e_ping", "T10_speed"]


def calibrate(noise_id, runs, seed_base=0, full=False):
    mn_cleanup()
    state = create_network()
    net = state["net"]
    ensure_iperf3_server(net)
    noise_applied = apply_noise(net, noise_id, state["links"])
    latencies = []
    losses = []
    speeds = []
    false_alarms = 0
    test_ids = None if full else _THRESHOLD_TESTS

    scope = "full OAM battery" if full else "T1 and T10 only"
    print(f"Calibrating {noise_id}: {runs} healthy runs ({scope}).", flush=True)
    try:
        for i in range(runs):
            print(f"  run {i + 1}/{runs} ...", flush=True)
            try:
                inj.reset_all(net)
                bundle = oam.run_all(net, test_ids=test_ids)
            except UnicodeDecodeError as exc:
                print(f"  run {i + 1}/{runs} skipped after binary shell output: {exc}", flush=True)
                recover_net(net)
                continue
            print(f"  run {i + 1}/{runs} done", flush=True)
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
        recover_net(net)
        try:
            clear_noise(net, noise_applied, state["links"])
        except Exception as exc:
            print(f"clear_noise failed: {exc}", flush=True)
        try:
            inj.reset_all(net)
        except Exception as exc:
            print(f"reset_all failed: {exc}", flush=True)
        try:
            net.stop()
        except Exception as exc:
            print(f"net.stop failed: {exc}", flush=True)
        mn_cleanup()

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
    if report.get("T10_speed_p05") is not None:
        level["T10_speed"] = {
            "min_mbit": max(0.1, report["T10_speed_p05"] * 0.9),
        }
    doc.setdefault("calibration", {})["last_run"] = report
    with path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--noise", default="N0")
    parser.add_argument("--full", action="store_true", help="run the whole OAM battery each sample")
    parser.add_argument("--write-thresholds", action="store_true")
    parser.add_argument("--report", default="results/calibration_report.json")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("calibrate_healthy.py requires root on the Mininet VM.", file=sys.stderr)
        sys.exit(2)

    report = calibrate(args.noise, args.runs, full=args.full)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.write_thresholds:
        update_thresholds_yaml(args.noise, report)


if __name__ == "__main__":
    main()
