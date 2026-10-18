"""
Calibrate healthy-network thresholds from H0 runs.

The fit sample and the false-alarm sample are different runs. Fault
ids, including the held-out set, are not inputs. A short or partial
sample cannot replace config/thresholds.yaml.

Offline check, from the repo root:

    python bench/calibrate_healthy.py

Live collection, on the VM as root. One traffic level is a long run.
All three levels is the Stage 5 sample:

    sudo python3 bench/calibrate_healthy.py --noise N0 --runs 200 --eval-runs 50
    sudo python3 bench/calibrate_healthy.py --noise all --runs 200 --eval-runs 50

--write-thresholds is refused unless that sample is a full battery of
at least 200 calibration runs plus a separate eval set. The configured
speed floor is kept when the healthy low tail sits below it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.contracts import TEST_IDS
from bench.outcomes import grade_result, is_degraded, load_thresholds

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = ROOT / "config" / "thresholds.yaml"
SAMPLES_PATH = ROOT / "results" / "calibration_samples.jsonl"
FULL_BATTERY = list(TEST_IDS)
MIN_CALIBRATION_RUNS = 200
CALIBRATION_PINGS = 3


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


def _numeric(values):
    present = [value for value in values if value is not None]
    if not present:
        return {"n": 0, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(present),
        "p05": _percentile(present, 5),
        "p50": _percentile(present, 50),
        "p95": _percentile(present, 95),
        "max": max(present),
    }


def _counts(values):
    found = {}
    for value in values:
        key = str(value)
        found[key] = found.get(key, 0) + 1
    return found


def _observe(result):
    """Pull the numbers a healthy threshold can use. No fault label."""
    test_id = result.get("test_id")
    observed = {"test_id": test_id}
    if test_id == "T1_e2e_ping":
        observed["latency_ms"] = result.get("latency_ms")
        observed["packet_loss_percent"] = result.get("packet_loss_percent")
    elif test_id == "T2_hop_ping":
        hops = list(result.get("hops_primary") or []) + list(result.get("hops_backup") or [])
        latencies = [hop.get("latency_ms") for hop in hops if hop.get("latency_ms") is not None]
        losses = [hop.get("packet_loss_percent") for hop in hops if hop.get("packet_loss_percent") is not None]
        observed["latency_ms"] = max(latencies) if latencies else None
        observed["packet_loss_percent"] = max(losses) if losses else None
    elif test_id == "T4_mtu":
        observed["df_ping_ok"] = bool(result.get("df_ping_ok"))
    elif test_id == "T5_if_errors":
        total = 0
        for stats in (result.get("interfaces") or {}).values():
            if isinstance(stats, dict):
                total += int(stats.get("rx_errors") or 0) + int(stats.get("tx_errors") or 0)
        observed["errors"] = total
    elif test_id == "T6_queue_stats":
        drops = 0
        for stats in (result.get("queues") or {}).values():
            if not isinstance(stats, dict):
                continue
            intentional_loss = stats.get("loss_rule") and not stats.get("rate_rule")
            if not intentional_loss:
                drops += int(stats.get("drops") or 0)
        observed["drops"] = drops
    elif test_id == "T7_route_check":
        observed["active_path"] = result.get("active_path")
    elif test_id == "T8_tcp_vs_ping":
        observed["ping_ok"] = bool(result.get("ping_ok"))
        observed["tcp_ok"] = bool(result.get("tcp_ok"))
    elif test_id == "T9_oneway_loss":
        observed["a2b_loss"] = (result.get("hostA_to_hostB") or {}).get("packet_loss_percent")
        observed["b2a_loss"] = (result.get("hostB_to_hostA") or {}).get("packet_loss_percent")
    elif test_id == "T10_speed":
        observed["mbit_per_sec"] = result.get("mbit_per_sec")
    return observed


def summarize(bundles):
    """Per-test distributions from the calibration bundles only."""
    grouped = {test_id: [] for test_id in FULL_BATTERY}
    for bundle in bundles:
        for result in bundle.get("results") or []:
            test_id = result.get("test_id")
            if test_id in grouped:
                grouped[test_id].append(_observe(result))
    tests = {}
    for test_id, rows in grouped.items():
        entry = {"n": len(rows)}
        if test_id in ("T1_e2e_ping", "T2_hop_ping"):
            entry["latency_ms"] = _numeric([row.get("latency_ms") for row in rows])
            entry["packet_loss_percent"] = _numeric([row.get("packet_loss_percent") for row in rows])
        elif test_id == "T4_mtu":
            entry["df_ping_ok"] = _counts([row.get("df_ping_ok") for row in rows])
        elif test_id == "T5_if_errors":
            entry["errors"] = _numeric([row.get("errors") for row in rows])
        elif test_id == "T6_queue_stats":
            entry["drops"] = _numeric([row.get("drops") for row in rows])
        elif test_id == "T7_route_check":
            entry["active_path"] = _counts([row.get("active_path") for row in rows])
        elif test_id == "T8_tcp_vs_ping":
            entry["ping_ok"] = _counts([row.get("ping_ok") for row in rows])
            entry["tcp_ok"] = _counts([row.get("tcp_ok") for row in rows])
        elif test_id == "T9_oneway_loss":
            entry["a2b_loss"] = _numeric([row.get("a2b_loss") for row in rows])
            entry["b2a_loss"] = _numeric([row.get("b2a_loss") for row in rows])
        elif test_id == "T10_speed":
            entry["mbit_per_sec"] = _numeric([row.get("mbit_per_sec") for row in rows])
        tests[test_id] = entry
    return tests


def _upper(measured, floor):
    if measured is None:
        return floor
    return max(floor, measured * 1.1)


def propose_thresholds(tests, existing):
    """Turn calibration distributions into one noise level's limits.

    Existing floors stay in place when the healthy tail is healthier
    than the floor, and when the speed tail is below the configured floor.
    """
    existing = existing or {}
    notes = []
    t1 = existing.get("T1_e2e_ping") or {}
    t1_fit = tests.get("T1_e2e_ping") or {}
    proposed = {
        "T1_e2e_ping": {
            "max_loss_percent": _upper(
                (t1_fit.get("packet_loss_percent") or {}).get("p95"),
                t1.get("max_loss_percent", 5.0),
            ),
            "max_latency_ms": _upper(
                (t1_fit.get("latency_ms") or {}).get("p95"),
                t1.get("max_latency_ms", 50.0),
            ),
        }
    }
    t2 = existing.get("T2_hop_ping") or {}
    t2_fit = tests.get("T2_hop_ping") or {}
    proposed["T2_hop_ping"] = {
        "max_loss_percent": _upper(
            (t2_fit.get("packet_loss_percent") or {}).get("p95"),
            t2.get("max_loss_percent", 10.0),
        ),
        "max_latency_ms": _upper(
            (t2_fit.get("latency_ms") or {}).get("p95"),
            t2.get("max_latency_ms", 80.0),
        ),
    }
    t8 = existing.get("T8_tcp_vs_ping") or {}
    proposed["T8_tcp_vs_ping"] = {"tcp_must_succeed": bool(t8.get("tcp_must_succeed", True))}
    t10 = existing.get("T10_speed") or {}
    floor = t10.get("min_mbit", 1.0)
    p05 = ((tests.get("T10_speed") or {}).get("mbit_per_sec") or {}).get("p05")
    if p05 is None or p05 * 0.9 < floor:
        proposed["T10_speed"] = {"min_mbit": floor}
        notes.append("T10 min_mbit kept at the configured floor")
    else:
        proposed["T10_speed"] = {"min_mbit": p05 * 0.9}
    return proposed, notes


def score_false_alarms(bundles, thresholds):
    """Count eval bundles whose graded outcome is not healthy."""
    alarms = 0
    for bundle in bundles:
        for result in bundle.get("results") or []:
            outcome = grade_result(result, thresholds)
            if is_degraded(result.get("test_id"), outcome, thresholds):
                alarms += 1
                break
    return alarms


def build_report(calibration, evaluation, noise_id, existing, full_battery, seed):
    tests = summarize(calibration)
    proposed, notes = propose_thresholds(tests, existing)
    alarms = score_false_alarms(evaluation, proposed)
    eval_runs = len(evaluation)
    return {
        "noise_id": noise_id,
        "calibration_runs": len(calibration),
        "eval_runs": eval_runs,
        "full_battery": full_battery,
        "seed": seed,
        "used_evaluation_faults": False,
        "tests": tests,
        "proposed_thresholds": proposed,
        "threshold_notes": notes,
        "eval_false_alarms": alarms,
        "eval_false_alarm_rate": (alarms / eval_runs) if eval_runs else None,
    }


def ready_to_write(report):
    return (
        report.get("full_battery") is True
        and report.get("calibration_runs", 0) >= MIN_CALIBRATION_RUNS
        and report.get("eval_runs", 0) > 0
        and report.get("used_evaluation_faults") is False
    )


def update_thresholds_yaml(reports, path=THRESHOLDS_PATH):
    """Replace limits for levels that passed the write guard."""
    refused = [noise for noise, report in reports.items() if not ready_to_write(report)]
    if refused:
        raise ValueError(
            "refusing to write thresholds for " + ", ".join(refused)
            + ": need a full battery, 200 calibration runs, and a separate eval set"
        )
    with path.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    levels = doc.setdefault("by_noise_level", {})
    for noise, report in reports.items():
        levels[noise] = report["proposed_thresholds"]
    doc.setdefault("calibration", {})["last_run"] = {
        noise: {
            "calibration_runs": report["calibration_runs"],
            "eval_runs": report["eval_runs"],
            "eval_false_alarm_rate": report["eval_false_alarm_rate"],
            "threshold_notes": report["threshold_notes"],
        }
        for noise, report in reports.items()
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False)


def _healthy_result(latency=40.0, loss=0.0, mbit=8.0):
    hop = {"dst_ip": "10.0.12.2", "latency_ms": 10.0, "packet_loss_percent": 0.0}
    return [
        {"test_id": "T1_e2e_ping", "latency_ms": latency, "packet_loss_percent": loss},
        {"test_id": "T2_hop_ping", "hops_primary": [hop], "hops_backup": [hop]},
        {"test_id": "T4_mtu", "df_ping_ok": True},
        {"test_id": "T5_if_errors", "interfaces": {"r1/r1-eth1": {"rx_errors": 0, "tx_errors": 0}}},
        {"test_id": "T6_queue_stats", "queues": {"r1/r1-eth1": {"drops": 0, "backlog_bytes": 0, "backlog_packets": 0}}},
        {"test_id": "T7_route_check", "r1_to_dest": "10.0.4.0/24 via 10.0.12.2", "r4_to_src": "10.0.1.0/24 via 10.0.24.1", "active_path": "primary"},
        {"test_id": "T8_tcp_vs_ping", "ping_ok": True, "tcp_ok": True},
        {"test_id": "T9_oneway_loss", "hostA_to_hostB": {"packet_loss_percent": 0.0}, "hostB_to_hostA": {"packet_loss_percent": 0.0}},
        {"test_id": "T10_speed", "mbit_per_sec": mbit},
    ]


def _self_check():
    """Prove the split, the floors, and the write guard. Does not collect or write."""
    import inspect

    before = THRESHOLDS_PATH.read_text(encoding="utf-8")
    existing = {
        "T1_e2e_ping": {"max_loss_percent": 5.0, "max_latency_ms": 50.0},
        "T2_hop_ping": {"max_loss_percent": 10.0, "max_latency_ms": 80.0},
        "T8_tcp_vs_ping": {"tcp_must_succeed": True},
        "T10_speed": {"min_mbit": 1.0},
    }
    calm = [{"results": _healthy_result()} for _ in range(20)]
    held_out_eval = [{"results": _healthy_result(latency=400.0)} for _ in range(4)]
    held_out_eval.append({"results": _healthy_result()})
    report = build_report(calm, held_out_eval, "N0", existing, True, 0)
    assert report["calibration_runs"] == 20
    assert report["eval_runs"] == 5
    assert report["used_evaluation_faults"] is False
    assert set(FULL_BATTERY) <= set(report["tests"])
    assert report["tests"]["T1_e2e_ping"]["latency_ms"]["p95"] == 40.0
    assert report["eval_false_alarms"] == 4
    assert report["proposed_thresholds"]["T10_speed"]["min_mbit"] == 7.2
    assert "T10 min_mbit kept at the configured floor" not in report["threshold_notes"]

    contaminated = build_report(calm + held_out_eval, [], "N0", existing, True, 0)
    assert contaminated["tests"]["T1_e2e_ping"]["latency_ms"]["p95"] != 40.0
    assert contaminated["eval_false_alarms"] == 0

    slow = build_report(
        [{"results": _healthy_result(mbit=0.3)} for _ in range(20)],
        [{"results": _healthy_result()}],
        "N0",
        existing,
        True,
        0,
    )
    assert slow["proposed_thresholds"]["T10_speed"]["min_mbit"] == 1.0
    assert "T10 min_mbit kept at the configured floor" in slow["threshold_notes"]

    fast = build_report(
        [{"results": _healthy_result(mbit=10.0)} for _ in range(20)],
        [{"results": _healthy_result()}],
        "N0",
        existing,
        True,
        0,
    )
    assert fast["proposed_thresholds"]["T10_speed"]["min_mbit"] == 9.0

    assert ready_to_write(report) is False
    try:
        update_thresholds_yaml({"N0": report}, THRESHOLDS_PATH)
        raise AssertionError("short sample was written")
    except ValueError:
        pass
    assert THRESHOLDS_PATH.read_text(encoding="utf-8") == before

    logic = inspect.getsource(summarize) + inspect.getsource(propose_thresholds) + inspect.getsource(score_false_alarms)
    assert "observability" not in logic
    assert "faults.yaml" not in logic
    live = inspect.getsource(collect_healthy)
    assert "inject(" not in live
    assert "reset_all" in live and "run_all" in live

    print("healthy calibration: PASS")


def _slim(bundle):
    results = []
    for result in bundle.get("results") or []:
        item = {key: value for key, value in result.items() if key not in ("raw", "tcp_raw")}
        queues = item.get("queues")
        if isinstance(queues, dict):
            item["queues"] = {
                name: {key: value for key, value in stats.items() if key != "raw"}
                for name, stats in queues.items()
                if isinstance(stats, dict)
            }
        results.append(item)
    return {"results": results}


def _load_samples(path, noise_id):
    if not path.exists():
        return []
    bundles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("noise_id") == noise_id:
            bundles.append(item["bundle"])
    return bundles


def _append_sample(path, noise_id, index, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"noise_id": noise_id, "index": index, "bundle": bundle}) + "\n")


def collect_healthy(net, runs, test_ids, noise_id="N0", sample_path=SAMPLES_PATH):
    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import recover_net

    bundles = _load_samples(sample_path, noise_id)
    if len(bundles) >= runs:
        print(f"  {noise_id} already has {len(bundles)} saved runs.", flush=True)
        return bundles[:runs]
    print(f"  resuming {noise_id} at {len(bundles) + 1}/{runs}.", flush=True)
    inj.reset_all(net)
    for index in range(len(bundles), runs):
        print(f"  run {index + 1}/{runs} ...", flush=True)
        started = time.monotonic()
        try:
            bundle = _slim(oam.run_all(
                net,
                test_ids=test_ids,
                ping_count=CALIBRATION_PINGS,
                ping_interval=0.2,
            ))
        except UnicodeDecodeError as exc:
            print(f"  run {index + 1}/{runs} skipped after binary shell output: {exc}", flush=True)
            recover_net(net)
            continue
        _append_sample(sample_path, noise_id, index, bundle)
        bundles.append(bundle)
        print(f"  run {index + 1}/{runs} done in {time.monotonic() - started:.0f}s", flush=True)
    return bundles


def calibrate(noise_ids, runs, eval_runs, seed=0, quick=False):
    from network.topology import create_network
    from bench import inject as inj
    from bench.traffic import apply_noise, clear_noise
    from bench.mininet_util import mn_cleanup, ensure_iperf3_server, recover_net

    test_ids = ["T1_e2e_ping", "T10_speed"] if quick else None
    full_battery = test_ids is None
    mn_cleanup()
    state = create_network()
    net = state["net"]
    ensure_iperf3_server(net)
    reports = {}
    try:
        for noise_id in noise_ids:
            total = runs + eval_runs
            scope = "full OAM battery" if full_battery else "T1 and T10 only"
            print(
                f"Calibrating {noise_id}: {runs} fit runs, {eval_runs} eval runs ({scope}).",
                flush=True,
            )
            noise_applied = apply_noise(net, noise_id, state["links"])
            try:
                collected = collect_healthy(net, total, test_ids, noise_id=noise_id)
            finally:
                try:
                    clear_noise(net, noise_applied, state["links"])
                except Exception as exc:
                    print(f"clear_noise failed: {exc}", flush=True)
                recover_net(net)
            reports[noise_id] = build_report(
                collected[:runs],
                collected[runs:],
                noise_id,
                load_thresholds(noise_id),
                full_battery,
                seed,
            )
    finally:
        try:
            inj.reset_all(net)
        except Exception as exc:
            print(f"reset_all failed: {exc}", flush=True)
        try:
            net.stop()
        except Exception as exc:
            print(f"net.stop failed: {exc}", flush=True)
        mn_cleanup()
    return reports


def _is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=MIN_CALIBRATION_RUNS)
    parser.add_argument("--eval-runs", type=int, default=50)
    parser.add_argument("--noise", default="N0", help="N0, N1, N2, or all")
    parser.add_argument("--quick", action="store_true", help="T1 and T10 only; cannot write thresholds")
    parser.add_argument("--write-thresholds", action="store_true")
    parser.add_argument("--report", default="results/calibration_report.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not _is_root():
        print("calibrate_healthy.py collects on the Mininet VM as root.", file=sys.stderr)
        sys.exit(2)

    noise_ids = ["N0", "N1", "N2"] if args.noise == "all" else [args.noise]
    reports = calibrate(noise_ids, args.runs, args.eval_runs, seed=args.seed, quick=args.quick)
    payload = {"version": 1, "levels": reports}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.write_thresholds:
        update_thresholds_yaml(reports)


if __name__ == "__main__":
    if "--check" in sys.argv or not _is_root():
        _self_check()
    else:
        main()
