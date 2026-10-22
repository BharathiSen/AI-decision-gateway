"""
Calibrate healthy-network thresholds from H0 runs.

The fit sample and the false-alarm sample are different runs. Fault
ids, including the held-out set, are not inputs. A short or partial
sample cannot replace config/thresholds.yaml.

Offline check, from the repo root:

    python bench/calibrate_healthy.py

Live collection, on the VM as root. Each launch saves 10 samples and exits
so the VM can be powered off. Run the same command again to continue:

    sudo python3 bench/calibrate_healthy.py --noise N0 --runs 200 --eval-runs 50

When the checkpoint already has the requested runs, the command reprints
that file and does not start Mininet, even if --fill-speed is passed.
Failed speed readings under 1 Mbit/s and readings above 15 Mbit/s are
left out of the distribution and are not counted as false alarms.

    python3 bench/calibrate_healthy.py --from-saved --noise all --runs 200 --eval-runs 50

--write-thresholds is refused unless that sample is a full battery of
at least 200 calibration runs plus a separate eval set. The configured
speed floor is kept when the healthy low tail sits below it.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.contracts import TEST_IDS
from bench.outcomes import grade_result, is_degraded, load_thresholds
from bench.tests import accepted_speed

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = ROOT / "config" / "thresholds.yaml"
SAMPLES_PATH = ROOT / "results" / "calibration_samples.jsonl"
FULL_BATTERY = list(TEST_IDS)
MIN_CALIBRATION_RUNS = 200
CALIBRATION_PINGS = 3
CALIBRATION_BATCH = 10
SAMPLE_TIMEOUT_SECONDS = 90
SPEED_FILL_CAP = 30


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


def _usable_loss(value):
    """A finished ping. Total loss and values above 100% are not a healthy sample."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value < 0 or value >= 100:
        return None
    return value


def _observe(result):
    """Pull the numbers a healthy threshold can use. No fault label."""
    test_id = result.get("test_id")
    observed = {"test_id": test_id}
    if test_id == "T1_e2e_ping":
        observed["latency_ms"] = result.get("latency_ms")
        observed["packet_loss_percent"] = _usable_loss(result.get("packet_loss_percent"))
    elif test_id == "T2_hop_ping":
        hops = list(result.get("hops_primary") or []) + list(result.get("hops_backup") or [])
        latencies = [hop.get("latency_ms") for hop in hops if hop.get("latency_ms") is not None]
        losses = [_usable_loss(hop.get("packet_loss_percent")) for hop in hops]
        losses = [value for value in losses if value is not None]
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
        observed["a2b_loss"] = _usable_loss((result.get("hostA_to_hostB") or {}).get("packet_loss_percent"))
        observed["b2a_loss"] = _usable_loss((result.get("hostB_to_hostA") or {}).get("packet_loss_percent"))
    elif test_id == "T10_speed":
        observed["mbit_per_sec"] = accepted_speed(result)
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
    if p05 is None:
        proposed["T10_speed"] = {"min_mbit": floor}
        notes.append("T10 had no bitrate; speed floor was not changed")
    elif p05 * 0.9 < floor:
        proposed["T10_speed"] = {"min_mbit": floor}
        notes.append("T10 min_mbit kept at the configured floor")
    else:
        proposed["T10_speed"] = {"min_mbit": p05 * 0.9}
    return proposed, notes


def _no_reading(result):
    """A probe that did not return a usable number is not a false alarm."""
    test_id = result.get("test_id")
    if test_id == "T10_speed":
        return accepted_speed(result) is None
    if test_id == "T1_e2e_ping":
        raw = result.get("packet_loss_percent")
        return raw is not None and _usable_loss(raw) is None
    if test_id == "T9_oneway_loss":
        for key in ("hostA_to_hostB", "hostB_to_hostA"):
            raw = (result.get(key) or {}).get("packet_loss_percent")
            if raw is not None and _usable_loss(raw) is None:
                return True
        return False
    if test_id == "T2_hop_ping":
        hops = list(result.get("hops_primary") or []) + list(result.get("hops_backup") or [])
        raw = [hop.get("packet_loss_percent") for hop in hops if hop.get("packet_loss_percent") is not None]
        return bool(raw) and all(_usable_loss(value) is None for value in raw)
    return False


def score_false_alarms(bundles, thresholds):
    """Count eval bundles whose graded outcome is not healthy."""
    alarms = 0
    for bundle in bundles:
        for result in bundle.get("results") or []:
            if _no_reading(result):
                continue
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

    slow_row = _healthy_result(mbit=0.3)
    for item in slow_row:
        if item["test_id"] == "T10_speed":
            item["checked"] = True
    slow = build_report(
        [{"results": slow_row} for _ in range(20)],
        [{"results": _healthy_result()}],
        "N0",
        existing,
        True,
        0,
    )
    assert slow["proposed_thresholds"]["T10_speed"]["min_mbit"] == 1.0
    assert "T10 min_mbit kept at the configured floor" in slow["threshold_notes"]
    assert slow["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 20

    junk_rows = [{"results": _healthy_result(mbit=8.0)} for _ in range(19)]
    junk_rows.append({"results": _healthy_result(mbit=2930.0)})
    junk_rows.append({"results": _healthy_result(mbit=0.05)})
    junk = build_report(
        junk_rows,
        [{"results": _healthy_result(mbit=0.05)}],
        "N0",
        existing,
        True,
        0,
    )
    assert junk["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 19
    assert junk["tests"]["T10_speed"]["mbit_per_sec"]["max"] == 8.0
    assert junk["eval_false_alarms"] == 0

    fast = build_report(
        [{"results": _healthy_result(mbit=10.0)} for _ in range(20)],
        [{"results": _healthy_result()}],
        "N0",
        existing,
        True,
        0,
    )
    assert fast["proposed_thresholds"]["T10_speed"]["min_mbit"] == 9.0

    dead_rows = [{"results": _healthy_result()} for _ in range(19)]
    dead_one = _healthy_result(loss=100.0)
    for item in dead_one:
        if item["test_id"] == "T2_hop_ping":
            for hop in item["hops_primary"] + item["hops_backup"]:
                hop["packet_loss_percent"] = 100.0
        if item["test_id"] == "T9_oneway_loss":
            item["hostA_to_hostB"]["packet_loss_percent"] = 3333.0
    dead_rows.append({"results": dead_one})
    dead = build_report(dead_rows, [{"results": dead_one}], "N0", existing, True, 0)
    assert dead["tests"]["T1_e2e_ping"]["packet_loss_percent"]["p95"] == 0.0
    assert dead["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == 5.0
    assert dead["proposed_thresholds"]["T2_hop_ping"]["max_loss_percent"] == 10.0
    assert dead["tests"]["T9_oneway_loss"]["a2b_loss"]["p95"] == 0.0
    assert dead["eval_false_alarms"] == 0

    partial_rows = [{"results": _healthy_result(loss=20.0)} for _ in range(20)]
    for bundle in partial_rows:
        for item in bundle["results"]:
            if item["test_id"] == "T2_hop_ping":
                item["hops_primary"][0]["packet_loss_percent"] = 20.0
    partial = build_report(partial_rows, [{"results": _healthy_result(loss=50.0)}], "N0", existing, True, 0)
    assert partial["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == 22.0
    assert partial["proposed_thresholds"]["T2_hop_ping"]["max_loss_percent"] == 22.0
    assert partial["eval_false_alarms"] == 1

    blank = _healthy_result()
    for item in blank:
        if item["test_id"] == "T10_speed":
            item["mbit_per_sec"] = None
    missing_speed = build_report(
        [{"results": blank} for _ in range(20)],
        [{"results": blank}],
        "N1",
        existing,
        True,
        0,
    )
    assert missing_speed["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 0
    assert missing_speed["proposed_thresholds"]["T10_speed"]["min_mbit"] == 1.0
    assert any("no bitrate" in note for note in missing_speed["threshold_notes"])
    assert missing_speed["eval_false_alarms"] == 0

    from bench.tests import _parse_mbit_intervals, _summary_mbit
    from bench.traffic import apply_noise

    summary = (
        "[  5]   0.00-0.01   sec  3.58 MBytes  2930 Mbits/sec\n"
        "[  5]   0.00-1.00   sec   448 KBytes  3.67 Mbits/sec                  sender\n"
        "[  5]   0.00-1.00   sec   439 KBytes  3.59 Mbits/sec                  receiver\n"
    )
    assert _summary_mbit(summary, 1) == 3.59
    assert _summary_mbit(summary + "iperf3: error - unable to send control message\n", 1) is None
    assert _summary_mbit("[  5]   0.00-1.00   sec  20 MBytes  167 Mbits/sec\n", 1) is None
    assert _summary_mbit("[  5]   0.00-1.00   sec  1.37 MBytes  11.6 Mbits/sec\n", 1) == 11.6
    assert _summary_mbit("[  5]   0.00-3.00   sec  2.00 MBytes  5.60 Mbits/sec\n", 3) == 5.6

    traffic = inspect.getsource(apply_noise)
    assert "5202" in traffic and "-u " in traffic
    parsed = _parse_mbit_intervals(
        "[  5]   0.00-1.00   sec  1.00 MBytes  8.00 Mbits/sec\n"
        "[  5]   1.00-2.00   sec  1.00 MBytes  7.50 Mbits/sec\n"
        "[  5]   0.00-2.00   sec  2.00 MBytes  7.75 Mbits/sec\n"
    )
    assert parsed == [8.0, 7.5]
    filler = inspect.getsource(fill_missing_speed)
    assert "measure_speed_once" in filler and "start_one_off_iperf3" in filler
    assert "SPEED_FILL_CAP" in filler
    assert "_call_with_timeout" not in filler
    from bench.mininet_util import start_one_off_iperf3
    assert '"-s", "-1"' in inspect.getsource(start_one_off_iperf3)
    assert "free_tcp_port" in inspect.getsource(start_one_off_iperf3)
    from bench.tests import T10_speed, measure_speed_once
    once = inspect.getsource(measure_speed_once)
    assert "timeout 8" in once and "-t 1" in once and "-i 1" not in once
    battery = inspect.getsource(T10_speed)
    assert "timeout 20 iperf3" in battery

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
    assert _session_goal(0, 250, 10) == 10
    assert _session_goal(10, 250, 10) == 20
    assert _session_goal(245, 250, 10) == 250

    import tempfile

    checkpoint = Path(tempfile.mkdtemp()) / "samples.jsonl"
    checkpoint.write_text('not-json\n{"noise_id": "N0", "index": 0, "bundle": {"results": []}}\n', encoding="utf-8")
    loaded = _load_samples(checkpoint, "N0")
    assert len(loaded) == 1 and loaded[0] == {"results": []}
    assert "not-json" not in checkpoint.read_text(encoding="utf-8")
    checkpoint.unlink()
    checkpoint.parent.rmdir()

    saved = Path(tempfile.mkdtemp()) / "samples.jsonl"
    rows = []
    for index, mbit in enumerate((8.0, 8.0, 0.05, 2930.0)):
        bundle = {"results": _healthy_result(mbit=mbit)}
        rows.append({"noise_id": "N0", "index": index, "bundle": bundle})
    saved.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    reprinted = report_from_saved(["N0"], 3, 1, sample_path=saved)
    assert reprinted["N0"]["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 2
    assert reprinted["N0"]["eval_false_alarms"] == 0
    assert "create_network" not in inspect.getsource(report_from_saved)
    assert _reprint_only(["N0"], 3, 1, saved, False) is True
    assert _reprint_only(["N0"], 200, 50, saved, False) is False
    assert _reprint_only(["N0"], 200, 50, saved, True) is True
    saved.unlink()
    saved.parent.rmdir()

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
    """Read saved samples. A blank or half-written line does not stop the run."""
    if not path.exists():
        return []
    kept = []
    skipped = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(item, dict) or "bundle" not in item:
            skipped += 1
            continue
        kept.append(item)
    if skipped:
        print(f"  ignored {skipped} unreadable checkpoint line(s) in {path.name}", flush=True)
        path.write_text("".join(json.dumps(item) + "\n" for item in kept), encoding="utf-8")
    return [item["bundle"] for item in kept if item.get("noise_id") == noise_id]


def _append_sample(path, noise_id, index, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"noise_id": noise_id, "index": index, "bundle": bundle}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_rows(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "bundle" in item:
            rows.append(item)
    return rows


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_noise_bundles(path, noise_id, updates):
    """Replace several saved bundles for one noise level in a single write."""
    rows = _read_rows(path)
    positions = [pos for pos, item in enumerate(rows) if item.get("noise_id") == noise_id]
    for index, bundle in updates.items():
        if index >= len(positions):
            continue
        rows[positions[index]]["bundle"] = bundle
        rows[positions[index]]["index"] = index
    _write_rows(path, rows)


def _t10_mbit(bundle):
    for result in bundle.get("results") or []:
        if result.get("test_id") == "T10_speed":
            return accepted_speed(result)
    return None


def _store_speeds(bundles, indexes, intervals):
    updates = {}
    for index, mbit in zip(indexes, intervals):
        results = [row for row in bundles[index].get("results") or [] if row.get("test_id") != "T10_speed"]
        results.append({"test_id": "T10_speed", "mbit_per_sec": mbit, "checked": True})
        bundles[index]["results"] = results
        updates[index] = bundles[index]
    return updates


def _print_iperf_tail(raw):
    lines = [line.strip() for line in (raw or "").splitlines() if "error" in line.lower()]
    if not lines:
        print("  no bitrate", flush=True)
        return
    for line in lines[-2:]:
        print(f"  {line}", flush=True)


def fill_missing_speed(net, bundles, noise_id, sample_path=SAMPLES_PATH, batch=CALIBRATION_BATCH):
    """Fill missing T10 readings, then return so the VM can stop."""
    from bench import tests as oam
    from bench.mininet_util import finish_one_off, start_one_off_iperf3

    missing = [index for index, bundle in enumerate(bundles) if _t10_mbit(bundle) is None]
    if not missing:
        return bundles, False
    count = min(len(missing), SPEED_FILL_CAP)
    print(
        f"  {noise_id} has {len(missing)} saved runs with no T10 bitrate. "
        f"Collecting {count}, then this command exits.",
        flush=True,
    )
    found = []
    attempts = 0
    attempt_limit = count + 20
    while len(found) < count and attempts < attempt_limit:
        attempts += 1
        server = start_one_off_iperf3(net, port=5201)
        time.sleep(0.25)
        try:
            mbit, raw = oam.measure_speed_once(net, port=5201)
        finally:
            finish_one_off(server)
        failed = mbit is None or "unable to send" in (raw or "") or "error -" in (raw or "")
        if failed:
            _print_iperf_tail(raw)
            time.sleep(0.3)
            continue
        print(f"  {len(found) + 1}/{count}  {mbit:.2f} Mbit/s", flush=True)
        slot = missing[len(found)]
        found.append(mbit)
        _replace_noise_bundles(sample_path, noise_id, _store_speeds(bundles, [slot], [mbit]))
    if not found:
        print("  speed fill got no bitrate. Not retrying this sitting.", flush=True)
        return bundles, True
    print(f"  saved {len(found)} speed readings.", flush=True)
    still = any(_t10_mbit(bundle) is None for bundle in bundles)
    return bundles, still


def _session_goal(done, total, batch):
    if batch is None:
        return total
    return min(total, done + batch)


def _call_with_timeout(seconds, fn):
    """Stop one sample if a Mininet shell never returns. No-op where SIGALRM is absent."""
    if not hasattr(signal, "SIGALRM"):
        return fn()

    def _handle(signum, frame):
        raise TimeoutError(f"sample exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def collect_healthy(net, runs, test_ids, noise_id="N0", sample_path=SAMPLES_PATH, batch=CALIBRATION_BATCH):
    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import recover_net

    bundles = _load_samples(sample_path, noise_id)
    goal = _session_goal(len(bundles), runs, batch)
    if len(bundles) >= runs:
        print(f"  {noise_id} already has {len(bundles)} saved runs.", flush=True)
        return bundles[:runs]
    print(f"  resuming {noise_id} at {len(bundles) + 1}/{runs}; this sitting stops at {goal}.", flush=True)
    inj.reset_all(net)
    for index in range(len(bundles), goal):
        print(f"  run {index + 1}/{runs} ...", flush=True)
        started = time.monotonic()
        try:
            bundle = _call_with_timeout(
                SAMPLE_TIMEOUT_SECONDS,
                lambda: _slim(oam.run_all(
                    net,
                    test_ids=test_ids,
                    ping_count=CALIBRATION_PINGS,
                    ping_interval=0.2,
                    speed_seconds=1,
                )),
            )
        except (UnicodeDecodeError, TimeoutError) as exc:
            print(f"  run {index + 1}/{runs} skipped: {exc}", flush=True)
            recover_net(net)
            continue
        _append_sample(sample_path, noise_id, index, bundle)
        bundles.append(bundle)
        print(f"  run {index + 1}/{runs} done in {time.monotonic() - started:.0f}s", flush=True)
    return bundles


def calibrate(noise_ids, runs, eval_runs, seed=0, quick=False, batch=CALIBRATION_BATCH, fill_speed=False):
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
            speed_left = False
            try:
                collected = collect_healthy(net, total, test_ids, noise_id=noise_id, batch=batch)
                if fill_speed and len(collected) >= total:
                    collected, speed_left = fill_missing_speed(
                        net, collected, noise_id, batch=batch,
                    )
            finally:
                try:
                    clear_noise(net, noise_applied, state["links"])
                except Exception as exc:
                    print(f"clear_noise failed: {exc}", flush=True)
                recover_net(net)
            if len(collected) < total or speed_left:
                if speed_left:
                    print(
                        f"  {noise_id} still has saved runs with no T10 bitrate. "
                        "Run the same command with --fill-speed again.",
                        flush=True,
                    )
                else:
                    print(
                        f"  saved {len(collected)}/{total} for {noise_id}. "
                        "Run the same command again to continue. The VM can be off between sittings.",
                        flush=True,
                    )
                return None
            missing_speed = sum(1 for bundle in collected if _t10_mbit(bundle) is None)
            if missing_speed:
                print(
                    f"  {noise_id} has {missing_speed} saved runs with no usable T10 bitrate. "
                    "Run the same command with --fill-speed.",
                    flush=True,
                )
                return None
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


def _bundles_for(path, noise_id):
    """Saved runs for one noise level, in collection order. The last copy of an index wins."""
    chosen = {}
    for item in _read_rows(path):
        if item.get("noise_id") != noise_id or "bundle" not in item:
            continue
        index = item.get("index")
        if not isinstance(index, int):
            index = len(chosen)
        chosen[index] = item["bundle"]
    return [chosen[index] for index in sorted(chosen)]


def _reprint_only(noise_ids, runs, eval_runs, sample_path=SAMPLES_PATH, from_saved=False):
    """True when this launch must not start Mininet."""
    if from_saved:
        return True
    total = runs + eval_runs
    return all(len(_bundles_for(sample_path, noise_id)) >= total for noise_id in noise_ids)


def report_from_saved(noise_ids, runs, eval_runs, seed=0, sample_path=SAMPLES_PATH):
    """Build the calibration report from the checkpoint. Does not start Mininet."""
    reports = {}
    total = runs + eval_runs
    for noise_id in noise_ids:
        bundles = _bundles_for(sample_path, noise_id)
        if len(bundles) < total:
            print(
                f"  {noise_id} has {len(bundles)}/{total} saved runs. No report.",
                flush=True,
            )
            continue
        bundles = bundles[:total]
        usable = sum(1 for bundle in bundles[:runs] if _t10_mbit(bundle) is not None)
        report = build_report(
            bundles[:runs],
            bundles[runs:],
            noise_id,
            load_thresholds(noise_id),
            True,
            seed,
        )
        reports[noise_id] = report
        print(
            f"  {noise_id}: T10 fit {usable}/{runs}, "
            f"eval false alarms {report['eval_false_alarms']}/{report['eval_runs']} "
            f"({report['eval_false_alarm_rate']})",
            flush=True,
        )
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
    parser.add_argument("--batch", type=int, default=CALIBRATION_BATCH, help="new samples to collect before exiting")
    parser.add_argument("--fill-speed", action="store_true", help="measure T10 only where a saved run has no bitrate")
    parser.add_argument("--from-saved", action="store_true", help="reprint the report from saved runs; does not start Mininet")
    args = parser.parse_args()

    noise_ids = ["N0", "N1", "N2"] if args.noise == "all" else [args.noise]
    if _reprint_only(noise_ids, args.runs, args.eval_runs, from_saved=args.from_saved):
        if args.write_thresholds:
            print("Refusing --write-thresholds on a reprint. Paste the report first.", file=sys.stderr)
            sys.exit(2)
        if not args.from_saved:
            print("Saved runs already cover this sample. Reprinting. Mininet will not start.", flush=True)
        reports = report_from_saved(noise_ids, args.runs, args.eval_runs, seed=args.seed)
        if not reports:
            return
        payload = {"version": 1, "levels": reports}
        out = Path(args.report)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    if not _is_root():
        print("calibrate_healthy.py collects on the Mininet VM as root.", file=sys.stderr)
        sys.exit(2)

    reports = calibrate(
        noise_ids,
        args.runs,
        args.eval_runs,
        seed=args.seed,
        quick=args.quick,
        batch=args.batch,
        fill_speed=args.fill_speed,
    )
    if reports is None:
        return
    payload = {"version": 1, "levels": reports}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.write_thresholds:
        update_thresholds_yaml(reports)


if __name__ == "__main__":
    if "--from-saved" in sys.argv:
        main()
    elif "--check" in sys.argv or not _is_root():
        _self_check()
    else:
        main()
