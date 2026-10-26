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

Episode-aligned collection (10 pings, 3 s T10) writes
results/calibration_samples_episode.jsonl. Repeat the same command until
that file has 250 runs and every T10 bitrate is present:

    sudo python3 bench/calibrate_healthy.py --noise N0 --runs 200 --eval-runs 50 --batch 10 --fill-speed --samples results/calibration_samples_episode.jsonl

--fill-speed starts Mininet when T10 bitrates are missing. It does not
reprint. --from-saved reprints without Mininet. --write-thresholds is
refused unless the episode battery, 200 fit runs, a held-out eval set,
and false_alarm_rate_target are all met.
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
from bench.tests import (
    EPISODE_BATTERY,
    EPISODE_PING_COUNT,
    EPISODE_PING_INTERVAL,
    EPISODE_SPEED_SECONDS,
    LEGACY_BATTERY,
)
from bench.outcomes import grade_result, is_degraded, load_thresholds, load_yaml_thresholds
from bench.tests import _parse_ping, accepted_speed

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = ROOT / "config" / "thresholds.yaml"
SAMPLES_PATH = ROOT / "results" / "calibration_samples.jsonl"
SAMPLES_EPISODE_PATH = ROOT / "results" / "calibration_samples_episode.jsonl"
FULL_BATTERY = list(TEST_IDS)
MIN_CALIBRATION_RUNS = 200
CALIBRATION_BATCH = 10
SAMPLE_TIMEOUT_SECONDS = 90
SPEED_FILL_CAP = 10
FILL_SITTING_SECONDS = 90


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
        return {key: (0 if key == "n" else None) for key in ("n", "min", "p005", "p01", "p05", "p50", "p95", "p99", "p995", "max")}
    return {
        "n": len(present),
        "min": min(present),
        "p005": _percentile(present, 0.5),
        "p01": _percentile(present, 1),
        "p05": _percentile(present, 5),
        "p50": _percentile(present, 50),
        "p95": _percentile(present, 95),
        "p99": _percentile(present, 99),
        "p995": _percentile(present, 99.5),
        "max": max(present),
    }


# Upper-tail quantile for latency/loss limits, with the matching lower
# tail for the speed floor. A run alarms if any of the nine tests does,
# so a per-test p95 cannot hold a per-run rate of 5%.
FIT_QUANTILES = (("p95", "p05"), ("p99", "p01"), ("p995", "p005"), ("max", "min"))


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


def propose_thresholds(tests, existing, upper="p95", lower="p05"):
    """Turn calibration distributions into one noise level's limits.

    upper is the fit quantile for latency and loss limits, lower the one
    for the speed floor. Existing floors stay in place when the healthy
    tail is healthier than the floor, and when the speed tail is below
    the configured floor.
    """
    existing = existing or {}
    notes = []
    t1 = existing.get("T1_e2e_ping") or {}
    t1_fit = tests.get("T1_e2e_ping") or {}
    proposed = {
        "T1_e2e_ping": {
            "max_loss_percent": _upper(
                (t1_fit.get("packet_loss_percent") or {}).get(upper),
                t1.get("max_loss_percent", 5.0),
            ),
            "max_latency_ms": _upper(
                (t1_fit.get("latency_ms") or {}).get(upper),
                t1.get("max_latency_ms", 50.0),
            ),
        }
    }
    t2 = existing.get("T2_hop_ping") or {}
    t2_fit = tests.get("T2_hop_ping") or {}
    proposed["T2_hop_ping"] = {
        "max_loss_percent": _upper(
            (t2_fit.get("packet_loss_percent") or {}).get(upper),
            t2.get("max_loss_percent", 10.0),
        ),
        "max_latency_ms": _upper(
            (t2_fit.get("latency_ms") or {}).get(upper),
            t2.get("max_latency_ms", 80.0),
        ),
    }
    t8 = existing.get("T8_tcp_vs_ping") or {}
    proposed["T8_tcp_vs_ping"] = {"tcp_must_succeed": bool(t8.get("tcp_must_succeed", True))}
    t10 = existing.get("T10_speed") or {}
    floor = t10.get("min_mbit", 1.0)
    tail = ((tests.get("T10_speed") or {}).get("mbit_per_sec") or {}).get(lower)
    if tail is None:
        proposed["T10_speed"] = {"min_mbit": floor}
        notes.append("T10 had no bitrate; speed floor was not changed")
    elif tail * 0.9 < floor:
        proposed["T10_speed"] = {"min_mbit": floor}
        notes.append("T10 min_mbit kept at the configured floor")
    else:
        proposed["T10_speed"] = {"min_mbit": tail * 0.9}
    return proposed, notes


def fit_thresholds(calibration, existing, noise_id, target):
    """Smallest fit quantile whose fit-set per-run alarm rate meets target.

    Uses the calibration bundles only. Falls back to the max/min fit.
    """
    tests = summarize(calibration)
    runs = len(calibration)
    chosen = None
    for upper, lower in FIT_QUANTILES:
        proposed, notes = propose_thresholds(tests, existing, upper, lower)
        rate = (score_false_alarms(calibration, proposed, noise_id) / runs) if runs else None
        chosen = (upper, proposed, notes, rate)
        if rate is not None and rate <= target:
            break
    upper, proposed, notes, rate = chosen
    if rate is None or rate > target:
        notes = list(notes) + [f"fit-set alarm rate {rate} stays above {target} at the max fit"]
    return tests, upper, proposed, notes, rate


def _failed_read(row):
    """Saved ping output with no summary line: the read failed, ping did not report loss."""
    return isinstance(row, dict) and isinstance(row.get("raw"), str) and "packet loss" not in row["raw"]


def _no_reading(result):
    """A probe that did not return a usable number is not a false alarm."""
    test_id = result.get("test_id")
    if test_id == "T2_hop_ping":
        hops = list(result.get("hops_primary") or []) + list(result.get("hops_backup") or [])
        if any(_failed_read(hop) for hop in hops):
            return True
    if test_id == "T9_oneway_loss":
        if any(_failed_read(result.get(key)) for key in ("hostA_to_hostB", "hostB_to_hostA")):
            return True
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


def _background_enabled(noise_id):
    """True when config/noise.yaml turns background traffic on for this level."""
    if not noise_id:
        return False
    from bench.traffic import load_noise_levels

    level = load_noise_levels().get(noise_id) or {}
    return bool((level.get("background_traffic") or {}).get("enabled"))


def _counts_as_alarm(result, outcome, thresholds, background=False):
    """A healthy-network alarm.

    With background traffic enabled, packets waiting in a queue are the load
    from config/noise.yaml. A drop is still an alarm. With background off,
    backlog counts.
    """
    if _no_reading(result):
        return False
    if (
        background
        and result.get("test_id") == "T6_queue_stats"
        and str(outcome).startswith("backlog")
    ):
        return False
    return is_degraded(result.get("test_id"), outcome, thresholds)


def score_false_alarms(bundles, thresholds, noise_id=None):
    """Count eval bundles whose graded outcome is not healthy."""
    background = _background_enabled(noise_id)
    alarms = 0
    for bundle in bundles:
        for result in bundle.get("results") or []:
            outcome = grade_result(result, thresholds)
            if _counts_as_alarm(result, outcome, thresholds, background):
                alarms += 1
                break
    return alarms


def _alarm_key(test_id, outcome):
    """test_id plus the failing part. T2 names each failing hop, e.g. primary:r2=loss."""
    outcome = str(outcome)
    if test_id == "T2_hop_ping":
        bad = []
        for chunk in outcome.split(";"):
            path, _, hops = chunk.partition(":")
            bad += [f"{path}:{part}" for part in hops.split(",") if part and not part.endswith("=ok")]
        return f"{test_id}:{'+'.join(bad)}" if bad else test_id
    kind = outcome.split(":", 1)[0]
    return test_id if kind in ("", "ok") else f"{test_id}:{kind}"


def _alarm_values(result):
    """The numbers behind one alarm, so a stall, a lost reply and a real fault can be told apart."""
    def _ping(row):
        row = row or {}
        values = {
            key: row.get(key)
            for key in ("latency_ms", "latency_avg_ms", "packet_loss_percent", "rtt_ms", "latency_basis")
            if row.get(key) is not None
        }
        if not row.get("rtt_ms") and isinstance(row.get("raw"), str):
            values["raw"] = row["raw"][:300]
        return values

    test_id = result.get("test_id")
    if test_id == "T1_e2e_ping":
        return _ping(result)
    if test_id == "T2_hop_ping":
        return {
            f"{path}:{hop.get('dst_ip')}": _ping(hop)
            for path in ("primary", "backup")
            for hop in result.get(f"hops_{path}") or []
        }
    if test_id == "T9_oneway_loss":
        return {key: _ping(result.get(key)) for key in ("hostA_to_hostB", "hostB_to_hostA")}
    return {}


def _alarm_runs(bundles, thresholds, noise_id=None, first_index=0):
    """One entry per alarming run: its position and the first test that alarmed."""
    background = _background_enabled(noise_id)
    runs = []
    for offset, bundle in enumerate(bundles):
        for result in bundle.get("results") or []:
            outcome = grade_result(result, thresholds)
            if not _counts_as_alarm(result, outcome, thresholds, background):
                continue
            test_id = result.get("test_id")
            runs.append({
                "run": first_index + offset,
                "alarm": _alarm_key(test_id, outcome),
                "outcome": outcome,
                "values": _alarm_values(result),
            })
            break
    return runs


def _alarm_tests(bundles, thresholds, noise_id=None):
    found = {}
    for run in _alarm_runs(bundles, thresholds, noise_id):
        found[run["alarm"]] = found.get(run["alarm"], 0) + 1
    return found


def _battery_profile(bundles):
    for bundle in bundles or []:
        battery = bundle.get("battery")
        if isinstance(battery, dict) and battery.get("ping_count") is not None:
            return dict(battery)
    return dict(LEGACY_BATTERY)


def _episode_battery(profile):
    return (
        profile.get("ping_count") == EPISODE_PING_COUNT
        and profile.get("speed_seconds") == EPISODE_SPEED_SECONDS
    )


def _false_alarm_target():
    with THRESHOLDS_PATH.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return float((doc.get("calibration") or {}).get("false_alarm_rate_target", 0.05))


def build_report(calibration, evaluation, noise_id, existing, full_battery, seed):
    calibration = [_median_latency(bundle) for bundle in calibration]
    evaluation = [_median_latency(bundle) for bundle in evaluation]
    target = _false_alarm_target()
    tests, fit_quantile, proposed, notes, fit_rate = fit_thresholds(calibration, existing, noise_id, target)
    alarms = score_false_alarms(evaluation, proposed, noise_id)
    eval_runs = len(evaluation)
    battery_profile = _battery_profile(calibration)
    average_runs = _average_latency_runs(list(calibration) + list(evaluation))
    if average_runs:
        notes = list(notes) + [
            f"{average_runs} saved run(s) have no per-reply RTTs; their ping latency is the average, not the median"
        ]
    return {
        "noise_id": noise_id,
        "calibration_runs": len(calibration),
        "eval_runs": eval_runs,
        "full_battery": full_battery,
        "seed": seed,
        "used_evaluation_faults": False,
        "battery_profile": battery_profile,
        "episode_aligned_battery": _episode_battery(battery_profile),
        "tests": tests,
        "proposed_thresholds": proposed,
        "fit_quantile": fit_quantile,
        "fit_false_alarm_rate": fit_rate,
        "latency_average_runs": average_runs,
        "failed_read_runs": sum(
            1 for bundle in list(calibration) + list(evaluation)
            if any(
                result.get("test_id") in ("T2_hop_ping", "T9_oneway_loss") and _no_reading(result)
                for result in bundle.get("results") or []
            )
        ),
        "threshold_notes": notes,
        "eval_false_alarms": alarms,
        "eval_false_alarm_rate": (alarms / eval_runs) if eval_runs else None,
        "eval_alarm_tests": _alarm_tests(evaluation, proposed, noise_id),
        "eval_alarm_runs": _alarm_runs(evaluation, proposed, noise_id, first_index=len(calibration)),
        "false_alarm_rate_target": target,
    }


def ready_to_write(report):
    target = report.get("false_alarm_rate_target", _false_alarm_target())
    rate = report.get("eval_false_alarm_rate")
    return (
        report.get("full_battery") is True
        and report.get("calibration_runs", 0) >= MIN_CALIBRATION_RUNS
        and report.get("eval_runs", 0) > 0
        and report.get("used_evaluation_faults") is False
        and report.get("episode_aligned_battery") is True
        and rate is not None
        and rate <= target
    )


def update_thresholds_yaml(reports, path=THRESHOLDS_PATH):
    """Replace limits for levels that passed the write guard."""
    refused = []
    for noise, report in reports.items():
        if ready_to_write(report):
            continue
        rate = report.get("eval_false_alarm_rate")
        target = report.get("false_alarm_rate_target", _false_alarm_target())
        if not report.get("episode_aligned_battery"):
            refused.append(f"{noise} (legacy 3-ping/1s battery; re-collect to {SAMPLES_EPISODE_PATH.name})")
        elif rate is None or rate > target:
            refused.append(f"{noise} (false-alarm rate {rate} > target {target})")
        else:
            refused.append(noise)
    if refused:
        raise ValueError(
            "refusing to write thresholds for " + ", ".join(refused)
            + ": need episode-aligned battery, 200 fit runs, eval set, and false-alarm rate at or below target"
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
            "fit_quantile": report.get("fit_quantile"),
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
    assert junk["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 20
    assert junk["tests"]["T10_speed"]["mbit_per_sec"]["max"] == 8.0
    assert junk["eval_false_alarms"] == 1

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
    assert report["fit_quantile"] == "p95" and report["fit_false_alarm_rate"] == 0.0

    # Five T1 tails and five different T2 tails: each test is under its p95
    # limit 95% of the time, but 10% of runs alarm. p99 holds the target.
    tails = (300.0, 400.0, 500.0, 600.0, 700.0)
    spread = []
    for index in range(100):
        rows = _healthy_result()
        if index < 5:
            rows[0]["latency_ms"] = tails[index]
        if 5 <= index < 10:
            rows[1] = dict(rows[1], hops_primary=[{"dst_ip": "10.0.12.2", "latency_ms": tails[index - 5], "packet_loss_percent": 0.0}])
        spread.append({"results": rows})
    quiet_eval = [{"results": _healthy_result()}]
    loud_eval = [{"results": _healthy_result(latency=10000.0)}]
    wide = build_report(spread, quiet_eval, "N0", existing, True, 0)
    assert wide["fit_quantile"] == "p99", wide["fit_quantile"]
    assert wide["fit_false_alarm_rate"] == 0.02, wide["fit_false_alarm_rate"]
    assert score_false_alarms(spread, propose_thresholds(wide["tests"], existing)[0], "N0") == 10
    loud = build_report(spread, loud_eval, "N0", existing, True, 0)
    assert loud["proposed_thresholds"] == wide["proposed_thresholds"]
    assert loud["eval_false_alarms"] == 1 and wide["eval_false_alarms"] == 0

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
    from network.controller import BASELINE_LINK_PARAMS
    link = float(BASELINE_LINK_PARAMS["bw"])
    assert _summary_mbit(f"[  5]   0.00-1.00   sec  1.00 MBytes  {link} Mbits/sec\n", 1) == link
    assert _summary_mbit(f"[  5]   0.00-1.00   sec  1.00 MBytes  {link + 1} Mbits/sec\n", 1) is None
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
    from bench.tests import _IPERF_TIMEOUT, T10_speed, measure_speed_once
    once = inspect.getsource(measure_speed_once)
    assert "_IPERF_TIMEOUT} 8 iperf3" in once and "-t 1" in once and "-i 1" not in once
    battery = inspect.getsource(T10_speed)
    assert "_IPERF_TIMEOUT} 20 iperf3" in battery
    assert _IPERF_TIMEOUT.startswith("timeout -s INT -k ")
    assert "wait_for_listener" in filler

    assert ready_to_write(report) is False
    high_far = dict(report)
    high_far.update(
        {
            "full_battery": True,
            "calibration_runs": MIN_CALIBRATION_RUNS,
            "eval_runs": 50,
            "episode_aligned_battery": True,
            "eval_false_alarm_rate": 0.25,
            "false_alarm_rate_target": 0.05,
        }
    )
    assert ready_to_write(high_far) is False
    try:
        update_thresholds_yaml({"N0": report}, THRESHOLDS_PATH)
        raise AssertionError("short sample was written")
    except ValueError:
        pass
    assert THRESHOLDS_PATH.read_text(encoding="utf-8") == before

    logic = (
        inspect.getsource(summarize)
        + inspect.getsource(propose_thresholds)
        + inspect.getsource(score_false_alarms)
        + inspect.getsource(fit_thresholds)
    )
    assert "observability" not in logic
    assert "faults.yaml" not in logic
    assert "evaluation" not in inspect.getsource(fit_thresholds)
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
    assert reprinted["N0"]["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 3
    assert reprinted["N0"]["eval_false_alarms"] == 0
    assert "create_network" not in inspect.getsource(report_from_saved)
    waiting = _healthy_result()
    for item in waiting:
        if item["test_id"] == "T6_queue_stats":
            item["queues"]["r1/r1-eth1"]["backlog_packets"] = 4
    assert score_false_alarms([{"results": waiting}], existing, "N0") == 1
    assert score_false_alarms([{"results": waiting}], existing, "N2") == 0
    dropped = _healthy_result()
    for item in dropped:
        if item["test_id"] == "T6_queue_stats":
            item["queues"]["r1/r1-eth1"]["drops"] = 3
    assert score_false_alarms([{"results": dropped}], existing, "N2") == 1
    assert _reprint_only(["N0"], 3, 1, saved, False) is True
    assert _reprint_only(["N0"], 200, 50, saved, False) is False
    assert _reprint_only(["N0"], 200, 50, saved, True) is True
    assert _checkpoint_complete(["N0"], 3, 1, saved) is True
    assert _needs_speed_fill(["N0"], 3, 1, saved) is True
    saved.unlink()
    saved.parent.rmdir()

    import bench.outcomes as outcomes

    report_path = Path(tempfile.mkdtemp()) / "calibration_report.json"
    report_path.write_text(
        json.dumps(
            {
                "version": 1,
                "levels": {
                    "N0": {
                        "proposed_thresholds": {
                            "T1_e2e_ping": {"max_loss_percent": 9.0, "max_latency_ms": 99.0},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    previous = outcomes._REPORT
    outcomes._REPORT = report_path
    try:
        assert load_thresholds("N0")["T1_e2e_ping"]["max_latency_ms"] == 99.0
    finally:
        outcomes._REPORT = previous
    report_path.unlink()
    report_path.parent.rmdir()

    # A run skipped mid-sitting must not make the next sitting reuse an index.
    gapped = Path(tempfile.mkdtemp()) / "samples.jsonl"
    gap_rows = [
        {"noise_id": "N0", "index": index, "bundle": {"results": _healthy_result(mbit=mbit)}}
        for index, mbit in ((0, 8.0), (1, 8.0), (3, None), (3, 7.0))
    ]
    gap_rows.append({"noise_id": "N2", "index": 9, "bundle": {"results": []}})
    gapped.write_text("".join(json.dumps(item) + "\n" for item in gap_rows), encoding="utf-8")
    assert len(_bundles_for(gapped, "N0")) == 3
    assert _next_index(gapped, "N0") == 4 and _next_index(gapped, "N1") == 0
    assert _t10_mbit(_bundles_for(gapped, "N0")[2]) == 7.0
    _replace_noise_bundles(gapped, "N0", {2: {"results": _healthy_result(mbit=6.0)}})
    assert _t10_mbit(_bundles_for(gapped, "N0")[2]) == 6.0
    assert _t10_mbit(_read_rows(gapped)[2]["bundle"]) is None
    assert _bundles_for(gapped, "N2") == [{"results": []}]
    live_source = inspect.getsource(collect_healthy)
    assert "_next_index" in live_source and "_bundles_for" in live_source
    gapped.unlink()
    gapped.parent.rmdir()

    merged_path = Path(tempfile.mkdtemp()) / "calibration_report.json"
    _emit_reports({"N0": {"proposed_thresholds": {"T10_speed": {"min_mbit": 1.0}}}}, merged_path)
    _emit_reports({"N2": {"proposed_thresholds": {"T10_speed": {"min_mbit": 0.2}}}}, merged_path)
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert set(merged["levels"]) == {"N0", "N2"}, merged
    merged_path.unlink()
    merged_path.parent.rmdir()
    assert THRESHOLDS_PATH.read_text(encoding="utf-8") == before

    stall = "".join(f"icmp_seq={i} ttl=63 time={t} ms\n" for i, t in enumerate([11.0, 11.2, 240.0], 1))
    stall += "rtt min/avg/max/mdev = 11.0/87.4/240.0/107.9 ms\n"
    saved = _median_latency({"results": [
        {"test_id": "T1_e2e_ping", "latency_ms": 87.4, "packet_loss_percent": 0.0},
        {"test_id": "T2_hop_ping", "hops_primary": [{"latency_ms": 87.4, "raw": stall}]},
        {"test_id": "T9_oneway_loss", "hostA_to_hostB": {"latency_ms": 87.4, "raw": stall}},
    ]})
    t1, t2, t9 = saved["results"]
    assert t1["latency_ms"] == 87.4 and t1["latency_basis"] == "average"
    assert t2["hops_primary"][0]["latency_ms"] == 11.2
    assert t9["hostA_to_hostB"]["latency_ms"] == 11.2
    fresh = {"test_id": "T1_e2e_ping", "latency_ms": 11.2, "rtt_ms": [11.0, 11.2, 240.0]}
    _median_latency({"results": [fresh]})
    assert "latency_basis" not in fresh
    assert _average_latency_runs([saved, {"results": [fresh]}]) == 1
    assert "rtt_ms" in _slim({"results": [dict(fresh, raw="x")]})["results"][0]
    old_run = {"results": [
        {"test_id": "T1_e2e_ping", "latency_ms": 87.4, "packet_loss_percent": 0.0},
        {"test_id": "T2_hop_ping", "hops_primary": [{"latency_ms": 87.4, "packet_loss_percent": 0.0, "raw": stall}]},
    ]}
    from_disk = build_report([json.loads(json.dumps(old_run))], [json.loads(json.dumps(old_run))], "N0", {}, True, 0)
    assert from_disk["tests"]["T2_hop_ping"]["latency_ms"]["max"] == 11.2, from_disk["tests"]["T2_hop_ping"]
    assert from_disk["latency_average_runs"] == 2
    assert _alarm_key("T2_hop_ping", "primary:r1=ok,r2=loss;backup:r1=ok,r3=delay") == (
        "T2_hop_ping:primary:r2=loss+backup:r3=delay"
    )
    assert _alarm_key("T1_e2e_ping", "latency") == "T1_e2e_ping:latency"
    lossy = {"results": [{"test_id": "T2_hop_ping", "hops_primary": [
        {"dst_ip": "10.0.12.2", "latency_ms": 11.0, "packet_loss_percent": 30.0, "rtt_ms": [11.0]},
    ]}]}
    flagged = _alarm_runs([lossy], {"T2_hop_ping": {"max_loss_percent": 10.0, "max_latency_ms": 80.0}}, first_index=200)
    assert flagged[0]["run"] == 200 and flagged[0]["alarm"].endswith("=loss"), flagged
    assert flagged[0]["values"]["primary:10.0.12.2"]["packet_loss_percent"] == 30.0
    empty_read = {"results": [{"test_id": "T2_hop_ping", "hops_primary": [
        {"dst_ip": "10.0.1.1", "latency_ms": None, "packet_loss_percent": 100.0, "raw": ""},
        {"dst_ip": "10.0.12.2", "latency_ms": 11.0, "packet_loss_percent": 0.0, "raw": stall},
    ]}]}
    real_loss = {"results": [{"test_id": "T2_hop_ping", "hops_primary": [
        {"dst_ip": "10.0.1.1", "latency_ms": None, "packet_loss_percent": 100.0,
         "raw": "10 packets transmitted, 0 received, 100% packet loss, time 9000ms\n"},
        {"dst_ip": "10.0.12.2", "latency_ms": 11.0, "packet_loss_percent": 0.0, "raw": stall},
    ]}]}
    summary = stall + "3 packets transmitted, 3 received, 0% packet loss, time 400ms\n"
    for bundle in (empty_read, real_loss):
        bundle["results"][0]["hops_primary"][1]["raw"] = summary
    limits = {"T2_hop_ping": {"max_loss_percent": 10.0, "max_latency_ms": 80.0}}
    assert score_false_alarms([empty_read], limits) == 0
    assert score_false_alarms([real_loss], limits) == 1
    assert _alarm_runs([real_loss], limits)[0]["values"]["primary:10.0.1.1"]["raw"].startswith("10 packets")
    assert build_report([empty_read], [real_loss], "N0", {}, True, 0)["failed_read_runs"] == 1

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


def _median_row(row):
    """Re-derive latency_ms as the median reply RTT for a row saved before
    that change. A row without raw ping output keeps the ping average."""
    if not isinstance(row, dict) or "rtt_ms" in row or row.get("latency_ms") is None:
        return
    if not row.get("raw"):
        row["latency_basis"] = "average"
        return
    parsed = _parse_ping(row["raw"])
    for key in ("latency_ms", "latency_avg_ms", "rtt_ms"):
        row[key] = parsed[key]


def _median_latency(bundle):
    """Saved T2 hops and T9 directions kept raw output. Saved T1 did not."""
    for result in bundle.get("results") or []:
        test_id = result.get("test_id")
        if test_id == "T1_e2e_ping":
            _median_row(result)
        elif test_id == "T2_hop_ping":
            for key in ("hops", "hops_primary", "hops_backup"):
                for hop in result.get(key) or []:
                    _median_row(hop)
        elif test_id == "T9_oneway_loss":
            for key in ("hostA_to_hostB", "hostB_to_hostA"):
                _median_row(result.get(key))
    return bundle


def _average_latency_runs(bundles):
    """Runs where some ping latency is still the average, not the median."""
    def _marked(result):
        rows = [result, result.get("hostA_to_hostB"), result.get("hostB_to_hostA")]
        for key in ("hops", "hops_primary", "hops_backup"):
            rows += list(result.get(key) or [])
        return any(isinstance(row, dict) and row.get("latency_basis") == "average" for row in rows)

    return sum(
        1 for bundle in bundles
        if any(_marked(result) for result in bundle.get("results") or [])
    )


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


def _append_sample(path, noise_id, index, bundle, battery=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"noise_id": noise_id, "index": index, "bundle": bundle}
    if battery:
        row["battery"] = battery
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
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
    """Replace several saved bundles for one noise level in a single write.

    Keys are positions in the _bundles_for() list. Each maps to a saved
    run index; the last row with that index is the one replaced.
    """
    rows = _read_rows(path)
    last_row = {}
    for pos, item in enumerate(rows):
        if item.get("noise_id") == noise_id and isinstance(item.get("index"), int):
            last_row[item["index"]] = pos
    ordered = sorted(last_row)
    for position, bundle in updates.items():
        if position >= len(ordered):
            continue
        rows[last_row[ordered[position]]]["bundle"] = bundle
    _write_rows(path, rows)


def _next_index(path, noise_id):
    """One past the highest saved run index, so a skipped run never reuses one."""
    indexes = [
        item["index"]
        for item in _read_rows(path)
        if item.get("noise_id") == noise_id and isinstance(item.get("index"), int)
    ]
    return max(indexes) + 1 if indexes else 0


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
    text = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    lines = [line for line in text if "error" in line.lower() or "interrupt" in line.lower()]
    if not lines:
        tail = " | ".join(text[-2:])
        print(f"  no bitrate{': ' + tail if tail else ' (no client output)'}", flush=True)
        return
    for line in lines[-2:]:
        print(f"  {line}", flush=True)


def fill_missing_speed(net, bundles, noise_id, sample_path=SAMPLES_PATH, batch=CALIBRATION_BATCH):
    """Fill missing T10 readings, then return so the VM can stop.

    A long-lived iperf3 on 5201 dies after a few clients (Broken pipe).
    Each fill uses a one-off `iperf3 -s -1` after freeing that port.
    """
    del batch
    from bench import tests as oam
    from bench.mininet_util import (
        finish_one_off,
        free_tcp_port,
        one_off_log_tail,
        start_one_off_iperf3,
        stop_iperf3_server,
        wait_for_listener,
    )

    missing = [index for index, bundle in enumerate(bundles) if _t10_mbit(bundle) is None]
    if not missing:
        return bundles, False
    count = min(len(missing), SPEED_FILL_CAP)
    print(
        f"  {noise_id} has {len(missing)} saved runs with no T10 bitrate. "
        f"Collecting up to {count} in {FILL_SITTING_SECONDS}s, then this command exits.",
        flush=True,
    )
    stop_iperf3_server(net, port=5201)
    free_tcp_port(net, 5201)
    found = []
    attempts = 0
    deadline = time.monotonic() + FILL_SITTING_SECONDS
    while len(found) < count and attempts < count + 8 and time.monotonic() < deadline:
        attempts += 1
        server = start_one_off_iperf3(net, port=5201)
        listening = wait_for_listener(net, 5201)
        try:
            mbit, raw = oam.measure_speed_once(net, port=5201)
        finally:
            finish_one_off(server)
            free_tcp_port(net, 5201)
        failed = mbit is None or "unable to send" in (raw or "") or "error -" in (raw or "")
        if failed:
            if not listening:
                print("  server was not listening on 5201", flush=True)
            _print_iperf_tail(raw)
            server_tail = one_off_log_tail(5201, lines=2)
            if server_tail:
                print("  server: " + server_tail.replace("\n", " | "), flush=True)
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

    bundles = _bundles_for(sample_path, noise_id)
    goal = _session_goal(len(bundles), runs, batch)
    if len(bundles) >= runs:
        print(f"  {noise_id} already has {len(bundles)} saved runs.", flush=True)
        return bundles[:runs]
    print(f"  resuming {noise_id} at {len(bundles) + 1}/{runs}; this sitting stops at {goal}.", flush=True)
    inj.reset_all(net)
    next_index = _next_index(sample_path, noise_id)
    for _attempt in range(goal - len(bundles)):
        index = next_index
        print(f"  run {len(bundles) + 1}/{runs} ...", flush=True)
        started = time.monotonic()
        try:
            raw = _call_with_timeout(
                SAMPLE_TIMEOUT_SECONDS,
                lambda: oam.run_all(
                    net,
                    test_ids=test_ids,
                    ping_count=EPISODE_PING_COUNT,
                    ping_interval=EPISODE_PING_INTERVAL,
                    speed_seconds=EPISODE_SPEED_SECONDS,
                ),
            )
            bundle = _slim(raw)
            bundle["battery"] = dict(EPISODE_BATTERY)
        except (UnicodeDecodeError, TimeoutError) as exc:
            print(f"  run {len(bundles) + 1}/{runs} skipped: {exc}", flush=True)
            recover_net(net)
            continue
        _append_sample(sample_path, noise_id, index, bundle, battery=EPISODE_BATTERY)
        next_index += 1
        bundles.append(bundle)
        print(f"  run {len(bundles)}/{runs} done in {time.monotonic() - started:.0f}s", flush=True)
    return bundles


def resolve_sample_path(args):
    if args.samples:
        path = Path(args.samples)
        return path if path.is_absolute() else ROOT / path
    if args.legacy_samples or args.from_saved:
        return SAMPLES_PATH
    return SAMPLES_EPISODE_PATH


def calibrate(
    noise_ids,
    runs,
    eval_runs,
    seed=0,
    quick=False,
    batch=CALIBRATION_BATCH,
    fill_speed=False,
    sample_path=SAMPLES_EPISODE_PATH,
):
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
                collected = collect_healthy(
                    net, total, test_ids, noise_id=noise_id, batch=batch, sample_path=sample_path,
                )
                if fill_speed and len(collected) >= total:
                    collected, speed_left = fill_missing_speed(
                        net, collected, noise_id, batch=batch, sample_path=sample_path,
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
                load_yaml_thresholds(noise_id),
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


def _bundles_for(path, noise_id, minimum=0):
    """Saved runs for one noise level. The last copy of an index wins."""
    del minimum
    chosen = {}
    for item in _read_rows(path):
        if item.get("noise_id") != noise_id or "bundle" not in item:
            continue
        index = item.get("index")
        if isinstance(index, int):
            chosen[index] = item["bundle"]
    return [chosen[index] for index in sorted(chosen)]


def _missing_t10_in_checkpoint(sample_path, noise_id, total=None):
    bundles = _bundles_for(sample_path, noise_id)
    if total is not None and len(bundles) < total:
        return 0
    return sum(1 for bundle in bundles if _t10_mbit(bundle) is None)


def _checkpoint_complete(noise_ids, runs, eval_runs, sample_path):
    total = runs + eval_runs
    return all(len(_bundles_for(sample_path, noise_id)) >= total for noise_id in noise_ids)


def _needs_speed_fill(noise_ids, runs, eval_runs, sample_path):
    """Saved runs are complete but some T10 bitrates are still missing."""
    total = runs + eval_runs
    if not _checkpoint_complete(noise_ids, runs, eval_runs, sample_path):
        return False
    return any(_missing_t10_in_checkpoint(sample_path, noise_id, total) > 0 for noise_id in noise_ids)


def _reprint_only(noise_ids, runs, eval_runs, sample_path=SAMPLES_PATH, from_saved=False):
    """True when this launch must not start Mininet."""
    if from_saved:
        return True
    return _checkpoint_complete(noise_ids, runs, eval_runs, sample_path)


def _emit_reports(reports, report_path, write_thresholds=False):
    if not reports:
        return
    out = Path(report_path)
    if not out.is_absolute():
        out = ROOT / out
    levels = {}
    try:
        previous = json.loads(out.read_text(encoding="utf-8"))
        if isinstance(previous.get("levels"), dict):
            levels.update(previous["levels"])
    except (OSError, ValueError, AttributeError):
        pass
    levels.update(reports)
    payload = {"version": 1, "levels": levels}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if write_thresholds:
        update_thresholds_yaml(reports)


def report_from_saved(noise_ids, runs, eval_runs, seed=0, sample_path=SAMPLES_PATH):
    """Build the calibration report from the checkpoint. Does not start Mininet."""
    reports = {}
    total = runs + eval_runs
    for noise_id in noise_ids:
        bundles = _bundles_for(sample_path, noise_id, total)
        if len(bundles) < runs + 1:
            print(
                f"  {noise_id} has {len(bundles)}/{total} saved runs. No report.",
                flush=True,
            )
            continue
        eval_have = min(eval_runs, len(bundles) - runs)
        if len(bundles) < total:
            print(
                f"  {noise_id}: {len(bundles)} saved runs. Using {runs} fit and {eval_have} eval.",
                flush=True,
            )
        bundles = bundles[: runs + eval_have]
        report = build_report(
            bundles[:runs],
            bundles[runs:],
            noise_id,
            load_yaml_thresholds(noise_id),
            True,
            seed,
        )
        reports[noise_id] = report
        usable = report["tests"]["T10_speed"]["mbit_per_sec"]["n"]
        print(
            f"  {noise_id}: T10 fit {usable}/{runs}, "
            f"eval false alarms {report['eval_false_alarms']}/{report['eval_runs']} "
            f"({report['eval_false_alarm_rate']}) {report['eval_alarm_tests']}",
            flush=True,
        )
        if not report.get("episode_aligned_battery"):
            print(
                f"  {noise_id}: legacy battery profile {report.get('battery_profile')}. "
                f"Episode grading uses 10 pings and 3 s speed. Re-collect into "
                f"{SAMPLES_EPISODE_PATH.name} before --write-thresholds.",
                flush=True,
            )
        target = report.get("false_alarm_rate_target", _false_alarm_target())
        rate = report.get("eval_false_alarm_rate")
        if rate is not None and rate > target:
            print(
                f"  {noise_id}: false-alarm rate {rate} is above target {target}. "
                "Thresholds will not be written until this improves.",
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
    parser.add_argument(
        "--legacy-samples",
        action="store_true",
        help=f"read/write {SAMPLES_PATH.name} (VM checkpoint with 3 ping / 1 s speed)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="",
        help="checkpoint jsonl path (default: legacy file for --from-saved, episode file for new collection)",
    )
    args = parser.parse_args()
    sample_path = resolve_sample_path(args)

    noise_ids = ["N0", "N1", "N2"] if args.noise == "all" else [args.noise]
    if args.from_saved and not sample_path.is_file():
        print(
            f"No checkpoint at {sample_path}.\n"
            "On the VM, episode samples live under results/calibration_samples_episode.jsonl.",
            file=sys.stderr,
        )
        sys.exit(2)

    total = args.runs + args.eval_runs
    complete = _checkpoint_complete(noise_ids, args.runs, args.eval_runs, sample_path)
    missing = {
        noise_id: _missing_t10_in_checkpoint(sample_path, noise_id, total)
        for noise_id in noise_ids
    }
    fill_now = args.fill_speed and not args.from_saved and complete and any(missing.values())
    reprint = args.from_saved or (complete and not fill_now)

    if fill_now:
        print(
            "Checkpoint has missing T10 bitrates; starting Mininet speed fill "
            f"(up to {SPEED_FILL_CAP} per launch): {missing}",
            flush=True,
        )
    if reprint:
        if not args.from_saved:
            print("Saved runs already cover this sample. Reprinting. Mininet will not start.", flush=True)
        reports = report_from_saved(
            noise_ids, args.runs, args.eval_runs, seed=args.seed, sample_path=sample_path,
        )
        _emit_reports(reports, args.report, write_thresholds=args.write_thresholds)
        return

    if not _is_root():
        print("calibrate_healthy.py collects on the Mininet VM as root.", file=sys.stderr)
        sys.exit(2)

    print(f"Saving episode-aligned samples to {sample_path}", flush=True)
    reports = calibrate(
        noise_ids,
        args.runs,
        args.eval_runs,
        seed=args.seed,
        quick=args.quick,
        batch=args.batch,
        fill_speed=args.fill_speed,
        sample_path=sample_path,
    )
    if reports is None:
        return
    _emit_reports(reports, args.report, write_thresholds=args.write_thresholds)


if __name__ == "__main__":
    live_flags = ("-h", "--help", "--from-saved", "--fill-speed", "--write-thresholds")
    if any(flag in sys.argv for flag in live_flags):
        main()
    elif "--check" in sys.argv:
        _self_check()
    elif not _is_root():
        _self_check()
    else:
        main()
