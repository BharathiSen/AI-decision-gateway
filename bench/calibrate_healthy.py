"""
Calibrate healthy-network thresholds from H0 runs.

The fit sample and the false-alarm sample are different runs. Fault
ids, including the held-out set, are not inputs. A short or partial
sample cannot replace config/thresholds.yaml.

Offline check, from the repo root (a bare invocation is ALWAYS the offline
check, also under sudo):

    python bench/calibrate_healthy.py

Every collection or reprint writes its report to --report; the committed
results/calibration_report.json is protected and refused. Example reprint:

    python bench/calibrate_healthy.py --from-saved --noise all --report results/calibration/<run-id>.json

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

Methodology
-----------
1. H0 only: no fault is injected. Fault ids, including held-out ids, are
   never inputs.
2. Split: the first --runs bundles fit thresholds; the next --eval-runs
   bundles are held out and only scored.
3. Fit: per test and traffic level, the smallest upper quantile (p95 and
   up) whose fit-set alarm rate meets false_alarm_rate_target. Latency/loss
   limits follow the fit and may tighten a permissive configured floor only
   when the fit metric has at least MIN_N_TO_TIGHTEN samples. Loss limits are
   never finer than one lost probe in the episode ping battery
   (100/EPISODE_PING_COUNT percent). T10 min_mbit is never lowered below the
   configured floor when the healthy lower tail is slower than that floor.
   Backfilled/--fill-speed 1 s T10 rows are excluded from the fit distribution.
4. Score metrics (kept distinct):
   - eval_false_alarm_rate: graded degradation, rows with `error`, or
     unusable T10 (missing/errored/backfilled). Failed ping *reads* without
     `error` are not graded alarms (see eval_no_reading_*).
   - eval_no_reading_runs / eval_no_reading_tests: unusable readings.
   - eval_error_inclusive_false_alarm_rate: WRITE DECISION METRIC — any
     unusable reading or graded alarm. Episode no-op sees these as missing
     or degraded evidence.
5. --write-thresholds uses eval_error_inclusive_false_alarm_rate, requires
   first-try T10 completeness >= 0.95 (every run in the denominator; missing
   T10 object counts as unusable; backfill does not count as first-try), and
   requires episode-aligned battery.
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
# Empirical quantiles used to tighten below a configured floor need enough
# points; below this, keep the configured floor and record why.
MIN_N_TO_TIGHTEN = 30
# Episode ping battery reports loss in steps of one probe. A limit below
# that cannot be justified from a single T1/T2 reading (0% vs 10% for n=10).
LOSS_RESOLUTION_PERCENT = 100.0 / float(EPISODE_PING_COUNT)
# Report field naming the metric that gates --write-thresholds.
WRITE_DECISION_METRIC = "eval_error_inclusive_false_alarm_rate"


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


def _backfilled(result):
    """A T10 row filled after its first reading failed.

    Legacy fills (before filled_after_failure existed) are the bare
    {test_id, mbit_per_sec, checked} shape that only _store_speeds wrote.
    One-second --fill-speed readings are never first-try episode T10.
    """
    if result.get("test_id") != "T10_speed":
        return False
    if result.get("filled_after_failure") is True:
        return True
    if result.get("fill_not_episode_equivalent") is True:
        return True
    if result.get("checked") is not True:
        return False
    if "filled_after_failure" in result:
        return bool(result["filled_after_failure"])
    return set(result) <= {"test_id", "mbit_per_sec", "checked"}


def _observe(result):
    """Pull the numbers a healthy threshold can use. No fault label.

    Errored and backfilled T10 rows contribute no bitrate: a 1 s fill is not
    a 3 s episode measurement and must not tighten or loosen the floor.
    """
    test_id = result.get("test_id")
    observed = {"test_id": test_id}
    if result.get("error"):
        return observed
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
        if _backfilled(result):
            observed["mbit_per_sec"] = None
        else:
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


def _upper(measured, floor, *, n=None, kind="generic"):
    """Data-driven upper limit from the fit quantile.

    Returns (limit, note_or_None).

    Tightening below a configured floor needs n >= MIN_N_TO_TIGHTEN. Loss
    candidates are never finer than LOSS_RESOLUTION_PERCENT (one lost probe
    in the episode ping battery), so a zero-loss fit cannot propose
    max_loss_percent=0. A configured floor already below that resolution is
    kept when the fit is zero rather than loosened up to the resolution.
    """
    if measured is None:
        return floor, None
    raw = float(measured) * 1.1
    if kind == "loss":
        candidate = max(raw, LOSS_RESOLUTION_PERCENT)
    else:
        candidate = raw
    if floor is None:
        return candidate, None
    if candidate >= floor:
        if kind == "loss" and measured == 0 and floor < LOSS_RESOLUTION_PERCENT:
            return floor, (
                f"kept configured floor {floor}: zero-loss fit cannot justify a "
                f"limit below ping resolution {LOSS_RESOLUTION_PERCENT:g}%"
            )
        return candidate, None
    # candidate < floor: tighten only with enough samples.
    if n is not None and int(n) < MIN_N_TO_TIGHTEN:
        return floor, (
            f"kept configured floor {floor}: n={int(n)} < {MIN_N_TO_TIGHTEN} "
            f"needed to tighten toward {candidate:g}"
        )
    note = f"configured floor {floor} was more permissive than fit {candidate:g}"
    if kind == "loss" and raw < LOSS_RESOLUTION_PERCENT:
        note += (
            f"; clamped to ping resolution {LOSS_RESOLUTION_PERCENT:g}% "
            f"(one of {int(EPISODE_PING_COUNT)} episode probes)"
        )
    return candidate, note


def propose_thresholds(tests, existing, upper="p95", lower="p05"):
    """Turn calibration distributions into one noise level's limits.

    Latency/loss limits follow the fit quantile and may tighten a permissive
    configured floor when sample size allows (see _upper). The T10 min_mbit
    floor is never lowered below the configured value when the healthy lower
    tail is slower than that floor.
    """
    existing = existing or {}
    notes = []
    t1 = existing.get("T1_e2e_ping") or {}
    t1_fit = tests.get("T1_e2e_ping") or {}
    t1_loss_n = ((t1_fit.get("packet_loss_percent") or {}).get("n")) or 0
    t1_lat_n = ((t1_fit.get("latency_ms") or {}).get("n")) or 0
    t1_loss, note = _upper(
        (t1_fit.get("packet_loss_percent") or {}).get(upper),
        t1.get("max_loss_percent", 5.0),
        n=t1_loss_n,
        kind="loss",
    )
    if note:
        notes.append("T1 max_loss_percent: " + note)
    t1_lat, note = _upper(
        (t1_fit.get("latency_ms") or {}).get(upper),
        t1.get("max_latency_ms", 50.0),
        n=t1_lat_n,
        kind="latency",
    )
    if note:
        notes.append("T1 max_latency_ms: " + note)
    proposed = {
        "T1_e2e_ping": {
            "max_loss_percent": t1_loss,
            "max_latency_ms": t1_lat,
        }
    }
    t2 = existing.get("T2_hop_ping") or {}
    t2_fit = tests.get("T2_hop_ping") or {}
    t2_loss_n = ((t2_fit.get("packet_loss_percent") or {}).get("n")) or 0
    t2_lat_n = ((t2_fit.get("latency_ms") or {}).get("n")) or 0
    t2_loss, note = _upper(
        (t2_fit.get("packet_loss_percent") or {}).get(upper),
        t2.get("max_loss_percent", 10.0),
        n=t2_loss_n,
        kind="loss",
    )
    if note:
        notes.append("T2 max_loss_percent: " + note)
    t2_lat, note = _upper(
        (t2_fit.get("latency_ms") or {}).get(upper),
        t2.get("max_latency_ms", 80.0),
        n=t2_lat_n,
        kind="latency",
    )
    if note:
        notes.append("T2 max_latency_ms: " + note)
    proposed["T2_hop_ping"] = {
        "max_loss_percent": t2_loss,
        "max_latency_ms": t2_lat,
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

    Measurement failures are not healthy: a row with `error`, or a T10 with
    no usable first-try bitrate (missing/errored/backfilled), counts as an
    alarm. Other probes that returned no usable number (failed ping read
    without `error`) stay out of this counter and appear in no-reading /
    error-inclusive rates instead.

    With background traffic enabled, packets waiting in a queue are the load
    from config/noise.yaml. A drop is still an alarm. With background off,
    backlog counts.
    """
    if result.get("error"):
        return True
    if result.get("test_id") == "T10_speed" and _unusable(result):
        return True
    if _no_reading(result):
        return False
    if outcome is None:
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


def _unusable(result):
    """No usable first reading: errored, no number, or backfilled after a failure."""
    return bool(result.get("error")) or _no_reading(result) or _backfilled(result)


def score_no_readings(bundles):
    """(runs with at least one unusable reading, unusable count per test_id).

    These runs are healthy-network runs where a probe failed. In an episode
    the same failure grades as degraded (T10 'slow') or is missing evidence,
    so they are reported next to the false-alarm rate, never dropped.
    """
    runs = 0
    per_test = {}
    for bundle in bundles:
        bad = [r for r in bundle.get("results") or [] if _unusable(r)]
        if bad:
            runs += 1
        for row in bad:
            tid = row.get("test_id") or "?"
            per_test[tid] = per_test.get(tid, 0) + 1
    return runs, dict(sorted(per_test.items()))


def score_error_inclusive_alarms(bundles, thresholds, noise_id=None):
    """Runs that alarm OR carry an unusable reading: the episode-facing rate."""
    background = _background_enabled(noise_id)
    count = 0
    for bundle in bundles:
        rows = bundle.get("results") or []
        if any(_unusable(r) for r in rows):
            count += 1
            continue
        if any(_counts_as_alarm(r, grade_result(r, thresholds), thresholds, background) for r in rows):
            count += 1
    return count


FIRST_TRY_T10_TARGET = 0.95


def first_try_t10_completeness(bundles):
    """Share of runs whose T10 first reading is usable.

    Every run is in the denominator. A run with no T10 row, or with an
    errored/missing/backfilled T10, counts as unusable. Empty input → None.
    """
    if not bundles:
        return None
    ok = 0
    for bundle in bundles:
        rows = [r for r in bundle.get("results") or [] if r.get("test_id") == "T10_speed"]
        if rows and not _unusable(rows[0]):
            ok += 1
    return ok / len(bundles)


def _infer_full_battery(bundles):
    """True when every run carries the full OAM test id set."""
    if not bundles:
        return False
    needed = set(FULL_BATTERY)
    for bundle in bundles:
        ids = {row.get("test_id") for row in bundle.get("results") or []}
        if not needed.issubset(ids):
            return False
    return True


def _false_alarm_target():
    with THRESHOLDS_PATH.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return float((doc.get("calibration") or {}).get("false_alarm_rate_target", 0.05))


def build_report(
    calibration,
    evaluation,
    noise_id,
    existing,
    full_battery,
    seed,
    checkpoint_skipped_lines=0,
):
    calibration = [_median_latency(bundle) for bundle in calibration]
    evaluation = [_median_latency(bundle) for bundle in evaluation]
    all_runs = list(calibration) + list(evaluation)
    if full_battery is None:
        full_battery = _infer_full_battery(all_runs)
    target = _false_alarm_target()
    tests, fit_quantile, proposed, notes, fit_rate = fit_thresholds(calibration, existing, noise_id, target)
    alarms = score_false_alarms(evaluation, proposed, noise_id)
    eval_runs = len(evaluation)
    no_reading_runs, no_reading_tests = score_no_readings(evaluation)
    fit_no_reading_runs, _ = score_no_readings(calibration)
    inclusive = score_error_inclusive_alarms(evaluation, proposed, noise_id)
    graded_rate = (alarms / eval_runs) if eval_runs else None
    inclusive_rate = (inclusive / eval_runs) if eval_runs else None
    t10_completeness = first_try_t10_completeness(all_runs)
    battery_profile = _battery_profile(calibration)
    episode_aligned = _episode_battery(battery_profile)
    average_runs = _average_latency_runs(all_runs)
    if average_runs:
        notes = list(notes) + [
            f"{average_runs} saved run(s) have no per-reply RTTs; their ping latency is the average, not the median"
        ]
    if t10_completeness is not None and t10_completeness < FIRST_TRY_T10_TARGET:
        notes = list(notes) + [
            f"first-try T10 completeness {t10_completeness:.3f} < {FIRST_TRY_T10_TARGET}; "
            "backfill must not count toward acceptance"
        ]
    t10_backfilled = sum(
        1 for bundle in all_runs
        for result in bundle.get("results") or []
        if result.get("test_id") == "T10_speed" and _backfilled(result)
    )
    if t10_backfilled:
        notes = list(notes) + [
            f"{t10_backfilled} T10 row(s) are --fill-speed / backfill (1 s); "
            "not equivalent to episode 3 s first-try measurements"
        ]
    return {
        "noise_id": noise_id,
        "calibration_runs": len(calibration),
        "eval_runs": eval_runs,
        "full_battery": bool(full_battery),
        "seed": seed,
        "used_evaluation_faults": False,
        "battery_profile": battery_profile,
        "episode_aligned_battery": episode_aligned,
        "checkpoint_skipped_lines": int(checkpoint_skipped_lines or 0),
        "tests": tests,
        "proposed_thresholds": proposed,
        "fit_quantile": fit_quantile,
        "fit_false_alarm_rate": fit_rate,
        "latency_average_runs": average_runs,
        "failed_read_runs": sum(
            1 for bundle in all_runs
            if any(
                result.get("test_id") in ("T2_hop_ping", "T9_oneway_loss") and _no_reading(result)
                for result in bundle.get("results") or []
            )
        ),
        "t10_backfilled_runs": t10_backfilled,
        "threshold_notes": notes,
        # Graded FAR: degradation / error rows / unusable T10. Reported, not the write gate.
        "eval_false_alarms": alarms,
        "eval_false_alarm_rate": graded_rate,
        "eval_no_reading_runs": no_reading_runs,
        "eval_no_reading_tests": no_reading_tests,
        "fit_no_reading_runs": fit_no_reading_runs,
        # Inclusive FAR: WRITE DECISION METRIC (also see write_decision_metric).
        "eval_error_inclusive_alarms": inclusive,
        "eval_error_inclusive_false_alarm_rate": inclusive_rate,
        "write_decision_metric": WRITE_DECISION_METRIC,
        "write_decision_false_alarm_rate": inclusive_rate,
        "first_try_t10_completeness": t10_completeness,
        "first_try_t10_target": FIRST_TRY_T10_TARGET,
        "eval_alarm_tests": _alarm_tests(evaluation, proposed, noise_id),
        "eval_alarm_runs": _alarm_runs(evaluation, proposed, noise_id, first_index=len(calibration)),
        "false_alarm_rate_target": target,
    }


def ready_to_write(report):
    """True when --write-thresholds may replace committed limits for this level.

    Gated on write_decision_metric (error-inclusive FAR), first-try T10
    completeness, episode-aligned battery, and sample size — not on graded FAR alone.
    """
    target = report.get("false_alarm_rate_target", _false_alarm_target())
    rate = report.get(WRITE_DECISION_METRIC)
    if rate is None:
        rate = report.get("eval_error_inclusive_false_alarm_rate", report.get("eval_false_alarm_rate"))
    t10_target = report.get("first_try_t10_target", FIRST_TRY_T10_TARGET)
    t10_completeness = report.get("first_try_t10_completeness")
    return (
        report.get("full_battery") is True
        and report.get("calibration_runs", 0) >= MIN_CALIBRATION_RUNS
        and report.get("eval_runs", 0) > 0
        and report.get("used_evaluation_faults") is False
        and report.get("episode_aligned_battery") is True
        and rate is not None
        and rate <= target
        and t10_completeness is not None
        and t10_completeness >= t10_target
    )


def update_thresholds_yaml(reports, path=THRESHOLDS_PATH):
    """Replace limits for levels that passed the write guard."""
    refused = []
    for noise, report in reports.items():
        if ready_to_write(report):
            continue
        rate = report.get("eval_error_inclusive_false_alarm_rate", report.get("eval_false_alarm_rate"))
        target = report.get("false_alarm_rate_target", _false_alarm_target())
        t10 = report.get("first_try_t10_completeness")
        t10_target = report.get("first_try_t10_target", FIRST_TRY_T10_TARGET)
        if not report.get("episode_aligned_battery"):
            refused.append(f"{noise} (legacy 3-ping/1s battery; re-collect to {SAMPLES_EPISODE_PATH.name})")
        elif rate is None or rate > target:
            refused.append(f"{noise} (false-alarm rate {rate} > target {target})")
        elif t10 is None or t10 < t10_target:
            refused.append(f"{noise} (first-try T10 completeness {t10} < {t10_target})")
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
    doc.setdefault("calibration", {}).setdefault("last_run", {}).update({
        noise: {
            "calibration_runs": report["calibration_runs"],
            "eval_runs": report["eval_runs"],
            "eval_false_alarm_rate": report["eval_false_alarm_rate"],
            "eval_error_inclusive_false_alarm_rate": report.get("eval_error_inclusive_false_alarm_rate"),
            "write_decision_metric": report.get("write_decision_metric", WRITE_DECISION_METRIC),
            "write_decision_false_alarm_rate": report.get("write_decision_false_alarm_rate"),
            "first_try_t10_completeness": report.get("first_try_t10_completeness"),
            "t10_backfilled_runs": report.get("t10_backfilled_runs"),
            "checkpoint_skipped_lines": report.get("checkpoint_skipped_lines"),
            "episode_aligned_battery": report.get("episode_aligned_battery"),
            "full_battery": report.get("full_battery"),
            "fit_quantile": report.get("fit_quantile"),
            "threshold_notes": report["threshold_notes"],
        }
        for noise, report in reports.items()
    })
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
    # n=20 < MIN_N_TO_TIGHTEN: keep configured floors rather than over-tighten.
    assert report["proposed_thresholds"]["T1_e2e_ping"]["max_latency_ms"] == 50.0
    assert report["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == 5.0
    assert any("n=20" in note and "tighten" in note for note in report["threshold_notes"]), report["threshold_notes"]
    assert report["write_decision_metric"] == WRITE_DECISION_METRIC
    assert report["write_decision_false_alarm_rate"] == report["eval_error_inclusive_false_alarm_rate"]
    assert "T10 min_mbit kept at the configured floor" not in report["threshold_notes"]

    contaminated = build_report(calm + held_out_eval, [], "N0", existing, True, 0)
    assert contaminated["tests"]["T1_e2e_ping"]["latency_ms"]["p95"] != 40.0
    assert contaminated["eval_false_alarms"] == 0

    # Genuine slow first-try T10 (not a --fill-speed backfill): never lower the floor.
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
    # Small zero-loss fit must not propose max_loss=0; keep floor until n is enough.
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
    # Missing T10 is not a healthy outcome for FAR.
    assert missing_speed["eval_false_alarms"] == 1
    assert missing_speed["first_try_t10_completeness"] == 0.0
    # H3-A: runs with no T10 object are unusable in the denominator.
    assert first_try_t10_completeness([]) is None
    no_t10 = [{"results": [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0}]} for _ in range(100)]
    assert first_try_t10_completeness(no_t10) == 0.0
    mixed_denom = list(no_t10) + [{"results": _healthy_result(mbit=8.0)} for _ in range(5)]
    assert abs(first_try_t10_completeness(mixed_denom) - (5 / 105)) < 1e-9
    mixed_kinds = [
        {"results": _healthy_result(mbit=8.0)},
        {"results": [{"test_id": "T10_speed", "mbit_per_sec": None, "error": "timeout"}]},
        {"results": [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0}]},
        {"results": [{"test_id": "T10_speed", "mbit_per_sec": 8.0, "checked": True, "filled_after_failure": True}]},
    ]
    assert first_try_t10_completeness(mixed_kinds) == 0.25
    legacy_incomplete = [{"results": [{"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0}, {"test_id": "T10_speed", "mbit_per_sec": 8.0}], "battery": dict(LEGACY_BATTERY)} for _ in range(10)]
    assert first_try_t10_completeness(legacy_incomplete) == 1.0
    assert _infer_full_battery(legacy_incomplete) is False
    assert _infer_full_battery([{"results": _healthy_result()} for _ in range(3)]) is True
    # H3-B: floor already below ping resolution stays put on zero-loss fit.
    big_calm = [{"results": _healthy_result()} for _ in range(MIN_N_TO_TIGHTEN)]
    big = build_report(big_calm, [{"results": _healthy_result()}], "N0", existing, None, 0)
    assert big["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == 5.0
    assert big["proposed_thresholds"]["T1_e2e_ping"]["max_latency_ms"] == 44.0
    assert any("more permissive than fit" in note for note in big["threshold_notes"]), big["threshold_notes"]
    assert any("ping resolution" in note or "zero-loss" in note for note in big["threshold_notes"]), big["threshold_notes"]
    assert big["full_battery"] is True
    # Permissive loss floor (above resolution) tightens to resolution, never to 0.
    loose_existing = dict(existing)
    loose_existing["T1_e2e_ping"] = {"max_loss_percent": 25.0, "max_latency_ms": 50.0}
    loose = build_report(big_calm, [{"results": _healthy_result()}], "N0", loose_existing, None, 0)
    assert loose["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == LOSS_RESOLUTION_PERCENT
    assert any("ping resolution" in note for note in loose["threshold_notes"]), loose["threshold_notes"]
    assert 0.0 not in (
        loose["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"],
        loose["proposed_thresholds"]["T2_hop_ping"]["max_loss_percent"],
    )
    # Insufficient n refuses tighten of a permissive floor.
    tiny = build_report(
        [{"results": _healthy_result()} for _ in range(5)],
        [{"results": _healthy_result()}],
        "N0",
        loose_existing,
        None,
        0,
    )
    assert tiny["proposed_thresholds"]["T1_e2e_ping"]["max_loss_percent"] == 25.0
    assert tiny["full_battery"] is True
    assert any("n=5" in note for note in tiny["threshold_notes"]), tiny["threshold_notes"]
    # Errors are counted, never dropped: a failed T10 and an errored T1 on a
    # healthy run raise both the plain FAR and the error-inclusive rate.
    ok_t1 = {"test_id": "T1_e2e_ping", "packet_loss_percent": 0.0, "latency_ms": 10.0, "raw": "0% packet loss"}
    failed_t10 = {"test_id": "T10_speed", "mbit_per_sec": None, "raw": "iperf3: error - unable to send"}
    errored_t1 = dict(ok_t1, error="ping command_error")
    mixed = [{"results": [ok_t1]}, {"results": [ok_t1, failed_t10]}, {"results": [errored_t1]}]
    runs_bad, per_test = score_no_readings(mixed)
    assert runs_bad == 2 and per_test == {"T10_speed": 1, "T1_e2e_ping": 1}, (runs_bad, per_test)
    assert score_false_alarms(mixed, existing, "N0") == 2, "missing/errored rows are not healthy"
    filled = _store_speeds([{"results": [ok_t1, failed_t10]}], [0], [8.5], failed_attempts=[2])[0]
    t10_row = next(r for r in filled["results"] if r["test_id"] == "T10_speed")
    assert t10_row["filled_after_failure"] and t10_row["fill_attempts_failed"] == 2
    assert t10_row.get("fill_not_episode_equivalent") is True
    assert t10_row.get("fill_speed_seconds") == 1
    assert "unable to send" in t10_row["original_error"]
    assert _backfilled(t10_row) and _backfilled({"test_id": "T10_speed", "mbit_per_sec": 8.0, "checked": True})
    assert not _backfilled({"test_id": "T10_speed", "mbit_per_sec": 8.0, "raw": "ok"})
    assert score_no_readings([filled])[0] == 1, "a backfilled run is not a clean first reading"
    assert first_try_t10_completeness([filled]) == 0.0, "backfill is not a first-try T10 reading"
    assert score_false_alarms([filled], existing, "N0") == 1, "1s fill is not a healthy first-try T10"
    # Backfilled bitrate must not enter the fit distribution.
    fill_only = build_report(
        [{"results": _healthy_result(mbit=8.0)} for _ in range(19)] + [filled],
        [{"results": _healthy_result()}],
        "N0",
        existing,
        True,
        0,
    )
    assert fill_only["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 19, fill_only["tests"]["T10_speed"]
    assert fill_only["t10_backfilled_runs"] >= 1
    assert any("fill-speed" in note or "backfill" in note for note in fill_only["threshold_notes"])
    assert score_error_inclusive_alarms(mixed, existing, "N0") == 2
    inclusive_report = dict(
        report,
        full_battery=True,
        calibration_runs=MIN_CALIBRATION_RUNS,
        eval_runs=50,
        episode_aligned_battery=True,
        eval_error_inclusive_false_alarm_rate=0.5,
        eval_false_alarm_rate=0.0,
        first_try_t10_completeness=1.0,
        first_try_t10_target=FIRST_TRY_T10_TARGET,
    )
    assert not ready_to_write(inclusive_report), "error-inclusive rate must gate --write-thresholds"
    low_t10 = dict(
        report,
        full_battery=True,
        calibration_runs=MIN_CALIBRATION_RUNS,
        eval_runs=50,
        episode_aligned_battery=True,
        eval_error_inclusive_false_alarm_rate=0.0,
        eval_false_alarm_rate=0.0,
        first_try_t10_completeness=0.5,
        first_try_t10_target=FIRST_TRY_T10_TARGET,
    )
    assert not ready_to_write(low_t10), "first-try T10 completeness must gate --write-thresholds"
    missing_completeness = dict(low_t10, first_try_t10_completeness=None, eval_error_inclusive_false_alarm_rate=0.0)
    assert not ready_to_write(missing_completeness), "absent T10 completeness must refuse write"

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
    # Fit: two good (above committed N0 floor) + one rejected junk bitrate.
    # Eval: healthy so FAR stays clean when T10 is present.
    for index, mbit in enumerate((9.0, 9.0, 2930.0, 9.0)):
        bundle = {"results": _healthy_result(mbit=mbit)}
        rows.append({"noise_id": "N0", "index": index, "bundle": bundle})
    saved.write_text(
        "not-json\n" + "".join(json.dumps(item) + "\n" for item in rows),
        encoding="utf-8",
    )
    reprinted = report_from_saved(["N0"], 3, 1, sample_path=saved)
    assert reprinted["N0"]["tests"]["T10_speed"]["mbit_per_sec"]["n"] == 2
    assert reprinted["N0"]["eval_false_alarms"] == 0
    assert reprinted["N0"]["first_try_t10_completeness"] == 0.75
    assert reprinted["N0"]["checkpoint_skipped_lines"] == 1
    assert reprinted["N0"]["full_battery"] is True
    assert reprinted["N0"]["write_decision_metric"] == WRITE_DECISION_METRIC
    assert "create_network" not in inspect.getsource(report_from_saved)
    # Explicit: rejected/missing T10 in eval counts as an alarm, not healthy.
    junk_eval = Path(tempfile.mkdtemp()) / "junk.jsonl"
    junk_rows = []
    for index, mbit in enumerate((9.0, 9.0, 9.0, 2930.0)):
        junk_rows.append({
            "noise_id": "N0",
            "index": index,
            "bundle": {"results": _healthy_result(mbit=mbit)},
        })
    junk_eval.write_text("".join(json.dumps(item) + "\n" for item in junk_rows), encoding="utf-8")
    junk_report = report_from_saved(["N0"], 3, 1, sample_path=junk_eval)
    assert junk_report["N0"]["eval_false_alarms"] == 1
    assert junk_report["N0"]["first_try_t10_completeness"] == 0.75
    junk_eval.unlink()
    junk_eval.parent.rmdir()
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
    yaml_latency = load_yaml_thresholds("N0")["T1_e2e_ping"]["max_latency_ms"]
    # Unaccepted report must not override committed YAML for grading.
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
        unaccepted = load_thresholds("N0")
        assert unaccepted["T1_e2e_ping"]["max_latency_ms"] == yaml_latency, unaccepted
        assert unaccepted["T1_e2e_ping"]["max_latency_ms"] != 99.0
        # Accepted level (ready_to_write) may override YAML.
        accepted_level = {
            "full_battery": True,
            "calibration_runs": MIN_CALIBRATION_RUNS,
            "eval_runs": 50,
            "used_evaluation_faults": False,
            "episode_aligned_battery": True,
            "eval_error_inclusive_false_alarm_rate": 0.0,
            "eval_false_alarm_rate": 0.0,
            "first_try_t10_completeness": 1.0,
            "first_try_t10_target": FIRST_TRY_T10_TARGET,
            "proposed_thresholds": {
                "T1_e2e_ping": {"max_loss_percent": 9.0, "max_latency_ms": 99.0},
            },
        }
        assert ready_to_write(accepted_level)
        report_path.write_text(
            json.dumps({"version": 1, "levels": {"N0": accepted_level}}),
            encoding="utf-8",
        )
        accepted = load_thresholds("N0")
        assert accepted["T1_e2e_ping"]["max_latency_ms"] == 99.0, accepted
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
    # Failed readings keep their evidence; successful ones stay slim.
    good_t10 = {"test_id": "T10_speed", "mbit_per_sec": 8.4, "raw": "x" * 5000}
    bad_t10 = {"test_id": "T10_speed", "mbit_per_sec": None, "raw": "y" * 5000 + "Broken pipe",
               "server_log": "iperf3: error - unable to receive"}
    errored = {"test_id": "T1_e2e_ping", "packet_loss_percent": 100.0, "error": "ping command_error", "raw": "bash: ping: not found"}
    slim = _slim({"results": [good_t10, bad_t10, errored]})["results"]
    assert "raw" not in slim[0] and "failure_evidence" not in slim[0]
    assert slim[1]["failure_evidence"]["raw"].endswith("Broken pipe")
    assert len(slim[1]["failure_evidence"]["raw"]) == RAW_TAIL_CHARS
    assert "unable to receive" in slim[1]["failure_evidence"]["server_log"]
    assert "server_log" not in slim[1] and "raw" not in slim[1]
    assert slim[2]["failure_evidence"]["raw"] == "bash: ping: not found"
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


RAW_TAIL_CHARS = 400


def _failure_evidence(result):
    """Raw-output tail for a result with no usable reading, else None.

    Successful rows are stored without raw output to keep the checkpoint
    small. A failed reading keeps the end of its output (and the iperf3
    server log) so the cause can be diagnosed later; without it, 101 VM T10
    failures could not be explained.
    """
    failed = bool(result.get("error")) or _no_reading(result)
    if result.get("test_id") == "T10_speed" and accepted_speed(result) is None:
        failed = True
    if not failed:
        return None
    evidence = {}
    for key in ("raw", "tcp_raw", "server_log"):
        text = result.get(key)
        if isinstance(text, str) and text.strip():
            evidence[key] = text[-RAW_TAIL_CHARS:]
    return evidence or {"raw": ""}


def _slim(bundle):
    results = []
    for result in bundle.get("results") or []:
        item = {key: value for key, value in result.items() if key not in ("raw", "tcp_raw", "server_log")}
        evidence = _failure_evidence(result)
        if evidence is not None:
            item["failure_evidence"] = evidence
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


def _read_rows_with_skips(path):
    """Return (rows, skipped_unreadable_line_count)."""
    rows = []
    skipped = 0
    if not path.exists():
        return rows, 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(item, dict) and "bundle" in item:
            rows.append(item)
        else:
            skipped += 1
    return rows, skipped


def _read_rows(path):
    rows, _skipped = _read_rows_with_skips(path)
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


def _store_speeds(bundles, indexes, intervals, failed_attempts=None):
    """Put a filled T10 bitrate into each bundle.

    The first reading failed on a healthy network, so the row keeps that:
    filled_after_failure, the original error text, and how many fill
    attempts failed before this one. Calibration counts these runs in the
    error-inclusive false-alarm rate instead of treating them as clean.
    """
    updates = {}
    failed_attempts = list(failed_attempts or [0] * len(indexes))
    for pos, (index, mbit) in enumerate(zip(indexes, intervals)):
        old = next((row for row in bundles[index].get("results") or [] if row.get("test_id") == "T10_speed"), {})
        results = [row for row in bundles[index].get("results") or [] if row.get("test_id") != "T10_speed"]
        results.append({
            "test_id": "T10_speed",
            "mbit_per_sec": mbit,
            "checked": True,
            "filled_after_failure": True,
            # measure_speed_once is a 1 s probe; episode T10 uses 3 s.
            "fill_speed_seconds": 1,
            "fill_not_episode_equivalent": True,
            "original_error": old.get("error") or (str(old.get("raw") or "")[-200:] or None),
            "fill_attempts_failed": int(failed_attempts[pos] if pos < len(failed_attempts) else 0),
        })
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
    if batch is None:
        count = len(missing)
        deadline = float("inf")
        print(f"  {noise_id} has {len(missing)} saved runs with no T10 bitrate. Filling all of them.", flush=True)
    else:
        count = min(len(missing), SPEED_FILL_CAP)
        deadline = time.monotonic() + FILL_SITTING_SECONDS
        print(
            f"  {noise_id} has {len(missing)} saved runs with no T10 bitrate. "
            f"Collecting up to {count} in {FILL_SITTING_SECONDS}s, then this command exits.",
            flush=True,
        )
    stop_iperf3_server(net, port=5201)
    free_tcp_port(net, 5201)
    found = []
    attempts = 0
    failed_since_last = 0
    max_attempts = count * 2 + 8
    while len(found) < count and attempts < max_attempts and time.monotonic() < deadline:
        attempts += 1
        server = start_one_off_iperf3(net, port=5201)
        listening = wait_for_listener(net, 5201)
        try:
            mbit, raw, _run = oam.measure_speed_once(net, port=5201)
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
            failed_since_last += 1
            continue
        print(f"  {len(found) + 1}/{count}  {mbit:.2f} Mbit/s", flush=True)
        slot = missing[len(found)]
        found.append(mbit)
        _replace_noise_bundles(
            sample_path, noise_id, _store_speeds(bundles, [slot], [mbit], failed_attempts=[failed_since_last])
        )
        failed_since_last = 0
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
    from bench.contracts import refuse_protected_path

    # The committed calibration report is frozen evidence; reprints go to a new --report.
    refuse_protected_path(out)
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
        passing = {noise: report for noise, report in reports.items() if ready_to_write(report)}
        for noise in reports:
            if noise not in passing:
                print(f"  {noise}: not written to {THRESHOLDS_PATH.name}; it did not pass the write guard.", flush=True)
        if passing:
            update_thresholds_yaml(passing)
            print(f"  wrote {', '.join(passing)} to {THRESHOLDS_PATH.name}", flush=True)


def report_from_saved(noise_ids, runs, eval_runs, seed=0, sample_path=SAMPLES_PATH):
    """Build the calibration report from the checkpoint. Does not start Mininet."""
    reports = {}
    total = runs + eval_runs
    _rows, checkpoint_skipped = _read_rows_with_skips(sample_path)
    if checkpoint_skipped:
        print(
            f"  ignored {checkpoint_skipped} unreadable checkpoint line(s) in {Path(sample_path).name}",
            flush=True,
        )
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
        full_battery = _infer_full_battery(bundles)
        report = build_report(
            bundles[:runs],
            bundles[runs:],
            noise_id,
            load_yaml_thresholds(noise_id),
            full_battery,
            seed,
            checkpoint_skipped_lines=checkpoint_skipped,
        )
        reports[noise_id] = report
        usable = report["tests"]["T10_speed"]["mbit_per_sec"]["n"]
        t10_c = report.get("first_try_t10_completeness")
        t10_target = report.get("first_try_t10_target", FIRST_TRY_T10_TARGET)
        print(
            f"  {noise_id}: T10 fit {usable}/{runs}, "
            f"first-try T10 completeness {t10_c} (target {t10_target}), "
            f"backfilled T10 {report.get('t10_backfilled_runs')}, "
            f"full_battery={report.get('full_battery')}, "
            f"skipped_lines={report.get('checkpoint_skipped_lines')}, "
            f"graded FAR {report['eval_false_alarms']}/{report['eval_runs']} "
            f"({report['eval_false_alarm_rate']}) {report['eval_alarm_tests']}; "
            f"no-reading runs {report.get('eval_no_reading_runs')} {report.get('eval_no_reading_tests')}, "
            f"write-metric {report.get('write_decision_metric')}="
            f"{report.get('write_decision_false_alarm_rate')}",
            flush=True,
        )
        if not report.get("episode_aligned_battery"):
            print(
                f"  {noise_id}: legacy battery profile {report.get('battery_profile')}. "
                f"Episode grading uses 10 pings and 3 s speed. Re-collect into "
                f"{SAMPLES_EPISODE_PATH.name} before --write-thresholds.",
                flush=True,
            )
        if t10_c is None or t10_c < t10_target:
            print(
                f"  {noise_id}: first-try T10 completeness {t10_c} is below target {t10_target}. "
                "Backfilled 1 s --fill-speed readings do not count. Thresholds will not be written.",
                flush=True,
            )
        target = report.get("false_alarm_rate_target", _false_alarm_target())
        rate = report.get(
            "eval_error_inclusive_false_alarm_rate",
            report.get("eval_false_alarm_rate"),
        )
        if rate is not None and rate > target:
            print(
                f"  {noise_id}: error-inclusive false-alarm rate {rate} is above target {target}. "
                "Thresholds will not be written until this improves.",
                flush=True,
            )
        if not ready_to_write(report):
            print(
                f"  {noise_id}: not ready_to_write "
                f"(episode_aligned={report.get('episode_aligned_battery')}, "
                f"fit={report.get('calibration_runs')}, eval={report.get('eval_runs')}).",
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
    parser.add_argument(
        "--report",
        default="results/calibration_report.json",
        help="new report path; the committed results/calibration_report.json is protected and refused",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=CALIBRATION_BATCH, help="new samples to collect before exiting")
    parser.add_argument("--fill-speed", action="store_true", help="measure T10 only where a saved run has no bitrate")
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="one sitting: collect every missing run, fill every T10 bitrate, then report",
    )
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
    if args.until_done:
        args.batch = None
        args.fill_speed = True
    from bench.contracts import refuse_protected_path

    report_target = Path(args.report) if Path(args.report).is_absolute() else ROOT / args.report
    try:
        refuse_protected_path(report_target)
    except ValueError as exc:
        print(f"refusing: {exc}. Pass --report results/calibration/<run-id>.json", file=sys.stderr)
        sys.exit(2)
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
    # A bare invocation is always the offline self-check, as root or not.
    # Collection and reprints need explicit arguments (see the docstring).
    if len(sys.argv) == 1 or "--check" in sys.argv:
        _self_check()
    else:
        main()
