"""Grade OAM results into compact outcomes that can separate faults.

Outcomes are strings the observability matrix and the consistency check
both use.

Research grading thresholds (protocol v1 §3, §8) come ONLY from a noise
level that passed calibration acceptance (`calibrate_healthy.ready_to_write`):
either `accepted_by_noise_level` in config/thresholds.yaml (written by
`--write-thresholds` together with its acceptance evidence) or an accepted
level in results/calibration_report.json. Grading re-checks that evidence
with the same contract the writer used. A level without accepted thresholds
is unavailable: `load_thresholds` raises `ThresholdsUnavailable`. There is
no fallback to provisional or withdrawn values.

Offline self-checks grade with bench/test_thresholds.yaml. Those numbers are
test fixtures only and are used only inside `test_thresholds()` /
`@using_test_thresholds`, which restore research mode on exit.
"""

import contextlib
import contextvars
import copy
import functools
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "config"
_THRESHOLDS = _CONFIG / "thresholds.yaml"
_REPORT = _ROOT / "results" / "calibration_report.json"
_TEST_THRESHOLDS = Path(__file__).resolve().parent / "test_thresholds.yaml"

# Every accepted level must define all of these. A partial set would let the
# graders fall back to their built-in defaults for the missing tests.
REQUIRED_THRESHOLDS = {
    "T1_e2e_ping": ("max_loss_percent", "max_latency_ms"),
    "T2_hop_ping": ("max_loss_percent", "max_latency_ms"),
    "T8_tcp_vs_ping": ("tcp_must_succeed",),
    "T10_speed": ("min_mbit",),
}

# Evidence fields bound into acceptance_sha256 with the threshold set.
# Keep in sync with bench.calibrate_healthy.ACCEPTANCE_FIELDS.
ACCEPTANCE_BINDING_FIELDS = (
    "noise_id",
    "full_battery",
    "calibration_runs",
    "eval_runs",
    "used_evaluation_faults",
    "episode_aligned_battery",
    "eval_error_inclusive_false_alarm_rate",
    "false_alarm_rate_target",
    "first_try_t10_completeness",
    "first_try_t10_target",
    "report_sha256",
    "samples_sha256",
    "report_path",
    "samples_path",
)

# Offline / historical analysis only. Never a research-grade mark.
DIAGNOSTIC_UNACCEPTED_LABEL = "thresholds unaccepted — not research data"


def file_sha256(path):
    """SHA-256 of a file's bytes, or None if missing/unreadable."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def acceptance_binding_hash(noise_id, acceptance, thresholds):
    """SHA-256 binding noise identity, acceptance evidence, file hashes, thresholds."""
    payload = {
        "noise_id": noise_id,
        "acceptance": {key: (acceptance or {}).get(key) for key in ACCEPTANCE_BINDING_FIELDS},
        "thresholds": thresholds,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_acceptance_file_hashes(acceptance, where="acceptance"):
    """Re-hash report/sample files on disk. [] when digests match; else reasons.

    Missing digests, missing paths, unreadable files, or digest mismatch all refuse.
    """
    acceptance = acceptance or {}
    errors = []
    for digest_key, path_key in (
        ("report_sha256", "report_path"),
        ("samples_sha256", "samples_path"),
    ):
        digest = acceptance.get(digest_key)
        path = acceptance.get(path_key)
        if not isinstance(digest, str) or not digest:
            errors.append(f"{where}: {digest_key} missing")
            continue
        if not path:
            errors.append(f"{where}: {path_key} missing")
            continue
        actual = file_sha256(path)
        if actual is None:
            errors.append(f"{where}: {path_key} unreadable ({path})")
        elif actual != digest:
            errors.append(f"{where}: {digest_key} does not match bytes at {path}")
    return errors


def refuse_live_if_unaccepted(noise_ids, stream=None):
    """Live/campaign preflight. Return exit code 2 when calibration is unaccepted."""
    if test_mode_active():
        print("refusing: live entry under test-mode thresholds", file=stream or sys.stderr)
        return 2
    refusal = require_accepted_thresholds(noise_ids)
    if refusal:
        print(refusal, file=stream or sys.stderr)
        return 2
    return None


class ThresholdsUnavailable(RuntimeError):
    """No accepted calibration for this noise level. Research grading refuses.

    status is "unaccepted" (no level passed ready_to_write). reasons lists
    why each source was rejected.
    """

    def __init__(self, noise_id, status, reasons):
        self.noise_id = noise_id
        self.status = status
        self.reasons = list(reasons or [])
        detail = "; ".join(self.reasons) or "no accepted calibration"
        super().__init__(
            f"thresholds {status} for {noise_id}: {detail}. Research grading is "
            "unavailable until calibrate_healthy.py --write-thresholds passes ready_to_write"
        )


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def threshold_set_errors(thresholds, tests=None):
    """Why a threshold set is not a complete, well-formed T1/T2/T8/T10 set. [] if it is.

    tests restricts the check to named test ids (plus callers that need T1 for
    T9). Research acceptance and noop_justified pass tests=None so the full
    set is required. Per-row grading only demands the limits for that row.
    """
    if not isinstance(thresholds, dict):
        return ["threshold set is not a mapping"]
    errors = []
    wanted = set(REQUIRED_THRESHOLDS) if tests is None else set(tests)
    for test_id, keys in REQUIRED_THRESHOLDS.items():
        if test_id not in wanted:
            continue
        limits = thresholds.get(test_id)
        if not isinstance(limits, dict):
            errors.append(f"{test_id} missing")
            continue
        for key in keys:
            value = limits.get(key)
            if key == "tcp_must_succeed":
                if not isinstance(value, bool):
                    errors.append(f"{test_id}.{key} must be true or false")
                continue
            if not _finite_number(value):
                errors.append(f"{test_id}.{key} must be a finite number")
            elif key == "max_loss_percent" and not 0.0 <= value <= 100.0:
                errors.append(f"{test_id}.{key} outside 0..100")
            elif key in ("max_latency_ms", "min_mbit") and value <= 0.0:
                errors.append(f"{test_id}.{key} must be positive")
    return errors


# --- test-only thresholds -------------------------------------------------
# Context-local depth so concurrent contexts and nested with-blocks restore
# correctly. Live/research entrypoints refuse when this is active (L1).
# Generator functions decorated with @using_test_thresholds elevate depth only
# during each next()/send() step so suspension cannot leak into the caller.

_TEST_DEPTH = contextvars.ContextVar("verifix_test_threshold_depth", default=0)


def test_mode_active():
    return _TEST_DEPTH.get() > 0


@contextlib.contextmanager
def test_thresholds():
    """Grade with bench/test_thresholds.yaml inside this block only (offline checks)."""
    token = _TEST_DEPTH.set(_TEST_DEPTH.get() + 1)
    try:
        yield
    finally:
        _TEST_DEPTH.reset(token)


@contextlib.contextmanager
def research_mode():
    """Temporarily leave test mode, e.g. to prove research grading refuses."""
    token = _TEST_DEPTH.set(0)
    try:
        yield
    finally:
        _TEST_DEPTH.reset(token)


def _test_mode_generator(gen):
    """Drive a generator with test mode on only while a step is executing."""
    try:
        with test_thresholds():
            try:
                value = next(gen)
            except StopIteration:
                return
        while True:
            try:
                sent = yield value
            except GeneratorExit:
                with test_thresholds():
                    gen.close()
                return
            try:
                with test_thresholds():
                    value = gen.send(sent) if sent is not None else next(gen)
            except StopIteration:
                return
    finally:
        try:
            gen.close()
        except Exception:
            pass


def using_test_thresholds(fn):
    """Decorator for offline self-checks. Research mode is restored on return.

    Generator functions get step-scoped test mode so a suspended yield cannot
    leave test_mode_active() true in the caller's context (L1).
    """
    import inspect

    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen_wrapper(*args, **kwargs):
            return _test_mode_generator(fn(*args, **kwargs))

        return gen_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with test_thresholds():
            result = fn(*args, **kwargs)
        if inspect.isgenerator(result):
            return _test_mode_generator(result)
        return result

    return wrapper


def load_test_thresholds(noise_id="N0"):
    """Test fixture limits. Never valid for research grading."""
    with _TEST_THRESHOLDS.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    levels = doc.get("test_only_by_noise_level") or {}
    if noise_id not in levels:
        raise KeyError(f"no test-only thresholds for {noise_id}")
    return copy.deepcopy(levels[noise_id])


# --- research thresholds --------------------------------------------------

def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{Path(path).name} is not a mapping")
    return doc


def _accepted(candidate, noise_id, where):
    """(thresholds, []) when candidate passes ready_to_write for noise_id, else (None, reasons)."""
    from bench.calibrate_healthy import acceptance_errors

    if not isinstance(candidate, dict):
        return None, [f"{where}: level is not a mapping"]
    stated = candidate.get("noise_id")
    if stated != noise_id:
        return None, [f"{where}: acceptance noise_id {stated!r} does not exactly match {noise_id!r}"]
    errors = acceptance_errors(candidate)
    if errors:
        return None, [f"{where}: " + "; ".join(errors)]
    return copy.deepcopy(candidate["proposed_thresholds"]), []


def threshold_status(noise_id="N0"):
    """Research threshold status for one noise level. Never raises.

    {"noise_id", "status": "accepted"|"unaccepted", "source", "thresholds", "reasons"}
    """
    reasons = []
    try:
        doc = _read_yaml(_THRESHOLDS)
    except Exception as exc:
        doc = {}
        reasons.append(f"config/thresholds.yaml: {type(exc).__name__}: {exc}")
    accepted = doc.get("accepted_by_noise_level")
    entry = accepted.get(noise_id) if isinstance(accepted, dict) else None
    if entry is None:
        reasons.append("config/thresholds.yaml: no accepted level")
    else:
        try:
            if not isinstance(entry, dict):
                raise ValueError("accepted entry is not a mapping")
            acceptance = entry.get("acceptance") or {}
            thresholds = entry.get("thresholds")
            candidate = dict(acceptance, proposed_thresholds=thresholds)
            found, why = _accepted(candidate, noise_id, "config/thresholds.yaml")
            if found is not None:
                stored = entry.get("acceptance_sha256")
                if not isinstance(stored, str) or not stored:
                    found, why = None, ["config/thresholds.yaml: acceptance_sha256 missing"]
                else:
                    expected = acceptance_binding_hash(noise_id, acceptance, thresholds)
                    if stored != expected:
                        found, why = None, ["config/thresholds.yaml: acceptance_sha256 mismatch"]
                if found is not None:
                    file_why = verify_acceptance_file_hashes(acceptance, "config/thresholds.yaml")
                    if file_why:
                        found, why = None, file_why
        except Exception as exc:
            found, why = None, [f"config/thresholds.yaml: {type(exc).__name__}: {exc}"]
        if found is not None:
            return {"noise_id": noise_id, "status": "accepted", "source": "config/thresholds.yaml",
                    "thresholds": found, "reasons": []}
        reasons.extend(why)
    calibration = doc.get("calibration") if isinstance(doc.get("calibration"), dict) else {}
    if calibration.get("grading_from_report") is True:
        try:
            report = json.loads(_REPORT.read_text(encoding="utf-8"))
            levels = report.get("levels") if isinstance(report, dict) else None
            level = levels.get(noise_id) if isinstance(levels, dict) else None
            if level is None:
                raise ValueError("no level in report")
            found, why = _accepted(level, noise_id, _REPORT.name)
        except FileNotFoundError:
            found, why = None, [f"{_REPORT.name}: missing"]
        except Exception as exc:
            found, why = None, [f"{_REPORT.name}: {type(exc).__name__}: {exc}"]
        if found is not None:
            return {"noise_id": noise_id, "status": "accepted", "source": _REPORT.name,
                    "thresholds": found, "reasons": []}
        reasons.extend(why)
    return {"noise_id": noise_id, "status": "unaccepted", "source": None, "thresholds": None, "reasons": reasons}


def load_research_thresholds(noise_id="N0"):
    """Accepted thresholds for this level, or ThresholdsUnavailable. Ignores test mode."""
    status = threshold_status(noise_id)
    if status["status"] != "accepted":
        raise ThresholdsUnavailable(noise_id, status["status"], status["reasons"])
    return status["thresholds"]


def require_accepted_thresholds(noise_ids):
    """Preflight for research runners: None when every level is accepted, else a message."""
    missing = []
    for noise_id in sorted(set(noise_ids or [])):
        status = threshold_status(noise_id)
        if status["status"] != "accepted":
            missing.append(f"{noise_id} ({'; '.join(status['reasons'])})")
    if not missing:
        return None
    return (
        "refusing: calibration thresholds are unaccepted for " + ", ".join(missing)
        + ". Run calibrate_healthy.py on the VM and --write-thresholds once ready_to_write passes."
    )

HOP_NAMES = {
    "10.0.1.1": "hostA-r1",
    "10.0.12.2": "r1-r2",
    "10.0.24.2": "r2-r4",
    "10.0.4.2": "r4-hostB",
    "10.0.13.2": "r1-r3",
    "10.0.34.2": "r3-r4",
}

_HEALTHY = {
    "T1_e2e_ping": "ok",
    "T4_mtu": "df_ok",
    "T5_if_errors": "clean",
    "T6_queue_stats": "clear",
    "T7_route_check": "primary",
    "T8_tcp_vs_ping": "both_ok",
    "T9_oneway_loss": "symmetric",
    "T10_speed": "ok",
}


def load_thresholds(noise_id="N0"):
    """Grading limits for one noise level.

    Research mode (default): accepted calibration only, else
    ThresholdsUnavailable. Test mode (offline self-checks only): the
    test-only fixture limits.
    """
    if test_mode_active():
        return load_test_thresholds(noise_id)
    return load_research_thresholds(noise_id)


def parse_qdisc_stats(text):
    """Pull drop and backlog counters out of `tc -s qdisc` text."""
    drops = 0
    for match in re.finditer(r"dropped (\d+)", text or ""):
        drops += int(match.group(1))
    backlog_bytes = 0
    backlog_packets = 0
    backlog = re.search(r"backlog (\d+)b(?:\s+(\d+)p)?", text or "")
    if backlog:
        backlog_bytes = int(backlog.group(1))
        if backlog.group(2):
            backlog_packets = int(backlog.group(2))
    return {
        "drops": drops,
        "backlog_bytes": backlog_bytes,
        "backlog_packets": backlog_packets,
        # netem "loss N%" counts those discards as dropped. That is not a queue.
        "loss_rule": bool(re.search(r"\bloss\b", text or "")),
        "rate_rule": bool(re.search(r"\brate\b|\btbf\b", text or "")),
    }


def _hop_outcome(hop, thresholds):
    limits = thresholds["T2_hop_ping"]
    max_loss = limits["max_loss_percent"]
    max_lat = limits["max_latency_ms"]
    loss = hop.get("packet_loss_percent")
    # Missing loss is unavailable evidence, not an inferred link-down.
    if loss is None or not _finite_number(loss):
        return None
    if loss >= 100.0:
        return "down"
    if loss > max_loss:
        return "loss"
    latency = hop.get("latency_ms")
    if latency is not None:
        if not _finite_number(latency):
            return None
        if latency > max_lat:
            return "delay"
    return "ok"


def _grade_hops(hops, thresholds):
    parts = []
    for hop in hops or []:
        label = HOP_NAMES.get(hop.get("dst_ip"), hop.get("dst_ip") or "?")
        token = _hop_outcome(hop, thresholds)
        if token is None:
            return None
        parts.append(f"{label}={token}")
    return ",".join(parts)


def _grade_t1(result, thresholds):
    limits = thresholds["T1_e2e_ping"]
    max_loss = limits["max_loss_percent"]
    max_lat = limits["max_latency_ms"]
    loss = result.get("packet_loss_percent")
    # Missing loss is unavailable, not fabricated 100%/down.
    if loss is None or not _finite_number(loss):
        return None
    latency = result.get("latency_ms")
    if latency is not None and not _finite_number(latency):
        return None
    if loss >= 100.0:
        return "down"
    flags = []
    if loss > max_loss:
        flags.append("loss")
    if latency is not None and latency > max_lat:
        flags.append("latency")
    return "+".join(flags) if flags else "ok"


def _grade_t2(result, thresholds):
    primary = result.get("hops_primary")
    backup = result.get("hops_backup")
    if primary is None and result.get("path", "primary") == "primary":
        primary = result.get("hops")
    if backup is None and result.get("path") == "backup":
        backup = result.get("hops")
    chunks = []
    if primary is not None:
        graded = _grade_hops(primary, thresholds)
        if graded is None:
            return None
        chunks.append("primary:" + graded)
    if backup is not None:
        graded = _grade_hops(backup, thresholds)
        if graded is None:
            return None
        chunks.append("backup:" + graded)
    if chunks:
        return ";".join(chunks)
    path = result.get("path") or "primary"
    graded = _grade_hops(result.get("hops") or [], thresholds)
    if graded is None:
        return None
    return f"{path}:{graded}"


def _grade_t5(result):
    bad = []
    for name, stats in sorted((result.get("interfaces") or {}).items()):
        if not isinstance(stats, dict):
            continue
        if stats.get("rx_errors", 0) or stats.get("tx_errors", 0):
            bad.append(name)
    if not bad:
        return "clean"
    return "errors:" + "|".join(bad)


def _queue_maps(result):
    """Normalize T6 queues to {name: {drops, backlog_bytes, backlog_packets}}."""
    queues = result.get("queues") or {}
    parsed = {}
    for name, value in queues.items():
        if isinstance(value, dict) and "drops" in value:
            parsed[name] = value
        else:
            parsed[name] = parse_qdisc_stats(value if isinstance(value, str) else "")
    return parsed


def _grade_t6(result):
    drops = []
    backlog = []
    for name, stats in sorted(_queue_maps(result).items()):
        # Packet-loss netem reports drops too. Count them only when the
        # qdisc is a rate limit or has no loss rule.
        intentional_loss = stats.get("loss_rule") and not stats.get("rate_rule")
        if stats.get("drops", 0) and not intentional_loss:
            drops.append(name)
        elif stats.get("backlog_bytes", 0) or stats.get("backlog_packets", 0):
            backlog.append(name)
    if drops:
        return "drops:" + "|".join(drops)
    if backlog:
        return "backlog:" + "|".join(backlog)
    return "clear"


def _grade_t7(result):
    r1 = result.get("r1_to_dest") or ""
    r2 = result.get("r2_to_dest") or ""
    r4 = result.get("r4_to_src") or ""
    blob = f"{r1} {r2} {r4}"
    if "blackhole" in blob:
        return "blackhole"
    forward_primary = "10.0.12.2" in r1
    forward_backup = "10.0.13.2" in r1
    return_primary = "10.0.24.1" in r4
    return_backup = "10.0.34.1" in r4
    if (forward_primary and return_backup) or (forward_backup and return_primary):
        return "return_mismatch"
    if forward_backup or result.get("active_path") == "backup":
        return "backup"
    if forward_primary or result.get("active_path") == "primary":
        return "primary"
    return str(result.get("active_path") or "none")


def _grade_t8(result):
    ping_ok = bool(result.get("ping_ok"))
    tcp_ok = bool(result.get("tcp_ok"))
    if ping_ok and tcp_ok:
        return "both_ok"
    if ping_ok and not tcp_ok:
        return "ping_ok_tcp_fail"
    if tcp_ok:
        return "ping_fail_tcp_ok"
    return "both_fail"


def _grade_t9(result, thresholds):
    max_loss = thresholds["T1_e2e_ping"]["max_loss_percent"]
    a2b = (result.get("hostA_to_hostB") or {}).get("packet_loss_percent", 0.0)
    b2a = (result.get("hostB_to_hostA") or {}).get("packet_loss_percent", 0.0)
    if not _finite_number(a2b) or not _finite_number(b2a):
        return None
    margin = 5.0
    a_bad = a2b > max_loss
    b_bad = b2a > max_loss
    if a_bad and a2b > b2a + margin:
        return "a2b_loss"
    if b_bad and b2a > a2b + margin:
        return "b2a_loss"
    return "symmetric"


def _grade_t10(result, thresholds):
    minimum = thresholds["T10_speed"]["min_mbit"]
    mbit = result.get("mbit_per_sec")
    # Missing throughput is unavailable, not an inferred "slow".
    if mbit is None or not _finite_number(mbit):
        return None
    if mbit < minimum:
        return "slow"
    return "ok"


def grade_result(result, thresholds=None, noise_id="N0"):
    """Return one compact outcome string for a single OAM result dict.

    Contract: a row with truthy `error` is infrastructure/measurement
    failure, not a network observation. Return None (no grade). Callers
    must treat that as missing evidence, never as ok/slow/loss/etc.
    Incomplete or non-finite threshold dictionaries refuse grading (M1).
    Missing or non-finite loss/throughput measurements are unavailable
    (None), never inferred as down/slow/100% loss.
    """
    if not isinstance(result, dict):
        return "unknown"
    if result.get("error"):
        return None
    if thresholds is None:
        thresholds = load_thresholds(noise_id)
    test_id = result.get("test_id")
    needed = set()
    if test_id in REQUIRED_THRESHOLDS:
        needed.add(test_id)
    elif test_id == "T9_oneway_loss":
        needed.add("T1_e2e_ping")
    if needed:
        gaps = threshold_set_errors(thresholds, tests=needed)
        if gaps:
            raise ValueError("incomplete thresholds: " + "; ".join(gaps))
    if test_id == "T1_e2e_ping":
        return _grade_t1(result, thresholds)
    if test_id == "T2_hop_ping":
        return _grade_t2(result, thresholds)
    if test_id == "T4_mtu":
        return "df_ok" if result.get("df_ping_ok") else "mtu_fail"
    if test_id == "T5_if_errors":
        return _grade_t5(result)
    if test_id == "T6_queue_stats":
        return _grade_t6(result)
    if test_id == "T7_route_check":
        return _grade_t7(result)
    if test_id == "T8_tcp_vs_ping":
        return _grade_t8(result)
    if test_id == "T9_oneway_loss":
        return _grade_t9(result, thresholds)
    if test_id == "T10_speed":
        return _grade_t10(result, thresholds)
    return "unknown"


def signature(test_results, thresholds=None, noise_id="N0"):
    """Map test_id -> outcome for a battery of OAM results.

    Errored rows are omitted: missing from the map means unmeasured, not a
    fabricated network grade. Valid rows keep their graded tokens.
    """
    if thresholds is None:
        thresholds = load_thresholds(noise_id)
    out = {}
    for row in test_results or []:
        if not isinstance(row, dict):
            continue
        tid = row.get("test_id")
        if not tid or row.get("error"):
            continue
        out[tid] = grade_result(row, thresholds)
    return out


def healthy_outcome(test_id):
    return _HEALTHY.get(test_id, "ok")


def is_degraded(test_id, outcome, thresholds=None):
    """True when an outcome is not the healthy reading for that test.

    T2 is degraded when any hop is loss, down, or delay. T7 backup is a
    path choice, not a fault; blackhole and return_mismatch are faults.
    T8 may skip the TCP alarm when the traffic level says TCP need not succeed.
    None means unmeasured/errored — not a network degradation.
    """
    if outcome is None:
        return False
    if test_id == "T2_hop_ping":
        return any(token in (outcome or "") for token in ("=loss", "=down", "=delay"))
    if test_id == "T7_route_check":
        return outcome in ("blackhole", "return_mismatch")
    if test_id == "T8_tcp_vs_ping":
        if thresholds and not thresholds.get("T8_tcp_vs_ping", {}).get("tcp_must_succeed", True):
            return outcome in ("both_fail", "ping_fail_tcp_ok")
        return outcome != "both_ok"
    if test_id == "T6_queue_stats":
        return outcome != "clear"
    return outcome != healthy_outcome(test_id)
