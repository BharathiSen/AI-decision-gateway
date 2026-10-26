"""
Validate agent JSON proposals.

Malformed, oversized, or unsupported proposals are invalid. A cause id
must be one of the catalog ids in config/faults.yaml, including the
held-out U1–U5 ids.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.contracts import TEST_IDS

CANONICAL_ACTIONS = {"REROUTE", "DO_NOTHING", "ESCALATE"}
ALLOWED_ACTIONS = CANONICAL_ACTIONS
ALLOWED_TESTS = set(TEST_IDS)
SUPPORTED_TARGETS = {"HostA_to_HostB"}
MAX_EXPECTED_LEN = 80
MAX_PROPOSAL_BYTES = 8 * 1024
PROPOSAL_MAX_AGE_SECONDS = 120
_FAULTS = Path(__file__).resolve().parent.parent / "config" / "faults.yaml"


def _parse_iso_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def catalog_cause_ids():
    cached = getattr(catalog_cause_ids, "_ids", None)
    if cached is None:
        with _FAULTS.open(encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        cached = {item["id"] for item in (doc.get("faults") or []) if item.get("id")}
        catalog_cause_ids._ids = cached
    return cached


def _json_size(data):
    return len(json.dumps(data).encode("utf-8"))


def validate_proposal(data, allowed_actions=None, max_causes=5, max_bytes=MAX_PROPOSAL_BYTES):
    """
    Returns (ok: bool, errors: list[str]).
    """
    errors = []
    if not isinstance(data, dict):
        return False, ["root must be an object"]

    try:
        size = _json_size(data)
    except (TypeError, ValueError):
        return False, ["proposal is not JSON-serializable"]
    if size > max_bytes:
        errors.append(f"proposal exceeds {max_bytes} bytes")

    for key in ("action", "target", "reason"):
        if key not in data:
            errors.append(f"missing {key}")
        elif not isinstance(data.get(key), str) or not str(data.get(key)).strip():
            errors.append(f"{key} must be a non-empty string")
    target = data.get("target")
    if isinstance(target, str) and target.strip() and target not in SUPPORTED_TARGETS:
        errors.append(f"target {target!r} is not a supported flow")

    allowed = set(allowed_actions or CANONICAL_ACTIONS)
    action = data.get("action")
    if "action" in data and action not in allowed:
        errors.append(f"action {action!r} not in allowed set")

    if "causes" in data and not isinstance(data.get("causes"), list):
        errors.append("causes must be an array")
        return len(errors) == 0, errors

    causes = data.get("causes", [])
    if not isinstance(causes, list):
        errors.append("causes must be an array")
    elif len(causes) > max_causes:
        errors.append(f"at most {max_causes} causes allowed")
    else:
        for i, cause in enumerate(causes):
            if not isinstance(cause, dict):
                errors.append(f"causes[{i}] must be object")
                continue
            cause_id = cause.get("cause_id")
            if not isinstance(cause_id, str) or not cause_id:
                errors.append(f"causes[{i}] needs cause_id")
            elif cause_id not in catalog_cause_ids():
                errors.append(f"causes[{i}] cause_id {cause_id!r} is not in the fault catalog")
            if "predicted_tests" not in cause:
                errors.append(f"causes[{i}] needs predicted_tests")
                continue
            preds = cause.get("predicted_tests")
            if not isinstance(preds, list):
                errors.append(f"causes[{i}].predicted_tests must be array")
                continue
            for j, pred in enumerate(preds):
                if not isinstance(pred, dict):
                    errors.append(f"causes[{i}].predicted_tests[{j}] must be object")
                    continue
                tid = pred.get("test_id")
                if tid not in ALLOWED_TESTS:
                    errors.append(f"unknown test_id {tid!r}")
                expected = pred.get("expected")
                if "expected" not in pred:
                    errors.append(f"causes[{i}].predicted_tests[{j}] missing expected")
                elif not isinstance(expected, str):
                    errors.append(f"causes[{i}].predicted_tests[{j}].expected must be a string")
                elif len(expected) > MAX_EXPECTED_LEN:
                    errors.append(
                        f"causes[{i}].predicted_tests[{j}].expected exceeds {MAX_EXPECTED_LEN} characters"
                    )

    return len(errors) == 0, errors


def proposal_freshness_errors(proposal, snapshot=None, now=None, max_age_seconds=PROPOSAL_MAX_AGE_SECONDS):
    """Reject proposals not tied to the current telemetry snapshot or too old."""
    if not isinstance(proposal, dict):
        return ["proposal must be an object"]
    errors = []
    observed = proposal.get("observed_at")
    if not isinstance(observed, str) or not observed.strip():
        return ["missing observed_at"]
    snap_ts = (snapshot or {}).get("timestamp")
    if snap_ts and observed != snap_ts:
        errors.append("proposal stale: observed_at does not match telemetry timestamp")
    obs_dt = _parse_iso_timestamp(observed)
    if obs_dt is None:
        errors.append("observed_at is not a valid ISO timestamp")
        return errors
    now = now or datetime.now(timezone.utc)
    age = (now - obs_dt).total_seconds()
    if age > max_age_seconds:
        errors.append(f"proposal stale: age {age:.0f}s exceeds {max_age_seconds}s")
    return errors


def validate_batch(proposals):
    """Return validation rate for a list of proposals."""
    if not proposals:
        return 0.0, 0, 0
    ok_count = sum(1 for p in proposals if validate_proposal(p)[0])
    return ok_count / len(proposals), ok_count, len(proposals)
