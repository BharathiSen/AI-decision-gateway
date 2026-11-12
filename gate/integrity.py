"""Bind a gate decision to the snapshot, proposal, and evidence that produced it.

Signatures are optional. Unsigned tokens are valid unless require_signatures
is enabled in config/gate.yaml.

Evidence digest versions
  1  test_id, state_fingerprint and observed_at of each row, sorted by test_id.
     It does not cover measured values. Kept verbatim so stored version 1
     digests can still be checked. A binding without
     evidence_digest_version is version 1. Never accepted for execution.
  2  every field of every row, in the order given, in a canonical form.
     Required for execution.

What a digest does not do: it does not prove who produced the evidence. Code
running in this process can replace the evidence and the digest together.
Authenticating the decision is the job of gate-issued signed tokens, which
is separate work. The digest covers the rows handed to bind_gate_decision;
rows a gate runs itself during decide() are not part of it.
"""

import hashlib
import hmac
import json
import math
import os
import uuid

EVIDENCE_DIGEST_VERSION = 2

_consumed = set()


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def proposal_digest(proposal):
    if not isinstance(proposal, dict):
        return _digest(None)
    return _digest(
        {
            "action": proposal.get("action"),
            "target": proposal.get("target"),
            "proposed_path": proposal.get("proposed_path"),
            "path": proposal.get("path"),
            # Every field reroute_choice() reads must be covered, or the path
            # could be changed after the decision without changing the digest.
            "proposed_path_label": proposal.get("proposed_path_label"),
            "causes": proposal.get("causes"),
            "observed_at": proposal.get("observed_at"),
        }
    )


def evidence_digest_v1(test_results):
    """Version 1, unchanged. Only for checking digests stored by older records."""
    rows = []
    for row in test_results or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "test_id": row.get("test_id"),
                "state_fingerprint": row.get("state_fingerprint"),
                "observed_at": row.get("observed_at"),
            }
        )
    rows.sort(key=lambda row: str(row.get("test_id") or ""))
    return _digest(rows)


class EvidenceNotCanonical(ValueError):
    """An evidence value has no single canonical form."""


def _canonical(value, where="evidence"):
    """One unambiguous JSON tree per value.

    Lists and mappings are wrapped in a one-key tag, so a non-finite float
    (also a tag) can never equal a real list, mapping or string. A missing
    key, a null, False, 0, 0.0 and "" all stay distinct. Anything outside
    JSON types is refused, not stringified.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"float": "nan"}
        return {"float": "inf" if value > 0 else "-inf"}
    if isinstance(value, (list, tuple)):
        return {"list": [_canonical(item, f"{where}[{index}]") for index, item in enumerate(value)]}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceNotCanonical(f"{where} has a non-string key {key!r}")
            out[key] = _canonical(item, f"{where}.{key}")
        return {"dict": out}
    raise EvidenceNotCanonical(f"{where} has unsupported type {type(value).__name__}")


def evidence_digest_v2(test_results):
    """Version 2: every field of every row, in the order given.

    Order is part of the digest because the gate resolves duplicate test ids
    by position. None means no rows. Raises EvidenceNotCanonical.
    """
    if test_results is None:
        test_results = []
    if not isinstance(test_results, (list, tuple)):
        raise EvidenceNotCanonical("evidence must be a list of rows")
    payload = {
        "evidence_digest_version": 2,
        "rows": [_canonical(row, f"evidence[{index}]") for index, row in enumerate(test_results)],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_digest(test_results, version=EVIDENCE_DIGEST_VERSION):
    """Digest under an explicit version. An unknown version is an error."""
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError(f"unknown evidence digest version {version!r}")
    return evidence_digest_v1(test_results) if version == 1 else evidence_digest_v2(test_results)


def evidence_binding_errors(binding, test_results):
    """Why this evidence does not match the binding. Empty when it does."""
    version = binding.get("evidence_digest_version")
    if isinstance(version, bool) or version != EVIDENCE_DIGEST_VERSION:
        if version is None:
            return [
                "decision has no evidence digest version "
                f"(version 1 digests are not accepted for execution; need {EVIDENCE_DIGEST_VERSION})"
            ]
        return [f"evidence digest version {version!r} is not accepted for execution (need {EVIDENCE_DIGEST_VERSION})"]
    marked = binding.get("evidence_digest")
    if not isinstance(marked, str) or not marked:
        return ["decision is missing evidence digest"]
    try:
        current = evidence_digest_v2(test_results)
    except EvidenceNotCanonical as exc:
        return [f"evidence cannot be canonicalized: {exc}"]
    if not hmac.compare_digest(marked, current):
        return ["evidence changed since the gate decision"]
    return []


def bound_reroute_path(proposal):
    """The policy path a risky proposal resolves to (primary|backup), else None.

    Stored in the binding at decision time and re-derived at execution time.
    The controller resolves the path with the same rule, so it can only act
    on the path the decision was bound to.
    """
    if not isinstance(proposal, dict):
        return None
    from control.actions import RISKY_ACTIONS, normalize_action
    from gate.checks import reroute_choice

    try:
        if normalize_action(proposal.get("action")) not in RISKY_ACTIONS:
            return None
    except (ValueError, TypeError):
        return None
    return reroute_choice(proposal)


def signatures_required():
    try:
        from gate.inputs import load_gate_config

        cfg, _errors = load_gate_config()
    except Exception:
        return False
    return bool(cfg.get("require_signatures"))


def optional_sign(binding):
    secret = os.environ.get("VERIFIX_GATE_HMAC")
    if not secret or not isinstance(binding, dict):
        if isinstance(binding, dict):
            binding["signed"] = False
            binding["signature"] = None
        return binding
    payload = {key: value for key, value in binding.items() if key not in ("signed", "signature")}
    binding["signature"] = hmac.new(
        secret.encode("utf-8"),
        _digest(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    binding["signed"] = True
    return binding


def optional_verify_errors(binding):
    if not isinstance(binding, dict) or not binding.get("signed"):
        return []
    secret = os.environ.get("VERIFIX_GATE_HMAC")
    if not secret:
        # A token that claims a signature cannot be checked without the key.
        # Passing it would let anyone set signed=True and satisfy require_signatures.
        return ["decision signature cannot be verified: VERIFIX_GATE_HMAC is not set"]
    payload = {key: value for key, value in binding.items() if key not in ("signed", "signature")}
    expected = hmac.new(
        secret.encode("utf-8"),
        _digest(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    marked = binding.get("signature")
    if not isinstance(marked, str) or len(marked) != len(expected):
        return ["decision signature is invalid"]
    if not hmac.compare_digest(marked, expected):
        return ["decision signature is invalid"]
    return []


def bind_gate_decision(decision, snapshot, proposal, test_results=None):
    """Stamp an auditable bind. Existing binds are left unchanged."""
    if not isinstance(decision, dict):
        return decision
    existing = decision.get("binding")
    if isinstance(existing, dict) and existing.get("decision_id"):
        return decision
    from gate.decide import evidence_fingerprint

    bound = dict(decision)
    try:
        evidence_mark = evidence_digest_v2(test_results)
    except EvidenceNotCanonical:
        # No digest: binding_errors then refuses the decision.
        evidence_mark = None
    binding = {
        "decision_id": uuid.uuid4().hex,
        "snapshot_fingerprint": evidence_fingerprint(snapshot),
        "proposal_digest": proposal_digest(proposal),
        "evidence_digest_version": EVIDENCE_DIGEST_VERSION,
        "evidence_digest": evidence_mark,
        "reroute_path": bound_reroute_path(proposal),
        "snapshot_timestamp": (snapshot or {}).get("timestamp") if isinstance(snapshot, dict) else None,
        "signed": False,
        "signature": None,
    }
    optional_sign(binding)
    bound["binding"] = binding
    return bound


def binding_errors(decision, snapshot, proposal, test_results=None):
    """Allow requires a bind token. Bound tokens must match this snapshot,
    this proposal, this reroute path and this evidence (None means no rows).

    Evidence is compared under digest version 2: every field of every row, in
    order. A version 1 or unversioned binding is refused, never re-read as 2.
    """
    if not isinstance(decision, dict):
        return ["missing gate decision"]
    binding = decision.get("binding")
    if not isinstance(binding, dict):
        if str(decision.get("decision") or "").strip().lower() == "allow":
            return ["allow is missing decision binding"]
        return []
    from gate.decide import evidence_fingerprint

    errors = []
    marked = binding.get("snapshot_fingerprint")
    if not marked:
        errors.append("decision is missing snapshot fingerprint")
    elif marked != evidence_fingerprint(snapshot):
        errors.append("decision not bound to this snapshot")
    if binding.get("proposal_digest") != proposal_digest(proposal):
        errors.append("proposal changed since the gate decision")
    if "reroute_path" not in binding:
        errors.append("decision is missing bound reroute path")
    elif binding.get("reroute_path") != bound_reroute_path(proposal):
        errors.append("reroute path changed since the gate decision")
    errors.extend(evidence_binding_errors(binding, test_results))
    ts = binding.get("snapshot_timestamp")
    snap_ts = (snapshot or {}).get("timestamp") if isinstance(snapshot, dict) else None
    if ts and snap_ts and ts != snap_ts:
        errors.append("decision snapshot timestamp does not match")
    if signatures_required() and not binding.get("signed"):
        errors.append("unsigned decision")
    errors.extend(optional_verify_errors(binding))
    return errors


def consume_decision(decision):
    """Mark a bound decision as used. Unbound records are not tracked."""
    if not isinstance(decision, dict):
        return None
    binding = decision.get("binding")
    if not isinstance(binding, dict):
        return None
    decision_id = binding.get("decision_id")
    if not decision_id:
        return None
    if decision_id in _consumed:
        return "decision already used"
    _consumed.add(decision_id)
    return None
