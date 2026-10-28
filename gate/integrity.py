"""Bind a gate decision to the snapshot, proposal, and evidence that produced it.

Signatures are optional. Unsigned tokens are valid unless require_signatures
is enabled in config/gate.yaml.
"""

import hashlib
import hmac
import json
import os
import uuid

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
            "causes": proposal.get("causes"),
            "observed_at": proposal.get("observed_at"),
        }
    )


def evidence_digest(test_results):
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
        return []
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
    binding = {
        "decision_id": uuid.uuid4().hex,
        "snapshot_fingerprint": evidence_fingerprint(snapshot),
        "proposal_digest": proposal_digest(proposal),
        "evidence_digest": evidence_digest(test_results),
        "snapshot_timestamp": (snapshot or {}).get("timestamp") if isinstance(snapshot, dict) else None,
        "signed": False,
        "signature": None,
    }
    optional_sign(binding)
    bound["binding"] = binding
    return bound


def binding_errors(decision, snapshot, proposal):
    """Legacy unbound allows return no errors. Bound tokens must match."""
    if not isinstance(decision, dict):
        return ["missing gate decision"]
    binding = decision.get("binding")
    if not isinstance(binding, dict):
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
