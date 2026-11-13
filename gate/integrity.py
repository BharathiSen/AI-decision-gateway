"""Digests a gate authorization is bound to: proposal, reroute path, evidence.

The authorization itself is issued and Ed25519-sealed by the gate
(gate/authority.py) and verified at the execution boundary. Nothing here
creates or signs an authorization; the former orchestrator-side
bind_gate_decision / binding_errors / HMAC helpers are gone.

Evidence digest versions
  1  test_id, state_fingerprint and observed_at of each row, sorted by test_id.
     It does not cover measured values. Kept verbatim so stored version 1
     digests can still be checked. A binding without
     evidence_digest_version is version 1. Never accepted for execution.
  2  every field of every row, in the order given, in a canonical form.
     Required for execution.

What a digest does not do: it does not prove who produced the evidence. Code
running in this process can replace the evidence and the digest together.
Authenticating the decision is gate/authority.py. The digest covers the
rows handed to gate.decide, copied when decide() is entered, so a row changed
during or after the gate call no longer matches at the boundary. Rows a gate
runs itself during decide() are not part of it.
"""

import hashlib
import json
import math

EVIDENCE_DIGEST_VERSION = 2
# Deeper than any OAM row. A cyclic or absurdly nested value is refused
# instead of exhausting the stack.
_MAX_EVIDENCE_DEPTH = 32



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


def _canonical(value, where="evidence", depth=0):
    """One unambiguous JSON tree per value.

    Lists and mappings are wrapped in a one-key tag, so a non-finite float
    (also a tag) can never equal a real list, mapping or string. A missing
    key, a null, False, 0, 0.0 and "" all stay distinct. Anything outside
    JSON types is refused, not stringified.
    """
    if depth > _MAX_EVIDENCE_DEPTH:
        raise EvidenceNotCanonical(f"{where} is nested deeper than {_MAX_EVIDENCE_DEPTH} levels")
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
        return {"list": [_canonical(item, f"{where}[{index}]", depth + 1) for index, item in enumerate(value)]}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceNotCanonical(f"{where} has a non-string key {key!r}")
            out[key] = _canonical(item, f"{where}.{key}", depth + 1)
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
    # Exactly the integer 2: not 2.0, not True, not "2".
    if type(version) is not int or version != EVIDENCE_DIGEST_VERSION:
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
    # Plain comparison: the digest is not a secret, and compare_digest raises
    # on a non-ASCII string, which would abort the episode instead of refusing.
    if marked != current:
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
