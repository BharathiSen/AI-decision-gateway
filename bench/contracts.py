"""JSON schemas for the records that move between the existing pieces.

The episode runner, the agent, and the gate use these schemas. Catalog
rules stay in agent.validate. This module does not import the controller,
the fault injector, or the action executor.

Whitespace policy: NFKC plus dash/nbsp fold, collapse runs of spaces, then
trim whitespace around the delimiters , ; : = |. Interior spaces inside a
hop or interface name are not allowed. Path prefixes and outcomes are
matched case-insensitively; hop labels are canonicalized to topology
names (hostA-r1, not hosta-r1).

T2 path separator: semicolon is canonical. The parser may rewrite only
the exact transitions `,primary:` and `,backup:` (after whitespace trim)
to `;primary:` / `;backup:`. That rewrite is recorded, not hidden.

T6 coverage: queues are collected on r1, r2, and r3 only. r4 is not a
valid T6 interface. Grader precedence (unchanged): drops (excluding
intentional netem loss without a rate rule), else backlog, else clear.
"""

import os
import re
import unicodedata
from pathlib import Path

TEST_IDS = (
    "T1_e2e_ping",
    "T2_hop_ping",
    "T4_mtu",
    "T5_if_errors",
    "T6_queue_stats",
    "T7_route_check",
    "T8_tcp_vs_ping",
    "T9_oneway_loss",
    "T10_speed",
)

# Closed grade_result tokens. T2/T5/T6 also allow a structured suffix.
T1_GRADES = frozenset({"ok", "down", "loss", "latency", "loss+latency"})
T2_HOP_OUTCOMES = frozenset({"ok", "down", "loss", "delay"})
T2_HOPS_PRIMARY = ("hostA-r1", "r1-r2", "r2-r4", "r4-hostB")
T2_HOPS_BACKUP = ("hostA-r1", "r1-r3", "r3-r4", "r4-hostB")
T2_HOPS_SHARED = frozenset({"hostA-r1", "r4-hostB"})
T2_HOPS_UNIQUE_PRIMARY = frozenset({"r1-r2", "r2-r4"})
T2_HOPS_UNIQUE_BACKUP = frozenset({"r1-r3", "r3-r4"})
T2_HOPS = frozenset(T2_HOPS_PRIMARY) | frozenset(T2_HOPS_BACKUP)
T4_GRADES = frozenset({"df_ok", "mtu_fail"})
T5_GRADES = frozenset({"clean"})
T6_GRADES = frozenset({"clear"})
T6_QUEUE_NODES = ("r1", "r2", "r3")
T6_IFACES = frozenset(
    {
        "r1/r1-eth0",
        "r1/r1-eth1",
        "r1/r1-eth2",
        "r2/r2-eth0",
        "r2/r2-eth1",
        "r3/r3-eth0",
        "r3/r3-eth1",
    }
)
T7_GRADES = frozenset({"primary", "backup", "blackhole", "return_mismatch", "none"})
T8_GRADES = frozenset({"both_ok", "ping_ok_tcp_fail", "ping_fail_tcp_ok", "both_fail"})
T9_GRADES = frozenset({"symmetric", "a2b_loss", "b2a_loss"})
T10_GRADES = frozenset({"ok", "slow"})

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f"), " ")
_HOP_LABEL = r"[a-z0-9_./?-]+"
_NAMED_GRADE = re.compile(rf"^(?:errors|drops|backlog):{_HOP_LABEL}(?:\|{_HOP_LABEL})*$")
_DELIM_SPACE = re.compile(r"\s*([,;:=|])\s*")
_PATH_TRANSITION = re.compile(r",(primary|backup):", re.IGNORECASE)
_HOP_CANON = {name.lower(): name for name in T2_HOPS}
_IFACE_CANON = {name.lower(): name for name in T6_IFACES}


def normalize_prediction_text(text):
    """NFKC plus ASCII hyphen/space folding. Not a synonym table."""
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_SPACES)
    return " ".join(folded.split())


def _strip_delim_space(text):
    return _DELIM_SPACE.sub(r"\1", (text or "").strip())


def prepare_structured_text(text):
    """Fold unicode, then trim whitespace around structural delimiters."""
    return _strip_delim_space(normalize_prediction_text(text))


def _empty_t2_parse(original, normalized="", delimiter_normalized=False, errors=None):
    return {
        "ok": False,
        "pairs": {},
        "paths": frozenset(),
        "original": original,
        "normalized": normalized,
        "delimiter_normalized": delimiter_normalized,
        "errors": list(errors or []),
    }


def parse_t2_prediction(expected):
    """Parse a T2 expected or observed grade into (path, hop) -> outcome pairs.

    Partial predictions are allowed: any non-empty subset of hops. Shared hops
    (hostA-r1, r4-hostB) require a primary:/backup: prefix. Unique hops bind to
    their path when unprefixed (r2-r4 -> primary, r3-r4 -> backup).
    """
    original = expected if isinstance(expected, str) else ""
    prepared = prepare_structured_text(original)
    if not prepared:
        return _empty_t2_parse(original, errors=["empty T2 prediction"])
    rewritten = _PATH_TRANSITION.sub(lambda match: f";{match.group(1).lower()}:", prepared)
    delimiter_normalized = rewritten != prepared
    normalized = rewritten
    errors = []
    pairs = {}

    def add_pair(path, hop_label, outcome, where):
        hop = _HOP_CANON.get((hop_label or "").lower())
        outcome = (outcome or "").lower()
        if hop is None:
            errors.append(f"{where}: unknown hop {hop_label!r}")
            return
        if outcome not in T2_HOP_OUTCOMES:
            errors.append(f"{where}: unknown outcome {outcome!r}")
            return
        allowed = T2_HOPS_PRIMARY if path == "primary" else T2_HOPS_BACKUP
        if hop not in allowed:
            errors.append(f"{where}: hop {hop} is not on the {path} path")
            return
        key = (path, hop)
        if key in pairs and pairs[key] != outcome:
            errors.append(f"{where}: conflicting outcomes for {path}:{hop}")
            return
        if key in pairs:
            errors.append(f"{where}: duplicate hop {path}:{hop}")
            return
        pairs[key] = outcome

    def split_hops(body):
        parts = []
        for item in body.split(","):
            item = item.strip()
            if not item:
                errors.append("empty hop assignment")
                continue
            if "=" not in item:
                errors.append(f"hop assignment missing outcome: {item!r}")
                continue
            label, outcome = item.split("=", 1)
            parts.append((label.strip(), outcome.strip()))
        return parts

    for chunk in normalized.split(";"):
        chunk = chunk.strip()
        if not chunk:
            errors.append("empty T2 path chunk")
            continue
        lowered = chunk.lower()
        if lowered.startswith("primary:"):
            path = "primary"
            body = chunk.split(":", 1)[1]
            if not body.strip():
                errors.append("primary path has no hops")
                continue
            for label, outcome in split_hops(body):
                add_pair(path, label, outcome, "primary")
            continue
        if lowered.startswith("backup:"):
            path = "backup"
            body = chunk.split(":", 1)[1]
            if not body.strip():
                errors.append("backup path has no hops")
                continue
            for label, outcome in split_hops(body):
                add_pair(path, label, outcome, "backup")
            continue
        hops = split_hops(chunk)
        for label, outcome in hops:
            hop = _HOP_CANON.get((label or "").lower())
            if hop in T2_HOPS_SHARED:
                errors.append(f"shared hop {hop} requires a primary: or backup: prefix")
                continue
            if hop in T2_HOPS_UNIQUE_PRIMARY:
                add_pair("primary", label, outcome, "unprefixed")
            elif hop in T2_HOPS_UNIQUE_BACKUP:
                add_pair("backup", label, outcome, "unprefixed")
            else:
                add_pair("primary", label, outcome, "unprefixed")

    ok = not errors and bool(pairs)
    return {
        "ok": ok,
        "pairs": pairs if ok else {},
        "paths": frozenset(path for path, _hop in pairs) if ok else frozenset(),
        "original": original,
        "normalized": normalized,
        "delimiter_normalized": delimiter_normalized,
        "errors": errors,
    }


def parse_t6_prediction(expected):
    """Parse a T6 expected or observed grade.

    empty drops:/backlog: lists are invalid. Duplicate interface names are
    invalid. Comparison is a set of canonical rN/intf names.
    """
    original = expected if isinstance(expected, str) else ""
    prepared = prepare_structured_text(original).lower()
    result = {
        "ok": False,
        "kind": None,
        "ifaces": frozenset(),
        "original": original,
        "normalized": prepared,
        "delimiter_normalized": False,
        "errors": [],
    }
    if not prepared:
        result["errors"].append("empty T6 prediction")
        return result
    if prepared == "clear":
        result["ok"] = True
        result["kind"] = "clear"
        return result
    if prepared.startswith("drops:") or prepared.startswith("backlog:"):
        kind, _, rest = prepared.partition(":")
        if not rest:
            result["errors"].append(f"empty {kind} interface list")
            return result
        found = []
        for raw in rest.split("|"):
            token = raw.strip()
            if not token:
                result["errors"].append(f"empty {kind} interface name")
                return result
            canon = _IFACE_CANON.get(token)
            if canon is None:
                result["errors"].append(f"unknown T6 interface {token!r}")
                return result
            if canon in found:
                result["errors"].append(f"duplicate T6 interface {canon}")
                return result
            found.append(canon)
        result["ok"] = True
        result["kind"] = kind
        result["ifaces"] = frozenset(found)
        result["normalized"] = kind + ":" + "|".join(sorted(found))
        return result
    result["errors"].append(f"not a T6 grade token: {prepared!r}")
    return result


def t2_prediction_matches(expected, observed):
    """True when every predicted (path, hop, outcome) is in the observed grade.

    Unpredicted observed hops are ignored (partial prediction). A backup
    prediction cannot match an observed grade that has no backup section.
    An empty prediction does not match.
    """
    pred = parse_t2_prediction(expected)
    obs = parse_t2_prediction(observed)
    if not pred["ok"] or not pred["pairs"] or not obs["ok"]:
        return False
    if "backup" in pred["paths"] and "backup" not in obs["paths"]:
        return False
    if "primary" in pred["paths"] and "primary" not in obs["paths"]:
        return False
    for key, outcome in pred["pairs"].items():
        if obs["pairs"].get(key) != outcome:
            return False
    return True


def t6_prediction_matches(expected, observed):
    """Subset match within the same T6 kind (clear / drops / backlog)."""
    pred = parse_t6_prediction(expected)
    obs = parse_t6_prediction(observed)
    if not pred["ok"] or not obs["ok"]:
        return False
    if pred["kind"] != obs["kind"]:
        return False
    if pred["kind"] == "clear":
        return True
    return pred["ifaces"] <= obs["ifaces"]


def scalar_prediction_matches(expected, observed):
    """Whole-token match for closed-vocab tests (not T2/T6 structure)."""
    exp = normalize_prediction_text(expected).lower()
    obs = normalize_prediction_text(observed).lower()
    if not exp or not obs:
        return False
    if exp == obs:
        return True
    if obs.startswith(exp + "=") or exp.startswith(obs + "="):
        return True
    wanted = [token for token in exp.replace(";", " ").replace(",", " ").replace("=", " ").split() if token]
    got = [token for token in obs.replace(";", " ").replace(",", " ").replace("=", " ").split() if token]
    return bool(wanted) and all(token in got for token in wanted)


def prediction_matches(test_id, expected, observed):
    """Shared matcher. Callers must not reimplement T2/T6 parsing."""
    if test_id == "T2_hop_ping":
        return t2_prediction_matches(expected, observed)
    if test_id == "T6_queue_stats":
        return t6_prediction_matches(expected, observed)
    return scalar_prediction_matches(expected, observed)


def expected_grade_ok(test_id, expected):
    """True when expected is a grade_result token (or T2/T5/T6 structure)."""
    if test_id == "T2_hop_ping":
        return parse_t2_prediction(expected)["ok"] is True
    if test_id == "T6_queue_stats":
        return parse_t6_prediction(expected)["ok"] is True
    text = normalize_prediction_text(expected).lower()
    if not text:
        return False
    if test_id == "T1_e2e_ping":
        return text in T1_GRADES
    if test_id == "T4_mtu":
        return text in T4_GRADES
    if test_id == "T5_if_errors":
        return text in T5_GRADES or (text.startswith("errors:") and _NAMED_GRADE.fullmatch(text) is not None)
    if test_id == "T7_route_check":
        return text in T7_GRADES
    if test_id == "T8_tcp_vs_ping":
        return text in T8_GRADES
    if test_id == "T9_oneway_loss":
        return text in T9_GRADES
    if test_id == "T10_speed":
        return text in T10_GRADES
    return False


def grade_vocabulary_for_prompt():
    """Per-test tokens/formats the model may put in expected."""
    return {
        "T1_e2e_ping": sorted(T1_GRADES),
        "T2_hop_ping": [
            "r2-r4=ok|down|loss|delay",
            "primary:hostA-r1=ok,r2-r4=down;backup:r3-r4=ok",
        ],
        "T4_mtu": sorted(T4_GRADES),
        "T5_if_errors": ["clean", "errors:iface", "errors:iface|iface"],
        "T6_queue_stats": ["clear", "drops:r2/r2-eth1", "backlog:r2/r2-eth1"],
        "T7_route_check": sorted(T7_GRADES),
        "T8_tcp_vs_ping": sorted(T8_GRADES),
        "T9_oneway_loss": sorted(T9_GRADES),
        "T10_speed": sorted(T10_GRADES),
    }


_PREDICTED_TEST = {
    "type": "object",
    "required": ["test_id", "expected"],
    "additionalProperties": False,
    "properties": {
        "test_id": {"type": "string", "enum": list(TEST_IDS)},
        "expected": {
            "type": "string",
            "maxLength": 80,
            "description": (
                "grade_result token for this test_id. T1: ok|down|loss|latency|"
                "loss+latency. T2: real hop names with outcomes ok|down|loss|"
                "delay; semicolon between primary: and backup: (example "
                "primary:hostA-r1=ok,r2-r4=down;backup:r3-r4=ok). "
                "T4: df_ok|mtu_fail. T5: clean or errors:iface. T6: clear or "
                "drops:r2/r2-eth1 or backlog:r2/r2-eth1. T7: primary|backup|"
                "blackhole|return_mismatch|none. T8: both_ok|ping_ok_tcp_fail|"
                "ping_fail_tcp_ok|both_fail. T9: symmetric|a2b_loss|b2a_loss. "
                "T10: ok|slow. No English paraphrases."
            ),
        },
    },
}

_CAUSE = {
    "type": "object",
    "required": ["cause_id", "predicted_tests"],
    "additionalProperties": False,
    "properties": {
        "cause_id": {"type": "string", "minLength": 1},
        "predicted_tests": {"type": "array", "items": _PREDICTED_TEST, "maxItems": len(TEST_IDS)},
    },
}

SCHEMAS = {
    "telemetry": {
        "type": "object",
        "required": [
            "timestamp",
            "nodes",
            "links",
            "routes",
            "interface_counters",
            "queues",
            "metrics",
        ],
        "additionalProperties": True,
        "properties": {
            "timestamp": {"type": "string", "minLength": 1},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type", "ips"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "type": {"type": "string", "enum": ["host", "router"]},
                        "ips": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["intf", "ip"],
                                "properties": {
                                    "intf": {"type": "string"},
                                    "ip": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "up", "endpoints", "latency_ms", "packet_loss_percent"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "up": {"type": "boolean"},
                        "endpoints": {"type": "array", "items": {"type": "string"}},
                        "latency_ms": {"type": ["number", "null"]},
                        "packet_loss_percent": {"type": "number"},
                        "role": {"type": "string"},
                    },
                },
            },
            "routes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["node", "prefix", "route"],
                    "properties": {
                        "node": {"type": "string"},
                        "prefix": {"type": "string"},
                        "route": {"type": "string"},
                    },
                },
            },
            "interface_counters": {"type": "object"},
            "queues": {"type": "object"},
            "metrics": {
                "type": "object",
                "required": ["packet_loss_percent", "connectivity", "active_path"],
                "properties": {
                    "latency_ms": {"type": ["number", "null"]},
                    "packet_loss_percent": {"type": "number"},
                    "connectivity": {"type": "boolean"},
                    "active_path": {"type": "string"},
                },
            },
        },
    },
    "proposal": {
        "type": "object",
        "required": ["action", "target", "reason"],
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string", "minLength": 1},
            "target": {"type": "string", "minLength": 1},
            "reason": {"type": "string", "minLength": 1},
            "proposed_path": {"type": "array", "items": {"type": "string"}},
            "causes": {"type": "array", "maxItems": 5, "items": _CAUSE},
        },
    },
    "predicted_test": _PREDICTED_TEST,
    "gate_decision": {
        "type": "object",
        "required": ["decision", "reason", "proposal_action", "tests_run", "checks"],
        "additionalProperties": True,
        "properties": {
            "decision": {"enum": ["allow", "block", "escalate"]},
            "reason": {"type": "string"},
            "proposal_action": {"type": ["string", "null"]},
            "tests_run": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "object"},
        },
    },
    "execution_result": {
        "type": "object",
        "required": [
            "execution_attempted",
            "execution_result",
            "execution_error",
            "remediation_succeeded",
            "safeguard_missing",
            "state_verified",
        ],
        "additionalProperties": True,
        "properties": {
            "execution_attempted": {"type": "boolean"},
            "execution_result": {"type": ["object", "null"]},
            "execution_error": {"type": ["string", "null"]},
            "remediation_succeeded": {"type": "boolean"},
            "safeguard_missing": {"type": ["string", "null"]},
            "state_verified": {"type": "boolean"},
        },
    },
}


def llm_proposal_schema(allowed_actions, predictions=False, cause_ids=None, targets=None):
    """Schema sent to the model. Same fields as the stored proposal."""
    target_prop = {"type": "string"}
    if targets:
        target_prop["enum"] = list(targets)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "target", "proposed_path", "reason"],
        "properties": {
            "action": {"type": "string", "enum": list(allowed_actions)},
            "target": target_prop,
            "proposed_path": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
    }
    if predictions:
        cause_prop = {"type": "string", "minLength": 1}
        if cause_ids:
            cause_prop = {"type": "string", "enum": sorted(cause_ids)}
        cause_item = {
            "type": "object",
            "required": ["cause_id", "predicted_tests"],
            "additionalProperties": False,
            "properties": {
                "cause_id": cause_prop,
                "predicted_tests": _CAUSE["properties"]["predicted_tests"],
            },
        }
        schema["properties"]["causes"] = {
            "type": "array",
            "maxItems": 5,
            "items": cause_item,
        }
        schema["required"] = list(schema["required"]) + ["causes"]
    return schema


def _is_type(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "null":
        return value is None
    return False


def _check(schema, value, path):
    errors = []
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    if types and not any(_is_type(value, name) for name in types):
        errors.append(f"{path} has the wrong type")
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is not an allowed value")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path} is empty")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is too long")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} has too many items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(_check(item_schema, item, f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key} is missing")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")
        for key, item in value.items():
            if key in properties:
                errors.extend(_check(properties[key], item, f"{path}.{key}"))
    return errors


def validate_record(kind, data):
    """Return (ok, errors) for one named schema."""
    schema = SCHEMAS.get(kind)
    if schema is None:
        return False, [f"unknown record {kind}"]
    errors = _check(schema, data, kind)
    return len(errors) == 0, errors


_ROOT = Path(__file__).resolve().parent.parent
PROTECTED_RESULTS = (
    _ROOT / "results" / "calibration_report.json",
    _ROOT / "results" / "pilot" / "summary.json",
    _ROOT / "results" / "observability_matrix.csv",
    _ROOT / "results" / "experiment_summary.json",
    _ROOT / "results" / "pilot" / "groq_small.json",
    _ROOT / "results" / "pilot" / "groq_small_v2.json",
    _ROOT / "results" / "pilot" / "groq_small_v3.json",
    _ROOT / "results" / "pilot" / "groq_small_v3b.json",
    _ROOT / "results" / "pilot" / "groq_small_v3b_manifest.json",
    _ROOT / "results" / "pilot" / "groq_small_v4.json",
    _ROOT / "results" / "ground_truth.json",
    _ROOT / "results" / "inseparable_pairs.json",
    _ROOT / "results" / "calibration_samples_v2.jsonl",
    _ROOT / "results" / "live_oam_cost.json",
    _ROOT / "results" / "pilot_criteria.md",
)
PROTECTED_DIRS = (
    _ROOT / "results" / "episodes",
    _ROOT / "results" / "pilot" / "v3_episodes",
    _ROOT / "results" / "pilot" / "v3b_episodes",
    _ROOT / "results" / "pilot" / "v4_episodes",
)


def _norm(path):
    return os.path.normcase(str(Path(path).resolve()))


def atomic_write_text(path, text, encoding="utf-8", newline=None):
    """Write a whole file so a reader sees the old content or the new, never a part.

    The text goes to a temporary file in the same directory, is flushed to
    disk, and then replaces the target in one step. An interrupted write
    leaves the target untouched and removes the temporary file.
    """
    import uuid

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding=encoding, newline=newline) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def refuse_protected_path(out_path):
    """Fail closed instead of writing to or inside preserved evidence."""
    target = _norm(out_path)
    if target in {_norm(path) for path in PROTECTED_RESULTS}:
        raise ValueError(f"refusing to overwrite preserved artifact {out_path}")
    for folder in PROTECTED_DIRS:
        root = _norm(folder)
        if target == root or target.startswith(root + os.sep):
            raise ValueError(f"refusing to write inside preserved evidence {folder}")
    return Path(out_path)
