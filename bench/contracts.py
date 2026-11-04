"""JSON schemas for the records that move between the existing pieces.

The episode runner, the agent, and the gate use these schemas. Catalog
rules stay in agent.validate. This module does not import the controller,
the fault injector, or the action executor.
"""

import re
import unicodedata

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
T4_GRADES = frozenset({"df_ok", "mtu_fail"})
T5_GRADES = frozenset({"clean"})
T6_GRADES = frozenset({"clear"})
T7_GRADES = frozenset({"primary", "backup", "blackhole", "return_mismatch", "none"})
T8_GRADES = frozenset({"both_ok", "ping_ok_tcp_fail", "ping_fail_tcp_ok", "both_fail"})
T9_GRADES = frozenset({"symmetric", "a2b_loss", "b2a_loss"})
T10_GRADES = frozenset({"ok", "slow"})

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f"), " ")
_HOP_LABEL = r"[a-z0-9_./?-]+"
_HOP_PART = rf"{_HOP_LABEL}=(?:ok|down|loss|delay)"
_PATH_CHUNK = rf"(?:primary|backup):{_HOP_PART}(?:,{_HOP_PART})*"
_T2_COMPOSED = re.compile(rf"^(?:{_PATH_CHUNK}(?:;{_PATH_CHUNK})*|{_HOP_PART})$")
_NAMED_GRADE = re.compile(rf"^(?:errors|drops|backlog):{_HOP_LABEL}(?:\|{_HOP_LABEL})*$")


def normalize_prediction_text(text):
    """NFKC plus ASCII hyphen/space folding. Not a synonym table."""
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_SPACES)
    return " ".join(folded.split())


def expected_grade_ok(test_id, expected):
    """True when expected is a grade_result token (or T2/T5/T6 structure)."""
    text = normalize_prediction_text(expected).lower()
    if not text:
        return False
    if test_id == "T1_e2e_ping":
        return text in T1_GRADES
    if test_id == "T2_hop_ping":
        return _T2_COMPOSED.fullmatch(text) is not None
    if test_id == "T4_mtu":
        return text in T4_GRADES
    if test_id == "T5_if_errors":
        return text in T5_GRADES or (text.startswith("errors:") and _NAMED_GRADE.fullmatch(text))
    if test_id == "T6_queue_stats":
        return text in T6_GRADES or (
            (text.startswith("drops:") or text.startswith("backlog:")) and _NAMED_GRADE.fullmatch(text)
        )
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
            "hop=ok|down|loss|delay",
            "primary:hop=outcome,...;backup:hop=outcome,...",
        ],
        "T4_mtu": sorted(T4_GRADES),
        "T5_if_errors": ["clean", "errors:iface", "errors:iface|iface"],
        "T6_queue_stats": ["clear", "drops:iface", "backlog:iface"],
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
                "loss+latency. T2: hop=ok|down|loss|delay or primary:...;backup:...."
                " T4: df_ok|mtu_fail. T5: clean or errors:iface. T6: clear or "
                "drops:iface or backlog:iface. T7: primary|backup|blackhole|"
                "return_mismatch|none. T8: both_ok|ping_ok_tcp_fail|"
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
