"""JSON schemas for the records that move between the existing pieces.

The episode runner, the agent, and the gate use these schemas. Catalog
rules stay in agent.validate. This module does not import the controller,
the fault injector, or the action executor.
"""

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

_PREDICTED_TEST = {
    "type": "object",
    "required": ["test_id", "expected"],
    "additionalProperties": False,
    "properties": {
        "test_id": {"type": "string", "enum": list(TEST_IDS)},
        "expected": {"type": "string", "maxLength": 80},
    },
}

_CAUSE = {
    "type": "object",
    "required": ["cause_id", "predicted_tests"],
    "additionalProperties": False,
    "properties": {
        "cause_id": {"type": "string", "minLength": 1},
        "predicted_tests": {"type": "array", "items": _PREDICTED_TEST, "maxItems": 8},
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


def llm_proposal_schema(allowed_actions, predictions=False):
    """Schema sent to the model. Same fields as the stored proposal."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "target", "proposed_path", "reason"],
        "properties": {
            "action": {"type": "string", "enum": list(allowed_actions)},
            "target": {"type": "string"},
            "proposed_path": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
    }
    if predictions:
        schema["properties"]["causes"] = {
            "type": "array",
            "maxItems": 5,
            "items": _CAUSE,
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
