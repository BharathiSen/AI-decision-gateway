"""
AI decision agent.

propose_decision() sends an LLM (via OpenRouter) the current network
state (telemetry + topology, as returned by
telemetry.collector.collect_current_state()), the action types the
agent is allowed to choose from, and policy context, and returns the
model's single proposed action as a plain dict:

    {"action": ..., "target": ..., "proposed_path": [...], "reason": ...}

Important rule: this module can only PROPOSE. It has no import of, and
no access to, anything that can change the network (network.controller,
network.fault_injector).
"""

import json
import os

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "openai/gpt-4o-mini"

DEFAULT_ALLOWED_ACTIONS = ["reroute_traffic", "no_action"]

# Router names here match the real Mininet node names (network/topology.py)

DEFAULT_POLICY_CONTEXT = {
    "primary_path": ["r1", "r2", "r4"],
    "backup_path": ["r1", "r3", "r4"],
    "rules": [
        "Only reroute hostA<->hostB traffic between the pre-approved "
        "primary path (r1-r2-r4) and backup path (r1-r3-r4). No other "
        "path is permitted.",
        "Propose reroute_traffic only when telemetry shows the "
        "currently active path is degraded or down (packet loss, high "
        "latency, or a link that is administratively down) and the "
        "other path is currently healthy.",
        "Propose no_action when the active path is healthy, or when "
        "neither path is currently usable (rerouting to an equally "
        "broken path helps nobody).",
    ],
}

SYSTEM_PROMPT = """You are a network-reliability decision agent for a small \
dual-path network: hostA -- r1 -- {r2 or r3} -- r4 -- hostB.

On each call you are given the network's CURRENT TELEMETRY, CURRENT \
TOPOLOGY, the ALLOWED ACTION TYPES you may choose from, and POLICY \
CONTEXT describing the operator's constraints.

Your only job is to PROPOSE exactly one action. You never execute \
anything yourself -- a separate, independent verification and control \
system decides whether your proposal is ever applied to the real \
network. Base your decision only on the data you were given; do not \
assume state that isn't reported. Name the specific target and (when \
the action involves one) the concrete router path using the real node \
names given to you, and give a short, factual reason grounded in the \
telemetry."""


def _build_schema(allowed_actions):
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(allowed_actions)},
            "target": {"type": "string"},
            "proposed_path": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["action", "target", "proposed_path", "reason"],
        "additionalProperties": False,
    }


def propose_decision(
    network_state,
    allowed_actions=None,
    policy_context=None,
    target="HostA_to_HostB",
    client=None,
):
    """Ask the LLM to propose one action for the given network state.

    network_state: the dict returned by
        telemetry.collector.collect_current_state() -- current
        telemetry and current topology combined into one live snapshot.
    allowed_actions: action-type strings the agent may choose from
        (defaults to DEFAULT_ALLOWED_ACTIONS).
    policy_context: dict describing operator policy/constraints
        (defaults to DEFAULT_POLICY_CONTEXT).
    target: identifier for the traffic flow this decision concerns.
    client: an openai.OpenAI() instance pointed at OpenRouter to reuse;
        created fresh if omitted, reading the API key from the
        OPENROUTER_API_KEY env var.

    Returns a plain dict: {"action", "target", "proposed_path", "reason"}.
    Raises ValueError if the model's chosen action somehow isn't one of
    allowed_actions (belt-and-suspenders on top of the enforced schema).
    Never applies anything to the network.
    """
    allowed_actions = list(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    policy_context = policy_context or DEFAULT_POLICY_CONTEXT
    client = client or OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    user_payload = {
        "current_network_state": network_state,
        "allowed_action_types": allowed_actions,
        "policy_context": policy_context,
        "target": target,
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "network_decision",
                "strict": True,
                "schema": _build_schema(allowed_actions),
            },
        },
    )

    decision = json.loads(response.choices[0].message.content)

    if decision["action"] not in allowed_actions:
        raise ValueError(
            f"model proposed disallowed action {decision['action']!r}, "
            f"expected one of {allowed_actions}"
        )

    return decision
