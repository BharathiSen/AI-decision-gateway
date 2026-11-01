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
import logging
import os
import time
from pathlib import Path

import yaml
from openai import OpenAI

from bench.contracts import TEST_IDS, llm_proposal_schema

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# LLM_PROVIDER picks the backend; LLM_MODEL overrides its default model.
# Both speak the OpenAI chat-completions API, so one client serves both.
PROVIDERS = {
    "openrouter": {
        "base_url": OPENROUTER_BASE_URL,
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
    "gemini": {
        "base_url": GEMINI_BASE_URL,
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
        # Default thinking took 36-87 s per call, past the 120 s proposal
        # freshness window once telemetry and OAM time are added.
        "reasoning_effort": "low",
    },
}

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


def provider_config():
    """{"provider", "base_url", "key_env", "api_key", "model", "reasoning_effort"}
    from the environment. LLM_REASONING_EFFORT overrides the provider default;
    "default" sends nothing and leaves the model's own setting."""
    name = (os.environ.get("LLM_PROVIDER") or "openrouter").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    cfg = dict(PROVIDERS[name], provider=name)
    cfg.setdefault("reasoning_effort", None)
    cfg["model"] = (os.environ.get("LLM_MODEL") or "").strip() or cfg["model"]
    effort = (os.environ.get("LLM_REASONING_EFFORT") or "").strip().lower()
    if effort == "default":
        cfg["reasoning_effort"] = None
    elif effort:
        if effort not in REASONING_EFFORTS:
            raise ValueError(
                f"LLM_REASONING_EFFORT must be one of {list(REASONING_EFFORTS) + ['default']}, got {effort!r}"
            )
        cfg["reasoning_effort"] = effort
    cfg["api_key"] = os.environ.get(cfg["key_env"])
    return cfg


def _request_args():
    """Per-request arguments for the configured provider."""
    cfg = provider_config()
    args = {"model": cfg["model"]}
    if cfg["reasoning_effort"]:
        args["reasoning_effort"] = cfg["reasoning_effort"]
    return args


LOG = logging.getLogger("gemini")


def _setup_logging():
    """GEMINI_LOG=info shows each HTTP request, retry wait and call latency;
    GEMINI_LOG=debug adds full request/response detail. Unset: silent."""
    level = (os.environ.get("GEMINI_LOG") or "").strip().lower()
    if level not in ("info", "debug"):
        return
    if not logging.getLogger().handlers:
        logging.basicConfig(format="  [%(asctime)s %(name)s] %(message)s", datefmt="%H:%M:%S")
    value = logging.DEBUG if level == "debug" else logging.INFO
    # The HTTP client libraries log the requests and retries; "gemini" logs our timings.
    for name in ("gemini", "openai", "httpx"):
        logging.getLogger(name).setLevel(value)


def _client():
    _setup_logging()
    cfg = provider_config()
    # Extra retries ride out per-minute rate limits on free tiers.
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], max_retries=5)


def _model_cost(response, started):
    """Latency and token counts from a completed model call. Unknown stays null."""
    usage = getattr(response, "usage", None)

    def field(name):
        if usage is None:
            return None
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    cost = {
        "latency_seconds": time.monotonic() - started,
        "prompt_tokens": field("prompt_tokens"),
        "completion_tokens": field("completion_tokens"),
    }
    details = None
    if usage is not None:
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else getattr(
            usage, "completion_tokens_details", None
        )
    reasoning = None
    if details is not None:
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(
            details, "reasoning_tokens", None
        )
    LOG.info(
        "model call took %.1fs (prompt_tokens=%s, completion_tokens=%s, thinking_tokens=%s)",
        cost["latency_seconds"],
        cost["prompt_tokens"],
        cost["completion_tokens"],
        reasoning,
    )
    return cost

DEFAULT_ALLOWED_ACTIONS = ["REROUTE", "DO_NOTHING", "ESCALATE"]
DEFAULT_OAM_CATALOG = list(TEST_IDS)
DEFAULT_TARGET = "HostA_to_HostB"


def _canonical_target(value, default=DEFAULT_TARGET):
    """Map common flow spellings onto the single supported target token."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == default:
        return text
    tokens = [
        part
        for part in text.lower().replace("-", " ").replace("_", " ").replace(">", " ").replace("<", " ").split()
        if part
    ]
    if tokens in (["hosta", "to", "hostb"], ["hosta", "hostb"], ["hosta"], ["a", "to", "b"]):
        return default
    return text


def _apply_target(decision, target):
    if isinstance(decision, dict):
        decision["target"] = _canonical_target(decision.get("target"), target)
    return decision

# Policy context now lives in policy/policy.yaml (sibling of this file's
# package), not as a literal here, so it can be edited without touching
# code. Router names in it match the real Mininet node names
# (network/topology.py), not the capitalized "R1"/"R2" illustration in
# docs/phases.md, so a proposed_path can be handed straight to
# network.controller.set_active_path() without a translation step.
_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "policy.yaml"

with open(_POLICY_PATH, "r") as _f:
    DEFAULT_POLICY_CONTEXT = yaml.safe_load(_f)

SYSTEM_PROMPT = """You are a network-reliability decision agent for a small \
dual-path network: hostA -- r1 -- {r2 or r3} -- r4 -- hostB.

On each call you are given the network's CURRENT TELEMETRY, CURRENT \
TOPOLOGY, the ALLOWED ACTION TYPES you may choose from, and POLICY \
CONTEXT describing the operator's constraints.

Your only job is to PROPOSE exactly one action. You never execute \
anything yourself -- a separate, independent verification and control \
system decides whether your proposal is ever applied to the real \
network. Base your decision only on the data you were given; do not \
assume state that isn't reported. target must be exactly the \
supported_target from the user payload (HostA_to_HostB). When the \
action involves a path, use the real node names given to you. Give a \
short, factual reason grounded in the telemetry."""

PREDICTION_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Also return up to five candidate causes. Each cause_id must be one of \
the ids in fault_catalog (H0, F1–F12, U1–U5). Do not invent prose \
labels, link names, or compound ids. For each cause, predict how named \
OAM tests from oam_test_catalog would read if that cause were true \
(use test_id and a short expected field). You still do not execute \
anything or choose which tests the gate runs."""


def propose_decision(
    network_state,
    allowed_actions=None,
    policy_context=None,
    target=DEFAULT_TARGET,
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
    from agent.validate import SUPPORTED_TARGETS, validate_proposal

    allowed_actions = list(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    policy_context = policy_context or DEFAULT_POLICY_CONTEXT
    target = target or DEFAULT_TARGET
    client = client or _client()

    user_payload = {
        "current_network_state": network_state,
        "allowed_action_types": allowed_actions,
        "policy_context": policy_context,
        "supported_target": target,
        "target": target,
    }

    started = time.monotonic()
    response = client.chat.completions.create(
        **_request_args(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "network_decision",
                "strict": True,
                "schema": llm_proposal_schema(allowed_actions, targets=sorted(SUPPORTED_TARGETS)),
            },
        },
    )

    try:
        decision = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid agent output: malformed JSON: {exc}") from exc

    _apply_target(decision, target)
    ok, errors = validate_proposal(decision, allowed_actions=allowed_actions)
    if not ok:
        raise ValueError(f"invalid agent output: {errors}")

    decision["_cost"] = _model_cost(response, started)
    return decision


def propose_with_predictions(
    network_state,
    allowed_actions=None,
    policy_context=None,
    oam_test_catalog=None,
    target=DEFAULT_TARGET,
    stress_mode=None,
    ticket_text=None,
    client=None,
):
    """Phase 8: fix proposal plus structured OAM predictions per cause."""
    from agent.validate import SUPPORTED_TARGETS, catalog_cause_ids, catalog_entries, validate_proposal

    allowed_actions = list(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    oam_test_catalog = oam_test_catalog or DEFAULT_OAM_CATALOG
    policy_context = policy_context or DEFAULT_POLICY_CONTEXT
    target = target or DEFAULT_TARGET
    client = client or _client()

    user_payload = {
        "current_network_state": network_state,
        "allowed_action_types": allowed_actions,
        "policy_context": policy_context,
        "oam_test_catalog": oam_test_catalog,
        "supported_target": target,
        "target": target,
        "fault_catalog": catalog_entries(),
        "stress_mode": stress_mode,
        "ticket_text": ticket_text,
    }

    started = time.monotonic()
    response = client.chat.completions.create(
        **_request_args(),
        messages=[
            {"role": "system", "content": PREDICTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "network_decision_with_predictions",
                "strict": True,
                "schema": llm_proposal_schema(
                    allowed_actions,
                    predictions=True,
                    cause_ids=catalog_cause_ids(),
                    targets=sorted(SUPPORTED_TARGETS),
                ),
            },
        },
    )

    try:
        decision = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid agent output: malformed JSON: {exc}") from exc
    _apply_target(decision, target)
    ok, errors = validate_proposal(decision, allowed_actions=allowed_actions)
    if not ok:
        raise ValueError(f"invalid agent output: {errors}")
    decision["_cost"] = _model_cost(response, started)
    return decision
