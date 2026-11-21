"""
AI decision agent.

propose_decision() sends an LLM (Groq by default; see PROVIDERS) the current network
state (telemetry + topology, as returned by
telemetry.collector.collect_current_state()), the action types the
agent is allowed to choose from, and policy context, and returns the
model's single proposed action as a plain dict:

    {"action": ..., "target": ..., "proposed_path": [...], "reason": ...}

Important rule: this module can only PROPOSE. It has no import of, and
no access to, anything that can change the network (network.controller,
network.fault_injector).
"""

import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.contracts import TEST_IDS, grade_vocabulary_for_prompt, llm_proposal_schema

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
XAI_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_PROVIDER = "groq"
_ENV_LOADED = False

# LLM_PROVIDER picks the backend; LLM_MODEL overrides its default model.
# LLM_FALLBACK_PROVIDER (e.g. groq) is tried once when the primary hits a
# rate/quota limit. Keys may live in the process env or a repo-root .env.
# All speak the OpenAI chat-completions API, so one client serves them all.
# Note: Groq (groq.com, keys start gsk_) and Grok (xAI, keys start xai-) are
# different services.
PROVIDERS = {
    "groq": {
        "base_url": GROQ_BASE_URL,
        "key_env": "GROQ_API_KEY",
        # gpt-oss supports strict json_schema output on Groq; low effort keeps
        # it fast. Groq does not accept "none" for this model.
        "model": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
    },
    "grok": {
        "base_url": XAI_BASE_URL,
        "key_env": "XAI_API_KEY",
        # grok-4-fast-non-reasoning retired 2026-05-15 and now redirects here.
        # none-effort keeps the proposal inside the 120 s freshness window.
        "model": "grok-4.3",
        "reasoning_effort": "none",
    },
    "openrouter": {
        "base_url": OPENROUTER_BASE_URL,
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
    "gemini": {
        "base_url": GEMINI_BASE_URL,
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
        # none keeps proposals inside the 120 s freshness window; "low"
        # thinking often pushes total age past 120 s once OAM is included.
        "reasoning_effort": "none",
    },
}

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
_RATE_LIMIT_MARKERS = (
    "RateLimitError",
    "Error code: 429",
    "RESOURCE_EXHAUSTED",
    "rate_limit",
    "quota",
    "tokens per day",
    "TPD",
    "exceeded your current quota",
    # Transient capacity / overload — keep trying alt models and fallback.
    "Error code: 503",
    "UNAVAILABLE",
    "high demand",
    "currently experiencing high demand",
    "overloaded",
    # Transport hangs must fail over quickly; the proposal freshness window is 120s.
    "APITimeoutError",
    "Request timed out",
    "timed out",
    "TimeoutError",
    "Error code: 504",
)


def load_dotenv(path=None):
    """Load KEY=VAL lines from repo-root .env into os.environ (no overwrite)."""
    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
    if path is None:
        _ENV_LOADED = True


def _clean_api_key(value):
    """Strip whitespace, wrapping quotes, and a pasted export/$env assignment."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    lowered = text.lower()
    if lowered.startswith("export ") or lowered.startswith("$env:"):
        _, _, rest = text.partition("=")
        text = rest.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            text = text[1:-1].strip()
    return text


def _key_mismatch_hint(cfg):
    key = cfg.get("api_key") or ""
    provider = cfg.get("provider")
    if not key:
        return None
    if provider == "groq":
        if key.startswith("xai-"):
            return "GROQ_API_KEY holds an xAI key; for Grok set LLM_PROVIDER=grok and XAI_API_KEY"
        if key.lower().startswith("export ") or key.lower().startswith("$env:"):
            return (
                "GROQ_API_KEY still contains the export/$env assignment. "
                "In PowerShell set only the secret: $env:GROQ_API_KEY = 'gsk_...'"
            )
        if not key.startswith("gsk_"):
            return (
                "Groq keys start with gsk_. Create one at https://console.groq.com/keys "
                "and in PowerShell set $env:GROQ_API_KEY = 'gsk_...'"
            )
        return None
    if provider != "grok":
        return None
    if key.startswith("gsk_"):
        return (
            "that is a Groq key (gsk_), not an xAI Grok key. Use it with "
            "LLM_PROVIDER=groq and GROQ_API_KEY=gsk_..."
        )
    if key.startswith("sk-or-"):
        return (
            "XAI_API_KEY looks like an OpenRouter key; Grok needs a key from "
            "https://console.x.ai (usually starts with xai-)"
        )
    if not key.startswith("xai-"):
        return (
            "xAI keys usually start with xai-. Create one at https://console.x.ai "
            "and export it without quotes"
        )
    return None


def provider_config(name=None):
    """{"provider", "base_url", "key_env", "api_key", "model", "reasoning_effort"}
    from the environment. LLM_REASONING_EFFORT overrides the provider default;
    "default" sends nothing and leaves the model's own setting.

    When name is set, resolve that provider (used for LLM_FALLBACK_PROVIDER).
    LLM_MODEL only overrides the primary LLM_PROVIDER, not the fallback.
    """
    load_dotenv()
    if name is None:
        name = (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        primary = True
    else:
        name = str(name).strip().lower()
        primary = False
    name = {"xai": "grok", "groqcloud": "groq"}.get(name, name)
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    cfg = dict(PROVIDERS[name], provider=name)
    cfg.setdefault("reasoning_effort", None)
    if primary:
        cfg["model"] = (os.environ.get("LLM_MODEL") or "").strip() or cfg["model"]
        effort = (os.environ.get("LLM_REASONING_EFFORT") or "").strip().lower()
        if effort == "default":
            cfg["reasoning_effort"] = None
        elif effort:
            if effort not in REASONING_EFFORTS:
                raise ValueError(
                    f"LLM_REASONING_EFFORT must be one of {list(REASONING_EFFORTS) + ['default']}, got {effort!r}"
                )
            # Groq gpt-oss rejects "none"; Gemini is faster when the field is omitted.
            if effort == "none" and name in ("groq", "gemini"):
                cfg["reasoning_effort"] = None
            else:
                cfg["reasoning_effort"] = effort
    else:
        # Prefer a different Groq model than the primary so TPD quotas do not collide.
        # Keep the fallback provider's own default effort (do not inherit primary override).
        cfg["model"] = (os.environ.get("LLM_FALLBACK_MODEL") or "").strip() or cfg["model"]
    cfg["api_key"] = _clean_api_key(os.environ.get(cfg["key_env"]))
    if not cfg["api_key"] and name == "grok":
        cfg["api_key"] = _clean_api_key(os.environ.get("GROK_API_KEY"))
        if cfg["api_key"]:
            cfg["key_env"] = "GROK_API_KEY"
    return cfg


def _request_args(cfg=None):
    """Per-request arguments for the configured provider."""
    cfg = cfg or provider_config()
    args = {"model": cfg["model"]}
    if cfg["reasoning_effort"]:
        args["reasoning_effort"] = cfg["reasoning_effort"]
    return args


# Per-attempt budget so primary hang + alt/fallback can still fit in the
# 120 s proposal freshness window (OAM already consumed part of it).
_LLM_ATTEMPT_TIMEOUT_S = 45.0
_LLM_MAX_RETRIES = 0


def _client_for(cfg):
    _setup_logging()
    from openai import OpenAI

    # max_retries=5 previously let one Gemini hang burn >15 min and then
    # fail freshness (age > 120 s) even when a later attempt succeeded.
    return OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=_LLM_ATTEMPT_TIMEOUT_S,
        max_retries=_LLM_MAX_RETRIES,
    )


def _is_rate_limited(exc):
    text = f"{type(exc).__name__}: {exc}"
    lower = text.lower()
    return any(marker.lower() in lower for marker in _RATE_LIMIT_MARKERS)


def _fallback_provider_name(primary_name):
    raw = (os.environ.get("LLM_FALLBACK_PROVIDER") or "").strip().lower()
    if not raw:
        return None
    name = {"xai": "grok", "groqcloud": "groq"}.get(raw, raw)
    if name == primary_name or name not in PROVIDERS:
        return None
    return name


def _alt_models(cfg):
    """Same-provider alternate models when the configured model hits a free-tier cap."""
    if cfg["provider"] == "gemini":
        raw = os.environ.get(
            "LLM_GEMINI_ALT_MODELS",
            "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        )
    elif cfg["provider"] == "groq":
        raw = os.environ.get(
            "LLM_GROQ_ALT_MODELS",
            "llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen/qwen3-32b",
        )
    else:
        return []
    out = []
    for part in raw.split(","):
        model = part.strip()
        if model and model != cfg["model"] and model not in out:
            out.append(model)
    return out


def _parse_retry_wait_seconds(message, default=60, cap=3 * 3600):
    """Parse provider retry hints like '2h24m20s' or '13m0.6s'."""
    import re

    text = str(message or "")
    m = re.search(
        r"retry in\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*([\d.]+)\s*s",
        text,
        re.I,
    )
    if m:
        hours = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = float(m.group(3) or 0)
        return min(int(hours * 3600 + mins * 60 + secs) + 5, cap)
    m = re.search(r"try again in\s+(\d+)m([\d.]+)s", text, re.I)
    if m:
        return min(int(m.group(1)) * 60 + int(float(m.group(2))) + 5, cap)
    m = re.search(r"try again in\s+([\d.]+)s", text, re.I)
    if m:
        return min(int(float(m.group(1))) + 5, cap)
    return min(default, cap)


def _try_provider_models(base_cfg, config, user_payload, payload, models, label):
    """Try models in order; return (decision, response, fingerprint, args, cfg) or raise last error."""
    last = None
    for model in models:
        cfg = dict(base_cfg)
        cfg["model"] = model
        try:
            print(f"  trying {label}: {cfg['provider']}/{model}", flush=True)
            decision, response, fingerprint, args = _complete_with_provider(
                cfg, config, user_payload, payload, client=None
            )
            return decision, response, fingerprint, args, cfg
        except Exception as exc:
            last = exc
            if _is_rate_limited(exc):
                print(
                    f"  {cfg['provider']}/{model} rate-limited ({type(exc).__name__})",
                    flush=True,
                )
                continue
            raise
    if last is not None:
        raise last
    raise RuntimeError(f"no models to try for {label}")


LOG_ENVS = ("GROQ_LOG", "GROK_LOG", "GEMINI_LOG", "LLM_LOG")


def _log():
    # Named after the active provider, e.g. "[12:00:00 grok] model call took ...".
    try:
        return logging.getLogger(provider_config()["provider"])
    except ValueError:
        return logging.getLogger("llm")


def _setup_logging():
    """GROK_LOG=info (or GEMINI_LOG / LLM_LOG) shows each HTTP request, retry
    wait and call latency; =debug adds full request/response detail. Unset: silent."""
    level = next((os.environ[n].strip().lower() for n in LOG_ENVS if os.environ.get(n)), "")
    if level not in ("info", "debug"):
        return
    if not logging.getLogger().handlers:
        logging.basicConfig(format="  [%(asctime)s %(name)s] %(message)s", datefmt="%H:%M:%S")
    value = logging.DEBUG if level == "debug" else logging.INFO
    # The HTTP client libraries log the requests and retries; the provider logger logs our timings.
    for name in tuple(PROVIDERS) + ("llm", "openai", "httpx"):
        logging.getLogger(name).setLevel(value)


def _client():
    # Lazy import path kept via _client_for; episode records need PROMPT_*
    # without the SDK installed (live-accept demos use fixed proposals).
    return _client_for(provider_config())


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
    _log().info(
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
PROMPT_VERSION = "v3"
_QUEUE_LEAK_KEYS = ("raw", "loss_rule", "rate_rule")


class AgentOutputError(ValueError):
    """Model returned JSON that failed validation, or non-JSON content.

    Distinct from transport/API failures (auth, timeout, 429), which stay
    as the client exception. Episode records store raw and parsed on this
    error so invalid output is not discarded.
    """

    def __init__(self, message, raw=None, parsed=None, errors=None, error_class="invalid_output"):
        super().__init__(message)
        self.raw = raw
        self.parsed = parsed
        self.errors = list(errors or [])
        self.error_class = error_class


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


def _message_content(response):
    choice = (getattr(response, "choices", None) or [None])[0]
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", None) if message is not None else None
    return content if isinstance(content, str) else ""


def _decision_from_response(response, allowed_actions, target):
    """Parse and validate model JSON. Always keep the raw text."""
    from agent.validate import validate_proposal

    raw_text = _message_content(response)
    try:
        decision = json.loads(raw_text) if raw_text else None
    except json.JSONDecodeError as exc:
        raise AgentOutputError(
            f"invalid agent output: malformed JSON: {exc}",
            raw=raw_text,
            error_class="invalid_output",
        ) from exc
    if not isinstance(decision, dict):
        raise AgentOutputError(
            "invalid agent output: root must be an object",
            raw=raw_text,
            parsed=decision,
            errors=["root must be an object"],
            error_class="invalid_output",
        )
    _apply_target(decision, target)
    ok, errors = validate_proposal(decision, allowed_actions=allowed_actions)
    if not ok:
        raise AgentOutputError(
            f"invalid agent output: {errors}",
            raw=raw_text,
            parsed=decision,
            errors=errors,
            error_class="invalid_output",
        )
    decision["_raw"] = {"content": raw_text}
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

Also return one to five candidate causes, ordered most likely first. \
causes[0] is your primary diagnosis and must be consistent with the \
action you propose. Each cause_id must be one of the ids in \
fault_catalog (H0, F1–F12). Do not invent prose \
labels, link names, held-out ids, or compound ids. For each cause, predict how named \
OAM tests from oam_test_catalog would read if that cause were true. \
expected must be exactly a grade token from oam_grade_vocabulary for \
that test_id (T1: ok, down, loss, latency, loss+latency; T4: df_ok or \
mtu_fail; T5: clean or errors:iface; T6: clear, drops:r2/r2-eth1, or \
backlog:r2/r2-eth1; T7: primary, backup, blackhole, return_mismatch, none; \
T8: both_ok, ping_ok_tcp_fail, ping_fail_tcp_ok, both_fail; T9: \
symmetric, a2b_loss, b2a_loss; T10: ok or slow). For T2_hop_ping use \
real hop names (hostA-r1, r1-r2, r2-r4, r4-hostB, r1-r3, r3-r4) with \
outcomes ok|down|loss|delay. Separate primary and backup with a \
semicolon: primary:hostA-r1=ok,r2-r4=down;backup:r3-r4=ok. Shared hops \
hostA-r1 and r4-hostB need a path prefix. Do not write hop= as a \
placeholder. Do not paraphrase (not success, fail, timeout, no errors, or \
packet loss 100%). You still do not execute anything or choose which \
tests the gate runs."""

SCHEMA_NAME = "network_decision_with_predictions"
TELEMETRY_KEY = "current_network_state"
SNAPSHOT_VIEW = {"version": 1, "queue_keys_dropped": list(_QUEUE_LEAK_KEYS)}


def request_config(
    allowed_actions=None,
    policy_context=None,
    oam_test_catalog=None,
    target=DEFAULT_TARGET,
    stress_mode=None,
    ticket_text=None,
    provider_name=None,
):
    """Everything propose_with_predictions sends except the live telemetry."""
    from agent.validate import SUPPORTED_TARGETS, catalog_cause_ids, catalog_entries

    allowed_actions = list(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    target = target or DEFAULT_TARGET
    # Gemini's OpenAI-compatible endpoint rejects this proposal schema under
    # strict json_schema ("too many states"). json_object still returns JSON;
    # validate_proposal enforces the contract after the call.
    provider = provider_config(provider_name)["provider"]
    system = PREDICTION_SYSTEM_PROMPT
    if provider == "gemini":
        response_format = {"type": "json_object"}
        system = (
            PREDICTION_SYSTEM_PROMPT
            + "\n\nReturn ONLY a JSON object with keys action, target, "
            "proposed_path, reason, causes. Every causes[i] MUST include "
            "cause_id and a non-empty predicted_tests array of "
            "{test_id, expected} objects. Omit prose outside the JSON."
        )
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": llm_proposal_schema(
                    allowed_actions,
                    predictions=True,
                    cause_ids=sorted(catalog_cause_ids()),
                    targets=sorted(SUPPORTED_TARGETS),
                ),
            },
        }
    return {
        "system": system,
        "telemetry_key": TELEMETRY_KEY,
        "snapshot_view": SNAPSHOT_VIEW,
        "payload": {
            "allowed_action_types": allowed_actions,
            "policy_context": policy_context or DEFAULT_POLICY_CONTEXT,
            "oam_test_catalog": list(oam_test_catalog or DEFAULT_OAM_CATALOG),
            "oam_grade_vocabulary": grade_vocabulary_for_prompt(),
            "supported_target": target,
            "target": target,
            "fault_catalog": catalog_entries(),
            "prompt_version": PROMPT_VERSION,
            "stress_mode": stress_mode,
            "ticket_text": ticket_text,
        },
        "response_format": response_format,
        "provider": provider,
    }


def _sha256(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def prompt_bundle():
    """The default request (no stress ticket), excluding live telemetry and provider."""
    return request_config()


def prompt_sha256():
    return _sha256(prompt_bundle())


PROMPT_SHA256 = prompt_sha256()


def request_fingerprint(config, request_args, cfg=None):
    """Identity of one actual model request: full config plus provider, model and effort."""
    cfg = cfg or provider_config()
    provider = {
        "provider": cfg["provider"],
        "base_url": cfg["base_url"],
        "model": request_args.get("model"),
        "reasoning_effort": request_args.get("reasoning_effort"),
    }
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "request_sha256": _sha256({"config": config, "provider": provider}),
        **provider,
        "stress_mode": config["payload"]["stress_mode"],
    }


def model_snapshot_view(snapshot):
    """Copy telemetry for the model. Drop raw tc and derived injection flags.

    Grader and episode JSON keep the unsanitized snapshot.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    view = copy.deepcopy(snapshot)
    queues = view.get("queues")
    if isinstance(queues, dict):
        for row in queues.values():
            if isinstance(row, dict):
                for key in _QUEUE_LEAK_KEYS:
                    row.pop(key, None)
    return view


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
    client: an openai.OpenAI() instance to reuse; created fresh if
        omitted from provider_config() (Groq: GROQ_API_KEY).

    Returns a plain dict: {"action", "target", "proposed_path", "reason"}.
    Raises ValueError if the model's chosen action somehow isn't one of
    allowed_actions (belt-and-suspenders on top of the enforced schema).
    Never applies anything to the network.
    """
    from agent.validate import SUPPORTED_TARGETS

    allowed_actions = list(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    policy_context = policy_context or DEFAULT_POLICY_CONTEXT
    target = target or DEFAULT_TARGET
    client = client or _client()

    user_payload = {
        "current_network_state": model_snapshot_view(network_state),
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

    decision = _decision_from_response(response, allowed_actions, target)
    decision["_cost"] = _model_cost(response, started)
    return decision


def _complete_with_provider(
    cfg,
    config,
    user_payload,
    payload,
    client=None,
):
    """One provider attempt, including a Gemini json_object repair pass."""
    args = _request_args(cfg)
    fingerprint = request_fingerprint(config, args, cfg=cfg)
    messages = [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]
    client = client or _client_for(cfg)
    try:
        response = client.chat.completions.create(
            **args,
            messages=messages,
            response_format=config["response_format"],
        )
        decision = _decision_from_response(response, payload["allowed_action_types"], payload["target"])
    except AgentOutputError as first_err:
        if cfg["provider"] != "gemini":
            first_err.request_fingerprint = fingerprint
            raise
        repair = {
            "role": "user",
            "content": (
                "Your previous JSON was invalid: "
                + "; ".join(first_err.errors or [str(first_err)])
                + ". Resend the full JSON object; every cause must include "
                "predicted_tests with {test_id, expected} entries."
            ),
        }
        try:
            response = client.chat.completions.create(
                **args,
                messages=messages
                + [
                    {"role": "assistant", "content": first_err.raw or "{}"},
                    repair,
                ],
                response_format=config["response_format"],
            )
            decision = _decision_from_response(
                response, payload["allowed_action_types"], payload["target"]
            )
        except Exception as exc:
            exc.request_fingerprint = fingerprint
            raise
    return decision, response, fingerprint, args


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
    """Phase 8: fix proposal plus structured OAM predictions per cause.

    On primary rate/quota limits, retries once with LLM_FALLBACK_PROVIDER
    (typically groq) when that key is set.
    """
    primary = provider_config()
    config = request_config(
        allowed_actions,
        policy_context,
        oam_test_catalog,
        target,
        stress_mode,
        ticket_text,
        provider_name=primary["provider"],
    )
    payload = config["payload"]
    user_payload = {config["telemetry_key"]: model_snapshot_view(network_state), **payload}

    started = time.monotonic()
    used = primary
    try:
        decision, response, fingerprint, args = _complete_with_provider(
            primary, config, user_payload, payload, client=client
        )
    except Exception as primary_exc:
        if not _is_rate_limited(primary_exc):
            primary_exc.request_fingerprint = getattr(primary_exc, "request_fingerprint", None)
            raise

        # 1) Same-provider alternate models (separate free-tier quotas).
        alts = _alt_models(primary)
        if alts:
            try:
                decision, response, fingerprint, args, used = _try_provider_models(
                    primary, config, user_payload, payload, alts, "alt-model"
                )
            except Exception as alt_exc:
                if not _is_rate_limited(alt_exc):
                    alt_exc.request_fingerprint = getattr(alt_exc, "request_fingerprint", None)
                    raise
                primary_exc = alt_exc
            else:
                decision["_cost"] = _model_cost(response, started)
                decision["_fingerprint"] = fingerprint
                decision["_provider"] = used["provider"]
                decision["_model"] = used["model"]
                return decision

        # 2) Cross-provider fallback (e.g. groq) plus that provider's alt models.
        fallback_name = _fallback_provider_name(primary["provider"])
        if not fallback_name:
            primary_exc.request_fingerprint = getattr(primary_exc, "request_fingerprint", None)
            raise
        fallback = provider_config(fallback_name)
        if not fallback.get("api_key"):
            primary_exc.request_fingerprint = getattr(primary_exc, "request_fingerprint", None)
            raise
        print(
            f"  provider fallback: {primary['provider']} -> "
            f"{fallback['provider']}/{fallback['model']} ({type(primary_exc).__name__})",
            flush=True,
        )
        fb_config = request_config(
            allowed_actions,
            policy_context,
            oam_test_catalog,
            target,
            stress_mode,
            ticket_text,
            provider_name=fallback["provider"],
        )
        fb_payload = fb_config["payload"]
        fb_user = {fb_config["telemetry_key"]: model_snapshot_view(network_state), **fb_payload}
        fb_models = [fallback["model"]] + _alt_models(fallback)
        try:
            decision, response, fingerprint, args, used = _try_provider_models(
                fallback, fb_config, fb_user, fb_payload, fb_models, "fallback"
            )
        except Exception as fb_exc:
            # Do not sleep for provider retry hints here: the proposal must
            # stay inside the 120 s freshness window, so a multi-minute wait
            # only produces stale invalid_output. Fail the episode fast.
            fb_exc.request_fingerprint = getattr(fb_exc, "request_fingerprint", None)
            raise
    decision["_cost"] = _model_cost(response, started)
    decision["_fingerprint"] = fingerprint
    decision["_provider"] = used["provider"]
    decision["_model"] = used["model"]
    return decision


def _speed_check():
    """One live propose_with_predictions call. Prints provider and latency."""
    cfg = provider_config()
    print(
        f"provider={cfg['provider']} model={cfg['model']} "
        f"effort={cfg['reasoning_effort']} {cfg['key_env']}="
        f"{'set' if cfg['api_key'] else 'missing'}"
    )
    if not cfg["api_key"]:
        raise SystemExit(f"set {cfg['key_env']} before a speed check (LLM_PROVIDER={cfg['provider']})")
    key = cfg["api_key"]
    hint = _key_mismatch_hint(cfg)
    if hint:
        raise SystemExit(hint)
    if len(key) < 20 or key in {"xai-...", "xai-xxx", "gsk_...", "gsk_xxx", "your-key"}:
        raise SystemExit(
            f"that is a placeholder, not a real key. Export the full secret in {cfg['key_env']}"
        )
    snapshot = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "metrics": {
            "packet_loss_percent": 0.0,
            "connectivity": True,
            "active_path": "primary",
            "latency_ms": 50.0,
        },
        "nodes": [],
        "links": [],
        "routes": [],
    }
    decision = propose_with_predictions(snapshot)
    cost = decision.get("_cost") or {}
    print(
        f"latency_seconds={cost.get('latency_seconds')} "
        f"prompt_tokens={cost.get('prompt_tokens')} "
        f"completion_tokens={cost.get('completion_tokens')}"
    )
    print(
        f"action={decision.get('action')} "
        f"causes={[c.get('cause_id') for c in decision.get('causes') or []]}"
    )


if __name__ == "__main__":
    _speed_check()

