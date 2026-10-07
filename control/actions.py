"""
Canonical fix actions for agent, gate, and controller.

Maps high-level actions to network.controller primitives.
"""

from enum import Enum

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

# Legacy agent strings → canonical enum names
AGENT_ACTION_ALIASES = {
    "reroute_traffic": "REROUTE",
    "no_action": "DO_NOTHING",
    "do_nothing": "DO_NOTHING",
    "escalate": "ESCALATE",
    "rebalance": "REBALANCE",
}


class Action(str, Enum):
    REROUTE = "REROUTE"
    REBALANCE = "REBALANCE"
    ESCALATE = "ESCALATE"
    DO_NOTHING = "DO_NOTHING"


RISKY_ACTIONS = {Action.REROUTE, Action.REBALANCE}


def normalize_action(name):
    if name is None:
        raise ValueError("action is required")
    if name in AGENT_ACTION_ALIASES:
        name = AGENT_ACTION_ALIASES[name]
    return Action(name)


def execute(net, action, proposal=None):
    """Apply an allowed action to the live network."""
    action = normalize_action(action if isinstance(action, str) else action.value)
    proposal = proposal or {}

    if action == Action.DO_NOTHING:
        return {"executed": False, "action": action.value}

    if action == Action.ESCALATE:
        return {"executed": False, "action": action.value, "escalated": True}

    if action == Action.REROUTE:
        path = proposal.get("path") or proposal.get("proposed_path_label")
        if path in ("primary", "backup"):
            ctl.set_active_path(net, path)
            return {"executed": True, "action": action.value, "path": path}
        # Infer from proposed_path node list
        nodes = proposal.get("proposed_path") or []
        if "r3" in nodes:
            ctl.set_active_path(net, "backup")
            return {"executed": True, "action": action.value, "path": "backup"}
        if "r2" in nodes:
            ctl.set_active_path(net, "primary")
            return {"executed": True, "action": action.value, "path": "primary"}
        raise ValueError(f"REROUTE needs path primary|backup or proposed_path; got {proposal!r}")

    if action == Action.REBALANCE:
        # Diamond topology: rebalance == switch to less-loaded path (backup for now)
        current = ctl.get_active_path(net)
        new_path = "backup" if current == "primary" else "primary"
        ctl.set_active_path(net, new_path)
        return {"executed": True, "action": action.value, "path": new_path}

    raise ValueError(f"Unhandled action {action}")


def proposal_to_action(proposal):
    """Convert agent proposal dict to Action + execution payload."""
    action = normalize_action(proposal.get("action"))
    payload = dict(proposal)
    if action == Action.REROUTE and "proposed_path" in proposal:
        payload["proposed_path"] = proposal["proposed_path"]
    return action, payload
