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


# Modules that may run actions without a gate authorization (baselines and
# ground-truth tooling). Enforced by `python -m control`; it is a guard against
# accidental use, not a security boundary.
UNGATED_CALLERS = frozenset({
    "baselines/b0_ungated.py",
    "baselines/b1_fixed_rules.py",
    "bench/verify_fault_observability.py",
    "bench/run_episode.py",  # _self_check only: ungated path resolution matches the gate's
})


def execute_gated(net, capability):
    """Run one gate-authorized action. Requires a capability minted after verification.

    Route changes go through checked per-router commands (control.routes):
    a non-zero exit raises at that router, so r1-done-r4-failed is visible.
    Compensation is not done here; control.gated.run_lifecycle does it.
    """
    from control import gated, routes

    errors = gated.capability_errors(capability)
    if errors:
        raise PermissionError(errors[0])
    action = normalize_action(capability.action)
    if action == Action.DO_NOTHING:
        return {"executed": False, "action": action.value}
    if action == Action.ESCALATE:
        return {"executed": False, "action": action.value, "escalated": True}
    if action == Action.REROUTE:
        from gate.checks import reroute_choice

        path = reroute_choice(capability.proposal)
        if path is None or path != capability.reroute_path:
            raise PermissionError(f"reroute path {path!r} is not the authorized path {capability.reroute_path!r}")
        routes.apply_path_checked(net, path)
        return {"executed": True, "action": action.value, "path": path}
    raise PermissionError(f"{action.value} is not executable through the gate")


def execute_ungated(net, action, proposal=None, *, reason):
    """Apply an action with NO gate authorization. Baselines and tooling only.

    Behaviour is unchanged from the former execute(): set_active_path issues
    unchecked commands. reason names the caller (e.g. "B0 ungated baseline").
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("execute_ungated requires a reason naming the ungated caller")
    action = normalize_action(action if isinstance(action, str) else action.value)
    proposal = proposal or {}

    if action == Action.DO_NOTHING:
        return {"executed": False, "action": action.value}

    if action == Action.ESCALATE:
        return {"executed": False, "action": action.value, "escalated": True}

    if action == Action.REROUTE:
        # Resolve the path with the same rule the gate's safety check and the
        # execution recheck use, so the controller can never act on a path the
        # gate did not evaluate (no substring inference, no label/node conflict).
        from gate.checks import reroute_choice

        path = reroute_choice(proposal)
        if path is None:
            raise ValueError(
                "REROUTE needs an exact policy path (label primary|backup, or the "
                f"policy node list, and the two must agree); got {proposal!r}"
            )
        ctl.set_active_path(net, path)
        return {"executed": True, "action": action.value, "path": path}

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
