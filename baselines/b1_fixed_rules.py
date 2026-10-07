"""B1: fixed if-loss-then-reroute rules."""

from control.actions import Action, execute


def decide(proposal, snapshot):
    metrics = (snapshot or {}).get("metrics", {})
    loss = metrics.get("packet_loss_percent", 0)
    if loss > 15:
        action = Action.REROUTE
        path = ["r1", "r3", "r4"]
    else:
        action = Action.DO_NOTHING
        path = []
    decision = "allow" if action != Action.DO_NOTHING else "allow"
    return {"method": "B1", "decision": decision, "action": action.value, "path": path}


def run(net, proposal, snapshot):
    d = decide(proposal, snapshot)
    if d["action"] == "REROUTE":
        execute(net, d["action"], {"proposed_path": d["path"]})
    return d
