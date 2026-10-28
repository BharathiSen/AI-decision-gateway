"""B1: fixed if-loss-then-reroute rules on the snapshot. No extra OAM tests."""

from control.actions import Action, execute

from baselines.common import BACKUP_PATH, PERMITTED_ACTIONS


def decide(proposal, snapshot, test_results=None):
    metrics = (snapshot or {}).get("metrics", {})
    loss = metrics.get("packet_loss_percent", 0) or 0
    if loss > 15:
        action = Action.REROUTE
        path = list(BACKUP_PATH)
    else:
        action = Action.DO_NOTHING
        path = []
    if action.value not in PERMITTED_ACTIONS:
        action = Action.ESCALATE
        path = []
    return {
        "method": "B1",
        "decision": "allow",
        "action": action.value,
        "path": path,
        "tests_run": [],
        "reason": "loss>15 reroute" if loss > 15 else "else do_nothing",
    }


def run(net, proposal, snapshot):
    d = decide(proposal, snapshot)
    if d["action"] == "REROUTE":
        execute(net, d["action"], {"proposed_path": d["path"]})
    return d
