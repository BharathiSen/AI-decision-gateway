"""B0: execute agent proposal without gate."""

from control.actions import execute, proposal_to_action


def run(net, proposal):
    action, payload = proposal_to_action(proposal)
    result = execute(net, action, payload)
    return {"method": "B0", "decision": "allow", "execution": result}
