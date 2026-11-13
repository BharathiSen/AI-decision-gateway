"""Execution recheck shared by the in-process boundary and the executor service.

Compares the state the gate decided on with a fresh read: the r1/r4 routes
to the far LANs, the four path links and the path qdisc configuration. Moved
here from bench/run_episode.py so the root executor process does not import
the orchestrator; bench.run_episode re-exports these names.
"""

import re

CONTROL_ROUTES = (("r1", "10.0.4.0/24"), ("r4", "10.0.1.0/24"))
PATH_LINKS = ("r1-r2", "r2-r4", "r1-r3", "r3-r4")
PATH_INTERFACES = (
    ("r1", "r1-eth1"),
    ("r2", "r2-eth0"),
    ("r2", "r2-eth1"),
    ("r4", "r4-eth0"),
    ("r1", "r1-eth2"),
    ("r3", "r3-eth0"),
    ("r3", "r3-eth1"),
    ("r4", "r4-eth1"),
)
_QDISC_COUNTERS = re.compile(r"\b(direct_packets_stat|refcnt|seed)\s+\d+")


def qdisc_config(text):
    """The configured qdisc lines of `tc qdisc show`, without counters."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("qdisc "):
            lines.append(" ".join(_QDISC_COUNTERS.sub("", line).split()))
    return "|".join(lines)


def _route_text(route):
    return " ".join((route or "").split())


def execution_fingerprint(snapshot):
    """Routes and path-link state that an allow decision is bound to."""
    wanted = set(CONTROL_ROUTES)
    routes = {}
    for item in (snapshot or {}).get("routes") or []:
        key = (item.get("node"), item.get("prefix"))
        if key in wanted:
            routes[f"{key[0]} {key[1]}"] = _route_text(item.get("route"))
    links = {}
    for link in (snapshot or {}).get("links") or []:
        name = link.get("name")
        if name in PATH_LINKS:
            links[name] = bool(link.get("up"))
    qdiscs = (snapshot or {}).get("qdiscs")
    return {"routes": routes, "links": links, "qdiscs": qdiscs if isinstance(qdiscs, dict) else None}


def assess_recheck(approved_snapshot, current_snapshot, proposal):
    """Compare the gate snapshot with a fresh read. Does not trust a caller flag."""
    approved = execution_fingerprint(approved_snapshot)
    current = execution_fingerprint(current_snapshot)
    expected = {f"{node} {prefix}" for node, prefix in CONTROL_ROUTES}
    if set(approved["routes"]) != expected or any(not approved["routes"][key] for key in expected):
        return {"state_verified": False, "reason": "gate snapshot has no route fingerprint to recheck"}
    if set(current["routes"]) != expected or any(not current["routes"][key] for key in expected):
        return {"state_verified": False, "reason": "execution recheck could not read routes"}
    if approved["routes"] != current["routes"]:
        return {"state_verified": False, "reason": "route changed since the gate decision"}
    if approved["links"] != current["links"]:
        return {"state_verified": False, "reason": "path state changed since the gate decision"}
    if current["qdiscs"] is not None and approved["qdiscs"] is None:
        return {"state_verified": False, "reason": "gate snapshot has no path qdisc fingerprint"}
    if approved["qdiscs"] is not None and current["qdiscs"] is None:
        return {"state_verified": False, "reason": "execution recheck could not read path qdiscs"}
    if approved["qdiscs"] != current["qdiscs"]:
        return {"state_verified": False, "reason": "path qdisc changed since the gate decision"}
    from gate.checks import path_links_up, reroute_choice

    choice = reroute_choice(proposal)
    action = (proposal or {}).get("action")
    if choice is None and action in ("REROUTE", "REBALANCE", "reroute_traffic", "rebalance"):
        return {"state_verified": False, "reason": "path not in policy-approved routes"}
    if choice is not None:
        up, _down = path_links_up(current_snapshot, choice)
        if not up:
            return {"state_verified": False, "reason": f"{choice} path is down"}
    return {"state_verified": True, "reason": None}
