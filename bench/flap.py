"""
Intermittent link flapping for F8 / U4.

Runs toggles in a daemon thread; stop_flap() sets an event, joins the
thread, and ensures the link endpoint is left administratively up.
"""

import threading
import time

try:
    from bench.inject import link_up_one_end, _first_node
    from network.interfaces import get_interface
except ImportError:
    from inject import link_up_one_end, _first_node
    from interfaces import get_interface

_ACTIVE_FLAPS = {}


def _set_admin(net, link_name, node, state):
    """Bring one interface up or down without node.cmd().

    The flap loop runs on a background thread while OAM tests use the
    Mininet shell. node.cmd() from both at once raises AssertionError.
    popen() uses the network namespace directly and leaves the shell alone.
    """
    intf = get_interface(link_name, node)
    proc = net.get(node).popen(["ip", "link", "set", "dev", intf, state])
    proc.wait(timeout=10)


def start_link_flap(net, link_name, node=None, interval_seconds=2.0, cycles=3):
    """Start flapping one endpoint of link_name. Returns a stop callable."""
    node = node or _first_node(link_name)
    stop_event = threading.Event()
    flap_id = f"{link_name}-{node}-{time.time()}"

    def _run():
        for _ in range(int(cycles)):
            if stop_event.is_set():
                break
            _set_admin(net, link_name, node, "down")
            if stop_event.wait(timeout=float(interval_seconds)):
                break
            _set_admin(net, link_name, node, "up")
            if stop_event.wait(timeout=float(interval_seconds)):
                break
        _set_admin(net, link_name, node, "up")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    _ACTIVE_FLAPS[flap_id] = {"thread": thread, "stop": stop_event, "link": link_name, "node": node}
    return flap_id


def stop_flap(flap_id):
    """Stop one flap task and leave the link up."""
    entry = _ACTIVE_FLAPS.pop(flap_id, None)
    if not entry:
        return
    entry["stop"].set()
    entry["thread"].join(timeout=30)


def stop_all_flaps(net):
    """Stop every active flap and ensure endpoints are up."""
    for flap_id in list(_ACTIVE_FLAPS):
        entry = _ACTIVE_FLAPS.get(flap_id)
        if entry:
            stop_flap(flap_id)
            link_up_one_end(net, entry["link"], entry["node"])
