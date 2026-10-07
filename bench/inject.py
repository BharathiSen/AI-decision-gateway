"""
Phase 4 fault-injection primitives and dispatcher.

Deliberately separate from network.controller's link_down()/link_up()
(both-ended, used by Phase 1's interactive fault injector in
network/fault_injector.py). That existing behavior is left untouched
here -- fail_primary/fail_backup in the main.py CLI still bring down
both ends of a link, exactly as they always have.

The benchmark's own fault semantics are single-ended: a link failure
here means exactly one named endpoint's interface goes administratively
down. Its peer's own admin state is never touched -- only its carrier
drops, as a normal consequence of that peer being the other end of the
same veth pair. That's a materially different (and more realistic:
most single link failures aren't symmetric on both routers at once)
failure mode than taking both ends down together, so it gets its own
mechanism rather than silently changing what Phase 1 already does and
has already been validated against.

Interface names come from network.interfaces.get_interface() -- the
frozen, verified map -- not from a live net.linksBetween() lookup.

inject()/reset_all() dispatch by the `injection.type`/`cleanup.type`
declared in config/faults.yaml, tracking every active injection in
_ACTIVE so reset_all() can undo exactly what's currently applied
without the caller having to remember what it asked for.

Two injection types are honestly NOT implementable via Mininet's own
TCIntf (confirmed against its actual source): netem_corrupt (F4) has
no `corrupt` parameter at all, so it's done via a raw tc command that
temporarily replaces the interface's whole qdisc (losing the bandwidth
cap for that duration -- noted where it happens, not hidden). MTU
(U1) has no support anywhere in TCIntf/Intf either, so it's a plain
`ip link set mtu`, bypassing Mininet's API entirely. link_flap (F8)
raises NotImplementedError: it depends on bench/flap.py, which
docs/phases.md calls for as its own separate deliverable and hasn't
been built.
"""

import random
import re
import uuid
from pathlib import Path

import yaml

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

try:
    from network.interfaces import get_interface, LINKS
except ImportError:
    from interfaces import get_interface, LINKS


# ---------------------------------------------------------------------------
# Single-ended link down/up (the benchmark's own fault semantics)
# ---------------------------------------------------------------------------

def link_down_one_end(net, link_name, node):
    """Bring down exactly one endpoint of a named link.

    Only `node`'s interface goes administratively down. The peer is
    left alone -- not brought down, not reconfigured.
    """
    intf = get_interface(link_name, node)
    net.get(node).cmd(f"ip link set dev {intf} down")
    return {"link": link_name, "node": node, "interface": intf, "action": "down"}


def link_up_one_end(net, link_name, node):
    """Restore exactly the endpoint link_down_one_end() brought down."""
    intf = get_interface(link_name, node)
    net.get(node).cmd(f"ip link set dev {intf} up")
    return {"link": link_name, "node": node, "interface": intf, "action": "up"}


def _routes_via_interface(net, node, intf_name):
    """All of `node`'s current routes whose egress device is intf_name,
    captured as complete, replayable 'ip route replace' argument
    strings. Used so link_down's cleanup can restore exactly whatever
    routes existed through this interface -- including a route
    reflecting whichever path (primary/backup) happened to be active
    at injection time -- instead of a hardcoded table that would have
    to guess which path was active and could silently override an
    unrelated reroute decision made independently of this fault.
    """
    out = net.get(node).cmd(f"ip route show dev {intf_name}")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    # `ip route show dev X` omits the trailing "dev X" from each line;
    # add it back so each line is a complete spec ip route replace
    # can take as-is.
    return [f"{line} dev {intf_name}" for line in lines]


def _restore_routes(net, node, route_specs):
    node_obj = net.get(node)
    for spec in route_specs:
        node_obj.cmd(f"ip route replace {spec}")


def _operstate(net, link_name, node):
    intf = get_interface(link_name, node)
    return net.get(node).cmd(f"cat /sys/class/net/{intf}/operstate").strip()


def _peer_ipv4(net, link_name, peer_node):
    intf = get_interface(link_name, peer_node)
    out = net.get(peer_node).cmd(f"ip -4 -o addr show {intf}")
    # e.g. "3: r2-eth1    inet 10.0.24.1/24 brd ... scope global r2-eth1"
    return out.split("inet ")[1].split("/")[0].split()[0]


def verify_restore(net, link_name, node, peer_node):
    """Inject a single-ended down/up cycle on (link_name, node) and
    confirm the link returns to exactly its pre-injection state:

    - both endpoints' operstate match what they were before injection
    - the link passes a real ping test afterward (not just "up" on paper)

    Returns a dict describing the result. Raises AssertionError with a
    specific, actionable message if restoration isn't exact -- this is
    meant to be run once per fault type before it's trusted for real
    experiments, per "test that cleanup restores the exact pre-injection
    state."
    """
    before = {
        node: _operstate(net, link_name, node),
        peer_node: _operstate(net, link_name, peer_node),
    }

    link_down_one_end(net, link_name, node)
    link_up_one_end(net, link_name, node)

    after = {
        node: _operstate(net, link_name, node),
        peer_node: _operstate(net, link_name, peer_node),
    }

    if after != before:
        raise AssertionError(
            f"{link_name}/{node}: operstate after restore {after} "
            f"does not match state before injection {before}"
        )

    peer_ip = _peer_ipv4(net, link_name, peer_node)
    ping_output = net.get(node).cmd(f"ping -c 2 -W 1 {peer_ip}")
    if "0% packet loss" not in ping_output:
        raise AssertionError(
            f"{link_name}/{node}: interfaces report up but the link failed "
            f"a real connectivity check against {peer_node} ({peer_ip}):\n"
            f"{ping_output}"
        )

    return {"link": link_name, "node": node, "peer": peer_node, "restored": True, "state": after}


# ---------------------------------------------------------------------------
# Egress-only (directional) tc config -- network.controller's
# configure_link() applies symmetrically to both ends of a link, but
# netem only shapes traffic *leaving* an interface, and faults.yaml's
# directional faults (F3-F7, F11, U2) need exactly one side touched.
# ---------------------------------------------------------------------------

def _intf_obj(net, link_name, node):
    intf_name = get_interface(link_name, node)
    for intf in net.get(node).intfList():
        if intf.name == intf_name:
            return intf
    raise RuntimeError(f"{link_name}/{node}: interface {intf_name} not found on node")


def _check_tc_result(link_name, node, intf_name, result):
    """TCIntf.config() does NOT raise when a tc command fails -- verified
    against Mininet's actual source: it only does
    `for output in tcoutputs: if output != '': error(...)`, a log line,
    never an exception. That means a failed rebuild command (e.g. from
    a malformed value) can leave an interface silently half-configured
    -- the root cause of the F6 bug this exists to catch: cleanup
    appeared to run, but the qdisc it was supposed to rebuild was never
    actually created, because the prior `tc qdisc del ... root` had
    already succeeded while the rebuild silently failed.
    """
    if not result:
        return
    for output in result.get("tcoutputs", []):
        if output:
            raise RuntimeError(
                f"{link_name}/{node} ({intf_name}): tc command failed "
                f"during configure_one_end(): {output.strip()}"
            )


def configure_one_end(net, link_name, node, bw=None, delay=None, loss=None):
    """Apply tc bandwidth/delay/loss to only one node's interface on a
    link, via Mininet's own TCIntf.config() (so qdisc replacement stays
    safe/idempotent, same guarantee network.controller.configure_link()
    relies on) -- just targeted at one side instead of both.
    """
    intf = _intf_obj(net, link_name, node)
    params = {}
    if bw is not None:
        params["bw"] = bw
    if delay is not None:
        params["delay"] = delay
    if loss is not None:
        params["loss"] = loss
    result = intf.config(**params)
    _check_tc_result(link_name, node, intf.name, result)
    return {"link": link_name, "node": node, "interface": intf.name, **params}


def reset_one_end(net, link_name, node):
    """Restore one endpoint to the topology's baseline bw/delay/loss.

    Kept as-is for existing callers (e.g. verify_restore-style testing)
    that genuinely want "reset to the topology's known baseline." The
    dispatcher itself no longer uses this for netem/tbf cleanup -- see
    _capture_link_params() below.
    """
    return configure_one_end(net, link_name, node, **ctl.BASELINE_LINK_PARAMS)


def _capture_link_params(net, link_name, node):
    """Best-effort capture of one interface's CURRENT bw/delay/loss, by
    parsing `tc class show` / `tc qdisc show` output, so cleanup can
    restore what was actually configured instead of assuming
    ctl.BASELINE_LINK_PARAMS is always the right thing to go back to.

    IMPORTANT LIMITATION (raised explicitly, not discovered later):
    this is a heuristic text-scrape of Mininet's own htb+netem output
    shape (`rate <N>Mbit` for bandwidth, `delay <N>ms` / `loss <N>%`
    for netem), not a general-purpose read of arbitrary qdisc state.
    There is no stable, general Linux/Mininet API for reading back
    "whatever tc state currently exists" in structured form. Consequences:

    - A parameter this regex doesn't recognize (jitter, a corruption
      percentage left over from a *different*, uncleaned fault, a
      hand-built qdisc from outside this codebase) is invisible to
      this function. It will not be captured, and configure_one_end()'s
      cleanup call -- which goes through TCIntf.config(), which always
      does a full `tc qdisc del dev <intf> root` before rebuilding --
      will silently discard it on cleanup regardless of what this
      function found.
    - Any field this can't parse falls back to
      ctl.BASELINE_LINK_PARAMS's value for that field. That fallback
      may not be correct for a given interface; there is no way for
      this function to know. Fields that fell back (as opposed to
      being genuinely read off the interface) are reported back to the
      caller so this isn't silently papered over.

    Returns (params, guessed_fields): params is a dict with bw/delay/
    loss keys; guessed_fields lists which of those keys could not be
    parsed and are therefore baseline fallbacks, not real captures.
    """
    intf = _intf_obj(net, link_name, node)
    node_obj = net.get(node)

    params = dict(ctl.BASELINE_LINK_PARAMS)
    guessed = set(params)

    class_out = node_obj.cmd(f"tc class show dev {intf.name}")
    rate_match = re.search(r"rate (\d+(?:\.\d+)?)Mbit", class_out)
    if rate_match:
        params["bw"] = float(rate_match.group(1))
        guessed.discard("bw")

    qdisc_out = node_obj.cmd(f"tc qdisc show dev {intf.name}")
    delay_match = re.search(r"delay (\d+(?:\.\d+)?)ms", qdisc_out)
    if delay_match:
        params["delay"] = f"{delay_match.group(1)}ms"
        guessed.discard("delay")
    loss_match = re.search(r"loss (?:\S+ )?(\d+(?:\.\d+)?)%", qdisc_out)
    if loss_match:
        params["loss"] = float(loss_match.group(1))
        guessed.discard("loss")

    return params, sorted(guessed)


def _direction_from_node(target):
    """faults.yaml direction strings are '<from>_to_<to>' (e.g.
    'r2_to_r4'); the egress side to configure is always the 'from' node.
    """
    from_node, _, _to_node = target["direction"].partition("_to_")
    return from_node


def _first_node(link_name):
    """Convention for faults that name a link but not a node (F1, F2,
    F12, U1): use whichever node is listed first in network.interfaces
    for that link.
    """
    return next(iter(LINKS[link_name]))


# ---------------------------------------------------------------------------
# Raw tc for what TCIntf genuinely can't do: corruption (F4)
# ---------------------------------------------------------------------------

def _inject_reorder_dup(net, link_name, node, reorder_percent, duplicate_percent):
    """netem reorder/duplicate — raw tc (not in TCIntf.config())."""
    intf = _intf_obj(net, link_name, node)
    node_obj = net.get(node)
    node_obj.cmd(f"tc qdisc del dev {intf.name} root 2>/dev/null")
    node_obj.cmd(
        f"tc qdisc replace dev {intf.name} root netem "
        f"reorder {reorder_percent}% 50% duplicate {duplicate_percent}%"
    )
    return {"link": link_name, "node": node, "interface": intf.name}


def _inject_wrong_return_route(net, node, destination, wrong_via, wrong_dev):
    node_obj = net.get(node)
    original = node_obj.cmd(f"ip route show {destination}").strip()
    node_obj.cmd(
        f"ip route replace {destination} via {wrong_via} dev {wrong_dev}"
    )
    return original


def _cleanup_wrong_return_route(net, node, destination, original_route):
    node_obj = net.get(node)
    if original_route:
        node_obj.cmd(f"ip route replace {original_route}")
    else:
        node_obj.cmd(f"ip route del {destination} 2>/dev/null")


def _inject_corrupt(net, link_name, node, corrupt_percent):
    """netem 'corrupt' has no equivalent in TCIntf.config() (confirmed
    against Mininet's actual source -- its parameter list is bw, delay,
    jitter, loss, gro, txo, rxo, speedup, use_hfsc, use_tbf, latency_ms,
    enable_ecn, enable_red, max_queue_size; no corrupt). This replaces
    the interface's whole qdisc with a bare netem, which means the
    interface's bandwidth cap is NOT enforced while this fault is
    active -- a real, deliberate tradeoff, not an oversight: rebuilding
    a combined htb+corrupt hierarchy by hand is out of scope here.
    """
    intf = _intf_obj(net, link_name, node)
    node_obj = net.get(node)
    node_obj.cmd(f"tc qdisc del dev {intf.name} root 2>/dev/null")
    node_obj.cmd(f"tc qdisc replace dev {intf.name} root netem corrupt {corrupt_percent}%")
    return {"link": link_name, "node": node, "interface": intf.name, "corrupt_percent": corrupt_percent}


# ---------------------------------------------------------------------------
# Raw ip link for what Mininet genuinely can't do: MTU (U1)
# ---------------------------------------------------------------------------

def _inject_mtu(net, link_name, node, mtu_bytes):
    """No MTU support anywhere in TCIntf or the base Intf class
    (confirmed against Mininet's source) -- plain ip link, no Mininet
    API involved.
    """
    intf_name = get_interface(link_name, node)
    node_obj = net.get(node)
    original = node_obj.cmd(f"cat /sys/class/net/{intf_name}/mtu").strip()
    node_obj.cmd(f"ip link set dev {intf_name} mtu {mtu_bytes}")
    return original


def _cleanup_mtu(net, link_name, node, original_mtu):
    intf_name = get_interface(link_name, node)
    net.get(node).cmd(f"ip link set dev {intf_name} mtu {original_mtu}")


# ---------------------------------------------------------------------------
# Blackhole route (F9) -- cleanup must restore the ORIGINAL route, not
# just delete the blackhole, since the destination prefix (10.0.4.0/24)
# is a real, needed route the node had before injection.
# ---------------------------------------------------------------------------

def _inject_blackhole(net, node, destination):
    node_obj = net.get(node)
    original = node_obj.cmd(f"ip route show {destination}").strip()
    node_obj.cmd(f"ip route replace blackhole {destination}")
    return original


def _cleanup_blackhole(net, node, destination, original_route):
    node_obj = net.get(node)
    if original_route:
        node_obj.cmd(f"ip route replace {original_route}")
    else:
        node_obj.cmd(f"ip route del blackhole {destination} 2>/dev/null")


# ---------------------------------------------------------------------------
# ACL drop (F10) -- cleanup deletes the one rule this fault added
# (`iptables -D ...`), not a blanket `-F FORWARD` flush that would
# remove any other rule someone else's code had in that chain.
# ---------------------------------------------------------------------------

def _inject_acl(net, node, protocol, port):
    net.get(node).cmd(f"iptables -A FORWARD -p {protocol} --dport {port} -j DROP")


def _cleanup_acl(net, node, protocol, port):
    net.get(node).cmd(f"iptables -D FORWARD -p {protocol} --dport {port} -j DROP")


# ---------------------------------------------------------------------------
# Best-effort background traffic for F6/F7's `background_traffic: true`.
# Requires an iperf3 server already listening on the destination host
# (docs/phases.md Phase 4's environment checks, step 8) -- if there
# isn't one, the client simply fails to connect; it doesn't block the
# rest of the fault.
# ---------------------------------------------------------------------------

def _start_background_traffic(net, from_node, to_node, rate_mbit):
    dest_ip = net.get(to_node).IP()
    pid = net.get(from_node).cmd(
        f"iperf3 -c {dest_ip} -t 3600 -b {rate_mbit}M "
        f"> /tmp/inject_bg_{from_node}.log 2>&1 & echo $!"
    ).strip()
    return pid


def _stop_background_traffic(net, from_node, pid):
    if pid:
        net.get(from_node).cmd(f"kill {pid} 2>/dev/null")


# ---------------------------------------------------------------------------
# faults.yaml loading + light validation
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_ALLOWED_INJECTIONS = {
    "none", "link_down", "netem_loss", "netem_corrupt",
    "netem_delay", "tbf_congestion", "link_flap",
    "blackhole_route", "acl_drop", "mtu_mismatch",
    "netem_reorder_dup", "wrong_return_route",
}

# Required target.* fields per injection type, checked at load time so
# a malformed fault fails loudly and specifically here rather than as
# a raw KeyError deep inside inject() the first time that fault runs.
# "node" is intentionally absent for link_down/mtu_mismatch: it's
# optional there (falls back to _first_node()).
_REQUIRED_TARGET_KEYS = {
    "none": [],
    "link_down": ["link"],
    "netem_loss": ["link", "direction"],
    "netem_delay": ["link", "direction"],
    "netem_corrupt": ["link", "direction"],
    "tbf_congestion": ["link", "direction"],
    "link_flap": ["link"],
    "blackhole_route": ["node", "destination"],
    "acl_drop": ["node", "protocol", "port"],
    "mtu_mismatch": ["link"],
    "netem_reorder_dup": ["link", "direction"],
    "wrong_return_route": ["node", "destination", "wrong_via", "wrong_dev"],
}


def _load_faults():
    with (_CONFIG_DIR / "faults.yaml").open() as f:
        doc = yaml.safe_load(f)

    faults = doc["faults"]
    ids = [f["id"] for f in faults]
    if len(ids) != len(set(ids)):
        raise ValueError(f"config/faults.yaml: duplicate fault ids: {ids}")

    by_id = {}
    for fault in faults:
        injection = fault.get("injection", {}).get("type")
        if injection not in _ALLOWED_INJECTIONS:
            raise ValueError(f"{fault['id']}: invalid injection type {injection!r}")

        target = fault.get("target") or {}
        required = _REQUIRED_TARGET_KEYS.get(injection, [])
        missing = [k for k in required if k not in target]
        if missing:
            raise ValueError(
                f"{fault['id']}: injection type {injection!r} is missing "
                f"required target field(s) {missing} (target={target!r})"
            )

        by_id[fault["id"]] = fault

    return by_id


_FAULTS = _load_faults()


def _sample_param(value, rng):
    """A [low, high] pair is a severity range to sample from; anything
    else (a bool, a bare number, a string) is used as-is.
    """
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        lo, hi = value
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return rng.uniform(lo, hi)
    return value


def _resolve_params(raw_params, seed):
    rng = random.Random(seed)
    return {k: _sample_param(v, rng) for k, v in (raw_params or {}).items()}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ACTIVE = {}  # correlation_id -> {"cleanup": callable, "details": dict}


def inject(net, fault_id, params=None, seed=None):
    """Inject one fault from config/faults.yaml by id.

    params: explicit values overriding the fault's own declared params
        (still passed through severity-range sampling, so a caller can
        pass a narrower range too). Defaults to the fault's own params.
    seed: seeds the sampling of any [low, high] severity range, for
        reproducible episodes. None means non-reproducible (system
        randomness).

    Returns a correlation_id (str). Pass it to reset() to undo just
    this injection, or call reset_all() to undo every currently active
    one regardless of id.
    """
    if fault_id not in _FAULTS:
        raise ValueError(f"Unknown fault id {fault_id!r}; known ids: {sorted(_FAULTS)}")

    fault = _FAULTS[fault_id]
    injection = fault["injection"]["type"]
    target = fault.get("target") or {}
    resolved = _resolve_params(params if params is not None else fault.get("params"), seed)

    details = {"fault_id": fault_id, "injection": injection, "params": resolved}
    cleanup_fn = None

    if injection == "none":
        cleanup_fn = lambda: None

    elif injection == "link_down":
        link_name = target["link"]
        node = target.get("node") or _first_node(link_name)
        intf_name = get_interface(link_name, node)
        # Capture whatever routes currently transit this interface
        # BEFORE bringing it down, so cleanup can restore exactly that
        # -- not a hardcoded "assume primary" guess (see
        # _routes_via_interface's docstring for why that would be
        # unsafe: it could silently override an unrelated reroute).
        saved_routes = _routes_via_interface(net, node, intf_name)
        link_down_one_end(net, link_name, node)
        cleanup_fn = lambda: (
            link_up_one_end(net, link_name, node),
            _restore_routes(net, node, saved_routes),
        )
        details["node"] = node
        details["restored_routes"] = saved_routes

    elif injection in ("netem_loss", "netem_delay"):
        link_name = target["link"]
        from_node = _direction_from_node(target)
        original, guessed = _capture_link_params(net, link_name, from_node)
        if injection == "netem_loss":
            configure_one_end(net, link_name, from_node, loss=resolved["loss_percent"])
        else:
            configure_one_end(net, link_name, from_node, delay=f"{resolved['delay_ms']}ms")
        cleanup_fn = lambda: configure_one_end(net, link_name, from_node, **original)
        details["node"] = from_node
        details["pre_injection_params"] = original
        if guessed:
            details["params_not_captured"] = guessed

    elif injection == "netem_corrupt":
        link_name = target["link"]
        from_node = _direction_from_node(target)
        original, guessed = _capture_link_params(net, link_name, from_node)
        _inject_corrupt(net, link_name, from_node, resolved["corrupt_percent"])
        cleanup_fn = lambda: configure_one_end(net, link_name, from_node, **original)
        details["node"] = from_node
        details["pre_injection_params"] = original
        if guessed:
            details["params_not_captured"] = guessed

    elif injection == "tbf_congestion":
        link_name = target["link"]
        from_node, _, to_node = target["direction"].partition("_to_")
        # Diagnostic only -- NOT used for cleanup below. A prior version
        # of this branch used _capture_link_params()'s regex-parsed
        # result for cleanup instead of the topology's known baseline;
        # that's the confirmed root cause of F6 leaving the qdisc empty
        # on reset (see investigation notes): the parsed value fed a
        # malformed tc command, TCIntf.config() silently logs tc
        # failures instead of raising (confirmed against its source),
        # and the preceding `qdisc del` had already succeeded -- so the
        # interface was left with nothing. Keeping the capture here only
        # so params_not_captured/pre_injection_params stay informative;
        # restoration itself uses reset_one_end()'s hardcoded
        # ctl.BASELINE_LINK_PARAMS, the same values a direct
        # configure_one_end() call already proved work.
        original, guessed = _capture_link_params(net, link_name, from_node)
        configure_one_end(net, link_name, from_node, bw=resolved["rate_mbit"])

        details["node"] = from_node
        details["pre_injection_params"] = original
        if guessed:
            details["params_not_captured"] = guessed

        # Register the bandwidth mutation's cleanup and track it in
        # _ACTIVE *immediately* -- before the optional background-
        # traffic step below, which can fail (e.g. no iperf3 server
        # listening). Audit finding: previously, a failure there left
        # this real mutation applied to the network but untracked,
        # because _ACTIVE was only ever populated at the end of
        # inject(), after every branch. `cleanups` is a list the
        # registered lambda closes over by reference, so appending to
        # it below still affects what that already-stored cleanup does
        # when it's eventually called -- no re-registration needed.
        cleanups = [lambda: reset_one_end(net, link_name, from_node)]
        correlation_id = f"{fault_id}-{uuid.uuid4().hex[:8]}"
        _ACTIVE[correlation_id] = {
            "cleanup": lambda: [c() for c in cleanups],
            "details": dict(details),
        }

        if resolved.get("background_traffic"):
            pid = _start_background_traffic(net, from_node, to_node, resolved["rate_mbit"])
            cleanups.append(lambda: _stop_background_traffic(net, from_node, pid))

        return correlation_id

    elif injection == "link_flap":
        try:
            from bench import flap as flap_mod
        except ImportError:
            import flap as flap_mod

        link_name = target["link"]
        node = target.get("node") or _first_node(link_name)
        flap_id = flap_mod.start_link_flap(
            net,
            link_name,
            node=node,
            interval_seconds=resolved.get("interval_seconds", 2),
            cycles=resolved.get("cycles", 3),
        )
        cleanup_fn = lambda: (
            flap_mod.stop_flap(flap_id),
            link_up_one_end(net, link_name, node),
        )
        details.update(node=node, flap_id=flap_id)

    elif injection == "netem_reorder_dup":
        link_name = target["link"]
        from_node = _direction_from_node(target)
        original, guessed = _capture_link_params(net, link_name, from_node)
        _inject_reorder_dup(
            net,
            link_name,
            from_node,
            resolved["reorder_percent"],
            resolved["duplicate_percent"],
        )
        cleanup_fn = lambda: configure_one_end(net, link_name, from_node, **original)
        details["node"] = from_node
        details["pre_injection_params"] = original
        if guessed:
            details["params_not_captured"] = guessed

    elif injection == "wrong_return_route":
        node = target["node"]
        destination = target["destination"]
        original = _inject_wrong_return_route(
            net,
            node,
            destination,
            target["wrong_via"],
            target["wrong_dev"],
        )
        cleanup_fn = lambda: _cleanup_wrong_return_route(
            net, node, destination, original
        )
        details.update(node=node, destination=destination)

    elif injection == "blackhole_route":
        node, destination = target["node"], target["destination"]
        original = _inject_blackhole(net, node, destination)
        cleanup_fn = lambda: _cleanup_blackhole(net, node, destination, original)
        details.update(node=node, destination=destination)

    elif injection == "acl_drop":
        node, protocol, port = target["node"], target["protocol"], target["port"]
        _inject_acl(net, node, protocol, port)
        cleanup_fn = lambda: _cleanup_acl(net, node, protocol, port)
        details.update(node=node, protocol=protocol, port=port)

    elif injection == "mtu_mismatch":
        link_name = target["link"]
        node = target.get("node") or _first_node(link_name)
        original = _inject_mtu(net, link_name, node, resolved["mtu_bytes"])
        cleanup_fn = lambda: _cleanup_mtu(net, link_name, node, original)
        details["node"] = node

    else:
        # Unreachable: _load_faults() already rejects unknown injection
        # types at import time.
        raise ValueError(f"{fault_id}: unhandled injection type {injection!r}")

    correlation_id = f"{fault_id}-{uuid.uuid4().hex[:8]}"
    _ACTIVE[correlation_id] = {"cleanup": cleanup_fn, "details": details}
    return correlation_id


def reset(correlation_id):
    """Undo one specific injection returned by inject().

    Safe to call more than once on the same id: a repeat call (or a
    call on an id that was never active) is a no-op that reports
    already_reset=True, not an exception -- deliberately: once an
    entry is popped from _ACTIVE, there is no way to distinguish
    "already cleaned up" from "never existed" anyway, so treating both
    as "nothing left to do here" is the honest option, not a weaker one.
    """
    entry = _ACTIVE.pop(correlation_id, None)
    if entry is None:
        return {"reset": correlation_id, "already_reset": True}
    entry["cleanup"]()
    return {"reset": correlation_id, "already_reset": False, **entry["details"]}


def _restore_routing(net):
    """Re-assert every static route this topology depends on, regardless
    of which specific link was toggled during fault injection. Cheap
    insurance against exactly the class of bug network/fault_injector.py's
    own reset_all() already had to defend against: bringing one
    interface back up does not reliably restore every route that used
    to transit it.
    """
    r1, r2, r3, r4 = (net.get(n) for n in ("r1", "r2", "r3", "r4"))

    r2.cmd("ip route replace 10.0.1.0/24 via 10.0.12.1 dev r2-eth0")
    r2.cmd("ip route replace 10.0.4.0/24 via 10.0.24.2 dev r2-eth1")
    r3.cmd("ip route replace 10.0.1.0/24 via 10.0.13.1 dev r3-eth0")
    r3.cmd("ip route replace 10.0.4.0/24 via 10.0.34.2 dev r3-eth1")

    r1.cmd("ip route replace 10.0.24.0/24 via 10.0.12.2 dev r1-eth1")
    r2.cmd("ip route replace 10.0.13.0/24 via 10.0.12.1 dev r2-eth0")
    r3.cmd("ip route replace 10.0.12.0/24 via 10.0.13.1 dev r3-eth0")
    r3.cmd("ip route replace 10.0.24.0/24 via 10.0.34.2 dev r3-eth1")
    r4.cmd("ip route replace 10.0.12.0/24 via 10.0.24.1 dev r4-eth0")
    r4.cmd("ip route replace 10.0.13.0/24 via 10.0.34.1 dev r4-eth1")

    ctl.set_active_path(net, "primary")


def _load_compounds():
    with (_CONFIG_DIR / "compounds.yaml").open() as f:
        return {c["id"]: c for c in yaml.safe_load(f)["compounds"]}


_COMPOUNDS = _load_compounds()


def inject_compound(net, compound_id, seed=None):
    """Inject a compound scenario; returns list of correlation_ids."""
    if compound_id not in _COMPOUNDS:
        raise ValueError(
            f"Unknown compound id {compound_id!r}; known: {sorted(_COMPOUNDS)}"
        )
    spec = _COMPOUNDS[compound_id]
    cids = []
    for fault_id in spec["injection_order"]:
        cids.append(inject(net, fault_id, seed=seed))
    return {"compound_id": compound_id, "correlation_ids": cids}


def reset_compound(net, compound_id, correlation_ids):
    """Reset compound faults in reverse cleanup order."""
    spec = _COMPOUNDS[compound_id]
    cleared = []
    cid_by_fault = dict(zip(spec["injection_order"], correlation_ids))
    for fault_id in spec["cleanup_order"]:
        cid = cid_by_fault.get(fault_id)
        if cid:
            cleared.append(reset(cid))
    return cleared


def reset_all(net):
    """Undo every currently active injection, restore routing, and
    verify the network is actually healthy again with a real ping --
    Phase 4's stated requirement ("reset must restore a clean network,
    verify with a healthy ping check"), not just "every cleanup ran
    without an exception."

    Resilient to a single injection's cleanup raising: every remaining
    entry still gets attempted, and routing restoration + the health
    ping still run, regardless. Failures are reported in the returned
    "errors" list, not swallowed -- previously, one bad cleanup in a
    list comprehension would abort the whole call, silently skipping
    every entry after it and never reaching the routing/ping check at
    all.
    """
    try:
        from bench import flap as flap_mod
    except ImportError:
        import flap as flap_mod
    flap_mod.stop_all_flaps(net)

    cleared = []
    errors = []
    for cid in list(_ACTIVE):
        try:
            cleared.append(reset(cid))
        except Exception as exc:
            errors.append({"correlation_id": cid, "error": f"{type(exc).__name__}: {exc}"})

    _restore_routing(net)

    hostB_ip = net.get("hostB").IP()
    ping_output = net.get("hostA").cmd(f"ping -c 3 -W 1 {hostB_ip}")
    healthy = "0% packet loss" in ping_output

    return {"cleared": cleared, "errors": errors, "healthy": healthy, "ping": ping_output.strip()}
