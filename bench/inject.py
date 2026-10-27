"""
Fault injection for the benchmark: inject() and reset() for the faults in
config/faults.yaml and the compounds in config/compounds.yaml.

Link faults are single-ended: one named interface goes down and its peer
only loses carrier. network.controller's link_down()/link_up() take both
ends down and stay that way for the interactive CLI in
network/fault_injector.py.

Every command goes through _sh(), a fresh process in the node's network
namespace with its own exit status. node.cmd() shares one PTY per node
and can return output left over from an earlier command. That is where
the F6/F7 "tc command failed" errors came from: they were stale
`tc qdisc show` lines, not failures.

Before a fault changes anything, inject() snapshots the touched nodes'
links, routes, qdiscs, htb classes, interface settings and iptables
rules, and records one undo step per change. reset() runs only those
steps, newest first, then compares the nodes with the snapshot and
reports any difference as residual.

Shaping faults edit the interface's existing netem in place (adding
loss, delay, rate, corruption or reordering) and keep the htb bandwidth
cap, so a fault changes only its own parameter and a second fault on the
same interface composes with the first.
"""

import itertools
import random
import re
import time
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


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

NODES = ("hostA", "hostB", "r1", "r2", "r3", "r4")
BACKGROUND_PORT = 5202
PURPOSES = {"evaluation", "validation", "calibration", "training", "tuning"}
TUNING_PURPOSES = {"calibration", "training", "tuning"}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _sh(node_obj, command):
    """Run one shell command in the node's namespace. Returns (output, exit code)."""
    if hasattr(node_obj, "pexec"):
        out, err, code = node_obj.pexec(["bash", "-c", command])
        return (out or "") + (err or ""), code
    return node_obj.cmd(command), 0


def _run(node_obj, command):
    """_sh() that raises on a non-zero exit status."""
    out, code = _sh(node_obj, command)
    if code != 0:
        raise RuntimeError(f"{node_obj.name}: `{command}` exited {code}: {out.strip()}")
    return out


def _first_node(link_name):
    """Faults that name a link but not a node (F1, F2, F12, U1, F8, U4)
    use the node listed first for that link in network.interfaces."""
    return next(iter(LINKS[link_name]))


def _peer(link_name, node):
    return next((n for n in LINKS.get(link_name, {}) if n != node), None)


def _direction_from_node(target):
    """'<from>_to_<to>': the egress side to shape is the 'from' node."""
    return target["direction"].partition("_to_")[0]


# ---------------------------------------------------------------------------
# Link state and routes
# ---------------------------------------------------------------------------

def _operstate(net, link_name, node):
    intf = get_interface(link_name, node)
    out, _ = _sh(net.get(node), f"cat /sys/class/net/{intf}/operstate")
    return out.strip()


def _wait_link_up(net, link_name, timeout=3.0):
    """Linkwatch reports operstate up to about a second after `ip link set up`."""
    deadline = time.monotonic() + timeout
    while True:
        if all(_operstate(net, link_name, n) == "up" for n in LINKS[link_name]):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def link_down_one_end(net, link_name, node):
    """Bring down only `node`'s interface on link_name."""
    intf = get_interface(link_name, node)
    _run(net.get(node), f"ip link set dev {intf} down")
    return {"link": link_name, "node": node, "interface": intf, "action": "down"}


def link_up_one_end(net, link_name, node):
    """Bring `node`'s interface back up and wait for both ends to report up."""
    intf = get_interface(link_name, node)
    _run(net.get(node), f"ip link set dev {intf} up")
    _wait_link_up(net, link_name)
    return {"link": link_name, "node": node, "interface": intf, "action": "up"}


_ROUTE_FLAGS = {"linkdown", "dead", "offload", "trap", "onlink"}


def _route_spec(line):
    return " ".join(t for t in line.split() if t not in _ROUTE_FLAGS)


def _routes_via(net, node, intf):
    """Every route through intf as an `ip route replace` spec.

    The kernel deletes them when the interface goes admin down and only
    re-adds the connected one on up, so link_down and flap put these back.
    Connected routes come first so the gateways are reachable again.
    """
    out, _ = _sh(net.get(node), f"ip route show dev {intf}")
    specs = [f"{_route_spec(line)} dev {intf}" for line in out.splitlines() if line.strip()]
    return sorted(specs, key=lambda spec: " via " in f" {spec} ")


def _restore_routes(net, node, specs):
    node_obj = net.get(node)
    for spec in specs:
        _run(node_obj, f"ip route replace {spec}")


def _route_line(net, node, destination):
    out, _ = _sh(net.get(node), f"ip route show {destination}")
    lines = [_route_spec(line) for line in out.splitlines() if line.strip()]
    if len(lines) > 1:
        raise RuntimeError(f"{node}: more than one route for {destination}: {lines}")
    return lines[0] if lines else ""


def _put_route(net, node, destination, original):
    node_obj = net.get(node)
    if original:
        _run(node_obj, f"ip route replace {original}")
    else:
        _sh(node_obj, f"ip route del {destination}")


# ---------------------------------------------------------------------------
# Traffic control shape: htb root + class 5:1 + netem child, or a root netem
# ---------------------------------------------------------------------------

_NETEM_KEYS = {
    "limit", "delay", "loss", "corrupt", "duplicate", "reorder",
    "rate", "gap", "slot", "ecn", "distribution", "seed",
}
_QDISC_LINE = re.compile(r"^qdisc (\S+) (\S+) (?:dev \S+ )?(root|parent (\S+))\s*(.*)$")
_CLASS_LINE = re.compile(r"^class htb (\S+) .*?\brate (\S+)(?:.*?\bburst (\S+))?")
_NUMBER = re.compile(r"^(\d+(?:\.\d+)?)([a-zA-Z%]*)$")
_UNITS = {
    "": 1.0, "%": 1.0, "us": 0.001, "ms": 1.0, "s": 1000.0,
    "bit": 1e-6, "kbit": 1e-3, "mbit": 1.0, "gbit": 1e3,
}


def _netem_fields(args):
    fields, key = {}, None
    for token in (args or "").split():
        if token in _NETEM_KEYS:
            key = token
            fields[key] = []
        elif key is not None:
            fields[key].append(token)
    fields.pop("seed", None)
    return fields


def _netem_args(fields):
    return " ".join(" ".join([key, *values]) for key, values in fields.items())


def _number(token):
    match = _NUMBER.match(token)
    if not match or match.group(2).lower() not in _UNITS:
        return token
    return float(match.group(1)) * _UNITS[match.group(2).lower()]


def _close(a, b):
    a, b = _number(a), _number(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= max(1e-3, 1e-3 * abs(b))
    return a == b


def _parse_shape(intf, qdisc_out, class_out):
    shape = {"htb": None, "netem": None}
    for line in (qdisc_out or "").splitlines():
        match = _QDISC_LINE.match(line.strip())
        if not match:
            continue
        kind, handle, parent, rest = match.group(1), match.group(2), match.group(4) or "root", match.group(5)
        if parent.startswith("ffff:"):
            continue
        rest = re.sub(r"^refcnt \d+\s*", "", rest)
        if parent == "root" and handle == "0:":
            continue
        if kind == "htb" and parent == "root":
            default = re.search(r"\bdefault (\S+)", rest)
            shape["htb"] = {
                "handle": handle,
                "default": default.group(1) if default else "0x1",
                "classid": None,
                "rate": None,
                "burst": "15k",
            }
        elif kind == "netem" and shape["netem"] is None:
            fields = _netem_fields(rest)
            shape["netem"] = {"handle": handle, "parent": parent, "args": _netem_args(fields)}
        else:
            raise RuntimeError(f"{intf}: unsupported qdisc layout: {line.strip()}")
    if shape["htb"]:
        classes = [m for m in (_CLASS_LINE.match(l.strip()) for l in (class_out or "").splitlines()) if m]
        if len(classes) != 1:
            raise RuntimeError(f"{intf}: expected one htb class, found {len(classes)}")
        shape["htb"].update(classid=classes[0].group(1), rate=classes[0].group(2))
        if classes[0].group(3):
            shape["htb"]["burst"] = classes[0].group(3)
    netem = shape["netem"]
    if netem and netem["parent"] != "root" and netem["parent"] != (shape["htb"] or {}).get("classid"):
        raise RuntimeError(f"{intf}: netem under unknown parent {netem['parent']}")
    return shape


def read_shape(net, link_name, node):
    """The interface's current htb rate and netem parameters."""
    intf = get_interface(link_name, node)
    node_obj = net.get(node)
    qdisc_out, _ = _sh(node_obj, f"tc qdisc show dev {intf}")
    class_out, _ = _sh(node_obj, f"tc class show dev {intf}")
    return _parse_shape(intf, qdisc_out, class_out)


def _comparable_netem(args):
    """netem fields with the kernel defaults tc may or may not print:
    limit 1000, and gap 1 once reordering is set."""
    fields = _netem_fields(args)
    fields.setdefault("limit", ["1000"])
    if fields.get("gap") == ["1"]:
        fields.pop("gap")
    return fields


def same_shape(a, b):
    for part in ("htb", "netem"):
        if (a.get(part) is None) != (b.get(part) is None):
            return False
    if a["htb"] and not _close(a["htb"]["rate"], b["htb"]["rate"]):
        return False
    if a["netem"]:
        if a["netem"]["parent"] != b["netem"]["parent"]:
            return False
        fa, fb = _comparable_netem(a["netem"]["args"]), _comparable_netem(b["netem"]["args"])
        if set(fa) != set(fb):
            return False
        for key in fa:
            if len(fa[key]) != len(fb[key]) or not all(map(_close, fa[key], fb[key])):
                return False
    return True


def describe_shape(shape):
    parts = []
    if shape.get("htb"):
        parts.append(f"htb rate {shape['htb']['rate']}")
    if shape.get("netem"):
        parts.append(f"netem {shape['netem']['args']}")
    return " + ".join(parts) or "no shaping"


def apply_shape(net, link_name, node, shape):
    """Rebuild one interface's qdiscs and confirm by reading them back.

    Success means every tc command exited 0 and the readback matches.
    Text on stdout is not an error.
    """
    intf = get_interface(link_name, node)
    node_obj = net.get(node)
    _sh(node_obj, f"tc qdisc del dev {intf} root")
    htb, netem = shape.get("htb"), shape.get("netem")
    if htb:
        default = int(str(htb["default"]), 16)
        _run(node_obj, f"tc qdisc add dev {intf} root handle {htb['handle']} htb default {default:x}")
        _run(
            node_obj,
            f"tc class add dev {intf} parent {htb['handle']} classid {htb['classid']} "
            f"htb rate {htb['rate']} burst {htb['burst']}",
        )
    if netem:
        where = "root" if netem["parent"] == "root" else f"parent {netem['parent']}"
        _run(node_obj, f"tc qdisc add dev {intf} {where} handle {netem['handle']} netem {netem['args']}")
    got = read_shape(net, link_name, node)
    if not same_shape(got, shape):
        raise RuntimeError(
            f"{link_name}/{node} ({intf}): tc reads back {describe_shape(got)}, "
            f"wanted {describe_shape(shape)}"
        )
    return got


def baseline_shape():
    params = ctl.BASELINE_LINK_PARAMS
    fields = {"limit": ["1000"], "delay": [params["delay"]]}
    if params.get("loss"):
        fields["loss"] = [f"{params['loss']:g}%"]
    return {
        "htb": {"handle": "5:", "default": "0x1", "classid": "5:1", "rate": f"{params['bw']:g}Mbit", "burst": "15k"},
        "netem": {"handle": "10:", "parent": "5:1", "args": _netem_args(fields)},
    }


def _netem_with(shape, **changes):
    """Copy of shape with netem fields set; a value of None removes the field."""
    new = {
        "htb": dict(shape["htb"]) if shape.get("htb") else None,
        "netem": dict(shape["netem"]) if shape.get("netem") else None,
    }
    if new["netem"] is None:
        parent = new["htb"]["classid"] if new["htb"] else "root"
        new["netem"] = {"handle": "10:", "parent": parent, "args": "limit 1000"}
    fields = _netem_fields(new["netem"]["args"])
    for key, value in changes.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = str(value).split()
    new["netem"]["args"] = _netem_args(fields)
    return new


def _netem_value(shape, key):
    fields = _netem_fields((shape.get("netem") or {}).get("args"))
    return " ".join(fields[key]) if fields.get(key) else None


# ---------------------------------------------------------------------------
# Snapshots and residual detection
# ---------------------------------------------------------------------------

_SNAPSHOT = {
    "links": "ip -o link show",
    "routes": "ip route show",
    "qdiscs": "tc qdisc show",
    "classes": 'for d in $(ls /sys/class/net); do tc class show dev "$d" | sed "s|^|$d |"; done',
    "settings": (
        'for d in $(ls /sys/class/net); do [ "$d" = lo ] && continue; '
        'echo "$d mtu=$(cat /sys/class/net/$d/mtu) '
        'ipv6_off=$(cat /proc/sys/net/ipv6/conf/$d/disable_ipv6 2>/dev/null) '
        "rx_csum=$(ethtool -k $d 2>/dev/null | awk '/^rx-checksumming:/{print $2}')\"; done"
    ),
    "firewall": "iptables-save 2>/dev/null",
}
_LINK_LINE = re.compile(r"^\d+:\s+([^:@\s]+)(?:@\S+)?:\s+<([^>]*)>.*?\bmtu (\d+).*?\bstate (\S+)")
_QDISC_COUNTERS = re.compile(r"\b(direct_packets_stat|refcnt|seed)\s+\d+")
_IPT_COUNTERS = re.compile(r"\s*\[\d+:\d+\]")


def _normalize(key, text):
    lines = [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]
    if key == "links":
        out = []
        for line in lines:
            match = _LINK_LINE.match(line)
            if match:
                out.append(f"{match.group(1)} <{match.group(2)}> mtu {match.group(3)} state {match.group(4)}")
        return sorted(out)
    if key == "qdiscs":
        return sorted(" ".join(_QDISC_COUNTERS.sub("", l).split()) for l in lines if l.startswith("qdisc "))
    if key == "firewall":
        return [_IPT_COUNTERS.sub("", l) for l in lines if not l.startswith("#")]
    return sorted(lines)


def snapshot(net, nodes=NODES):
    """Links, routes, qdiscs, classes, interface settings and iptables per node."""
    state = {}
    for name in nodes:
        node_obj = net.get(name)
        state[name] = {key: _normalize(key, _sh(node_obj, command)[0]) for key, command in _SNAPSHOT.items()}
    return state


def diff_snapshots(before, after):
    """Lines present in one snapshot and not the other, as readable strings."""
    out = []
    for node in sorted(set(before) | set(after)):
        old_node, new_node = before.get(node, {}), after.get(node, {})
        for key in _SNAPSHOT:
            old, new = old_node.get(key, []), new_node.get(key, [])
            if old == new:
                continue
            gone = [line for line in old if line not in new]
            added = [line for line in new if line not in old]
            if not gone and not added:
                out.append(f"{node} {key}: order changed")
            out += [f"{node} {key}: - {line}" for line in gone]
            out += [f"{node} {key}: + {line}" for line in added]
    return out


# ---------------------------------------------------------------------------
# Background processes (F6/F7 background_traffic)
# ---------------------------------------------------------------------------

def _spawn(node_obj, command, log):
    out, code = _sh(node_obj, f"nohup {command} </dev/null >{log} 2>&1 & echo $!")
    lines = out.strip().splitlines()
    pid = lines[-1].strip() if lines else ""
    if code != 0 or not pid.isdigit():
        raise RuntimeError(f"{node_obj.name}: could not start `{command}`: {out.strip()}")
    return pid


def _alive(node_obj, pid):
    return _sh(node_obj, f"kill -0 {pid} 2>/dev/null")[1] == 0


def _stop_pid(node_obj, pid):
    """Stop one process this injection started, and confirm it is gone."""
    _sh(node_obj, f"kill {pid} 2>/dev/null")
    for _ in range(20):
        if not _alive(node_obj, pid):
            return
        time.sleep(0.1)
    _sh(node_obj, f"kill -9 {pid} 2>/dev/null")
    time.sleep(0.1)
    if _alive(node_obj, pid):
        raise RuntimeError(f"{node_obj.name}: process {pid} is still running")


def _wait_listening(node_obj, port, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        out, _ = _sh(node_obj, f"ss -ltnH 'sport = :{int(port)}'")
        if out.strip():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _start_background_traffic(net, from_node, to_node, rate_mbit, undo):
    """UDP from from_node to an iperf3 server on to_node, faster than the shaper.

    The server and client are this injection's own processes, stopped by pid.
    Noise traffic (hostA to hostB) and the T8/T10 server on 5201 are separate.
    """
    dest, src = net.get(to_node), net.get(from_node)
    server = _spawn(dest, f"iperf3 -s -p {BACKGROUND_PORT}", f"/tmp/inject_bg_server_{to_node}.log")
    undo.append(lambda: _stop_pid(dest, server))
    if not _wait_listening(dest, BACKGROUND_PORT):
        raise RuntimeError(f"{to_node}: iperf3 server on {BACKGROUND_PORT} did not start")
    send_mbit = max(float(rate_mbit) * 8, 20)
    client = _spawn(
        src,
        f"iperf3 -c {dest.IP()} -p {BACKGROUND_PORT} -u -b {send_mbit:.0f}M -t 3600",
        f"/tmp/inject_bg_{from_node}.log",
    )
    undo.append(lambda: _stop_pid(src, client))
    time.sleep(0.3)
    if not _alive(src, client):
        log, _ = _sh(src, f"tail -n 5 /tmp/inject_bg_{from_node}.log")
        raise RuntimeError(f"{from_node}: background iperf3 client exited: {log.strip()}")
    return {"server_pid": server, "client_pid": client, "from_node": from_node, "to_node": to_node}


# ---------------------------------------------------------------------------
# Fault types. Each records an undo step before making its change.
# ---------------------------------------------------------------------------

def _apply_none(net, target, params, details, undo):
    return None


def _apply_link_down(net, target, params, details, undo):
    link_name = target["link"]
    node = target.get("node") or _first_node(link_name)
    routes = _routes_via(net, node, get_interface(link_name, node))
    details.update(node=node, saved_routes=routes)
    undo.append(lambda: (link_up_one_end(net, link_name, node), _restore_routes(net, node, routes)))
    link_down_one_end(net, link_name, node)


def _shape_fault(build):
    def apply(net, target, params, details, undo):
        link_name = target["link"]
        node = _direction_from_node(target)
        original = read_shape(net, link_name, node)
        details.update(node=node, interface=get_interface(link_name, node), pre_injection_shape=original)
        undo.append(lambda: apply_shape(net, link_name, node, original))
        details["injected_shape"] = apply_shape(net, link_name, node, build(original, params))
    return apply


_apply_netem_loss = _shape_fault(lambda s, p: _netem_with(s, loss=f"{p['loss_percent']:g}%"))
_apply_netem_delay = _shape_fault(lambda s, p: _netem_with(s, delay=f"{p['delay_ms']:g}ms"))
_apply_netem_reorder_dup = _shape_fault(lambda s, p: _netem_with(
    s,
    delay=_netem_value(s, "delay") or "10ms",
    reorder=f"{p['reorder_percent']:g}% 50%",
    duplicate=f"{p['duplicate_percent']:g}%",
))
_apply_corrupt_shape = _shape_fault(lambda s, p: _netem_with(s, corrupt=f"{p['corrupt_percent']:g}%"))


def _rx_checksumming(node_obj, intf):
    out, _ = _sh(node_obj, f"ethtool -k {intf}")
    match = re.search(r"^rx-checksumming:\s+(\S+)", out, re.M)
    return match.group(1) if match else ""


def _set_rx_checksumming(node_obj, intf, state):
    _sh(node_obj, f"ethtool -K {intf} rx {state}")
    got = _rx_checksumming(node_obj, intf)
    if got != state:
        raise RuntimeError(f"{node_obj.name}: rx-checksumming on {intf} is {got or 'unknown'}, wanted {state}")


def _apply_netem_corrupt(net, target, params, details, undo):
    """veth accepts corrupted frames unless the receiver checks checksums,
    so the peer's rx checksum offload goes off for the duration. Otherwise
    T5 never sees InCsumErrors and corruption looks like a healthy link."""
    _apply_corrupt_shape(net, target, params, details, undo)
    link_name = target["link"]
    peer = _peer(link_name, details["node"])
    peer_obj, peer_intf = net.get(peer), get_interface(link_name, peer)
    original = _rx_checksumming(peer_obj, peer_intf)
    details.update(peer=peer, peer_intf=peer_intf, peer_rx_checksumming=original)
    if original == "on":
        undo.append(lambda: _set_rx_checksumming(peer_obj, peer_intf, "on"))
        _set_rx_checksumming(peer_obj, peer_intf, "off")


_apply_tbf_shape = _shape_fault(lambda s, p: _netem_with(s, rate=f"{p['rate_mbit']:g}Mbit"))


def _apply_tbf_congestion(net, target, params, details, undo):
    """netem rate under the existing htb class: the link keeps its delay and
    any loss already on it, and drains at rate_mbit."""
    _apply_tbf_shape(net, target, params, details, undo)
    if params.get("background_traffic"):
        from_node, _, to_node = target["direction"].partition("_to_")
        details["background"] = _start_background_traffic(net, from_node, to_node, params["rate_mbit"], undo)


def _apply_link_flap(net, target, params, details, undo):
    try:
        from bench import flap as flap_mod
    except ImportError:
        import flap as flap_mod

    link_name = target["link"]
    node = target.get("node") or _first_node(link_name)
    routes = _routes_via(net, node, get_interface(link_name, node))
    undo.append(lambda: (link_up_one_end(net, link_name, node), _restore_routes(net, node, routes)))
    flap_id = flap_mod.start_link_flap(
        net,
        link_name,
        node=node,
        interval_seconds=params.get("interval_seconds", 2),
        cycles=params.get("cycles", 3),
        on_up=lambda: _restore_routes(net, node, routes),
    )
    undo.append(lambda: flap_mod.stop_flap(flap_id))
    details.update(node=node, flap_id=flap_id, saved_routes=routes)


def _apply_route(kind):
    def apply(net, target, params, details, undo):
        node, destination = target["node"], target["destination"]
        original = _route_line(net, node, destination)
        details.update(node=node, destination=destination, original_route=original)
        undo.append(lambda: _put_route(net, node, destination, original))
        if kind == "blackhole":
            _run(net.get(node), f"ip route replace blackhole {destination}")
        else:
            _run(net.get(node), f"ip route replace {destination} via {target['wrong_via']} dev {target['wrong_dev']}")
    return apply


def _drop_rule(node_obj, rule):
    if _sh(node_obj, f"iptables -C {rule}")[1] == 0:
        _run(node_obj, f"iptables -D {rule}")


def _apply_acl_drop(net, target, params, details, undo):
    """Adds one FORWARD rule and removes exactly that rule; never a flush."""
    node_obj = net.get(target["node"])
    rule = f"FORWARD -p {target['protocol']} --dport {target['port']} -j DROP"
    details.update(node=target["node"], rule=rule)
    undo.append(lambda: _drop_rule(node_obj, rule))
    _run(node_obj, f"iptables -A {rule}")


def _apply_mtu_mismatch(net, target, params, details, undo):
    """Both ends get the lower MTU so a DF ping cannot pass through the
    peer's default. IPv6 refuses an MTU under 1280, so it is disabled on
    those interfaces for the duration."""
    link_name = target["link"]
    mtu = int(params["mtu_bytes"])
    saved = []
    for node, intf in LINKS[link_name].items():
        node_obj = net.get(node)
        original_mtu = _sh(node_obj, f"cat /sys/class/net/{intf}/mtu")[0].strip()
        original_v6 = _sh(node_obj, f"cat /proc/sys/net/ipv6/conf/{intf}/disable_ipv6")[0].strip() or "0"
        saved.append({"node": node, "interface": intf, "mtu": original_mtu, "disable_ipv6": original_v6})
        undo.append(lambda n=node_obj, i=intf, m=original_mtu, v=original_v6: (
            _run(n, f"ip link set dev {i} mtu {m}"),
            _run(n, f"sysctl -w net.ipv6.conf.{i}.disable_ipv6={v}"),
        ))
        _run(node_obj, f"sysctl -w net.ipv6.conf.{intf}.disable_ipv6=1")
        _run(node_obj, f"ip link set dev {intf} mtu {mtu}")
        got = _sh(node_obj, f"cat /sys/class/net/{intf}/mtu")[0].strip()
        if got != str(mtu):
            raise RuntimeError(f"{link_name}/{node} ({intf}): mtu stayed {got or 'unset'}, wanted {mtu}")
    details.update(node=_first_node(link_name), saved_mtu=saved)


_APPLY = {
    "none": _apply_none,
    "link_down": _apply_link_down,
    "netem_loss": _apply_netem_loss,
    "netem_delay": _apply_netem_delay,
    "netem_corrupt": _apply_netem_corrupt,
    "tbf_congestion": _apply_tbf_congestion,
    "link_flap": _apply_link_flap,
    "blackhole_route": _apply_route("blackhole"),
    "acl_drop": _apply_acl_drop,
    "mtu_mismatch": _apply_mtu_mismatch,
    "netem_reorder_dup": _apply_netem_reorder_dup,
    "wrong_return_route": _apply_route("wrong_return"),
}
_SHAPE_TYPES = {"netem_loss", "netem_delay", "netem_corrupt", "tbf_congestion", "netem_reorder_dup"}


def _resources(injection, target):
    """What an injection changes, for last-in-first-out reset ordering."""
    if injection == "none":
        return set()
    if injection in ("blackhole_route", "wrong_return_route"):
        return {("routes", target["node"], "")}
    if injection == "acl_drop":
        return {("iptables", target["node"], "")}
    link_name = target["link"]
    if injection in _SHAPE_TYPES:
        node = _direction_from_node(target)
        out = {("tc", node, get_interface(link_name, node))}
        if injection == "netem_corrupt":
            peer = _peer(link_name, node)
            out.add(("intf", peer, get_interface(link_name, peer)))
        return out
    if injection == "mtu_mismatch":
        return {("intf", n, i) for n, i in LINKS[link_name].items()}
    node = target.get("node") or _first_node(link_name)
    return {("link", node, get_interface(link_name, node)), ("routes", node, "")}


def _touched_nodes(injection, target, resources):
    nodes = {r[1] for r in resources}
    if injection != "none" and "link" in target:
        nodes |= set(LINKS[target["link"]])
    return sorted(nodes)


# ---------------------------------------------------------------------------
# faults.yaml / compounds.yaml
# ---------------------------------------------------------------------------

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
        if injection not in _APPLY:
            raise ValueError(f"{fault['id']}: invalid injection type {injection!r}")
        target = fault.get("target") or {}
        missing = [k for k in _REQUIRED_TARGET_KEYS.get(injection, []) if k not in target]
        if missing:
            raise ValueError(
                f"{fault['id']}: injection type {injection!r} is missing "
                f"required target field(s) {missing} (target={target!r})"
            )
        by_id[fault["id"]] = fault
    return by_id


def _load_compounds():
    with (_CONFIG_DIR / "compounds.yaml").open() as f:
        return {c["id"]: c for c in yaml.safe_load(f)["compounds"]}


_FAULTS = _load_faults()
_COMPOUNDS = _load_compounds()


def held_out_ids():
    """Fault and compound ids marked split: held_out."""
    return {i for i, f in {**_FAULTS, **_COMPOUNDS}.items() if f.get("split") == "held_out"}


def _check_purpose(item_id, split, purpose):
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(PURPOSES)}, got {purpose!r}")
    if split == "held_out" and purpose in TUNING_PURPOSES:
        raise ValueError(f"{item_id} is held out and cannot be injected for {purpose}")


def _sample_param(value, rng):
    """A [low, high] pair is a severity range to sample; anything else is used as-is."""
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
# inject / reset
# ---------------------------------------------------------------------------

_ACTIVE = {}  # correlation_id -> entry
_SEQ = itertools.count()


def _run_undo(steps):
    """Run undo steps newest first. Every step is attempted; failures are raised together.
    Steps are idempotent, so a failed reset can be retried."""
    errors = []
    for step in reversed(steps):
        try:
            step()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def inject(net, fault_id, params=None, seed=None, purpose="evaluation"):
    """Inject one fault from config/faults.yaml. Returns a correlation_id.

    params overrides the fault's own params (ranges are still sampled).
    seed makes the sampled severity reproducible. purpose "calibration",
    "training" or "tuning" refuses held-out faults.

    If any step fails, the steps already applied are undone before the
    error is raised, so the call can simply be retried. If that rollback
    also fails, the entry stays in _ACTIVE for reset_all().
    """
    if fault_id not in _FAULTS:
        raise ValueError(f"Unknown fault id {fault_id!r}; known ids: {sorted(_FAULTS)}")
    fault = _FAULTS[fault_id]
    _check_purpose(fault_id, fault.get("split"), purpose)
    injection = fault["injection"]["type"]
    target = fault.get("target") or {}
    resolved = _resolve_params(params if params is not None else fault.get("params"), seed)

    correlation_id = f"{fault_id}-{uuid.uuid4().hex[:8]}"
    resources = _resources(injection, target)
    nodes = _touched_nodes(injection, target, resources)
    details = {
        "correlation_id": correlation_id,
        "fault_id": fault_id,
        "injection": injection,
        "params": resolved,
        "seed": seed,
        "purpose": purpose,
    }
    entry = {
        "net": net,
        "seq": next(_SEQ),
        "details": details,
        "resources": resources,
        "nodes": nodes,
        "pre_state": snapshot(net, nodes),
        "undo": [],
    }
    _ACTIVE[correlation_id] = entry
    try:
        _APPLY[injection](net, target, resolved, details, entry["undo"])
    except Exception as exc:
        try:
            _run_undo(entry["undo"])
        except Exception as undo_exc:
            details["rollback_error"] = str(undo_exc)
            raise RuntimeError(
                f"{correlation_id}: inject failed ({exc}); rollback failed ({undo_exc}); "
                f"still tracked for reset_all()"
            ) from exc
        _ACTIVE.pop(correlation_id, None)
        raise
    return correlation_id


def _newer_overlap(correlation_id, entry):
    for other_id, other in _ACTIVE.items():
        if other_id != correlation_id and other["seq"] > entry["seq"] and other["resources"] & entry["resources"]:
            return other_id
    return None


def reset(correlation_id):
    """Undo one injection and compare its nodes with the pre-injection snapshot.

    Unknown or already-reset ids are a no-op (already_reset=True). If a
    newer active injection changed the same interface, routes or rules,
    that one must be reset first. If an undo step fails, the entry stays
    active and reset() can be called again.
    """
    entry = _ACTIVE.get(correlation_id)
    if entry is None:
        return {"reset": correlation_id, "already_reset": True}
    newer = _newer_overlap(correlation_id, entry)
    if newer:
        raise RuntimeError(f"reset {newer} before {correlation_id}: it changed the same state afterwards")
    _run_undo(entry["undo"])
    _ACTIVE.pop(correlation_id, None)
    residual = diff_snapshots(entry["pre_state"], snapshot(entry["net"], entry["nodes"]))
    return {
        "reset": correlation_id,
        "already_reset": False,
        "restored": not residual,
        "residual": residual,
        **entry["details"],
    }


def inject_compound(net, compound_id, seed=None, purpose="evaluation"):
    """Inject a compound in its injection_order. A failure resets the parts already injected."""
    if compound_id not in _COMPOUNDS:
        raise ValueError(f"Unknown compound id {compound_id!r}; known: {sorted(_COMPOUNDS)}")
    spec = _COMPOUNDS[compound_id]
    _check_purpose(compound_id, spec.get("split"), purpose)
    cids = []
    try:
        for fault_id in spec["injection_order"]:
            cid = inject(net, fault_id, seed=seed, purpose=purpose)
            _ACTIVE[cid]["details"]["compound_id"] = compound_id
            cids.append(cid)
    except Exception:
        for cid in reversed(cids):
            reset(cid)
        raise
    return {"compound_id": compound_id, "correlation_ids": cids}


def reset_compound(net, compound_id, correlation_ids):
    """Reset a compound's parts in its cleanup_order."""
    spec = _COMPOUNDS[compound_id]
    cid_by_fault = dict(zip(spec["injection_order"], correlation_ids))
    return [reset(cid_by_fault[f]) for f in spec["cleanup_order"] if f in cid_by_fault]


# ---------------------------------------------------------------------------
# Whole-network helpers
# ---------------------------------------------------------------------------

_STATIC_ROUTES = {
    "r1": ["10.0.24.0/24 via 10.0.12.2 dev r1-eth1"],
    "r2": [
        "10.0.1.0/24 via 10.0.12.1 dev r2-eth0",
        "10.0.4.0/24 via 10.0.24.2 dev r2-eth1",
        "10.0.13.0/24 via 10.0.12.1 dev r2-eth0",
    ],
    "r3": [
        "10.0.1.0/24 via 10.0.13.1 dev r3-eth0",
        "10.0.4.0/24 via 10.0.34.2 dev r3-eth1",
        "10.0.12.0/24 via 10.0.13.1 dev r3-eth0",
        "10.0.24.0/24 via 10.0.34.2 dev r3-eth1",
    ],
    "r4": [
        "10.0.12.0/24 via 10.0.24.1 dev r4-eth0",
        "10.0.13.0/24 via 10.0.34.1 dev r4-eth1",
    ],
}
_LOSS = re.compile(r"(\d+(?:\.\d+)?)% packet loss")


def ping_loss(text):
    """Loss percent from ping's summary line, or None when there is none."""
    match = _LOSS.search(text or "")
    return float(match.group(1)) if match else None


def health_check(net, count=3):
    """(loss percent or None, ping output) for hostA to hostB."""
    out, _ = _sh(net.get("hostA"), f"ping -c {count} -i 0.2 -W 1 {net.get('hostB').IP()}")
    return ping_loss(out), out.strip()


def restore_baseline(net):
    """Put every link up at its startup shape, every static route back and
    traffic on the primary path. This is episode teardown after a
    remediation moved traffic, not a fault reset: it also undoes noise."""
    try:
        from bench import flap as flap_mod
    except ImportError:
        import flap as flap_mod
    flap_mod.stop_all_flaps(net)
    shape = baseline_shape()
    for link_name, ends in LINKS.items():
        for node, intf in ends.items():
            _run(net.get(node), f"ip link set dev {intf} up")
            apply_shape(net, link_name, node, shape)
        _wait_link_up(net, link_name)
    for router, specs in _STATIC_ROUTES.items():
        _restore_routes(net, router, specs)
    for router, cfg in ctl.ROUTE_TARGETS.items():
        hop = cfg["primary"]
        _run(net.get(router), f"ip route replace {cfg['dest']} via {hop['via']} dev {hop['dev']}")


def reset_all(net, baseline=False):
    """Reset every active injection, newest first, then ping hostA to hostB.

    Only injected changes are undone, so noise and remediation state stay.
    baseline=True also runs restore_baseline() (episode teardown). A failed
    reset is reported in "errors" and stays active for a retry.
    """
    try:
        from bench import flap as flap_mod
    except ImportError:
        import flap as flap_mod
    flap_mod.stop_all_flaps(net)

    cleared, errors, residual = [], [], {}
    for cid in sorted(_ACTIVE, key=lambda c: _ACTIVE[c]["seq"], reverse=True):
        try:
            result = reset(cid)
            cleared.append(result)
            if result.get("residual"):
                residual[cid] = result["residual"]
        except Exception as exc:
            errors.append({"correlation_id": cid, "error": f"{type(exc).__name__}: {exc}"})

    if baseline:
        try:
            restore_baseline(net)
        except Exception as exc:
            errors.append({"correlation_id": None, "error": f"restore_baseline: {type(exc).__name__}: {exc}"})

    loss, ping_output = health_check(net)
    return {
        "cleared": cleared,
        "errors": errors,
        "residual": residual,
        "healthy": loss == 0.0,
        "loss_percent": loss,
        "ping": ping_output,
    }
