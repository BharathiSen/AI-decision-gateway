"""
Offline checks for bench/inject.py against a simulated network.

FakeNode keeps per-interface link, tc, route, iptables and process state
and answers the commands inject.py sends through pexec. These checks cover
what inject() changes and whether reset() puts back exactly that. They do
not prove the kernel accepts the commands; that is the VM run of
bench/validate_faults.py.

Run: python bench/test_inject.py
"""

import itertools
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

from bench import flap
from bench import inject as inj
from network.interfaces import LINKS


# ---------------------------------------------------------------------------
# Simulated network
# ---------------------------------------------------------------------------

IPS = {"hostA": "10.0.1.2", "hostB": "10.0.4.2", "r1": "10.0.1.1", "r2": "10.0.12.2", "r3": "10.0.13.2", "r4": "10.0.24.2"}
CONNECTED = {
    "hostA-eth0": "10.0.1.0/24 dev hostA-eth0 proto kernel scope link src 10.0.1.2",
    "r1-eth0": "10.0.1.0/24 dev r1-eth0 proto kernel scope link src 10.0.1.1",
    "r1-eth1": "10.0.12.0/24 dev r1-eth1 proto kernel scope link src 10.0.12.1",
    "r1-eth2": "10.0.13.0/24 dev r1-eth2 proto kernel scope link src 10.0.13.1",
    "r2-eth0": "10.0.12.0/24 dev r2-eth0 proto kernel scope link src 10.0.12.2",
    "r2-eth1": "10.0.24.0/24 dev r2-eth1 proto kernel scope link src 10.0.24.1",
    "r3-eth0": "10.0.13.0/24 dev r3-eth0 proto kernel scope link src 10.0.13.2",
    "r3-eth1": "10.0.34.0/24 dev r3-eth1 proto kernel scope link src 10.0.34.1",
    "r4-eth0": "10.0.24.0/24 dev r4-eth0 proto kernel scope link src 10.0.24.2",
    "r4-eth1": "10.0.34.0/24 dev r4-eth1 proto kernel scope link src 10.0.34.2",
    "r4-eth2": "10.0.4.0/24 dev r4-eth2 proto kernel scope link src 10.0.4.1",
    "hostB-eth0": "10.0.4.0/24 dev hostB-eth0 proto kernel scope link src 10.0.4.2",
}
STATIC = {
    "hostA": ["default via 10.0.1.1 dev hostA-eth0"],
    "hostB": ["default via 10.0.4.1 dev hostB-eth0"],
    "r1": ["10.0.4.0/24 via 10.0.12.2 dev r1-eth1", *inj._STATIC_ROUTES["r1"]],
    "r2": inj._STATIC_ROUTES["r2"],
    "r3": inj._STATIC_ROUTES["r3"],
    "r4": ["10.0.1.0/24 via 10.0.24.1 dev r4-eth0", *inj._STATIC_ROUTES["r4"]],
}


def _canon(spec):
    tokens = spec.split()
    if tokens[0] == "blackhole":
        return tokens[1], f"blackhole {tokens[1]}"
    prefix, rest = tokens[0], tokens[1:]
    via = dev = None
    others = []
    i = 0
    while i < len(rest):
        if rest[i] in ("via", "dev") and i + 1 < len(rest):
            if rest[i] == "via":
                via = rest[i + 1]
            else:
                dev = rest[i + 1]
            i += 2
            continue
        others.append(rest[i])
        i += 1
    line = prefix + (f" via {via}" if via else "") + (f" dev {dev}" if dev else "")
    return prefix, line + (" " + " ".join(others) if others else "")


class FakeProc:
    def __init__(self, code=0):
        self.returncode = code

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _baseline_tc():
    return {
        "htb": {"handle": "5:", "default": 1, "classid": "5:1", "rate": "10Mbit", "burst": "15Kb"},
        "netem": {"handle": "10:", "parent": "5:1", "args": "limit 1000 delay 5ms"},
        "other": None,
    }


class FakeNode:
    def __init__(self, net, name, intfs):
        self.net = net
        self.name = name
        self.cmds = []
        self.intfs = {
            i: {"up": True, "mtu": 1500, "ipv6_off": 0, "rx_csum": "on", "tc": _baseline_tc()}
            for i in intfs
        }
        self.routes = {}
        for spec in [CONNECTED[i] for i in intfs] + STATIC[name]:
            key, line = _canon(spec)
            self.routes[key] = line
        self.rules = []
        self.iptables_used = False
        self.counter = itertools.count(1)
        self.tc_fail = None      # (substring, message): tc exits 2 with message
        self.tc_ignore = None    # substring: tc exits 0 but changes nothing
        self.tc_echo = ""        # text every tc add prints with exit 0
        self.fail_once = {}      # substring -> message, for one command

    def IP(self):
        return IPS[self.name]

    def intfList(self):
        return list(self.intfs)

    # -- link helpers -------------------------------------------------------
    def _peer_up(self, intf):
        for ends in LINKS.values():
            if self.name in ends and ends[self.name] == intf:
                peer = next((n for n in ends if n != self.name), None)
                return self.net.get(peer).intfs[ends[peer]]["up"]
        return True

    def _oper(self, intf):
        if not self.intfs[intf]["up"]:
            return "down"
        return "up" if self._peer_up(intf) else "lowerlayerdown"

    def _set_link(self, intf, state):
        self.intfs[intf]["up"] = state == "up"
        if state == "down":
            self.routes = {k: v for k, v in self.routes.items() if f" dev {intf}" not in f"{v} "}
        elif intf in CONNECTED:
            key, line = _canon(CONNECTED[intf])
            self.routes[key] = line

    # -- tc rendering -------------------------------------------------------
    def _qdisc_lines(self, intf, with_dev):
        tc = self.intfs[intf]["tc"]
        dev = f"dev {intf} " if with_dev else ""
        if tc["other"]:
            return [tc["other"].replace("qdisc tbf 1: ", f"qdisc tbf 1: {dev}")]
        lines = []
        if tc["htb"]:
            h = tc["htb"]
            lines.append(
                f"qdisc htb {h['handle']} {dev}root refcnt 2 r2q 10 default 0x{h['default']:x} "
                f"direct_packets_stat {next(self.counter)} direct_qlen 1000"
            )
        if tc["netem"]:
            n = tc["netem"]
            where = "root refcnt 2" if n["parent"] == "root" else f"parent {n['parent']}"
            args = n["args"] if "limit" in n["args"] else f"limit 1000 {n['args']}"
            lines.append(f"qdisc netem {n['handle']} {dev}{where} {args} seed {next(self.counter)}")
        return lines or [f"qdisc noqueue 0: {dev}root refcnt 2"]

    def _class_lines(self, intf):
        tc = self.intfs[intf]["tc"]
        h = tc["htb"]
        if not h or not h["classid"]:
            return []
        leaf = "leaf 10: " if tc["netem"] and tc["netem"]["parent"] == h["classid"] else ""
        return [f"class htb {h['classid']} root {leaf}prio 0 rate {h['rate']} ceil {h['rate']} burst {h['burst']} cburst 1600b"]

    def _tc(self, c):
        if self.tc_fail and self.tc_fail[0] in c and not c.startswith("tc qdisc del"):
            return self.tc_fail[1], 2
        if self.tc_ignore and self.tc_ignore in c:
            return "", 0
        m = re.match(r"^tc qdisc del dev (\S+) root$", c)
        if m:
            tc = self.intfs[m.group(1)]["tc"]
            had = tc["htb"] or tc["netem"] or tc["other"]
            tc.update(htb=None, netem=None, other=None)
            return ("", 0) if had else ("Error: Cannot delete qdisc with handle of zero.", 2)
        m = re.match(r"^tc qdisc add dev (\S+) root handle (\S+) htb default (\S+)$", c)
        if m:
            tc = self.intfs[m.group(1)]["tc"]
            if tc["htb"] or tc["netem"]:
                return "RTNETLINK answers: File exists", 2
            tc["htb"] = {"handle": m.group(2), "default": int(m.group(3), 16), "classid": None, "rate": None, "burst": None}
            return self.tc_echo, 0
        m = re.match(r"^tc class add dev (\S+) parent (\S+) classid (\S+) htb rate (\S+) burst (\S+)$", c)
        if m:
            h = self.intfs[m.group(1)]["tc"]["htb"]
            if not h:
                return "RTNETLINK answers: Invalid argument", 2
            h.update(classid=m.group(3), rate=m.group(4), burst=m.group(5).replace("k", "Kb") if m.group(5).endswith("k") else m.group(5))
            return self.tc_echo, 0
        m = re.match(r"^tc qdisc add dev (\S+) (?:root|parent (\S+)) handle (\S+) netem (.*)$", c)
        if m:
            tc = self.intfs[m.group(1)]["tc"]
            parent = m.group(2) or "root"
            if "reorder" in m.group(4) and "delay" not in m.group(4):
                return "reordering not possible without specifying some delay", 1
            if parent != "root" and (not tc["htb"] or tc["htb"]["classid"] != parent):
                return "RTNETLINK answers: Invalid argument", 2
            tc["netem"] = {"handle": m.group(3), "parent": parent, "args": m.group(4)}
            return self.tc_echo, 0
        m = re.match(r"^tc qdisc show dev (\S+)$", c)
        if m:
            return "\n".join(self._qdisc_lines(m.group(1), False)) + "\n", 0
        if c == "tc qdisc show":
            return "\n".join(l for i in self.intfs for l in self._qdisc_lines(i, True)) + "\n", 0
        m = re.match(r"^tc class show dev (\S+)$", c)
        if m:
            return "\n".join(self._class_lines(m.group(1))), 0
        return f"unhandled tc: {c}", 1

    # -- command dispatch ---------------------------------------------------
    def _run(self, c):
        self.cmds.append(c)
        for key, message in list(self.fail_once.items()):
            if key in c:
                del self.fail_once[key]
                return message, 1
        if c.startswith("tc "):
            return self._tc(c)
        if c.startswith("for d in") and "tc class show" in c:
            return "\n".join(f"{i} {l}" for i in self.intfs for l in self._class_lines(i)), 0
        if c.startswith("for d in") and "mtu=" in c:
            return "\n".join(
                f"{i} mtu={s['mtu']} ipv6_off={s['ipv6_off']} rx_csum={s['rx_csum']}" for i, s in self.intfs.items()
            ), 0
        if c == "ip -o link show":
            lines = ["1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT"]
            for n, (i, s) in enumerate(self.intfs.items(), start=2):
                oper = self._oper(i)
                flags = {"down": "BROADCAST,MULTICAST", "lowerlayerdown": "NO-CARRIER,BROADCAST,MULTICAST,UP"}.get(
                    oper, "BROADCAST,MULTICAST,UP,LOWER_UP"
                )
                lines.append(f"{n}: {i}@if{n + 10}: <{flags}> mtu {s['mtu']} qdisc htb state {oper.upper()} mode DEFAULT")
            return "\n".join(lines), 0
        m = re.match(r"^ip link set dev (\S+) (up|down)$", c)
        if m:
            self._set_link(m.group(1), m.group(2))
            return "", 0
        m = re.match(r"^ip link set dev (\S+) mtu (\d+)$", c)
        if m:
            self.intfs[m.group(1)]["mtu"] = int(m.group(2))
            return "", 0
        m = re.match(r"^cat /sys/class/net/(\S+)/(mtu|operstate)$", c)
        if m:
            return (str(self.intfs[m.group(1)]["mtu"]) if m.group(2) == "mtu" else self._oper(m.group(1))) + "\n", 0
        m = re.match(r"^cat /proc/sys/net/ipv6/conf/(\S+)/disable_ipv6$", c)
        if m:
            return f"{self.intfs[m.group(1)]['ipv6_off']}\n", 0
        m = re.match(r"^sysctl -w net\.ipv6\.conf\.(\S+)\.disable_ipv6=(\d)$", c)
        if m:
            self.intfs[m.group(1)]["ipv6_off"] = int(m.group(2))
            return "", 0
        m = re.match(r"^ethtool -K (\S+) rx (on|off)$", c)
        if m:
            self.intfs[m.group(1)]["rx_csum"] = m.group(2)
            return "", 0
        m = re.match(r"^ethtool -k (\S+)$", c)
        if m:
            return f"Features for {m.group(1)}:\nrx-checksumming: {self.intfs[m.group(1)]['rx_csum']}\n", 0
        if c == "ip route show":
            return "\n".join(sorted(self.routes.values())), 0
        m = re.match(r"^ip route show dev (\S+)$", c)
        if m:
            dev = f" dev {m.group(1)}"
            out = []
            for line in self.routes.values():
                if dev in f"{line} ":
                    flag = "" if self.intfs[m.group(1)]["up"] and self._peer_up(m.group(1)) else " linkdown"
                    out.append(line.replace(dev, "") + flag)
            return "\n".join(out), 0
        m = re.match(r"^ip route show (\S+)$", c)
        if m:
            return self.routes.get(m.group(1), ""), 0
        m = re.match(r"^ip route replace (.+)$", c)
        if m:
            key, line = _canon(m.group(1))
            dev = re.search(r" dev (\S+)", line)
            if dev and not self.intfs[dev.group(1)]["up"]:
                return "Error: Nexthop device is not up.", 2
            self.routes[key] = line
            return "", 0
        m = re.match(r"^ip route del (\S+)$", c)
        if m:
            return ("", 0) if self.routes.pop(m.group(1), None) else ("RTNETLINK answers: No such process", 2)
        m = re.match(r"^iptables -([ACD]) (.+)$", c)
        if m:
            op, rule = m.group(1), m.group(2)
            self.iptables_used = True
            if op == "A":
                self.rules.append(rule)
                return "", 0
            if op == "C":
                return ("", 0) if rule in self.rules else ("iptables: Bad rule", 1)
            if rule in self.rules:
                self.rules.remove(rule)
                return "", 0
            return "iptables: Bad rule (does a matching rule exist in that chain?).", 1
        if c.startswith("iptables-save"):
            if not self.iptables_used:
                return "", 0
            lines = ["# Generated by iptables-save", "*filter", ":INPUT ACCEPT [0:0]",
                     f":FORWARD ACCEPT [{next(self.counter)}:{next(self.counter) * 60}]", ":OUTPUT ACCEPT [0:0]"]
            lines += [f"-A {r}" for r in self.rules] + ["COMMIT"]
            return "\n".join(lines), 0
        m = re.match(r"^nohup (.+?) </dev/null >\S+ 2>&1 & echo \$!$", c)
        if m:
            return self.net.spawn(self, m.group(1)), 0
        m = re.match(r"^kill (-0 |-9 )?(\d+)", c)
        if m:
            alive = m.group(2) in self.net.procs
            if m.group(1) == "-0 ":
                return "", 0 if alive else 1
            self.net.procs.pop(m.group(2), None)
            return "", 0 if alive else 1
        m = re.match(r"^ss -ltnH 'sport = :(\d+)'$", c)
        if m:
            listening = any(
                node is self and f"-s -p {m.group(1)}" in cmd for node, cmd in self.net.procs.values()
            )
            return (f"LISTEN 0 5 *:{m.group(1)} *:*" if listening else ""), 0
        if c.startswith("tail "):
            return "iperf3: error - unable to connect to server", 0
        if c.startswith("ping "):
            loss = self.net.ping_loss
            received = 5 - round(5 * loss / 100)
            replies = "".join(f"64 bytes from 10.0.4.2: icmp_seq={seq} ttl=61 time=42.0 ms\n" for seq in range(1, received + 1))
            return f"{replies}5 packets transmitted, {received} received, {loss:g}% packet loss, time 800ms", 0
        return "", 0

    def pexec(self, argv):
        out, code = self._run(argv[-1])
        return out, "", code

    def cmd(self, c):
        return self._run(c)[0]

    def popen(self, argv, **_kw):
        _out, code = self._run(" ".join(argv))
        return FakeProc(code)


NODE_INTFS = {
    "hostA": ["hostA-eth0"],
    "hostB": ["hostB-eth0"],
    "r1": ["r1-eth0", "r1-eth1", "r1-eth2"],
    "r2": ["r2-eth0", "r2-eth1"],
    "r3": ["r3-eth0", "r3-eth1"],
    "r4": ["r4-eth0", "r4-eth1", "r4-eth2"],
}


class FakeNet:
    def __init__(self):
        self.procs = {}
        self.pids = itertools.count(4000)
        self.client_dies = False
        self.ping_loss = 0
        self._nodes = {n: FakeNode(self, n, ifs) for n, ifs in NODE_INTFS.items()}

    def get(self, name):
        return self._nodes[name]

    def spawn(self, node, command):
        pid = str(next(self.pids))
        if not (self.client_dies and "iperf3 -c" in command):
            self.procs[pid] = (node, command)
        return f"{pid}\n"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_FAILURES = []
FAST_FLAP = {"interval_seconds": 0.01, "cycles": 1}


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def fresh():
    inj._ACTIVE.clear()
    flap._ACTIVE_FLAPS.clear()
    return FakeNet()


def _params(fault_id):
    if inj._FAULTS[fault_id]["injection"]["type"] == "link_flap":
        return FAST_FLAP
    return None


def _set_noise(net):
    """What traffic.apply_noise(N1) leaves on r1-r2 and r2-r4: netem only, no htb."""
    for node, intf in (("r1", "r1-eth1"), ("r2", "r2-eth0"), ("r2", "r2-eth1"), ("r4", "r4-eth0")):
        net.get(node).intfs[intf]["tc"] = {
            "htb": None, "netem": {"handle": "10:", "parent": "root", "args": "limit 1000 delay 10ms"}, "other": None,
        }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def test_every_fault_injects_and_resets_cleanly():
    for fault_id in sorted(inj._FAULTS):
        net = fresh()
        before = inj.snapshot(net)
        try:
            cid = inj.inject(net, fault_id, params=_params(fault_id), seed=1, purpose="validation")
            during = inj.snapshot(net)
            flapping = inj._FAULTS[fault_id]["injection"]["type"] == "link_flap"
            changed = bool(inj.diff_snapshots(before, during))
            if fault_id != "H0" and not flapping:
                check(f"{fault_id}: inject changes the network", changed)
            result = inj.reset(cid)
            check(f"{fault_id}: reset reports no residual", result["restored"] and result["residual"] == [], result.get("residual"))
            check(f"{fault_id}: whole network matches the pre-inject snapshot",
                  inj.diff_snapshots(before, inj.snapshot(net)) == [])
            check(f"{fault_id}: nothing left active or running", not inj._ACTIVE and not net.procs)
        except Exception as exc:
            check(f"{fault_id}: inject/reset ran", False, f"{type(exc).__name__}: {exc}")


def test_every_compound_injects_and_resets_cleanly():
    for compound_id in sorted(inj._COMPOUNDS):
        net = fresh()
        before = inj.snapshot(net)
        try:
            comp = inj.inject_compound(net, compound_id, seed=3, purpose="validation")
            check(f"{compound_id}: every part tracked", len(comp["correlation_ids"]) == len(inj._ACTIVE) >= 2)
            results = inj.reset_compound(net, compound_id, comp["correlation_ids"])
            check(f"{compound_id}: no residual", all(r["restored"] for r in results))
            check(f"{compound_id}: network matches the pre-inject snapshot",
                  inj.diff_snapshots(before, inj.snapshot(net)) == [] and not inj._ACTIVE)
        except Exception as exc:
            check(f"{compound_id}: inject/reset ran", False, f"{type(exc).__name__}: {exc}")


def test_shaping_changes_only_its_own_parameter():
    net = fresh()
    cid = inj.inject(net, "F3", params={"loss_percent": 12})
    shape = inj.read_shape(net, "primary_r2_r4", "r2")
    fields = inj._netem_fields(shape["netem"]["args"])
    check("F3 keeps the 10 Mbit htb cap", shape["htb"] and shape["htb"]["rate"] == "10Mbit")
    check("F3 keeps the 5 ms delay and adds 12% loss", fields.get("delay") == ["5ms"] and fields.get("loss") == ["12%"], fields)
    inj.reset(cid)
    cid = inj.inject(net, "F5", params={"delay_ms": 150})
    shape = inj.read_shape(net, "primary_r2_r4", "r2")
    check("F5 sets delay and keeps the htb cap",
          shape["htb"] and inj._netem_fields(shape["netem"]["args"]).get("delay") == ["150ms"])
    inj.reset(cid)


def test_compound_on_one_interface_composes_and_resets_in_order():
    net = fresh()
    before = inj.snapshot(net)
    loss = inj.inject(net, "F3", params={"loss_percent": 10})
    rate = inj.inject(net, "F6", params={"rate_mbit": 2, "background_traffic": False})
    fields = inj._netem_fields(inj.read_shape(net, "primary_r2_r4", "r2")["netem"]["args"])
    check("C3-style F3 then F6: netem has both the loss and the rate",
          fields.get("loss") == ["10%"] and fields.get("rate") == ["2Mbit"], fields)
    try:
        inj.reset(loss)
        check("resetting the older F3 first is refused", False)
    except RuntimeError as exc:
        check("resetting the older F3 first is refused", rate in str(exc), str(exc))
    result = inj.reset_all(net)
    check("reset_all resets newest first with no errors", result["errors"] == [] and not inj._ACTIVE, result["errors"])
    check("network back to the pre-inject snapshot", inj.diff_snapshots(before, inj.snapshot(net)) == [])


def test_reset_keeps_noise_that_was_there_before():
    net = fresh()
    _set_noise(net)
    noisy = inj.snapshot(net)
    inj.inject(net, "F7", params={"rate_mbit": 2, "background_traffic": False})
    inj.inject(net, "F3", params={"loss_percent": 8})
    shape = inj.read_shape(net, "primary_r2_r4", "r2")
    check("F3 on a noisy link keeps the noise delay (no htb added)",
          shape["htb"] is None and inj._netem_fields(shape["netem"]["args"]).get("delay") == ["10ms"])
    result = inj.reset_all(net)
    check("reset_all leaves the N1 jitter on r1-r2 and r2-r4", inj.diff_snapshots(noisy, inj.snapshot(net)) == [],
          inj.diff_snapshots(noisy, inj.snapshot(net))[:3])
    check("reset_all reports no residual", result["residual"] == {})


def test_reset_all_baseline_restores_startup_state():
    net = fresh()
    pristine = inj.snapshot(net)
    _set_noise(net)
    net.get("r1").routes["10.0.4.0/24"] = "10.0.4.0/24 via 10.0.13.2 dev r1-eth2"
    inj.reset_all(net, baseline=True)
    check("baseline=True puts shapes and the primary route back", inj.diff_snapshots(pristine, inj.snapshot(net)) == [],
          inj.diff_snapshots(pristine, inj.snapshot(net))[:3])


def test_failed_inject_rolls_back():
    net = fresh()
    before = inj.snapshot(net)
    net.client_dies = True
    try:
        inj.inject(net, "F6", params={"rate_mbit": 2, "background_traffic": True})
        check("F6 with a dead background client raises", False)
    except RuntimeError as exc:
        check("F6 with a dead background client raises", "background iperf3 client exited" in str(exc), str(exc))
    check("the rate change was rolled back", inj.diff_snapshots(before, inj.snapshot(net)) == [])
    check("the server it started was stopped and nothing is tracked", not net.procs and not inj._ACTIVE)


def test_failed_reset_can_be_retried():
    net = fresh()
    before = inj.snapshot(net)
    cid = inj.inject(net, "F10")
    net.get("r4").fail_once["iptables -D"] = "iptables: resource busy"
    try:
        inj.reset(cid)
        check("a failing undo step raises", False)
    except RuntimeError:
        check("a failing undo step raises", True)
    check("the injection stays active for a retry", cid in inj._ACTIVE)
    result = inj.reset(cid)
    check("the retry succeeds with no residual", result["restored"] and not inj._ACTIVE)
    check("the rule is gone exactly once", inj.diff_snapshots(before, inj.snapshot(net)) == [])


def test_tc_output_is_not_mistaken_for_failure():
    net = fresh()
    net.get("r2").tc_echo = (
        "qdisc htb 5: root refcnt 3 r2q 10 default 0x1 direct_packets_stat 0 direct_qlen 1000\n"
        "qdisc netem 10: parent 5:1 limit 1000 delay 5ms"
    )
    try:
        for _ in range(10):
            cid = inj.inject(net, "F6", params={"rate_mbit": 2, "background_traffic": False})
            inj.reset(cid)
        check("F6 x10 succeeds when tc prints qdisc listings with exit 0", not inj._ACTIVE)
    except Exception as exc:
        check("F6 x10 succeeds when tc prints qdisc listings with exit 0", False, str(exc))

    net = fresh()
    before = inj.snapshot(net)
    net.get("r2").tc_fail = ("loss", "RTNETLINK answers: Invalid argument")
    try:
        inj.inject(net, "F3")
        check("a tc command that exits non-zero raises", False)
    except RuntimeError as exc:
        check("a tc command that exits non-zero raises with tc's message", "Invalid argument" in str(exc), str(exc))
    check("and the interface is put back", inj.diff_snapshots(before, inj.snapshot(net)) == [] and not inj._ACTIVE)

    net = fresh()
    net.get("r2").tc_ignore = "loss"
    try:
        inj.inject(net, "F3")
        check("a tc change that silently did not happen is caught by readback", False)
    except RuntimeError as exc:
        check("a tc change that silently did not happen is caught by readback", "reads back" in str(exc), str(exc))


def test_unsupported_qdisc_is_refused():
    net = fresh()
    net.get("r2").intfs["r2-eth1"]["tc"]["other"] = "qdisc tbf 1: root refcnt 2 rate 1Mbit burst 1600b lat 50ms"
    before = inj.snapshot(net)
    try:
        inj.inject(net, "F3")
        check("an interface with a qdisc we cannot rebuild is refused", False)
    except RuntimeError as exc:
        check("an interface with a qdisc we cannot rebuild is refused", "unsupported qdisc" in str(exc), str(exc))
    check("and left untouched", inj.diff_snapshots(before, inj.snapshot(net)) == [])


def test_residual_is_detected():
    net = fresh()
    check("two snapshots of an unchanged network match (counters and netem seed ignored)",
          inj.diff_snapshots(inj.snapshot(net), inj.snapshot(net)) == [])
    cid = inj.inject(net, "F10")
    net.get("r4").rules.append("FORWARD -p udp --dport 9 -j DROP")
    result = inj.reset(cid)
    check("a change reset did not undo is reported as residual",
          not result["restored"] and any("udp --dport 9" in line for line in result["residual"]), result["residual"])


def test_health_check_parses_loss():
    check("100% loss is not 0% loss", inj.ping_loss("3 packets transmitted, 0 received, 100% packet loss") == 100.0)
    check("missing summary is no reading", inj.ping_loss("connect: Network is unreachable") is None)
    net = fresh()
    net.get("hostA").cmds.clear()
    inj.health_check(net)
    check("the health ping waits for every reply (-w deadline)", any(" -w " in c for c in net.get("hostA").cmds))
    in_flight = (
        "".join(f"64 bytes from 10.0.4.2: icmp_seq={s} ttl=61 time=300 ms\n" for s in range(1, 6))
        + "6 packets transmitted, 5 received, 16.6667% packet loss, time 1072ms\n"
    )
    check("a probe still in flight when ping exits is not loss", inj.probe_loss(in_flight, 5) == 0.0, inj.probe_loss(in_flight, 5))
    lost_one = (
        "".join(f"64 bytes from 10.0.4.2: icmp_seq={s} ttl=61 time=40 ms\n" for s in (1, 2, 4, 5, 6))
        + "6 packets transmitted, 5 received, 16.6667% packet loss, time 1072ms\n"
    )
    check("a lost probe among the first five is loss", inj.probe_loss(lost_one, 5) == 20.0, inj.probe_loss(lost_one, 5))
    check("a duplicate reply does not hide a loss", inj.probe_loss(lost_one + "64 bytes from 10.0.4.2: icmp_seq=1 ttl=61 time=40 ms (DUP!)\n", 5) == 20.0)
    check("no summary is still no reading", inj.probe_loss("connect: Network is unreachable", 5) is None)

    net = fresh()
    cid = inj.inject(net, "U1", purpose="validation")
    for name in inj.NODES:
        net.get(name).cmds.clear()
    inj.reset(cid)
    r2 = net.get("r2").cmds
    check("U1 reset flushes learned path MTUs on every node, after the MTU is back",
          all("ip route flush cache" in net.get(n).cmds for n in inj.NODES)
          and r2.index("ip route flush cache") > max(i for i, c in enumerate(r2) if " mtu " in c), str(r2))
    empty = inj._normalize("firewall", "")
    loaded = inj._normalize("firewall", "# Generated\n*filter\n:INPUT ACCEPT [0:0]\n:FORWARD ACCEPT [9:540]\n:OUTPUT ACCEPT [0:0]\nCOMMIT")
    check("an empty filter table equals no iptables output", empty == loaded == [], loaded)
    net = fresh()
    net.ping_loss = 100
    check("reset_all is not healthy at 100% loss", inj.reset_all(net)["healthy"] is False)
    net.ping_loss = 0
    check("reset_all is healthy at 0% loss", inj.reset_all(net)["healthy"] is True)


def test_held_out_kept_out_of_tuning():
    expected = {"U1", "U2", "U3", "U4", "U5", "C4", "C5"}
    check("held_out_ids() is U1-U5, C4 and C5", inj.held_out_ids() == expected, inj.held_out_ids())
    net = fresh()
    for purpose in ("calibration", "training", "tuning"):
        try:
            inj.inject(net, "U1", purpose=purpose)
            check(f"U1 refused for {purpose}", False)
        except ValueError:
            check(f"U1 refused for {purpose}", not inj._ACTIVE)
    try:
        inj.inject_compound(net, "C4", purpose="tuning")
        check("C4 refused for tuning", False)
    except ValueError:
        check("C4 refused for tuning", not inj._ACTIVE)
    cid = inj.inject(net, "F3", purpose="tuning")
    check("development faults are allowed for tuning", cid in inj._ACTIVE)
    inj.reset(cid)
    try:
        inj.inject(net, "F3", purpose="bogus")
        check("an unknown purpose is rejected", False)
    except ValueError:
        check("an unknown purpose is rejected", True)

    from gate import selector, grader

    check("the gate's fix table has no held-out labels", not (set(selector._load_fixes()) & expected))
    with mock.patch.object(grader, "_MODEL", Path("unused.json")):
        try:
            grader.train_grader([{"test_id": "T1_e2e_ping", "label": 1, "fault_id": "U2"}])
            check("the grader refuses held-out training samples", False)
        except ValueError:
            check("the grader refuses held-out training samples", not Path("unused.json").exists())


def test_reproducible_and_identifiable():
    net = fresh()
    a = inj.inject(net, "F3", seed=7)
    b = inj.inject(net, "F11", seed=7)
    c = inj.inject(net, "F5", seed=7)
    d = inj.inject(net, "F5", seed=8)
    details = {cid: inj._ACTIVE[cid]["details"] for cid in (a, b, c, d)}
    check("the same seed gives the same sampled severity",
          inj._resolve_params(inj._FAULTS["F3"]["params"], 7) == details[a]["params"])
    check("every injection has its own id carrying the fault id", len(set(details)) == 4 and a.startswith("F3-"))
    check("details record seed, purpose and id",
          all(v["seed"] in (7, 8) and v["purpose"] == "evaluation" and v["correlation_id"] == k for k, v in details.items()))
    inj.reset_all(net)


def test_link_down_restores_routes():
    net = fresh()
    r2 = net.get("r2")
    cid = inj.inject(net, "F1")
    check("F1 removes r2's routes through r2-eth1", "10.0.4.0/24" not in r2.routes)
    saved = inj._ACTIVE[cid]["details"]["saved_routes"]
    check("saved routes are replayable (no linkdown flag)", saved and not any("linkdown" in s for s in saved))
    inj.reset(cid)
    check("reset puts the static route back", r2.routes.get("10.0.4.0/24") == "10.0.4.0/24 via 10.0.24.2 dev r2-eth1")


def test_flap_restores_routes_after_each_up():
    net = fresh()
    cid = inj.inject(net, "F8", params={"interval_seconds": 0.01, "cycles": 2})
    entry = flap._ACTIVE_FLAPS[inj._ACTIVE[cid]["details"]["flap_id"]]
    entry["thread"].join(timeout=5)
    check("after the flap finishes, r2 still has its route to 10.0.4.0/24", "10.0.4.0/24" in net.get("r2").routes)
    check("the flap thread reported no errors", entry["errors"] == [], entry["errors"])
    inj.reset(cid)


def test_background_traffic_stops_only_its_own_processes():
    net = fresh()
    noise_pid = net.spawn(net.get("hostA"), "iperf3 -c 10.0.4.2 -p 5202 -u -b 2M -t 3600").strip()
    cid = inj.inject(net, "F6", params={"rate_mbit": 2, "background_traffic": True})
    background = inj._ACTIVE[cid]["details"]["background"]
    check("F6 started its own server and client", {background["server_pid"], background["client_pid"]} <= set(net.procs))
    inj.reset(cid)
    check("F6 reset stopped its processes and left the noise client running", set(net.procs) == {noise_pid})


def test_malformed_fault_definitions_rejected():
    cases = [
        ({"injection": {"type": "netem_loss"}, "target": {}, "params": {}}, "netem_loss missing link+direction"),
        ({"injection": {"type": "link_down"}, "target": {}, "params": {}}, "link_down missing link"),
        ({"injection": {"type": "acl_drop"}, "target": {"node": "r4"}, "params": {}}, "acl_drop missing protocol+port"),
        ({"injection": {"type": "blackhole_route"}, "target": {"node": "r2"}, "params": {}}, "blackhole_route missing destination"),
    ]
    for fault, label in cases:
        fault["id"] = "_TEST"
        with mock.patch("yaml.safe_load", return_value={"faults": [fault]}):
            try:
                inj._load_faults()
                check(f"rejects malformed fault ({label})", False)
            except ValueError as e:
                check(f"rejects malformed fault ({label}) with id and type", "_TEST" in str(e) and fault["injection"]["type"] in str(e))
    check("config/faults.yaml loads all 18 faults", len(inj._load_faults()) == 18)
    try:
        inj.inject(fresh(), "NOPE")
        check("unknown fault id raises ValueError", False)
    except ValueError:
        check("unknown fault id raises ValueError", not inj._ACTIVE)
    check("reset of an unknown id is a no-op", inj.reset("made-up")["already_reset"] is True)


def main():
    tests = [
        test_every_fault_injects_and_resets_cleanly,
        test_every_compound_injects_and_resets_cleanly,
        test_shaping_changes_only_its_own_parameter,
        test_compound_on_one_interface_composes_and_resets_in_order,
        test_reset_keeps_noise_that_was_there_before,
        test_reset_all_baseline_restores_startup_state,
        test_failed_inject_rolls_back,
        test_failed_reset_can_be_retried,
        test_tc_output_is_not_mistaken_for_failure,
        test_unsupported_qdisc_is_refused,
        test_residual_is_detected,
        test_health_check_parses_loss,
        test_held_out_kept_out_of_tuning,
        test_reproducible_and_identifiable,
        test_link_down_restores_routes,
        test_flap_restores_routes_after_each_up,
        test_background_traffic_stops_only_its_own_processes,
        test_malformed_fault_definitions_rejected,
    ]
    started = time.monotonic()
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print(f"\n{'=' * 60}")
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s):")
        for name in _FAILURES:
            print(" -", name)
        sys.exit(1)
    print(f"ALL CHECKS PASSED in {time.monotonic() - started:.1f}s (simulated network; live proof is bench/validate_faults.py on the VM)")


if __name__ == "__main__":
    main()
