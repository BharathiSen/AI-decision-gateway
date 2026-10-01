"""
Offline regression tests for bench/inject.py.

These exercise inject()/reset()/reset_all()'s Python control flow
against a mock Mininet network (FakeNet/FakeNode/FakeIntf below) --
they verify what the dispatcher *decides to do* (which commands it
issues, what it tracks in _ACTIVE, what it returns), not whether those
commands actually work against a real kernel. Nothing here touches
tc/ip/iptables for real; that still requires the Mininet VM.

Run: python3 bench/test_inject.py
"""

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import inject as inj


# ---------------------------------------------------------------------------
# Mock Mininet network
# ---------------------------------------------------------------------------

class FakeIntf:
    """fail_tc, when set, makes config() mimic a REAL TCIntf.config()
    that ran a tc command which failed -- i.e. it does NOT raise (that's
    the whole point: Mininet's own source confirms it never raises on a
    failed tc command), it just returns a result dict whose tcoutputs
    contains the error text, exactly like the real thing does.
    """
    def __init__(self, name):
        self.name = name
        self.config_calls = []
        self.fail_tc = None

    def config(self, **kw):
        self.config_calls.append(kw)
        if self.fail_tc:
            return {"tcoutputs": [self.fail_tc], "parent": " parent 5:1 "}
        return {"tcoutputs": [""], "parent": " parent 5:1 "}


class FakeNode:
    def __init__(self, name, intf_names, tc_class_out="", tc_qdisc_out="", route_out=""):
        self.name = name
        self.cmds = []
        self._intfs = [FakeIntf(n) for n in intf_names]
        self.tc_class_out = tc_class_out
        self.tc_qdisc_out = tc_qdisc_out
        self.route_out = route_out

    def cmd(self, c):
        self.cmds.append(c)
        if c.startswith("tc class show"):
            return self.tc_class_out
        if c.startswith("tc qdisc show"):
            return self.tc_qdisc_out
        if c.startswith("ip route show dev"):
            return self.route_out
        if "ip route show" in c:
            return "10.0.4.0/24 via 10.0.24.2 dev r2-eth1"
        if "operstate" in c:
            return "up"
        if "ip -4 -o addr show" in c:
            return "3: fake    inet 10.0.0.9/24 brd 10.0.0.255 scope global fake"
        if c.startswith("cat") and "mtu" in c:
            return "1500"
        return ""

    def intfList(self):
        return self._intfs

    def IP(self):
        return "10.0.4.2"


NODE_INTFS = {
    "hostA": ["hostA-eth0"], "hostB": ["hostB-eth0"],
    "r1": ["r1-eth0", "r1-eth1", "r1-eth2"],
    "r2": ["r2-eth0", "r2-eth1"],
    "r3": ["r3-eth0", "r3-eth1"],
    "r4": ["r4-eth0", "r4-eth1", "r4-eth2"],
}


class FakeNet:
    def __init__(self, node_kwargs=None):
        node_kwargs = node_kwargs or {}
        self._nodes = {
            n: FakeNode(n, ifs, **node_kwargs.get(n, {}))
            for n, ifs in NODE_INTFS.items()
        }

    def get(self, name):
        return self._nodes[name]


# ---------------------------------------------------------------------------
# Test harness (no pytest dependency -- matches config/validate.py's style)
# ---------------------------------------------------------------------------

_FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def reset_module_state():
    inj._ACTIVE.clear()


# ---------------------------------------------------------------------------
# Pre-existing behaviour (must still pass after the four fixes)
# ---------------------------------------------------------------------------

def test_dispatch_coverage():
    reset_module_state()
    net = FakeNet()
    ok = True
    for fid in sorted(inj._FAULTS):
        if fid in ("F8", "H0"):
            continue
        try:
            before = set(inj._ACTIVE)
            cid = inj.inject(net, fid, seed=1)
            new = set(inj._ACTIVE) - before
            if new != {cid}:
                ok = False
        except Exception as e:
            print(f"    {fid} raised unexpectedly: {e}")
            ok = False
    check("all 13 non-trivial fault types dispatch and register exactly one _ACTIVE entry", ok)
    result = inj.reset_all(net)
    check("reset_all() clears everything dispatch_coverage created", len(inj._ACTIVE) == 0)
    check("reset_all() reports no errors on a clean run", result["errors"] == [])


def test_f8_and_unknown_id_fail_safely():
    reset_module_state()
    net = FakeNet()
    try:
        inj.inject(net, "F8")
        check("F8 raises NotImplementedError", False)
    except NotImplementedError:
        check("F8 raises NotImplementedError", True)

    try:
        inj.inject(net, "NOPE")
        check("unknown fault id raises ValueError", False)
    except ValueError:
        check("unknown fault id raises ValueError", True)

    check("_ACTIVE untouched by either failure", len(inj._ACTIVE) == 0)


# ---------------------------------------------------------------------------
# Fix 1: tbf_congestion partial-mutation safety
# ---------------------------------------------------------------------------

def test_fix1_tbf_partial_failure_is_recoverable():
    reset_module_state()
    net = FakeNet()
    r2_eth1 = [i for i in net.get("r2").intfList() if i.name == "r2-eth1"][0]

    with mock.patch.object(inj, "_start_background_traffic", side_effect=RuntimeError("simulated")):
        raised = False
        try:
            inj.inject(net, "F6", seed=1)
        except RuntimeError:
            raised = True

    check("inject() still raises when background traffic fails to start", raised)
    check(
        "the bandwidth mutation was actually applied before the failure",
        any("bw" in c for c in r2_eth1.config_calls),
    )
    check(
        "FIX 1: the mutation is tracked in _ACTIVE despite the failure "
        "(previously this was the confirmed defect: 0 entries)",
        len(inj._ACTIVE) == 1,
    )

    # And it must actually be cleanable.
    result = inj.reset_all(net)
    check("the recovered mutation cleans up via reset_all() with no errors", result["errors"] == [] and len(inj._ACTIVE) == 0)


def test_fix1_successful_tbf_unaffected():
    """The fix must not change behaviour for the success path."""
    reset_module_state()
    net = FakeNet()
    cid = inj.inject(net, "F6", seed=1, params={"rate_mbit": 2, "background_traffic": False})
    check("successful tbf_congestion still returns a usable correlation_id", cid in inj._ACTIVE)
    result = inj.reset(cid)
    check("successful tbf_congestion still cleans up normally", result["already_reset"] is False)


# ---------------------------------------------------------------------------
# Fix 2: reset() routing-safety + repeated-reset safety
# ---------------------------------------------------------------------------

def test_fix2_individual_reset_restores_scoped_routes():
    reset_module_state()
    net = FakeNet(node_kwargs={
        "r2": {"route_out": "10.0.4.0/24 via 10.0.24.2\n10.0.13.0/24 via 10.0.12.1"},
    })
    cid = inj.inject(net, "F1", seed=1)  # link_down on primary_r2_r4, first_node="r2"

    r2_cmds_before = list(net.get("r2").cmds)
    result = inj.reset(cid)
    new_cmds = net.get("r2").cmds[len(r2_cmds_before):]

    check("reset() brought the interface back up", any("ip link set dev r2-eth1 up" in c for c in new_cmds))
    check(
        "FIX 2: reset() replayed the routes captured through that interface "
        "(previously: zero routing commands from reset() alone)",
        any(c.startswith("ip route replace") for c in new_cmds),
    )
    check("reset() did not touch unrelated nodes (r3/r4/hostA/hostB untouched)", all(
        len(net.get(n).cmds) == 0 for n in ("r3", "r4", "hostA", "hostB")
    ))
    check("reset() returns a normal (not already_reset) result", result["already_reset"] is False)


def test_fix2_repeated_reset_is_safe():
    reset_module_state()
    net = FakeNet()
    cid = inj.inject(net, "F1", seed=1)
    r1 = inj.reset(cid)
    r2 = inj.reset(cid)  # repeat call on an already-cleaned-up id
    check("first reset() succeeds normally", r1["already_reset"] is False)
    check("FIX 2: repeated reset() is a safe no-op, not an exception", r2["already_reset"] is True)

    r3 = inj.reset("totally-made-up-id")
    check("reset() on a never-issued id is also a safe no-op", r3["already_reset"] is True)


def test_fix2_reset_all_survives_a_bad_cleanup():
    reset_module_state()
    net = FakeNet()
    cid_good1 = inj.inject(net, "F1", seed=1)
    cid_bad = inj.inject(net, "F2", seed=1)
    cid_good2 = inj.inject(net, "F12", seed=1)

    # Force the middle entry's cleanup to explode.
    inj._ACTIVE[cid_bad]["cleanup"] = mock.Mock(side_effect=RuntimeError("boom"))

    result = inj.reset_all(net)

    check("FIX 2: reset_all() still processes entries after a failing cleanup", len(result["cleared"]) == 2)
    check("FIX 2: reset_all() reports the failure instead of hiding it", len(result["errors"]) == 1 and result["errors"][0]["correlation_id"] == cid_bad)
    check("FIX 2: reset_all() still runs the routing restore + health check despite the failure", "healthy" in result and "ping" in result)
    check("_ACTIVE is empty afterward (the failing entry was still popped, not left stuck)", len(inj._ACTIVE) == 0)


# ---------------------------------------------------------------------------
# Fix 3: required target key validation
# ---------------------------------------------------------------------------

def test_fix3_malformed_fault_definitions_rejected():
    cases = [
        ({"injection": {"type": "netem_loss"}, "target": {}, "params": {}}, "netem_loss missing link+direction"),
        ({"injection": {"type": "link_down"}, "target": {}, "params": {}}, "link_down missing link"),
        ({"injection": {"type": "acl_drop"}, "target": {"node": "r4"}, "params": {}}, "acl_drop missing protocol+port"),
        ({"injection": {"type": "blackhole_route"}, "target": {"node": "r2"}, "params": {}}, "blackhole_route missing destination"),
    ]
    for fault, label in cases:
        fault["id"] = "_TEST"
        doc = {"faults": [fault]}
        with mock.patch("yaml.safe_load", return_value=doc):
            try:
                inj._load_faults()
                check(f"FIX 3: rejects malformed fault ({label})", False)
            except ValueError as e:
                has_id = "_TEST" in str(e)
                has_type = fault["injection"]["type"] in str(e)
                check(f"FIX 3: rejects malformed fault ({label}) with an actionable message", has_id and has_type, str(e))


def test_fix3_valid_faults_still_load():
    # The real, shipped faults.yaml -- must be unaffected by the new check.
    by_id = inj._load_faults()
    check("FIX 3: the real config/faults.yaml still loads all 15 faults", len(by_id) == 15)
    check("FIX 3: H0 (target: null, injection: none) still loads", "H0" in by_id)


# ---------------------------------------------------------------------------
# Fix 4: capture actual pre-injection params instead of assuming baseline
# ---------------------------------------------------------------------------

def test_fix4_non_baseline_settings_are_captured_and_restored():
    reset_module_state()
    # r2-eth1 is NOT at the topology baseline (bw=10, delay=5ms, loss=0) --
    # simulate it already having a different configuration, e.g. left
    # over from an earlier, different fault.
    net = FakeNet(node_kwargs={
        "r2": {
            "tc_class_out": "class htb 5:1 root prio 0 rate 3Mbit ceil 3Mbit burst 15Kb",
            "tc_qdisc_out": "qdisc netem 10: parent 5:1 limit 1000 delay 40.0ms loss 2%",
        },
    })

    params, guessed = inj._capture_link_params(net, "primary_r2_r4", "r2")
    check("FIX 4: captures actual (non-baseline) bandwidth", params["bw"] == 3.0)
    check("FIX 4: captures actual (non-baseline) delay", params["delay"] == "40.0ms")
    check("FIX 4: captures actual (non-baseline) loss", params["loss"] == 2.0)
    check("FIX 4: reports nothing as 'guessed' when everything parsed", guessed == [])

    r2_eth1 = [i for i in net.get("r2").intfList() if i.name == "r2-eth1"][0]
    cid = inj.inject(net, "F3", seed=1)  # netem_loss on primary_r2_r4, r2_to_r4
    inj.reset(cid)

    restored = r2_eth1.config_calls[-1]
    check(
        "FIX 4: cleanup restores the ACTUAL prior bw/delay/loss, not ctl.BASELINE_LINK_PARAMS",
        restored == {"bw": 3.0, "delay": "40.0ms", "loss": 2.0},
        f"got {restored}",
    )


def test_fix4_unparseable_state_falls_back_and_says_so():
    reset_module_state()
    # No tc output at all (e.g. a genuinely bare/default interface) --
    # nothing to parse, so this must fall back to baseline AND say so.
    net = FakeNet()
    params, guessed = inj._capture_link_params(net, "primary_r2_r4", "r2")
    check("FIX 4: falls back to baseline bw when nothing can be parsed", params["bw"] == inj.ctl.BASELINE_LINK_PARAMS["bw"])
    check(
        "FIX 4: explicitly reports which fields were guessed, rather than "
        "silently pretending they were captured",
        set(guessed) == {"bw", "delay", "loss"},
    )


# ---------------------------------------------------------------------------
# F6 investigation: reset() was leaving the qdisc empty instead of
# restoring 10Mbit htb + 5ms netem. Root cause: cleanup used
# _capture_link_params()'s regex-parsed result (never verified against
# real tc output) instead of the topology's known baseline; a bad
# parsed value fed a malformed tc command, and TCIntf.config() -- per
# its actual source -- never raises on a failed tc command, just logs
# it, so the preceding `qdisc del` succeeded while the rebuild silently
# failed, leaving nothing behind.
# ---------------------------------------------------------------------------

def test_f6_reset_restores_baseline_qdisc_not_empty():
    """The actual regression test requested: inject F6, reset it, and
    confirm the interface is told to go back to exactly 10Mbit/5ms/0%
    -- not left empty, and not whatever a tc-output parse happened to
    produce.
    """
    reset_module_state()
    # Deliberately give the interface tc output that would mislead the
    # old (now-removed) capture-based cleanup if it were still in use --
    # proving the fix no longer depends on this at all for restoration.
    net = FakeNet(node_kwargs={
        "r2": {
            "tc_class_out": "class htb 5:1 root prio 0 rate 999Mbit ceil 999Mbit burst 15Kb",
            "tc_qdisc_out": "qdisc netem 10: parent 5:1 limit 1000 delay 9999.0ms loss 87%",
        },
    })
    r2_eth1 = [i for i in net.get("r2").intfList() if i.name == "r2-eth1"][0]

    cid = inj.inject(net, "F6", seed=1, params={"rate_mbit": 2, "background_traffic": False})
    check(
        "F6 injection applies the throttled bandwidth",
        r2_eth1.config_calls[-1] == {"bw": 2},
    )

    result = inj.reset(cid)
    restored = r2_eth1.config_calls[-1]

    check(
        "F6 FIX: reset() restores the topology's real baseline (10Mbit/5ms/0%), "
        "NOT the misleading tc output (999Mbit/9999ms/87%) and NOT empty",
        restored == dict(inj.ctl.BASELINE_LINK_PARAMS),
        f"got {restored}",
    )
    check(
        "connectivity proxy: the restored config matches a healthy link "
        "(bw/delay/loss all present and non-degenerate) -- NOTE: this "
        "confirms what configure_one_end() was told to apply, not that "
        "a real kernel applied it; that still needs the Mininet VM",
        restored.get("bw", 0) > 0 and restored.get("delay") and restored.get("loss") == 0,
    )
    check("reset() completed normally (not already_reset)", result["already_reset"] is False)


def test_f6_cleanup_no_longer_calls_capture_based_restore():
    """Confirm the fix structurally, not just by outcome: F6's cleanup
    must come from reset_one_end() (ctl.BASELINE_LINK_PARAMS), not from
    configure_one_end(**original) built off parsed tc output.
    """
    reset_module_state()
    net = FakeNet()
    with mock.patch.object(inj, "reset_one_end", wraps=inj.reset_one_end) as spy:
        cid = inj.inject(net, "F6", seed=1, params={"rate_mbit": 2, "background_traffic": False})
        inj.reset(cid)
    check("F6 FIX: cleanup goes through reset_one_end(), the proven baseline path", spy.called)


def test_configure_one_end_raises_on_silent_tc_failure():
    """Change 1: a tc command that fails (per real TCIntf.config()'s own
    behavior -- it logs, never raises) must now surface as an exception
    from configure_one_end(), instead of looking like success.
    """
    reset_module_state()
    net = FakeNet()
    r2_eth1 = [i for i in net.get("r2").intfList() if i.name == "r2-eth1"][0]
    r2_eth1.fail_tc = "RTNETLINK answers: Invalid argument"

    try:
        inj.configure_one_end(net, "primary_r2_r4", "r2", bw=5, delay="5ms", loss=0)
        check("CHANGE 1: configure_one_end() raises when the underlying tc command failed", False)
    except RuntimeError as e:
        check(
            "CHANGE 1: configure_one_end() raises when the underlying tc command failed",
            True,
        )
        check("the raised error includes the real tc failure text", "Invalid argument" in str(e))


def test_configure_one_end_still_succeeds_normally():
    """The new check must not false-positive on a normal, successful call."""
    reset_module_state()
    net = FakeNet()
    try:
        result = inj.configure_one_end(net, "primary_r2_r4", "r2", bw=10, delay="5ms", loss=0)
        check("CHANGE 1: a genuinely successful configure_one_end() still returns normally", result["bw"] == 10)
    except Exception as e:
        check("CHANGE 1: a genuinely successful configure_one_end() still returns normally", False, str(e))


# ---------------------------------------------------------------------------

def main():
    tests = [
        test_dispatch_coverage,
        test_f8_and_unknown_id_fail_safely,
        test_fix1_tbf_partial_failure_is_recoverable,
        test_fix1_successful_tbf_unaffected,
        test_fix2_individual_reset_restores_scoped_routes,
        test_fix2_repeated_reset_is_safe,
        test_fix2_reset_all_survives_a_bad_cleanup,
        test_fix3_malformed_fault_definitions_rejected,
        test_fix3_valid_faults_still_load,
        test_fix4_non_baseline_settings_are_captured_and_restored,
        test_fix4_unparseable_state_falls_back_and_says_so,
        test_f6_reset_restores_baseline_qdisc_not_empty,
        test_f6_cleanup_no_longer_calls_capture_based_restore,
        test_configure_one_end_raises_on_silent_tc_failure,
        test_configure_one_end_still_succeeds_normally,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()

    print(f"\n{'=' * 60}")
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s) did not pass:")
        for name in _FAILURES:
            print(" -", name)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED (mock-network only -- see report for what still needs a live Mininet VM)")


if __name__ == "__main__":
    main()
