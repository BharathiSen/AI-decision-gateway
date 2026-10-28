"""Offline: python -m baselines
Live (VM, root): sudo python3 -m baselines --live
"""

import os
import sys

from baselines import _self_check, run_method
from baselines.b0_ungated import run as b0_run


def _live():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("baselines --live must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    from bench import inject as inj
    from bench import tests as oam
    from bench.mininet_util import mn_cleanup
    from network import controller as ctl
    from network.topology import create_network
    from telemetry.collector import collect_current_state

    mn_cleanup()
    state = create_network()
    net, links = state["net"], state["links"]
    proposal = {
        "action": "REROUTE",
        "target": "HostA_to_HostB",
        "reason": "stage8 live",
        "proposed_path": ["r1", "r3", "r4"],
        "causes": [{"cause_id": "F12", "predicted_tests": []}],
    }
    failures = []
    try:
        inj.inject(net, "F12")
        live = b0_run(net, proposal)
        if live.get("harm") is not True:
            failures.append(f"B0 harm {live.get('harm')}")
        if live.get("rolled_back") is not True:
            failures.append(f"B0 rollback {live.get('rolled_back')}")
        if ctl.get_active_path(net) != "primary":
            failures.append(f"path after B0 {ctl.get_active_path(net)}")
        snap = collect_current_state(net, links)
        battery = oam.run_all(net)
        results = battery.get("results") or []
        for method in ("B0", "B1", "B2", "B3", "B4", "B5"):
            out = run_method(method, proposal, snap, results)
            if out.get("decision") not in {"allow", "block", "escalate"}:
                failures.append(f"{method} decision {out}")
        b2 = run_method("B2", proposal, snap, results)
        if b2.get("action") == "REROUTE":
            failures.append(f"B2 rerouted onto down backup {b2}")
        b3 = run_method("B3", proposal, snap, results)
        if not b3.get("budget", {}).get("within_budget"):
            failures.append(f"B3 over budget {b3.get('budget')}")
        if "T10_speed" in (b3.get("tests_run") or []):
            failures.append(f"B3 used T10 {b3.get('tests_run')}")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        live = None
    finally:
        inj.reset_all(net, baseline=True)
        net.stop()
        mn_cleanup()

    if failures:
        print("baselines live: FAIL " + "; ".join(failures))
        if live is not None:
            print(live)
        return 1
    print("baselines live: PASS")
    return 0


if __name__ == "__main__":
    _self_check()
    print("baselines: PASS")
    if "--live" in sys.argv:
        sys.exit(_live())
