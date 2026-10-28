"""
Live Mininet check of the fault emulator (checklist Stage 7).

For every fault and compound, at each noise level, run --iterations
inject/reset cycles. A cycle passes when:

- the inject changed the network (not required for H0, or for flaps,
  which change it on their own thread),
- every reset reports no residual difference on the nodes it touched,
- the whole network matches the snapshot taken just before the inject,
- hostA still reaches hostB within the noise level's T1 loss limit, on the
  first ping or on a second one SETTLE_SECONDS later (counted as "settled").
  If the second ping also fails, the cycle records qdiscs, neighbours and
  routes for diagnosis.

Even cycles reset with reset()/reset_compound(), odd cycles with
reset_all(). Held-out faults run here with purpose="validation"; they
are still refused for calibration, training and tuning.

Run on the Mininet VM as root:

    sudo python3 bench/validate_faults.py
    sudo python3 bench/validate_faults.py --fault F6 --noise N2 --iterations 3

Writes results/fault_validation.json.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from bench import inject as inj

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "fault_validation.json"
NOISE_IDS = ("N0", "N1", "N2")
SETTLE_SECONDS = 3.0
_DIAGNOSTICS = {
    "hostA": ["ip route get 10.0.4.2", "ip neigh show"],
    "r1": ["ip route get 10.0.4.2", "ip neigh show", "tc qdisc show"],
    "r2": ["ip route get 10.0.4.2", "ip neigh show", "tc qdisc show"],
    "r4": ["ip route get 10.0.1.2", "ip neigh show", "tc qdisc show"],
}


def _natural(item_id):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", item_id)]


def scenario_ids():
    faults = sorted((f for f in inj._FAULTS if f != "H0"), key=_natural)
    return ["H0", *faults, *sorted(inj._COMPOUNDS, key=_natural)]


def loss_limit(noise_id):
    with (ROOT / "config" / "thresholds.yaml").open() as f:
        doc = yaml.safe_load(f)
    return float(doc["by_noise_level"][noise_id]["T1_e2e_ping"]["max_loss_percent"])


def _inject(net, scenario, seed):
    if scenario == "H0":
        return []
    if scenario in inj._COMPOUNDS:
        return inj.inject_compound(net, scenario, seed=seed, purpose="validation")["correlation_ids"]
    return [inj.inject(net, scenario, seed=seed, purpose="validation")]


def _reset(net, scenario, cids, use_reset_all):
    problems = []
    if use_reset_all:
        result = inj.reset_all(net)
        problems += [f"reset_all: {e['error']}" for e in result["errors"]]
        residual = [line for lines in result["residual"].values() for line in lines]
    else:
        if scenario in inj._COMPOUNDS:
            results = inj.reset_compound(net, scenario, cids)
        else:
            results = [inj.reset(cid) for cid in cids]
        problems += [f"{r['reset']}: already reset" for r in results if r.get("already_reset")]
        residual = [line for r in results for line in r.get("residual", [])]
    problems += [f"residual on touched node: {line}" for line in residual]
    return problems


def _diagnostics(net):
    lines = []
    for node, commands in _DIAGNOSTICS.items():
        for command in commands:
            out, _ = inj._sh(net.get(node), command)
            lines += [f"{node} $ {command}"] + [f"    {line}" for line in out.strip().splitlines()]
    return lines


def check_health(net, max_loss):
    """(problems, settled). One retry after SETTLE_SECONDS; diagnostics if that fails too."""
    loss, ping = inj.health_check(net)
    if loss is not None and loss <= max_loss:
        return [], False
    time.sleep(SETTLE_SECONDS)
    again, ping_again = inj.health_check(net)
    if again is not None and again <= max_loss:
        return [], True
    return [
        f"hostA to hostB loss {loss}%, then {again}% after {SETTLE_SECONDS:g}s, over {max_loss}%: {ping_again[-160:]}",
        *_diagnostics(net),
    ], False


def run_cycle(net, scenario, seed, use_reset_all, max_loss, settled=None):
    problems = []
    before = inj.snapshot(net)
    cids = _inject(net, scenario, seed)
    entries = [inj._ACTIVE[cid]["details"] for cid in cids]
    flapping = any(d["injection"] == "link_flap" for d in entries)
    if cids and not flapping and not inj.diff_snapshots(before, inj.snapshot(net)):
        problems.append("inject changed nothing")
    for details in entries:
        background = details.get("background")
        if background and not inj._alive(net.get(background["from_node"]), background["client_pid"]):
            problems.append(f"{details['correlation_id']}: background client not running")
    problems += _reset(net, scenario, cids, use_reset_all)
    if inj._ACTIVE:
        problems.append(f"still active after reset: {sorted(inj._ACTIVE)}")
    problems += [f"differs from before inject: {line}" for line in inj.diff_snapshots(before, inj.snapshot(net))]
    health, was_settled = check_health(net, max_loss)
    problems += health
    if was_settled and settled is not None:
        settled.append(seed)
    return problems


def validate(net, scenario, iterations, max_loss, seed_base=0, settled=None):
    failures = []
    for i in range(iterations):
        try:
            problems = run_cycle(net, scenario, seed_base + i, use_reset_all=bool(i % 2), max_loss=max_loss, settled=settled)
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
            try:
                inj.reset_all(net)
            except Exception as reset_exc:
                problems.append(f"recovery reset_all: {reset_exc}")
        if problems:
            failures.append({"cycle": i, "problems": problems})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="append", help="Fault or compound id (repeatable); default all")
    parser.add_argument("--noise", action="append", choices=[*NOISE_IDS, "all"], help="Default all")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("validate_faults.py must run as root (sudo) on the Mininet VM.", file=sys.stderr)
        sys.exit(2)

    from network.topology import create_network
    from bench.mininet_util import mn_cleanup
    from bench.traffic import apply_noise, clear_noise

    known = scenario_ids()
    targets = args.fault or known
    unknown = [t for t in targets if t not in known]
    if unknown:
        print(f"Unknown ids: {unknown}; known: {known}", file=sys.stderr)
        sys.exit(2)
    noise_ids = list(NOISE_IDS) if not args.noise or "all" in args.noise else args.noise

    print("Cleaning stale Mininet state (mn -c)...", flush=True)
    mn_cleanup()
    state = create_network()
    net = state["net"]
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "iterations": args.iterations,
        "noise_levels": {},
        "noise_cleanup_residual": {},
    }
    try:
        pristine = inj.snapshot(net)
        for noise_id in noise_ids:
            max_loss = loss_limit(noise_id)
            applied = apply_noise(net, noise_id, state["links"])
            time.sleep(1.0)
            level = report["noise_levels"].setdefault(noise_id, {})
            try:
                for scenario in targets:
                    started = time.monotonic()
                    settled = []
                    failures = validate(net, scenario, args.iterations, max_loss, settled=settled)
                    passed = args.iterations - len(failures)
                    level[scenario] = {"passed": passed, "failed": len(failures), "settled": len(settled), "failures": failures[:5]}
                    status = "OK" if not failures else f"FAIL {len(failures)}/{args.iterations}"
                    note = f" [{len(settled)} settled after {SETTLE_SECONDS:g}s]" if settled else ""
                    print(f"{noise_id} {scenario:<4} {status} ({time.monotonic() - started:.0f}s){note}", flush=True)
                    if failures:
                        for line in failures[0]["problems"][:40]:
                            print(f"    {line}", flush=True)
            finally:
                clear_noise(net, applied, state["links"])
            report["noise_cleanup_residual"][noise_id] = inj.diff_snapshots(pristine, inj.snapshot(net))
    finally:
        try:
            inj.reset_all(net)
        except Exception as exc:
            print(f"reset_all during teardown: {exc}", file=sys.stderr)
        try:
            net.stop()
        except Exception as exc:
            print(f"net.stop(): {exc}", file=sys.stderr)
        mn_cleanup()

    report["pass"] = all(
        item["failed"] == 0 for level in report["noise_levels"].values() for item in level.values()
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for noise_id, lines in report["noise_cleanup_residual"].items():
        if lines:
            print(f"note: clear_noise({noise_id}) left {len(lines)} difference(s), e.g. {lines[0]}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if not report["pass"]:
        print("Fault validation: FAIL")
        sys.exit(1)
    print("Fault validation: PASS")


if __name__ == "__main__":
    main()
