"""
Live Mininet harness: inject → reset each fault 10 times.

Run on the Mininet VM as root:

    sudo python3 bench/validate_faults.py
    sudo python3 bench/validate_faults.py --fault F3 --iterations 10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench import inject as inj
from bench.mininet_util import mn_cleanup


def _fault_ids():
    return sorted(inj._FAULTS.keys())


def _inject_params(fault_id, seed):
    """Validation overrides: skip long iperf background during tc stress tests."""
    if fault_id == "F6":
        return {"rate_mbit": 2, "background_traffic": False}, seed
    if fault_id == "F7":
        return {"rate_mbit": 2, "background_traffic": False}, seed
    return None, seed


def validate_fault(net, fault_id, iterations=10, seed_base=0):
    failures = []
    for i in range(iterations):
        seed = seed_base + i
        cid = None
        try:
            if fault_id == "H0":
                result = inj.reset_all(net)
                if not result.get("healthy"):
                    failures.append((i, "H0 reset_all not healthy", result))
                continue
            params, seed = _inject_params(fault_id, seed)
            cid = inj.inject(net, fault_id, params=params, seed=seed)
            result = inj.reset(cid)
            if result.get("already_reset"):
                failures.append((i, "reset reported already_reset", result))
            cleanup = inj.reset_all(net)
            if not cleanup.get("healthy"):
                failures.append((i, "post reset_all not healthy", cleanup))
        except Exception as exc:
            failures.append((i, f"{type(exc).__name__}: {exc}", {"cid": cid}))
            try:
                inj.reset_all(net)
            except Exception:
                pass
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="append", help="Fault id (repeatable); default all")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("validate_faults.py must run as root (sudo) on the Mininet VM.", file=sys.stderr)
        sys.exit(2)

    print("Cleaning stale Mininet state (mn -c)...", flush=True)
    mn_cleanup()

    targets = args.fault or _fault_ids()
    state = create_network()
    net = state["net"]
    all_failures = {}

    try:
        for fault_id in targets:
            if fault_id not in inj._FAULTS:
                print(f"SKIP unknown fault {fault_id}")
                continue
            print(f"Validating {fault_id} x{args.iterations}...")
            fails = validate_fault(net, fault_id, iterations=args.iterations)
            if fails:
                all_failures[fault_id] = fails
                print(f"  FAIL {len(fails)} iteration(s)")
            else:
                print("  OK")
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

    if all_failures:
        for fid, fails in all_failures.items():
            print(f"\n{fid}:")
            for item in fails[:5]:
                print(" ", item)
        sys.exit(1)

    print("\nAll fault inject/reset cycles passed.")


if __name__ == "__main__":
    main()
