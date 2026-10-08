"""
Build fault × test observability matrix and inseparable pairs list.

    sudo python3 bench/build_observability_matrix.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench import inject as inj
from bench import tests as oam
from bench.mininet_util import recover_net

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _signature(test_results):
    """Compact outcome signature per test for discrimination."""
    sig = {}
    for r in test_results:
        tid = r["test_id"]
        if tid == "T1_e2e_ping":
            sig[tid] = f"loss={r.get('packet_loss_percent')},lat={r.get('latency_ms')}"
        elif tid == "T7_route_check":
            sig[tid] = r.get("active_path")
        elif tid == "T10_speed":
            sig[tid] = r.get("mbit_per_sec")
        else:
            sig[tid] = "ok"
    return sig


def load_correct_fixes():
    with (Path(__file__).resolve().parent.parent / "config" / "correct_fixes.yaml").open() as f:
        return yaml.safe_load(f)


def _write_pairs(rows):
    fixes = load_correct_fixes()
    fault_fix = {**fixes.get("faults", {}), **fixes.get("compounds", {})}
    pairs = []
    ids = list(fault_fix.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            fa = set(fault_fix.get(a, {}).get("correct", []))
            fb = set(fault_fix.get(b, {}).get("correct", []))
            if fa == fb:
                continue
            separating = []
            for tid in oam.DEFAULT_ORDER:
                va = next((r["outcome"] for r in rows if r["fault_id"] == a and r["test_id"] == tid), None)
                vb = next((r["outcome"] for r in rows if r["fault_id"] == b and r["test_id"] == tid), None)
                if va is not None and vb is not None and va != vb:
                    separating.append(tid)
            pairs.append({
                "fault_a": a,
                "fault_b": b,
                "different_fixes": True,
                "separating_tests": separating,
                "inseparable": len(separating) == 0,
            })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fault",
        action="append",
        help="Rebuild only these fault ids and merge them into the existing matrix",
    )
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("Requires root on Mininet VM.", file=sys.stderr)
        sys.exit(2)

    state = create_network()
    net = state["net"]
    fault_ids = args.fault or [f for f in inj._FAULTS if f != "H0"]
    rows = []
    if args.fault:
        matrix_path = RESULTS / "observability_matrix.csv"
        if matrix_path.exists():
            with matrix_path.open(newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["fault_id"] not in set(args.fault)]

    try:
        for fid in fault_ids:
            print(f"fault {fid} ...", flush=True)
            cid = None
            try:
                inj.reset_all(net)
                cid = inj.inject(net, fid, seed=42)
                bundle = oam.run_all(net)
                sig = _signature([r for r in bundle["results"]])
                for tid, val in sig.items():
                    rows.append({"fault_id": fid, "test_id": tid, "outcome": val})
                print(f"fault {fid} done", flush=True)
            except Exception as exc:
                print(f"fault {fid} FAILED: {exc}", flush=True)
                recover_net(net)
            finally:
                if cid:
                    try:
                        inj.reset(cid)
                    except Exception:
                        recover_net(net)
                try:
                    inj.reset_all(net)
                except Exception:
                    recover_net(net)
    finally:
        inj.reset_all(net)
        net.stop()

    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULTS / "observability_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fault_id", "test_id", "outcome"])
        w.writeheader()
        w.writerows(rows)

    pairs = _write_pairs(rows)

    inseparable_path = RESULTS / "inseparable_pairs.json"
    inseparable_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Wrote {matrix_path} ({len(rows)} rows)")
    print(f"Wrote {inseparable_path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
