"""
Build fault × test observability matrix and inseparable pairs list.

    sudo python3 bench/build_observability_matrix.py
"""

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


def main():
    if os.geteuid() != 0:
        print("Requires root on Mininet VM.", file=sys.stderr)
        sys.exit(2)

    state = create_network()
    net = state["net"]
    rows = []
    fault_ids = [f for f in inj._FAULTS if f != "H0"]

    try:
        for fid in fault_ids:
            inj.reset_all(net)
            cid = inj.inject(net, fid, seed=42)
            bundle = oam.run_all(net)
            sig = _signature([r for r in bundle["results"]])
            for tid, val in sig.items():
                rows.append({"fault_id": fid, "test_id": tid, "outcome": val})
            inj.reset(cid)
            inj.reset_all(net)
    finally:
        inj.reset_all(net)
        net.stop()

    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULTS / "observability_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fault_id", "test_id", "outcome"])
        w.writeheader()
        w.writerows(rows)

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

    inseparable_path = RESULTS / "inseparable_pairs.json"
    inseparable_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Wrote {matrix_path} ({len(rows)} rows)")
    print(f"Wrote {inseparable_path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
