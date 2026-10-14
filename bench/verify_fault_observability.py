"""Inject each catalog fault and record how OAM outcomes change.

Run on the Mininet VM as root. This overwrites results/observability_matrix.csv
with measured grades. The checked-in stub is left alone until this runs.

    sudo python3 bench/verify_fault_observability.py
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench import inject as inj
from bench import tests as oam
from bench.mininet_util import recover_net
from bench.outcomes import signature
from bench.signature_checks import load_expected, separation_errors

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _inject_params(fault_id):
    """Skip long background iperf so T10 can use the port itself."""
    if fault_id in ("F6", "F7"):
        return {"rate_mbit": 2, "background_traffic": False}
    return None


def main():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("verify_fault_observability.py must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    spec = load_expected()
    fault_ids = list(spec.get("faults") or [])
    state = create_network()
    net = state["net"]
    rows = []
    try:
        for fid in fault_ids:
            print(f"fault {fid} ...", flush=True)
            cid = None
            try:
                inj.reset_all(net)
                cid = inj.inject(net, fid, params=_inject_params(fid), seed=42)
                bundle = oam.run_all(net)
                sig = signature(bundle["results"])
                for tid, val in sig.items():
                    rows.append({"fault_id": fid, "test_id": tid, "outcome": val})
                    print(f"  {tid}: {val}", flush=True)
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
        try:
            inj.reset_all(net)
        except Exception:
            pass
        net.stop()

    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULTS / "observability_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fault_id", "test_id", "outcome"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {matrix_path} ({len(rows)} rows)")

    errors = separation_errors(rows, spec)
    if errors:
        print("\nObservability checks failed:")
        for err in errors:
            print(f"  {err}")
        return 1
    print("Each required fault changed the expected tests, and required pairs are separated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
