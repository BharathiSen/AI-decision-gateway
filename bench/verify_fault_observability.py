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
from bench.mininet_util import ensure_iperf3_server, recover_net
from bench.outcomes import signature
from bench.signature_checks import load_expected, separation_errors

RESULTS = Path(__file__).resolve().parent.parent / "results"


# Top of each catalog range, so the matrix is not a one-sample coin flip.
# F6/F7 keep background traffic, but on UDP/5202 (see inject.py), so T8/T10
# can still use the iperf3 server on 5201.
_OBSERVABLE_PARAMS = {
    "F3": {"loss_percent": 20},
    "F4": {"corrupt_percent": 5},
    "F6": {"rate_mbit": 1, "background_traffic": True},
    "F7": {"rate_mbit": 1, "background_traffic": True},
    "F11": {"loss_percent": 20},
    "U1": {"mtu_bytes": 1000},
    "U2": {"loss_percent": 20},
}


def _inject_params(fault_id):
    return _OBSERVABLE_PARAMS.get(fault_id)


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
        ensure_iperf3_server(net)
        for fid in fault_ids:
            print(f"fault {fid} ...", flush=True)
            cid = None
            try:
                inj.reset_all(net)
                cid = inj.inject(net, fid, params=_inject_params(fid), seed=42)
                ensure_iperf3_server(net)
                bundle = oam.run_all(net, ping_count=50, error_probe=80)
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
