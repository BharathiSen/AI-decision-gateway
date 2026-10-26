"""Assert the diamond topology, static routes, and hostA-hostB connectivity.

Run on the Mininet VM as root:

    sudo python3 bench/verify_baseline.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from network import controller as ctl
from telemetry.collector import collect_current_state
from bench.mininet_util import run_in_node

EXPECTED_NODES = ("hostA", "hostB", "r1", "r2", "r3", "r4")
EXPECTED_LINKS = {
    "hostA-r1": {"10.0.1.2", "10.0.1.1"},
    "r1-r2": {"10.0.12.1", "10.0.12.2"},
    "r1-r3": {"10.0.13.1", "10.0.13.2"},
    "r2-r4": {"10.0.24.1", "10.0.24.2"},
    "r3-r4": {"10.0.34.1", "10.0.34.2"},
    "r4-hostB": {"10.0.4.1", "10.0.4.2"},
}
COUNTER_FIELDS = ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")
QUEUE_FIELDS = ("drops", "backlog_bytes", "backlog_packets")


def _check(failures, label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def main():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("verify_baseline.py must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    failures = []
    state = create_network()
    net = state["net"]
    links = state["links"]
    try:
        for name in EXPECTED_NODES:
            try:
                net.get(name)
                _check(failures, f"node {name} exists", True)
            except Exception as exc:
                _check(failures, f"node {name} exists", False, str(exc))

        for link_name, expected_ips in EXPECTED_LINKS.items():
            link = links[link_name]
            up = ctl.is_link_up(link)
            got = {ip for ip in (link.intf1.IP(), link.intf2.IP()) if ip}
            states = f"{link.intf1.name}={ctl.intf_operstate(link.intf1)!r} {link.intf2.name}={ctl.intf_operstate(link.intf2)!r}"
            _check(failures, f"link {link_name} up", up, f"operstate not up: {states}")
            _check(
                failures,
                f"link {link_name} addresses",
                got == expected_ips,
                f"got {sorted(got)}",
            )

        r1_route = run_in_node(net.get("r1"), "ip route show 10.0.4.0/24")
        r4_route = run_in_node(net.get("r4"), "ip route show 10.0.1.0/24")
        _check(failures, "r1 route to 10.0.4.0/24 via 10.0.12.2", "10.0.12.2" in r1_route, r1_route.strip())
        _check(failures, "r4 route to 10.0.1.0/24 via 10.0.24.1", "10.0.24.1" in r4_route, r4_route.strip())

        hostB = net.get("hostB").IP()
        ping = run_in_node(net.get("hostA"), f"ping -c 3 -W 1 {hostB}")
        _check(failures, "hostA ping hostB 0% loss", "0% packet loss" in ping, ping.strip().splitlines()[-1] if ping.strip() else "no output")

        snap = collect_current_state(net, links)
        metrics = snap.get("metrics") or {}
        _check(failures, "telemetry e2e loss present", "packet_loss_percent" in metrics)
        _check(failures, "telemetry e2e latency present", "latency_ms" in metrics)
        _check(failures, "telemetry connectivity", metrics.get("connectivity") is True, str(metrics))
        _check(
            failures,
            "telemetry e2e loss is 0",
            metrics.get("packet_loss_percent") == 0.0,
            str(metrics.get("packet_loss_percent")),
        )

        routes = snap.get("routes") or []
        r1_snap = next((r for r in routes if r.get("node") == "r1" and r.get("prefix") == "10.0.4.0/24"), None)
        _check(
            failures,
            "telemetry r1 route via primary",
            r1_snap is not None and "10.0.12.2" in r1_snap.get("route", ""),
            str(r1_snap),
        )
        counters = snap.get("interface_counters") or {}
        _check(failures, "telemetry interface counters present", bool(counters))
        if counters:
            sample = next(iter(counters.values()))
            _check(
                failures,
                "telemetry counter fields",
                all(field in sample for field in COUNTER_FIELDS),
                str(sample),
            )
        queues = snap.get("queues") or {}
        _check(failures, "telemetry queues present", bool(queues))
        if queues:
            sample_q = next(iter(queues.values()))
            _check(
                failures,
                "telemetry queue fields",
                all(field in sample_q for field in QUEUE_FIELDS),
                str(sample_q),
            )
        for link in snap.get("links") or []:
            if "packet_loss_percent" not in link or "latency_ms" not in link:
                _check(failures, f"telemetry link {link.get('name')} loss and latency", False, str(link))
                break
        else:
            if snap.get("links"):
                _check(failures, "telemetry per-link loss and latency", True)
            else:
                _check(failures, "telemetry per-link loss and latency", False, "no links")
    finally:
        net.stop()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print("Baseline topology, routes, connectivity, and telemetry checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
