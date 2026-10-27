import time

from network.topology import create_network
from bench.inject import inject, reset


def main():
    state = create_network()
    net = state["net"]
    correlation_id = None
    server_pid = None

    r2 = net.get("r2")
    r4 = net.get("r4")
    host_a = net.get("hostA")

    try:
        print("\n=== START IPERF3 SERVER ON R4 ===")
        server_pid = r4.cmd(
            "iperf3 -s > /tmp/f6_iperf_server.log 2>&1 & echo $!"
        ).strip()
        time.sleep(1)

        print("Server PID:", server_pid)
        print("Server listener:")
        print(r4.cmd("ss -lntp | grep ':5201' || true"))

        print("\n=== BEFORE F6 ===")
        print("R2 qdisc:")
        print(r2.cmd("tc -s qdisc show dev r2-eth1"))
        print("R2 classes:")
        print(r2.cmd("tc -s class show dev r2-eth1"))
        print("Connectivity:")
        print(host_a.cmd("ping -c 10 -i 0.1 -W 2 10.0.4.2"))

        correlation_id = inject(net, "F6", seed=42)

        print("\n=== DURING F6 ===")
        print("Injection ID:", correlation_id)
        print("R2 qdisc:")
        print(r2.cmd("tc -s qdisc show dev r2-eth1"))
        print("R2 classes:")
        print(r2.cmd("tc -s class show dev r2-eth1"))

        time.sleep(5)

        print("\n=== BACKGROUND TRAFFIC LOG ===")
        print(r2.cmd("cat /tmp/inject_bg_r2.log"))

        print("\n=== QDISC AFTER BACKGROUND TRAFFIC ===")
        print(r2.cmd("tc -s qdisc show dev r2-eth1"))
        print(r2.cmd("tc -s class show dev r2-eth1"))

        print("\n=== CONNECTIVITY DURING F6 ===")
        print(host_a.cmd("ping -c 30 -i 0.1 -W 2 10.0.4.2"))

        print("\n=== RESET F6 ===")
        print(reset(correlation_id))
        correlation_id = None

        print("\n=== AFTER RESET ===")
        print("R2 qdisc:")
        print(r2.cmd("tc -s qdisc show dev r2-eth1"))
        print("R2 classes:")
        print(r2.cmd("tc -s class show dev r2-eth1"))
        print("Connectivity:")
        print(host_a.cmd("ping -c 10 -i 0.1 -W 2 10.0.4.2"))

    finally:
        if correlation_id is not None:
            print("Emergency cleanup:", reset(correlation_id))

        if server_pid:
            r4.cmd(f"kill {server_pid} 2>/dev/null")
        net.stop()


if __name__ == "__main__":
    main()
