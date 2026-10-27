from network.topology import create_network
from bench.inject import inject, reset

def main():
    state = create_network()
    net = state["net"]
    correlation_id = None

    try:
        r2 = net.get("r2")
        host_a = net.get("hostA")

        print("\n=== BEFORE F5 ===")
        print(r2.cmd("tc qdisc show dev r2-eth1"))
        print(host_a.cmd("ping -c 30 -i 0.1 -W 2 10.0.4.2"))

        correlation_id = inject(net, "F5", seed=42)

        print("\n=== DURING F5 ===")
        print(r2.cmd("tc qdisc show dev r2-eth1"))
        print(host_a.cmd("ping -c 30 -i 0.1 -W 2 10.0.4.2"))

        print("\n=== RESET F5 ===")
        print(reset(correlation_id))
        correlation_id = None

        print("\n=== AFTER RESET ===")
        print(r2.cmd("tc qdisc show dev r2-eth1"))
        print(host_a.cmd("ping -c 30 -i 0.1 -W 2 10.0.4.2"))

    finally:
        if correlation_id is not None:
            print("Emergency cleanup:", reset(correlation_id))
        net.stop()

if __name__ == "__main__":
    main()
