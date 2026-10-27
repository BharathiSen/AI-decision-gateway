from network.topology import create_network
from bench.inject import inject, reset


def main():
    state = create_network()
    net = state["net"]
    correlation_id = None

    try:
        host_a = net.get("hostA")
        host_b = net.get("hostB")
        r2 = net.get("r2")

        def check(label):
            print(f"\n=== {label} ===")
            print(host_a.cmd("ping -c 5 -W 2 10.0.4.2"))
            print(r2.cmd("tc qdisc show dev r2-eth1"))

        check("BEFORE F5")

        correlation_id = inject(net, "F5", seed=42)
        check("DURING F5")

        print("\n=== RESET F5 ===")
        print(reset(correlation_id))
        correlation_id = None

        check("AFTER RESET")

    finally:
        if correlation_id is not None:
            print("Emergency cleanup:", reset(correlation_id))
        net.stop()


if __name__ == "__main__":
    main()
