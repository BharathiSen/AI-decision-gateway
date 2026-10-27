from network.topology import create_network
from bench.inject import inject, reset

def main():
    state = create_network()
    net = state["net"]
    correlation_id = None

    try:
        r1 = net.get("r1")
        host_a = net.get("hostA")

        print("\n=== BEFORE F3 ===")
        print("R1 routes:")
        print(r1.cmd("ip route"))
        print("Backup interface:")
        print(r1.cmd("ip link show r1-eth2"))
        print(host_a.cmd("ping -c 3 -W 2 10.0.4.2"))

        correlation_id = inject(net, "F3", seed=42)

        print("\n=== DURING F3 ===")
        print("R1 routes:")
        print(r1.cmd("ip route"))
        print("Backup interface:")
        print(r1.cmd("ip link show r1-eth2"))
        print(host_a.cmd("ping -c 3 -W 2 10.0.4.2"))

        print("\n=== RESET F3 ===")
        print(reset(correlation_id))
        correlation_id = None

        print("\n=== AFTER RESET ===")
        print("R1 routes:")
        print(r1.cmd("ip route"))
        print("Backup interface:")
        print(r1.cmd("ip link show r1-eth2"))
        print(host_a.cmd("ping -c 3 -W 2 10.0.4.2"))

    finally:
        if correlation_id is not None:
            print("Emergency cleanup:", reset(correlation_id))
        net.stop()

if __name__ == "__main__":
    main()
