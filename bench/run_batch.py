"""
Resumable overnight batch runner for episodes.

    sudo python3 bench/run_batch.py --count 50 --resume
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.topology import create_network
from bench.run_episode import run_episode, save_episode, EPISODE_DIR
from bench import inject as inj

STATE_FILE = EPISODE_DIR / "_batch_state.json"


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"completed": 0, "episode_ids": []}


def _save_state(state):
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--noise", default="N0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("run_batch.py requires root on Mininet VM.", file=sys.stderr)
        sys.exit(2)

    state = _load_state() if args.resume else {"completed": 0, "episode_ids": []}
    faults = [f for f in inj._FAULTS if f != "H0"] + ["H0"]

    topo = create_network()
    net, links = topo["net"], topo["links"]
    rng = random.Random(args.seed)

    try:
        while state["completed"] < args.count:
            fault_id = rng.choice(faults)
            seed = rng.randint(0, 2**31 - 1)
            record = run_episode(
                net,
                links,
                fault_id=fault_id,
                noise_id=args.noise,
                seed=seed,
            )
            path = save_episode(record)
            state["completed"] += 1
            state["episode_ids"].append(record["episode_id"])
            _save_state(state)
            print(f"[{state['completed']}/{args.count}] {fault_id} -> {path.name}")
    finally:
        inj.reset_all(net)
        net.stop()

    print(f"Done. {state['completed']} episodes in {EPISODE_DIR}")


if __name__ == "__main__":
    main()
