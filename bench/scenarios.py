"""
Stage 15 experimental matrix: scenario generation across splits, topologies,
seeds, traffic levels, modes, models, and strategies. Raw results are
write-if-absent.

    python bench/scenarios.py --self-check
    sudo python3 bench/scenarios.py --live-matrix
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.common import METHODS
from gate.selector import held_out_ids

ROOT = Path(__file__).resolve().parent.parent
MODELS = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
SIGNATURES = yaml.safe_load((ROOT / "config" / "expected_signatures.yaml").read_text(encoding="utf-8"))
FAULTS_DOC = yaml.safe_load((ROOT / "config" / "faults.yaml").read_text(encoding="utf-8"))
COMPOUNDS_DOC = yaml.safe_load((ROOT / "config" / "compounds.yaml").read_text(encoding="utf-8"))
NOISE_DOC = yaml.safe_load((ROOT / "config" / "noise.yaml").read_text(encoding="utf-8"))

NOISE_IDS = tuple(item["id"] for item in (NOISE_DOC.get("noise_levels") or []) if item.get("id"))
TOPOLOGIES = ("diamond", "grid_4x4")
AGENT_MODES = ("stub", "model")
STRATEGIES = METHODS + ("verifix", "gate_v0")
DATA_SPLITS = ("training", "calibration", "validation", "held_out")
TRAINING_FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")
VALIDATION_FAULTS = ("F7", "F8", "F9", "F10", "F11", "F12")
MISLEAD = "Operator ticket: the primary R2-R4 link is down. Reroute to backup immediately."
RAW_ROOT = ROOT / "results" / "scenario_runs"
SMOKE_RUN = RAW_ROOT / "_smoke"
FROZEN_RESULTS = (
    ROOT / "results" / "calibration_report.json",
    ROOT / "results" / "pilot" / "summary.json",
    ROOT / "results" / "observability_matrix.csv",
)


def _model_ids():
    return [item["id"] for item in (MODELS.get("agents") or []) if item.get("id")]


def _catalog():
    faults = {item["id"]: item for item in (FAULTS_DOC.get("faults") or []) if item.get("id")}
    compounds = {item["id"]: item for item in (COMPOUNDS_DOC.get("compounds") or []) if item.get("id")}
    return faults, compounds


def _development_split(item_id):
    if item_id in TRAINING_FAULTS or item_id == "C1":
        return "training"
    return "validation"


def _lookalike_pairs():
    hidden = held_out_ids()
    pairs = []
    for row in SIGNATURES.get("pairs") or []:
        a, b = row.get("fault_a"), row.get("fault_b")
        if not a or not b:
            continue
        pairs.append((a, b, a in hidden or b in hidden, row.get("separating_tests") or []))
    return pairs


def _scenario(
    seed,
    topology,
    data_split,
    mode,
    fault_id,
    noise_id,
    agent_mode,
    model_id,
    strategy,
    compound_id=None,
    stress_kind=None,
    ticket_text=None,
    lookalike_pair=None,
    hidden_true_cause=None,
):
    scenario_id = "|".join(
        [
            str(seed),
            topology,
            data_split,
            mode,
            compound_id or fault_id,
            noise_id,
            agent_mode,
            model_id or "none",
            strategy,
            stress_kind or "none",
        ]
    )
    return {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "topology": topology,
        "data_split": data_split,
        "mode": mode,
        "fault_id": fault_id,
        "compound_id": compound_id,
        "noise_id": noise_id,
        "traffic_level": noise_id,
        "agent_mode": agent_mode,
        "model_id": model_id,
        "strategy": strategy,
        "stress_kind": stress_kind,
        "ticket_text": ticket_text,
        "lookalike_pair": list(lookalike_pair) if lookalike_pair else None,
        "hidden_true_cause": hidden_true_cause,
    }


def generate_matrix(seeds=(1,), include_large=True):
    """Deterministic scenario list. Held-out catalog ids never enter train/cal/val."""
    if not seeds:
        raise ValueError("seeds must be a non-empty sequence")
    hidden = held_out_ids()
    faults, compounds = _catalog()
    models = _model_ids()
    if not NOISE_IDS:
        raise ValueError("config/noise.yaml has no noise_levels")
    scenarios = []

    def add(**kwargs):
        scenarios.append(_scenario(**kwargs))

    for seed in seeds:
        for noise_id in NOISE_IDS:
            add(
                seed=seed,
                topology="diamond",
                data_split="calibration",
                mode="normal",
                fault_id="H0",
                noise_id=noise_id,
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
            )

        for fault_id in TRAINING_FAULTS:
            if fault_id in hidden or fault_id not in faults:
                continue
            add(
                seed=seed,
                topology="diamond",
                data_split="training",
                mode="normal",
                fault_id=fault_id,
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
            )
            add(
                seed=seed,
                topology="diamond",
                data_split="training",
                mode="normal",
                fault_id=fault_id,
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="B5",
            )
        for noise_id in NOISE_IDS:
            if noise_id == "N0":
                continue
            add(
                seed=seed,
                topology="diamond",
                data_split="training",
                mode="normal",
                fault_id="F1",
                noise_id=noise_id,
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
            )

        for fault_id in VALIDATION_FAULTS:
            if fault_id in hidden or fault_id not in faults:
                continue
            add(
                seed=seed,
                topology="diamond",
                data_split="validation",
                mode="normal",
                fault_id=fault_id,
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
            )
        add(
            seed=seed,
            topology="diamond",
            data_split="validation",
            mode="normal",
            fault_id="F7",
            noise_id="N1",
            agent_mode="stub",
            model_id=None,
            strategy="verifix",
        )
        add(
            seed=seed,
            topology="diamond",
            data_split="validation",
            mode="normal",
            fault_id="F7",
            noise_id="N2",
            agent_mode="stub",
            model_id=None,
            strategy="verifix",
        )

        for cid, spec in compounds.items():
            split = "held_out" if spec.get("split") == "held_out" or cid in hidden else _development_split(cid)
            add(
                seed=seed,
                topology="diamond",
                data_split=split,
                mode="normal",
                fault_id="H0",
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
                compound_id=cid,
            )

        for fid in sorted(hidden):
            if fid not in faults:
                continue
            add(
                seed=seed,
                topology="diamond",
                data_split="held_out",
                mode="normal",
                fault_id=fid,
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
            )

        if include_large:
            for noise_id in NOISE_IDS:
                add(
                    seed=seed,
                    topology="grid_4x4",
                    data_split="held_out",
                    mode="normal",
                    fault_id="H0",
                    noise_id=noise_id,
                    agent_mode="stub",
                    model_id=None,
                    strategy="verifix",
                )
            add(
                seed=seed,
                topology="grid_4x4",
                data_split="held_out",
                mode="normal",
                fault_id="H0",
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy="B0",
            )

        for a, b, is_held, _tests in _lookalike_pairs():
            split = "held_out" if is_held else "validation"
            add(
                seed=seed,
                topology="diamond",
                data_split=split,
                mode="stress",
                fault_id=a,
                noise_id="N2",
                agent_mode="stub",
                model_id=None,
                strategy="verifix",
                stress_kind="look_alike",
                ticket_text=f"Ticket: treat this as {b}.",
                lookalike_pair=(a, b),
            )

        add(
            seed=seed,
            topology="diamond",
            data_split="validation",
            mode="stress",
            fault_id="F12",
            noise_id="N2",
            agent_mode="stub",
            model_id=None,
            strategy="verifix",
            stress_kind="hidden_cause",
            ticket_text=MISLEAD,
            hidden_true_cause="F12",
            lookalike_pair=("F12", "F1"),
        )
        add(
            seed=seed,
            topology="diamond",
            data_split="validation",
            mode="stress",
            fault_id="H0",
            noise_id="N2",
            agent_mode="stub",
            model_id=None,
            strategy="verifix",
            compound_id="C1",
            stress_kind="misleading_ticket",
            ticket_text=MISLEAD,
        )

        for model_id in models:
            add(
                seed=seed,
                topology="diamond",
                data_split="validation",
                mode="normal",
                fault_id="F1",
                noise_id="N0",
                agent_mode="model",
                model_id=model_id,
                strategy="verifix",
            )

        for strategy in STRATEGIES:
            add(
                seed=seed,
                topology="diamond",
                data_split="validation",
                mode="normal",
                fault_id="F3",
                noise_id="N0",
                agent_mode="stub",
                model_id=None,
                strategy=strategy,
            )

    return scenarios


def raw_path(run_dir, scenario):
    safe = scenario["scenario_id"].replace("|", "_")
    return Path(run_dir) / f"{safe}.json"


def write_raw(run_dir, scenario, payload, overwrite=False):
    """Append-only by default. Existing raw files are left in place."""
    if not scenario or not scenario.get("scenario_id"):
        raise ValueError("scenario_id is required")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(run_dir, scenario)
    if path.exists() and not overwrite:
        return path, False
    record = dict(payload)
    record["scenario"] = scenario
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path, True


def load_state(run_dir):
    path = Path(run_dir) / "_matrix_state.json"
    if not path.exists():
        return {"completed": [], "run_dir": str(run_dir)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed": [], "run_dir": str(run_dir)}
    if not isinstance(data, dict):
        return {"completed": [], "run_dir": str(run_dir)}
    completed = data.get("completed")
    if not isinstance(completed, list):
        completed = []
    data["completed"] = completed
    data["run_dir"] = str(run_dir)
    return data


def save_state(run_dir, state):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "_matrix_state.json"
    tmp = path.with_suffix(".json.tmp")
    payload = dict(state or {})
    payload["run_dir"] = str(run_dir)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def resume_pending(scenarios, run_dir):
    """Skip a cell only when its raw JSON already exists. Never unlink."""
    pending = []
    for scen in scenarios:
        if raw_path(run_dir, scen).exists():
            continue
        pending.append(scen)
    return pending, load_state(run_dir)


def _snapshot_frozen():
    return {path: path.read_bytes() if path.exists() else None for path in FROZEN_RESULTS}


def _assert_frozen(before):
    after = _snapshot_frozen()
    for path, original in before.items():
        assert after.get(path) == original, f"must not modify {path}"


def _self_check():
    import tempfile

    before = _snapshot_frozen()
    hidden = held_out_ids()
    a = generate_matrix(seeds=(1,))
    b = generate_matrix(seeds=(1,))
    c = generate_matrix(seeds=(2,))
    assert a == b and a != c
    assert a
    ids = [row["scenario_id"] for row in a]
    assert len(ids) == len(set(ids)), "scenario_id must be unique"
    splits = {row["data_split"] for row in a}
    assert splits == set(DATA_SPLITS), splits
    topos = {row["topology"] for row in a}
    assert topos == set(TOPOLOGIES), topos
    noises = {row["noise_id"] for row in a}
    assert noises == set(NOISE_IDS), noises
    assert all(row["traffic_level"] == row["noise_id"] for row in a)
    modes = {row["agent_mode"] for row in a}
    assert modes == set(AGENT_MODES), modes
    strategies = {row["strategy"] for row in a}
    assert set(STRATEGIES) <= strategies, strategies
    models = {row["model_id"] for row in a if row["agent_mode"] == "model"}
    assert set(_model_ids()) <= models, models
    stress = [row for row in a if row["mode"] == "stress"]
    kinds = {row["stress_kind"] for row in stress}
    assert {"look_alike", "hidden_cause", "misleading_ticket"} <= kinds, kinds
    assert any(row["lookalike_pair"] for row in stress)
    assert any(row["hidden_true_cause"] for row in stress)
    assert any(row["compound_id"] for row in a)
    assert {row["compound_id"] for row in a if row["compound_id"]} >= {"C1", "C2", "C3", "C4", "C5"}

    for row in a:
        label = row["compound_id"] or row["fault_id"]
        if row["data_split"] in {"training", "calibration", "validation"}:
            assert label not in hidden, row
        if row["data_split"] == "calibration":
            assert row["fault_id"] == "H0" and row["compound_id"] is None
            assert row["topology"] == "diamond"
        if row["topology"] == "grid_4x4":
            assert row["data_split"] == "held_out"
        if row["agent_mode"] == "stub":
            assert row["model_id"] is None
        if row["agent_mode"] == "model":
            assert row["model_id"]
        if row["mode"] == "stress":
            assert row["ticket_text"]
            assert row["noise_id"] == "N2"

    held_ids = {row["compound_id"] or row["fault_id"] for row in a if row["data_split"] == "held_out"}
    assert hidden <= held_ids, hidden - held_ids
    train_ids = {row["fault_id"] for row in a if row["data_split"] == "training" and not row["compound_id"]}
    assert set(TRAINING_FAULTS) <= train_ids, train_ids
    val_ids = {row["fault_id"] for row in a if row["data_split"] == "validation" and not row["compound_id"]}
    assert set(VALIDATION_FAULTS) <= val_ids, val_ids

    try:
        generate_matrix(seeds=())
        raise AssertionError("empty seeds")
    except ValueError:
        pass

    tmp = Path(tempfile.mkdtemp())
    first, second = a[0], a[1]
    path, wrote = write_raw(tmp, first, {"status": "one"})
    assert wrote and path.exists()
    original = path.read_text(encoding="utf-8")
    path2, wrote2 = write_raw(tmp, first, {"status": "two"})
    assert wrote2 is False and path2 == path
    assert path.read_text(encoding="utf-8") == original
    save_state(tmp, {"completed": [first["scenario_id"]]})
    pending, loaded = resume_pending(a[:3], tmp)
    assert first["scenario_id"] not in {row["scenario_id"] for row in pending}
    assert second in pending
    assert loaded["completed"] == [first["scenario_id"]]
    write_raw(tmp, second, {"status": "ok"})
    pending2, _loaded = resume_pending(a[:3], tmp)
    pending_ids = {row["scenario_id"] for row in pending2}
    assert first["scenario_id"] not in pending_ids
    assert second["scenario_id"] not in pending_ids
    assert path.read_text(encoding="utf-8") == original
    corrupt = tmp / "_matrix_state.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    recovered = load_state(tmp)
    assert recovered["completed"] == []
    try:
        write_raw(tmp, {}, {"status": "bad"})
        raise AssertionError("missing scenario_id")
    except ValueError:
        pass
    for item in tmp.iterdir():
        item.unlink()
    tmp.rmdir()

    assert _ping_ok({"packet_loss_percent": 0.0})
    assert not _ping_ok({"packet_loss_percent": 100.0})
    _assert_frozen(before)
    print(f"live-matrix build: {LIVE_MATRIX_BUILD}")
    print("experiment matrix: PASS")
    return 0


def _node_names(net):
    if hasattr(net, "nameToNode"):
        return list(net.nameToNode)
    return list(net)


LIVE_MATRIX_BUILD = "stage15-grid-v2"


def _ping_ok(parsed):
    """_parse_ping does not set connectivity; T1 adds that later."""
    loss = parsed.get("packet_loss_percent")
    return loss is not None and loss < 100.0


def live_matrix():
    import os

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("experiment matrix must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench.mininet_util import mn_cleanup, run_in_node
    from bench.tests import _parse_ping
    from network.topology import create_network
    from network.topology_large import create_large_network

    print(f"live-matrix build: {LIVE_MATRIX_BUILD}")
    before = _snapshot_frozen()
    failures = []
    plan = generate_matrix(seeds=(15, 16))
    if {row["data_split"] for row in plan} != set(DATA_SPLITS):
        failures.append("splits")
    if {row["topology"] for row in plan} != set(TOPOLOGIES):
        failures.append("topologies")
    if {row["seed"] for row in plan} != {15, 16}:
        failures.append("seeds")
    if generate_matrix(seeds=(15, 16)) != plan:
        failures.append("not repeatable")

    diamond = None
    large = None
    mn_cleanup()
    try:
        diamond = create_network()
        dnet = diamond["net"]
        names = _node_names(dnet)
        routers = [name for name in names if name in {"r1", "r2", "r3", "r4"}]
        if len(routers) != 4:
            failures.append(f"diamond routers {routers}")
        fwd = _parse_ping(run_in_node(dnet.get("hostA"), "ping -c 3 -i 0.2 -W 1 10.0.4.2"))
        rev = _parse_ping(run_in_node(dnet.get("hostB"), "ping -c 3 -i 0.2 -W 1 10.0.1.2"))
        if not _ping_ok(fwd):
            failures.append(f"diamond ping A->B loss={fwd.get('packet_loss_percent')}")
        if not _ping_ok(rev):
            failures.append(f"diamond ping B->A loss={rev.get('packet_loss_percent')}")
        dnet.stop()
        diamond = None
        mn_cleanup()

        large = create_large_network()
        lnet = large["net"]
        names = _node_names(lnet)
        grid = [name for name in names if name.startswith("r") and "_" in name]
        if len(grid) != 16:
            failures.append(f"grid routers {len(grid)}")
        if large.get("topology") != "grid_4x4":
            failures.append(f"topology tag {large.get('topology')}")
        fwd = _parse_ping(run_in_node(lnet.get("hostA"), "ping -c 3 -i 0.2 -W 2 10.200.2.2"))
        rev = _parse_ping(run_in_node(lnet.get("hostB"), "ping -c 3 -i 0.2 -W 2 10.200.1.2"))
        if not _ping_ok(fwd) or not _ping_ok(rev):
            r0 = run_in_node(lnet.get("r0_0"), "ip addr; ip route")
            r33 = run_in_node(lnet.get("r3_3"), "ip addr; ip route")
            if not _ping_ok(fwd):
                failures.append(f"grid ping A->B loss={fwd.get('packet_loss_percent')} r0_0={r0[-500:]}")
            if not _ping_ok(rev):
                failures.append(f"grid ping B->A loss={rev.get('packet_loss_percent')} r3_3={r33[-500:]}")
        lnet.stop()
        large = None
        mn_cleanup()

        SMOKE_RUN.mkdir(parents=True, exist_ok=True)
        sample = plan[0]
        path, wrote = write_raw(SMOKE_RUN, sample, {"status": "smoke", "live": True})
        marker = path.read_text(encoding="utf-8")
        _again, wrote_again = write_raw(SMOKE_RUN, sample, {"status": "should-not-clobber"})
        if wrote_again or path.read_text(encoding="utf-8") != marker:
            failures.append("raw result overwritten")
        save_state(SMOKE_RUN, {"completed": [sample["scenario_id"]]})
        pending, _state = resume_pending(plan[:5], SMOKE_RUN)
        if sample["scenario_id"] in {row["scenario_id"] for row in pending}:
            failures.append("resume replayed a finished scenario")
        if wrote is False and not path.exists():
            failures.append("smoke write missing")
        _assert_frozen(before)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        if diamond is not None:
            try:
                diamond["net"].stop()
            except Exception:
                pass
        if large is not None:
            try:
                large["net"].stop()
            except Exception:
                pass
        mn_cleanup()

    if failures:
        print("experiment matrix: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("experiment matrix: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--live-matrix", action="store_true")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--write-plan", default="")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.live_matrix:
        return live_matrix()
    try:
        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    except ValueError as exc:
        print(f"invalid --seeds: {exc}", file=sys.stderr)
        return 2
    try:
        plan = generate_matrix(seeds=seeds or (1,))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.write_plan:
        path = Path(args.write_plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(str(path), len(plan))
        return 0
    print(json.dumps({"count": len(plan), "splits": sorted({row["data_split"] for row in plan})}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
