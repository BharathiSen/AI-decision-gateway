"""
Master dataset runner: resumable episodes, append-only JSONL, derived CSV.

    python bench/run_batch.py --self-check
    sudo python3 bench/run_batch.py --live-dataset
    sudo python3 bench/run_batch.py --count 50 --resume
"""

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.run_episode import run_episode, save_episode  # save_episode kept for orphan-json recovery in self-check
from bench.scenarios import generate_matrix, raw_path, resume_pending, write_raw
from gate.selector import held_out_ids

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "results" / "dataset"
RAW_DIR = DATASET / "raw"
PROCESSED_DIR = DATASET / "processed"
JSONL_NAME = "episodes.jsonl"
CSV_NAME = "episodes.csv"
STATE_NAME = "_batch_state.json"
SMOKE_DIR = DATASET / "_smoke"
FROZEN_RESULTS = (
    ROOT / "results" / "calibration_report.json",
    ROOT / "results" / "pilot" / "summary.json",
    ROOT / "results" / "observability_matrix.csv",
    ROOT / "results" / "experiment_summary.json",
)
LIVE_DATASET_BUILD = "stage17-dataset-v2"
CSV_FIELDS = (
    "episode_id",
    "scenario_id",
    "status",
    "seed",
    "topology",
    "fault_id",
    "compound_id",
    "traffic_level",
    "model_id",
    "strategy",
    "proposed_fix",
    "decision",
    "tests",
    "oam_cost_seconds",
    "decision_latency_seconds",
    "recovery_succeeded",
    "software_version",
    "error",
)
CONFIG_FILES = (
    "config/faults.yaml",
    "config/compounds.yaml",
    "config/costs.yaml",
    "config/models.yaml",
    "config/correct_fixes.yaml",
)


def software_version():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def config_fingerprint():
    digest = hashlib.sha256()
    for rel in CONFIG_FILES:
        path = ROOT / rel
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def provenance():
    return {
        "software_version": software_version(),
        "config_fingerprint": config_fingerprint(),
        "python": sys.version.split()[0],
    }


def jsonl_path(raw_dir):
    return Path(raw_dir) / JSONL_NAME


def csv_path(processed_dir):
    return Path(processed_dir) / CSV_NAME


def state_path(out_dir):
    return Path(out_dir) / STATE_NAME


def load_state(out_dir):
    path = state_path(out_dir)
    if not path.exists():
        return {"completed": [], "episode_ids": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed": [], "episode_ids": []}
    if not isinstance(data, dict):
        return {"completed": [], "episode_ids": []}
    data.setdefault("completed", [])
    data.setdefault("episode_ids", [])
    return data


def save_state(out_dir, state):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = state_path(out_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, record):
    """Append one complete JSON object. Never rewrites earlier lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def jsonl_ids(path):
    ids = set()
    path = Path(path)
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("episode_id"):
                ids.add(row["episode_id"])
    return ids


def tabular_row(record):
    tests = (record.get("gate_decision") or {}).get("tests_run") or []
    if not tests:
        tests = [row.get("test_id") for row in ((record.get("oam") or {}).get("results") or []) if row.get("test_id")]
    oam_cost = record.get("oam_cost") or {}
    gate_cost = record.get("gate_cost") or {}
    return {
        "episode_id": record.get("episode_id"),
        "scenario_id": record.get("scenario_id"),
        "status": record.get("status"),
        "seed": record.get("seed"),
        "topology": record.get("topology"),
        "fault_id": record.get("fault_id"),
        "compound_id": record.get("compound_id") or "",
        "traffic_level": record.get("traffic_level") or record.get("noise_id"),
        "model_id": record.get("model_id") or "",
        "strategy": record.get("strategy"),
        "proposed_fix": record.get("proposed_fix")
        or (record.get("agent_proposal") or {}).get("action"),
        "decision": record.get("decision") or (record.get("gate_decision") or {}).get("decision"),
        "tests": " ".join(str(tid) for tid in tests if tid),
        "oam_cost_seconds": oam_cost.get("total_duration_seconds"),
        "decision_latency_seconds": gate_cost.get("latency_seconds"),
        "recovery_succeeded": record.get("remediation_succeeded"),
        "software_version": record.get("software_version"),
        "error": record.get("error") or record.get("execution_error") or record.get("agent_error") or "",
    }


def write_processed_csv(records, processed_dir):
    """Rebuild tabular output from records. Processed data may be overwritten."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = csv_path(processed_dir)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(tabular_row(record))
    tmp.replace(path)
    return path


def load_raw_json(raw_dir):
    records = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("episode_id"):
            records.append(row)
    return records


def sync_jsonl(raw_dir):
    """Append any per-episode JSON missing from the JSONL. Never rewrite lines."""
    path = jsonl_path(raw_dir)
    have = jsonl_ids(path)
    for record in load_raw_json(raw_dir):
        eid = record.get("episode_id")
        if eid and eid not in have:
            append_jsonl(path, record)
            have.add(eid)
    return path


def attach_dataset_fields(record, scenario, extra=None):
    """Fill Stage 17 trace fields without changing the episode execution path."""
    record = dict(record or {})
    scenario = scenario or {}
    meta = provenance()
    proposal = record.get("agent_proposal") or {}
    gate = record.get("gate_decision") or {}
    record["scenario_id"] = scenario.get("scenario_id")
    record["topology"] = scenario.get("topology") or record.get("topology") or "diamond"
    record["traffic_level"] = scenario.get("traffic_level") or scenario.get("noise_id") or record.get("noise_id")
    record["model_id"] = scenario.get("model_id")
    record["agent_mode"] = scenario.get("agent_mode")
    record["strategy"] = scenario.get("strategy")
    record["data_split"] = scenario.get("data_split")
    record["mode"] = scenario.get("mode")
    record["stress_kind"] = scenario.get("stress_kind")
    if record.get("seed") is None:
        record["seed"] = scenario.get("seed")
    if not record.get("compound_id"):
        record["compound_id"] = scenario.get("compound_id")
    record["proposed_fix"] = proposal.get("action")
    record["decision"] = gate.get("decision")
    record["software_version"] = meta["software_version"]
    record["config_fingerprint"] = meta["config_fingerprint"]
    record["python"] = meta["python"]
    record["dataset_recorded_at"] = datetime.now(timezone.utc).isoformat()
    if extra:
        record.update(extra)
    record.setdefault("status", "ok")
    return record


def partial_record(scenario, status, error):
    meta = provenance()
    scenario = scenario or {}
    return {
        "episode_id": str(uuid.uuid4()),
        "status": status,
        "error": str(error),
        "scenario_id": scenario.get("scenario_id"),
        "seed": scenario.get("seed"),
        "topology": scenario.get("topology") or "diamond",
        "fault_id": scenario.get("fault_id"),
        "compound_id": scenario.get("compound_id"),
        "noise_id": scenario.get("noise_id"),
        "traffic_level": scenario.get("traffic_level") or scenario.get("noise_id"),
        "model_id": scenario.get("model_id"),
        "strategy": scenario.get("strategy"),
        "proposed_fix": None,
        "decision": None,
        "execution_attempted": False,
        "remediation_succeeded": False,
        "state_verified": False,
        "software_version": meta["software_version"],
        "config_fingerprint": meta["config_fingerprint"],
        "python": meta["python"],
        "dataset_recorded_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
    }


def run_with_timeout(fn, seconds):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()

    def _handler(_signum, _frame):
        raise TimeoutError("episode timed out")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def persist_record(record, scenario, raw_dir, processed_dir):
    """Write-if-absent JSON, append JSONL, rebuild CSV. Never unlink raw files."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path, wrote = write_raw(raw_dir, scenario, record)
    if wrote and record.get("episode_id") not in jsonl_ids(jsonl_path(raw_dir)):
        append_jsonl(jsonl_path(raw_dir), record)
    sync_jsonl(raw_dir)
    write_processed_csv(load_raw_json(raw_dir), processed_dir)
    return path, wrote


def _stub_and_gate(scenario, net=None):
    from bench.run_pilot import _stub_proposal
    from gate.decide import decide

    label = scenario.get("compound_id") or scenario.get("fault_id") or "H0"

    def agent_fn(snapshot, _fault=label):
        return _stub_proposal(snapshot, _fault)

    def gate_fn(episode, _net=net):
        oam_results = (episode.get("oam") or {}).get("results") or []
        return decide(
            _net,
            episode.get("agent_proposal"),
            episode.get("network_snapshot"),
            oam_results,
            mode="pilot_v0" if scenario.get("strategy") == "gate_v0" else "full",
            run_missing_tests=False,
            noise_id=episode.get("noise_id") or scenario.get("noise_id") or "N0",
        )

    return agent_fn, gate_fn


def run_one(net, links, scenario, timeout_seconds=0):
    agent_fn, gate_fn = _stub_and_gate(scenario, net=net)
    fault_id = scenario.get("fault_id") or "H0"
    compound_id = scenario.get("compound_id")

    def _run():
        return run_episode(
            net,
            links,
            fault_id=fault_id,
            noise_id=scenario.get("noise_id") or "N0",
            seed=scenario.get("seed"),
            compound_id=compound_id,
            run_oam=True,
            agent_fn=agent_fn,
            gate_fn=gate_fn,
        )

    try:
        record = run_with_timeout(_run, timeout_seconds)
        return attach_dataset_fields(record, scenario, extra={"status": "ok"})
    except TimeoutError as exc:
        return partial_record(scenario, "timeout", exc)
    except ValueError as exc:
        text = str(exc)
        status = "inject_failed" if "Unknown fault" in text or "held out" in text else "error"
        return partial_record(scenario, status, exc)
    except Exception as exc:
        return partial_record(scenario, "error", f"{type(exc).__name__}: {exc}")


def pending_scenarios(scenarios, raw_dir, state):
    pending, _loaded = resume_pending(scenarios, raw_dir)
    done = set(state.get("completed") or [])
    return [row for row in pending if row.get("scenario_id") not in done]


def _snapshot_frozen():
    return {path: path.read_bytes() if path.exists() else None for path in FROZEN_RESULTS}


def _assert_frozen(before):
    after = _snapshot_frozen()
    for path, original in before.items():
        assert after.get(path) == original, f"must not modify {path}"


def required_fields(record):
    return [
        "episode_id",
        "seed",
        "topology",
        "fault_id",
        "traffic_level",
        "strategy",
        "software_version",
        "config_fingerprint",
        "status",
    ]


def _self_check():
    before = _snapshot_frozen()
    meta = provenance()
    assert meta["software_version"]
    assert meta["config_fingerprint"]
    assert len(meta["config_fingerprint"]) == 64

    scenario = {
        "scenario_id": "1|diamond|validation|normal|F1|N0|stub|none|verifix|none",
        "seed": 1,
        "topology": "diamond",
        "data_split": "validation",
        "mode": "normal",
        "fault_id": "F1",
        "compound_id": None,
        "noise_id": "N0",
        "traffic_level": "N0",
        "agent_mode": "stub",
        "model_id": None,
        "strategy": "verifix",
    }
    record = attach_dataset_fields(
        {
            "episode_id": "ep-1",
            "seed": 1,
            "fault_id": "F1",
            "noise_id": "N0",
            "agent_proposal": {"action": "REROUTE"},
            "gate_decision": {"decision": "allow", "tests_run": ["T1_e2e_ping"]},
            "oam": {"results": [{"test_id": "T1_e2e_ping"}]},
            "oam_cost": {"total_duration_seconds": 2.5},
            "gate_cost": {"latency_seconds": 0.3},
            "remediation_succeeded": True,
            "execution_attempted": True,
        },
        scenario,
    )
    for key in required_fields(record):
        assert record.get(key) not in (None, ""), key
    assert record["proposed_fix"] == "REROUTE"
    assert record["decision"] == "allow"
    assert record["traffic_level"] == "N0"

    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw"
    processed = tmp / "processed"
    path, wrote = persist_record(record, scenario, raw, processed)
    assert wrote and path.exists()
    jsonl = jsonl_path(raw)
    first = jsonl.read_text(encoding="utf-8")
    assert first.count("\n") == 1
    path2, wrote2 = persist_record(dict(record, proposed_fix="SHOULD_NOT_CLOBBER"), scenario, raw, processed)
    assert wrote2 is False
    assert jsonl.read_text(encoding="utf-8") == first
    assert "SHOULD_NOT_CLOBBER" not in path.read_text(encoding="utf-8")

    failed = partial_record(dict(scenario, scenario_id=scenario["scenario_id"] + "|fail", fault_id="NOPE"), "inject_failed", "Unknown fault id 'NOPE'")
    persist_record(failed, failed["scenario"], raw, processed)
    timeout = partial_record(dict(scenario, scenario_id=scenario["scenario_id"] + "|timeout"), "timeout", "episode timed out")
    persist_record(timeout, timeout["scenario"], raw, processed)
    rows = load_raw_json(raw)
    assert any(row["status"] == "inject_failed" for row in rows)
    assert any(row["status"] == "timeout" for row in rows)
    table = csv_path(processed).read_text(encoding="utf-8")
    assert "ep-1" in table and "inject_failed" in table and "timeout" in table
    assert jsonl.read_text(encoding="utf-8").count("\n") == 3

    extra = attach_dataset_fields(dict(record, episode_id="ep-orphan"), dict(scenario, scenario_id="orphan"))
    save_episode(extra, raw)
    sync_jsonl(raw)
    assert extra["episode_id"] in jsonl_ids(jsonl)

    hidden = held_out_ids()
    overnight = [row for row in generate_matrix(seeds=(17,)) if (row.get("compound_id") or row.get("fault_id")) not in hidden]
    assert overnight
    assert all((row.get("compound_id") or row.get("fault_id")) not in hidden for row in overnight)

    agent_fn, gate_fn = _stub_and_gate({"fault_id": "H0", "strategy": "gate_v0", "noise_id": "N0"}, net=None)
    proposal = agent_fn({"metrics": {"packet_loss_percent": 0}})
    assert proposal.get("action")
    try:
        gate_fn(
            {
                "agent_proposal": proposal,
                "network_snapshot": {"links": [], "metrics": {"packet_loss_percent": 0, "connectivity": True, "active_path": "primary"}},
                "oam": {"results": []},
                "noise_id": "N0",
            }
        )
    except NameError:
        raise
    except Exception:
        pass

    def boom():
        raise TimeoutError("episode timed out")

    try:
        run_with_timeout(boom, 0)
        raise AssertionError("timeout helper")
    except TimeoutError:
        pass

    for item in raw.iterdir():
        item.unlink()
    if processed.exists():
        for item in processed.iterdir():
            item.unlink()
        processed.rmdir()
    raw.rmdir()
    tmp.rmdir()

    _assert_frozen(before)
    print(f"live-dataset build: {LIVE_DATASET_BUILD}")
    print("master dataset: PASS")
    return 0


def live_dataset():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("master dataset must run as root on the Mininet VM.", file=sys.stderr)
        return 2
    from bench import inject as inj
    from bench.mininet_util import mn_cleanup
    from network.topology import create_network

    print(f"live-dataset build: {LIVE_DATASET_BUILD}")
    before = _snapshot_frozen()
    failures = []
    raw = SMOKE_DIR / "raw"
    processed = SMOKE_DIR / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    plan = [
        row
        for row in generate_matrix(seeds=(17,))
        if row["topology"] == "diamond"
        and row["agent_mode"] == "stub"
        and row["strategy"] == "verifix"
        and row["fault_id"] in {"H0", "F12"}
        and row.get("compound_id") is None
        and row["noise_id"] == "N0"
        and row["mode"] == "normal"
    ][:2]
    if len(plan) < 2:
        plan = generate_matrix(seeds=(17,), include_large=False)[:2]
    bogus = dict(
        plan[0],
        scenario_id=plan[0]["scenario_id"] + "|bogus",
        fault_id="NOT_A_FAULT",
        compound_id=None,
    )
    mn_cleanup()
    state = {"net": None}
    try:
        topo = create_network()
        state["net"] = topo["net"]
        net, links = topo["net"], topo["links"]
        batch_state = load_state(SMOKE_DIR)
        first = run_one(net, links, plan[0], timeout_seconds=180)
        persist_record(first, plan[0], raw, processed)
        batch_state["completed"] = list(batch_state.get("completed") or []) + [plan[0]["scenario_id"]]
        batch_state["episode_ids"] = list(batch_state.get("episode_ids") or []) + [first["episode_id"]]
        save_state(SMOKE_DIR, batch_state)
        marker = jsonl_path(raw).read_text(encoding="utf-8")
        again, wrote_again = persist_record(dict(first, status="should-not-clobber"), plan[0], raw, processed)
        if wrote_again or jsonl_path(raw).read_text(encoding="utf-8") != marker:
            failures.append("raw overwritten")
        skipped = pending_scenarios(plan, raw, batch_state)
        if plan[0]["scenario_id"] in {row["scenario_id"] for row in skipped}:
            failures.append("resume replayed a finished scenario")
        second = run_one(net, links, plan[1], timeout_seconds=180)
        persist_record(second, plan[1], raw, processed)
        failed = run_one(net, links, bogus, timeout_seconds=30)
        persist_record(failed, bogus, raw, processed)
        if failed.get("status") not in {"inject_failed", "error"}:
            failures.append(f"expected inject failure, got {failed.get('status')} {failed.get('error')}")
        records = load_raw_json(raw)
        if len(records) < 3:
            failures.append(f"raw count {len(records)}")
        for rec in records:
            missing = [key for key in required_fields(rec) if rec.get(key) in (None, "")]
            if missing:
                failures.append(f"{rec.get('episode_id')} missing {missing}")
        if not csv_path(processed).exists():
            failures.append("processed csv missing")
        lines = [line for line in jsonl_path(raw).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 3:
            failures.append(f"jsonl {len(lines)}")
        if first.get("status") != "ok":
            failures.append(f"first status {first.get('status')} {first.get('error')}")
        _assert_frozen(before)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    finally:
        net = state.get("net")
        if net is not None:
            try:
                inj.reset_all(net, baseline=True)
            except Exception:
                pass
            try:
                net.stop()
            except Exception:
                pass
        mn_cleanup()

    if failures:
        print("master dataset: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("master dataset: PASS")
    return 0


def overnight(args):
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("run_batch.py requires root on Mininet VM.", file=sys.stderr)
        return 2
    from bench import inject as inj
    from bench.mininet_util import mn_cleanup, ensure_iperf3_server
    from network.topology import create_network

    hidden = held_out_ids()
    out_dir = Path(args.out_dir)
    raw = out_dir / "raw"
    processed = out_dir / "processed"
    scenarios = []
    rng = random.Random(args.seed)
    catalog = [fid for fid in inj._FAULTS if fid not in hidden]
    while len(scenarios) < args.count:
        fault_id = rng.choice(catalog)
        seed = rng.randint(0, 2**31 - 1)
        scenarios.append(
            {
                "scenario_id": f"{seed}|diamond|evaluation|normal|{fault_id}|{args.noise}|stub|none|verifix|none",
                "seed": seed,
                "topology": "diamond",
                "data_split": "evaluation",
                "mode": "normal",
                "fault_id": fault_id,
                "compound_id": None,
                "noise_id": args.noise,
                "traffic_level": args.noise,
                "agent_mode": "stub",
                "model_id": None,
                "strategy": "verifix",
            }
        )
    batch_state = load_state(out_dir) if args.resume else {"completed": [], "episode_ids": []}
    pending = pending_scenarios(scenarios, raw, batch_state)
    mn_cleanup()
    topo = create_network()
    net, links = topo["net"], topo["links"]
    ensure_iperf3_server(net)
    try:
        for scen in pending:
            record = run_one(net, links, scen, timeout_seconds=args.timeout)
            persist_record(record, scen, raw, processed)
            batch_state["completed"] = list(batch_state.get("completed") or []) + [scen["scenario_id"]]
            batch_state["episode_ids"] = list(batch_state.get("episode_ids") or []) + [record["episode_id"]]
            save_state(out_dir, batch_state)
            print(f"[{len(batch_state['completed'])}/{args.count}] {record.get('status')} {scen['fault_id']} {record['episode_id']}")
    except KeyboardInterrupt:
        save_state(out_dir, batch_state)
        print("interrupted; raw JSONL preserved", file=sys.stderr)
        return 130
    finally:
        inj.reset_all(net, baseline=True)
        net.stop()
        mn_cleanup()
    print(f"Done. {len(batch_state['episode_ids'])} episodes in {raw}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--live-dataset", action="store_true")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--noise", default="N0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-dir", default=str(DATASET))
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.live_dataset:
        return live_dataset()
    return overnight(args)


if __name__ == "__main__":
    sys.exit(main())
