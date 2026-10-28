"""Inject each fault and compound, record the OAM outcomes, then try every
remediation and measure whether traffic recovers.

Run on the Mininet VM as root. A full run overwrites
results/observability_matrix.csv, results/ground_truth.json and
results/inseparable_pairs.json with measured data. A --fault run prints and
writes nothing.

    sudo python3 bench/verify_fault_observability.py
    sudo python3 bench/verify_fault_observability.py --fault F3 --fault C2
    python bench/verify_fault_observability.py --self-check    # offline
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import inject as inj
from bench import tests as oam
from bench.outcomes import grade_result, signature
from bench.signature_checks import (
    RECOVERY_TESTS,
    REMEDIATIONS,
    degraded_tests,
    expected_actions,
    fix_pairs,
    load_expected,
    outcome_map,
    separation_errors,
    valid_fix_set,
)
from control.actions import execute
from network import controller as ctl

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 42
NOISE_ID = "N0"
MATRIX_PING_COUNT = 50
RECOVERY_PING_COUNT = 30
SETTLE_SECONDS = 2.0
SPEED_RETRIES = 2
MEASURED_ACTIONS = ("DO_NOTHING",) + REMEDIATIONS
FLAP_TYPES = {"link_flap"}


# Top of each catalog range, so the matrix is not a one-sample coin flip.
# F6/F7 keep background traffic, but on UDP/5202 (see inject.py), so T8/T10
# can still use the iperf3 server on 5201. Compound parts use the same values.
_OBSERVABLE_PARAMS = {
    "F3": {"loss_percent": 20},
    "F4": {"corrupt_percent": 5},
    "F6": {"rate_mbit": 1, "background_traffic": True},
    "F7": {"rate_mbit": 1, "background_traffic": True},
    "F11": {"loss_percent": 20},
    "U1": {"mtu_bytes": 1000},
    "U2": {"loss_percent": 20},
}

# What the action is asked to do. REBALANCE toggles the path, which is the
# backup path when the measurement starts on the primary.
_ACTION_PAYLOAD = {"REROUTE": {"path": "backup"}}


def _inject_params(fault_id):
    return _OBSERVABLE_PARAMS.get(fault_id)


def all_scenarios():
    return list(inj._FAULTS) + list(inj._COMPOUNDS)


def _parts(scenario_id):
    if scenario_id in inj._COMPOUNDS:
        return list(inj._COMPOUNDS[scenario_id]["injection_order"])
    return [scenario_id]


def _split(scenario_id):
    spec = inj._COMPOUNDS.get(scenario_id) or inj._FAULTS.get(scenario_id) or {}
    return spec.get("split")


def _timing_sensitive(scenario_id):
    return any(inj._FAULTS[f].get("injection", {}).get("type") in FLAP_TYPES for f in _parts(scenario_id))


def _far_routes(net):
    return {r: inj._route_line(net, r, cfg["dest"]) for r, cfg in ctl.ROUTE_TARGETS.items()}


def _put_far_routes(net, routes):
    errors = []
    for router, line in routes.items():
        try:
            inj._put_route(net, router, ctl.ROUTE_TARGETS[router]["dest"], line)
        except Exception as exc:
            errors.append(f"{router}: {type(exc).__name__}: {exc}")
    return errors


def _fresh_host_state(net):
    """Drop learned path MTUs and TCP metrics so one measurement does not shape the next.

    Linux keeps a destination's slow-start threshold after each connection,
    so a lossy transfer makes the next 3 s iperf3 slow on a healthy path.
    """
    for name in inj.NODES:
        inj._sh(net.get(name), "ip route flush cache")
    for name in ("hostA", "hostB"):
        inj._sh(net.get(name), "ip tcp_metrics flush all")


def _measure(net, test_ids=None, **kwargs):
    """(results, T10 Mbit/s samples). A slow T10 is retried and the fastest kept:
    noise only lowers a throughput sample, a fault keeps every one low."""
    _fresh_host_state(net)
    results = oam.run_all(net, test_ids=test_ids, **kwargs)["results"]
    samples = []
    for i, result in enumerate(results):
        if result.get("test_id") != "T10_speed":
            continue
        best = result
        samples.append(result.get("mbit_per_sec"))
        for _ in range(SPEED_RETRIES):
            if grade_result(best, noise_id=NOISE_ID) == "ok":
                break
            _fresh_host_state(net)
            again = oam.run_all(net, test_ids=["T10_speed"])["results"][0]
            samples.append(again.get("mbit_per_sec"))
            if (again.get("mbit_per_sec") or -1) > (best.get("mbit_per_sec") or -1):
                best = again
        results[i] = best
    return results, samples


def _raw(results, speed_samples):
    """Numbers behind the grades, for reading a result without rerunning it."""
    by_id = {r.get("test_id"): r for r in results}
    raw = {"T10_mbit_samples": speed_samples}
    t1 = by_id.get("T1_e2e_ping") or {}
    if t1:
        raw.update(T1_loss_percent=t1.get("packet_loss_percent"), T1_latency_ms=t1.get("latency_ms"))
    t9 = by_id.get("T9_oneway_loss") or {}
    if t9:
        raw.update(
            T9_a2b_loss_percent=(t9.get("hostA_to_hostB") or {}).get("packet_loss_percent"),
            T9_b2a_loss_percent=(t9.get("hostB_to_hostA") or {}).get("packet_loss_percent"),
        )
    errors = {tid: r["error"] for tid, r in by_id.items() if r.get("error")}
    if errors:
        raw["errors"] = errors
    return raw


def measure_action(net, action):
    """Run one action, grade the recovery tests, then put r1/r4's routes back.

    recovered is None when the action or the measurement raised.
    """
    before = _far_routes(net)
    out = {"executed": False, "recovered": None}
    try:
        out.update(execute(net, action, dict(_ACTION_PAYLOAD.get(action, {}))))
        if out["executed"]:
            time.sleep(SETTLE_SECONDS)
        out["path_after"] = {r: ctl.path_from_route(r, line) for r, line in _far_routes(net).items()}
        results, speed = _measure(net, test_ids=list(RECOVERY_TESTS), ping_count=RECOVERY_PING_COUNT)
        out["raw"] = _raw(results, speed)
        out["outcomes"] = signature(results, noise_id=NOISE_ID)
        out["degraded"] = degraded_tests(out["outcomes"])
        out["recovered"] = not out["degraded"]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        restore_errors = _put_far_routes(net, before)
        if restore_errors:
            out["restore_errors"] = restore_errors
        try:
            out["routes_restored"] = _far_routes(net) == before
        except Exception as exc:
            out["routes_restored"] = False
            out.setdefault("restore_errors", []).append(f"readback: {type(exc).__name__}: {exc}")
    return out


def _inject_scenario(net, scenario_id):
    """Inject a fault, or a compound's parts in injection_order. A failure resets the parts already in."""
    cids = []
    try:
        for fault_id in _parts(scenario_id):
            cids.append(inj.inject(net, fault_id, params=_inject_params(fault_id), seed=SEED))
    except Exception:
        for cid in reversed(cids):
            try:
                inj.reset(cid)
            except Exception:
                pass
        raise
    return cids


def _reset_scenario(net, scenario_id, cids):
    if scenario_id in inj._COMPOUNDS:
        return inj.reset_compound(net, scenario_id, cids)
    return [inj.reset(cid) for cid in cids]


def measure_scenario(net, scenario_id):
    """Matrix battery, then each action from the same faulted state, then reset and compare."""
    record = {
        "scenario": scenario_id,
        "kind": "compound" if scenario_id in inj._COMPOUNDS else "fault",
        "split": _split(scenario_id),
        "parts": _parts(scenario_id),
        "timing_sensitive": _timing_sensitive(scenario_id),
        "errors": [],
    }
    before = inj.snapshot(net)
    cids = []
    try:
        cids = _inject_scenario(net, scenario_id)
        results, speed = _measure(net, ping_count=MATRIX_PING_COUNT, error_probe=80)
        record["raw"] = _raw(results, speed)
        record["signature"] = signature(results, noise_id=NOISE_ID)
        record["actions"] = {action: measure_action(net, action) for action in MEASURED_ACTIONS}
        record["actions"]["ESCALATE"] = {"executed": False, "note": "no network change; traffic is as under DO_NOTHING"}
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            _reset_scenario(net, scenario_id, cids)
        except Exception as exc:
            record["errors"].append(f"reset: {type(exc).__name__}: {exc}")
        cleanup = inj.reset_all(net)
        if cleanup["errors"]:
            record["errors"].extend(f"reset_all: {e['error']}" for e in cleanup["errors"])
        residual = inj.diff_snapshots(before, inj.snapshot(net))
        if residual:
            record["residual"] = residual
            try:
                inj.restore_baseline(net)
            except Exception as exc:
                record["errors"].append(f"restore_baseline: {type(exc).__name__}: {exc}")
        record["health_loss_percent"] = cleanup["loss_percent"]
    return record


def _load_catalog_fixes():
    with (ROOT / "config" / "correct_fixes.yaml").open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return {**(doc.get("faults") or {}), **(doc.get("compounds") or {})}


def build_report(records, catalog=None):
    """Matrix rows, measured fix sets, fix pairs and per-scenario expected actions."""
    catalog = _load_catalog_fixes() if catalog is None else catalog
    rows = [
        {"fault_id": r["scenario"], "test_id": tid, "outcome": outcome}
        for r in records
        for tid, outcome in (r.get("signature") or {}).items()
    ]
    measured = {r["scenario"]: valid_fix_set(r["actions"]) if r.get("actions") else None for r in records}
    by_fault = {sid: tests for sid, tests in outcome_map(rows).items() if measured.get(sid)}
    pairs = fix_pairs(by_fault, measured)
    expected = expected_actions(measured, pairs)
    held_out = inj.held_out_ids()
    for pair in pairs:
        pair["held_out"] = pair["fault_a"] in held_out or pair["fault_b"] in held_out

    scenarios = {}
    for r in records:
        sid = r["scenario"]
        catalog_fixes = (catalog.get(sid) or {}).get("correct")
        scenarios[sid] = {
            **{k: r.get(k) for k in ("kind", "split", "parts", "timing_sensitive", "signature", "raw", "actions", "errors", "health_loss_percent")},
            "residual": r.get("residual", []),
            "measured_valid_fixes": measured[sid],
            "inseparable_from": sorted(
                p["fault_b"] if p["fault_a"] == sid else p["fault_a"]
                for p in pairs
                if p["inseparable"] and sid in (p["fault_a"], p["fault_b"])
            ),
            "expected_actions": expected.get(sid),
            "catalog_fixes": catalog_fixes,
            "matches_catalog": measured[sid] is not None and set(measured[sid]) == set(catalog_fixes or []),
        }
    return {"rows": rows, "scenarios": scenarios, "pairs": pairs}


def run_failures(report, spec, full_run=True):
    """Reasons the measured run cannot be used. Catalog mismatches are reported, not failures."""
    failures = []
    for sid, s in report["scenarios"].items():
        failures.extend(f"{sid}: {err}" for err in s["errors"])
        if s["residual"]:
            failures.append(f"{sid}: reset left {len(s['residual'])} difference(s), e.g. {s['residual'][0]}")
        if s["measured_valid_fixes"] is None:
            failures.append(f"{sid}: no valid fix set (an action measurement failed)")
        for action, result in (s["actions"] or {}).items():
            if result.get("error"):
                failures.append(f"{sid} {action}: {result['error']}")
            # A running flap rewrites r1/r4 routes itself; the reset diff above still checks it.
            if s["timing_sensitive"]:
                continue
            if result.get("restore_errors") or result.get("routes_restored") is False:
                failures.append(f"{sid} {action}: r1/r4 routes not restored: {result.get('restore_errors')}")
    if full_run:
        failures.extend(separation_errors(report["rows"], spec))
    return failures


def write_outputs(report, results_dir=RESULTS):
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "observability_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fault_id", "test_id", "outcome"])
        writer.writeheader()
        writer.writerows(report["rows"])
    ground_truth = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "noise_id": NOISE_ID,
        "seed": SEED,
        "params": _OBSERVABLE_PARAMS,
        "recovery_tests": list(RECOVERY_TESTS),
        "rules": {
            "recovered": "T1, T4 and T8 read healthy; T10 is recorded but a slow sample alone is not a failed recovery",
            "valid_fixes": "DO_NOTHING alone if traffic is fine unchanged; else each remediation that recovered; else ESCALATE",
            "different_fixes": "the two valid fix sets share no action",
            "inseparable": "different fixes and no matrix test differs; expected action ESCALATE",
        },
        "scenarios": report["scenarios"],
        "pairs": report["pairs"],
    }
    (results_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
    (results_dir / "inseparable_pairs.json").write_text(json.dumps(report["pairs"], indent=2) + "\n", encoding="utf-8")


def print_report(report):
    print("\nScenario   measured fixes            expected              catalog")
    for sid, s in report["scenarios"].items():
        measured = ",".join(s["measured_valid_fixes"] or ["?"])
        expected = ",".join(s["expected_actions"] or ["?"])
        catalog = ",".join(s["catalog_fixes"] or ["-"])
        flags = ("" if s["matches_catalog"] else "  differs from catalog") + ("  (flap: timing sensitive)" if s["timing_sensitive"] else "")
        print(f"  {sid:<8} {measured:<25} {expected:<21} {catalog}{flags}")
    inseparable = [p for p in report["pairs"] if p["inseparable"]]
    print(f"\n{len(report['pairs'])} pair(s) need different fixes; {len(inseparable)} inseparable (expected ESCALATE)")
    for p in inseparable:
        print(f"  {p['fault_a']} {p['fixes_a']} vs {p['fault_b']} {p['fixes_b']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fault", action="append", help="scenario id; repeatable. Prints only, writes nothing")
    parser.add_argument("--self-check", action="store_true", help="offline check of the runner with a fake network")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("verify_fault_observability.py must run as root on the Mininet VM.", file=sys.stderr)
        return 2

    known = all_scenarios()
    scenarios = args.fault or known
    unknown = [s for s in scenarios if s not in known]
    if unknown:
        print(f"Unknown scenario(s) {unknown}; known: {known}", file=sys.stderr)
        return 2

    from network.topology import create_network
    from bench.mininet_util import ensure_iperf3_server

    spec = load_expected()
    state = create_network()
    net = state["net"]
    records = []
    try:
        ensure_iperf3_server(net)
        inj.reset_all(net)
        for sid in scenarios:
            print(f"scenario {sid} ...", flush=True)
            record = measure_scenario(net, sid)
            records.append(record)
            for tid, outcome in (record.get("signature") or {}).items():
                print(f"  {tid}: {outcome}", flush=True)
            print(f"  raw: {record.get('raw')}", flush=True)
            for action, result in (record.get("actions") or {}).items():
                if action in MEASURED_ACTIONS:
                    status = {True: "recovered", False: f"not recovered {result.get('degraded')}", None: f"error {result.get('error')}"}[result["recovered"]]
                    print(f"  {action}: path {result.get('path_after')} {status} {result.get('raw')}", flush=True)
            for err in record["errors"] + record.get("residual", []):
                print(f"  ! {err}", flush=True)
    finally:
        try:
            inj.reset_all(net, baseline=True)
        finally:
            net.stop()

    report = build_report(records)
    print_report(report)
    failures = run_failures(report, spec, full_run=not args.fault)
    if args.fault:
        print("\nPartial run: nothing written.")
    else:
        write_outputs(report)
        print(f"\nWrote {RESULTS / 'observability_matrix.csv'} ({len(report['rows'])} rows), ground_truth.json, inseparable_pairs.json")
    if failures:
        print("\nObservability and remediation checks failed:")
        for err in failures:
            print(f"  {err}")
        return 1
    print("Every scenario measured; each required fault changed the expected tests and required pairs are separated.")
    return 0


# ---------------------------------------------------------------------------
# Offline self-check: a fake network where the primary path is broken by
# "primary" faults and the backup by "backup" faults.
# ---------------------------------------------------------------------------

def _self_check():
    import tempfile

    primary = {r: f"{cfg['dest']} via {cfg['primary']['via']} dev {cfg['primary']['dev']}" for r, cfg in ctl.ROUTE_TARGETS.items()}
    backup = {r: f"{cfg['dest']} via {cfg['backup']['via']} dev {cfg['backup']['dev']}" for r, cfg in ctl.ROUTE_TARGETS.items()}
    # Fake fault effects: which path it breaks, whether the battery shows it, whether TCP fails everywhere.
    effects = {"H0": set(), "F1": {"primary"}, "F12": {"backup"}, "F10": {"tcp"}, "F3": {"primary"}, "F11": {"last_mile"}}
    world = {"routes": dict(primary), "active": [], "log": [], "fail_battery": set(), "fail_restore": False,
             "slow_speeds": [], "flushes": 0}

    def fake_inject(net, fault_id, params=None, seed=None, purpose="evaluation"):
        world["active"].append(fault_id)
        world["log"].append(f"inject {fault_id}")
        return f"cid-{fault_id}"

    def fake_reset(cid):
        world["active"].remove(cid[4:])
        world["log"].append(f"reset {cid[4:]}")
        return {"reset": cid}

    def fake_reset_all(net, baseline=False):
        if baseline:
            world["routes"] = dict(primary)
        return {"errors": [], "loss_percent": 0.0}

    def fake_snapshot(net, nodes=None):
        return {"routes": dict(world["routes"]), "active": list(world["active"])}

    def fake_diff(before, after):
        return [] if before == after else [f"{before} != {after}"]

    def fake_route_line(net, node, destination):
        return world["routes"].get(node, "")

    def fake_put_route(net, node, destination, original):
        if world["fail_restore"]:
            raise RuntimeError("route replace failed")
        world["routes"][node] = original

    def fake_execute(net, action, proposal=None):
        world["log"].append(f"execute {action}")
        if action == "DO_NOTHING":
            return {"executed": False, "action": action}
        if world["routes"]["r1"] == primary["r1"]:
            world["routes"] = dict(backup)
        return {"executed": True, "action": action, "path": "backup"}

    def broken():
        on = "backup" if world["routes"]["r1"] == backup["r1"] else "primary"
        hits = set().union(*(effects.get(f, set()) for f in world["active"])) if world["active"] else set()
        return on in hits or "last_mile" in hits, "tcp" in hits

    def fake_run_all(net, test_ids=None, ping_count=None, error_probe=None, **_):
        if set(world["active"]) & world["fail_battery"]:
            raise RuntimeError("battery failed")
        down, tcp_fail = broken()
        ids = test_ids or oam.DEFAULT_ORDER
        tag = "+".join(sorted(world["active"])) or "H0"
        results = []
        for tid in ids:
            if tid == "T1_e2e_ping":
                results.append({"test_id": tid, "packet_loss_percent": 100 if down else 0, "latency_ms": None if down else 20})
            elif tid == "T8_tcp_vs_ping":
                results.append({"test_id": tid, "ping_ok": not down, "tcp_ok": not (down or tcp_fail)})
            elif tid == "T10_speed":
                mbit = 0.0 if (down or tcp_fail) else (world["slow_speeds"].pop(0) if world["slow_speeds"] else 9.5)
                results.append({"test_id": tid, "mbit_per_sec": mbit})
            elif tid == "T4_mtu":
                results.append({"test_id": tid, "df_ping_ok": not down})
            elif tid == "T7_route_check":
                # F3 and F1 read the same here, so only an inseparable pair can come from them.
                key = "F1" if tag in ("F1", "F3") else tag
                results.append({"test_id": tid, "r1_to_dest": key, "r4_to_src": "", "active_path": "primary"})
        return {"results": results}

    def fake_signature(results, thresholds=None, noise_id="N0"):
        out = {}
        for r in results:
            tid = r["test_id"]
            if tid == "T7_route_check":
                out[tid] = f"seen:{r['r1_to_dest']}"
            elif tid == "T1_e2e_ping":
                out[tid] = "down" if r["packet_loss_percent"] == 100 else "ok"
            elif tid == "T8_tcp_vs_ping":
                out[tid] = "both_ok" if r["tcp_ok"] else ("ping_ok_tcp_fail" if r["ping_ok"] else "both_fail")
            elif tid == "T10_speed":
                out[tid] = "ok" if r["mbit_per_sec"] >= 1 else "slow"
            elif tid == "T4_mtu":
                out[tid] = "df_ok" if r["df_ping_ok"] else "mtu_fail"
        return out

    def fake_fresh(net):
        world["flushes"] += 1

    g = globals()
    patches = [
        (inj, "inject", fake_inject), (inj, "reset", fake_reset),
        (inj, "reset_all", fake_reset_all), (inj, "snapshot", fake_snapshot), (inj, "diff_snapshots", fake_diff),
        (inj, "restore_baseline", lambda net: None), (inj, "_route_line", fake_route_line), (inj, "_put_route", fake_put_route),
        (oam, "run_all", fake_run_all), (time, "sleep", lambda s: None),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    saved_globals = {k: g[k] for k in ("execute", "signature", "_fresh_host_state")}
    saved_compounds = dict(inj._COMPOUNDS)

    def check(label, ok, detail=""):
        if not ok:
            raise AssertionError(f"{label}: {detail}")
        print(f"  PASS  {label}")

    try:
        for mod, name, fake in patches:
            setattr(mod, name, fake)
        g["execute"], g["signature"], g["_fresh_host_state"] = fake_execute, fake_signature, fake_fresh
        inj._COMPOUNDS["CX"] = {"id": "CX", "split": "development", "injection_order": ["F1", "F12"], "cleanup_order": ["F12", "F1"]}

        records = [measure_scenario(None, sid) for sid in ("H0", "F1", "F3", "F12", "F10", "F11")]
        start = len(world["log"])
        records.append(measure_scenario(None, "CX"))
        by_id = {r["scenario"]: r for r in records}
        check("every action is measured and ESCALATE is recorded", all(set(r["actions"]) == {"DO_NOTHING", "REROUTE", "REBALANCE", "ESCALATE"} for r in records))
        check("routes are back on the primary after every action", world["routes"] == primary and all(a.get("routes_restored") for r in records for n, a in r["actions"].items() if n != "ESCALATE"))
        check("no fault is left active", world["active"] == [], str(world["active"]))
        check("compound: parts in injection order, actions, reset in cleanup order",
              world["log"][start:] == ["inject F1", "inject F12", "execute DO_NOTHING", "execute REROUTE", "execute REBALANCE", "reset F12", "reset F1"],
              str(world["log"][start:]))
        check("primary fault recovers by reroute", by_id["F1"]["actions"]["REROUTE"]["recovered"] and not by_id["F1"]["actions"]["DO_NOTHING"]["recovered"])
        check("backup fault: reroute is not a recovery", not by_id["F12"]["actions"]["REROUTE"]["recovered"] and by_id["F12"]["actions"]["DO_NOTHING"]["recovered"])

        report = build_report(records, catalog={"H0": {"correct": ["DO_NOTHING"]}, "F1": {"correct": ["REBALANCE", "REROUTE"]}, "F3": {"correct": ["REROUTE"]}})
        s = report["scenarios"]
        check("healthy and backup-down need no change", s["H0"]["measured_valid_fixes"] == ["DO_NOTHING"] and s["F12"]["measured_valid_fixes"] == ["DO_NOTHING"])
        check("primary loss is fixed by REROUTE and REBALANCE", s["F1"]["measured_valid_fixes"] == ["REROUTE", "REBALANCE"], str(s["F1"]["measured_valid_fixes"]))
        check("no fix works: ESCALATE", all(s[x]["measured_valid_fixes"] == ["ESCALATE"] for x in ("F10", "F11", "CX")), str({x: s[x]["measured_valid_fixes"] for x in ("F10", "F11", "CX")}))
        check("catalog comparison", s["H0"]["matches_catalog"] and s["F1"]["matches_catalog"] and not s["F3"]["matches_catalog"])
        pair_keys = {(p["fault_a"], p["fault_b"]): p for p in report["pairs"]}
        check("pairs with a shared fix are skipped", ("F1", "F3") not in pair_keys and ("H0", "F12") not in pair_keys)
        check("different fixes separated by a test", pair_keys[("H0", "F1")]["separating_tests"] and not pair_keys[("H0", "F1")]["inseparable"])
        check("matrix rows cover every scenario", {r["fault_id"] for r in report["rows"]} == set(by_id))

        # F3 vs F12 is separable here; make F3 read exactly like F11 to force an inseparable pair.
        same = [dict(r) for r in records]
        f3 = next(r for r in same if r["scenario"] == "F3")
        f3["signature"] = dict(next(r for r in same if r["scenario"] == "F11")["signature"])
        report = build_report(same, catalog={})
        s = report["scenarios"]
        check("inseparable pair expects ESCALATE", s["F3"]["expected_actions"] == ["ESCALATE"] and "F11" in s["F3"]["inseparable_from"], str(s["F3"]))
        check("its partner keeps ESCALATE", s["F11"]["expected_actions"] == ["ESCALATE"])
        check("separable scenarios keep their measured fixes", s["F1"]["expected_actions"] == ["REROUTE", "REBALANCE"])
        check("a full run with no errors has no failures besides signature rules",
              run_failures(report, {"faults": {}, "pairs": []}) == [], str(run_failures(report, {"faults": {}, "pairs": []})))

        check("host caches are flushed before the battery and every action", world["flushes"] >= 4 * len(records), str(world["flushes"]))

        # The battery takes the first sample, DO_NOTHING the next ones.
        world["slow_speeds"] = [9.5, 3.0]
        healthy = measure_scenario(None, "H0")
        samples = healthy["actions"]["DO_NOTHING"]["raw"]["T10_mbit_samples"]
        check("a single slow T10 sample is retried and the best kept", samples == [3.0, 9.5] and healthy["actions"]["DO_NOTHING"]["recovered"], str(samples))
        world["slow_speeds"] = [9.5, 0.5, 0.3, 0.6]
        healthy = measure_scenario(None, "H0")
        samples = healthy["actions"]["DO_NOTHING"]["raw"]["T10_mbit_samples"]
        check("T10 slow on every retry stays slow but ping/TCP still count as recovered", samples == [0.5, 0.3, 0.6] and healthy["actions"]["DO_NOTHING"]["recovered"], str(samples))
        check("the battery's T10 is retried the same way", healthy["raw"]["T10_mbit_samples"] == [9.5])
        world["slow_speeds"] = []
        check("a broken path is not retried into recovery", by_id["F1"]["actions"]["DO_NOTHING"]["raw"]["T10_mbit_samples"] == [0.0, 0.0, 0.0])

        world["fail_battery"] = {"F1"}
        failed = measure_scenario(None, "F1")
        world["fail_battery"] = set()
        check("a failed battery still resets the fault", world["active"] == [] and failed["errors"] and "actions" not in failed, str(failed))
        report = build_report([failed], catalog={})
        check("a failed scenario is a run failure", any("battery failed" in f for f in run_failures(report, {}, full_run=False)))

        world["fail_restore"] = True
        broken_restore = measure_scenario(None, "F1")
        world["fail_restore"] = False
        world["routes"] = dict(primary)
        bad = [n for n, a in broken_restore["actions"].items() if a.get("restore_errors")]
        check("a failed route restore is recorded for each action", bad == list(MEASURED_ACTIONS), str(bad))
        check("and the leftover backup route is caught by the scenario diff", broken_restore.get("residual"), str(broken_restore))
        check("and fails the run", any("routes not restored" in f for f in run_failures(build_report([broken_restore], catalog={}), {}, full_run=False)))

        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(build_report(records, catalog={}), Path(tmp))
            rows = list(csv.DictReader((Path(tmp) / "observability_matrix.csv").open(encoding="utf-8")))
            gt = json.loads((Path(tmp) / "ground_truth.json").read_text(encoding="utf-8"))
            pairs = json.loads((Path(tmp) / "inseparable_pairs.json").read_text(encoding="utf-8"))
            check("outputs keep the matrix schema the gate reads", rows and set(rows[0]) == {"fault_id", "test_id", "outcome"})
            check("ground truth lists every scenario", set(gt["scenarios"]) == set(by_id) and gt["noise_id"] == "N0")
            check("pairs file uses the inseparable_pairs schema", all({"fault_a", "fault_b", "separating_tests", "inseparable"} <= set(p) for p in pairs))
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)
        g.update(saved_globals)
        inj._COMPOUNDS.clear()
        inj._COMPOUNDS.update(saved_compounds)

    check("real scenario list covers faults and compounds", {"H0", "F1", "U5", "C1", "C5"} <= set(all_scenarios()) and "CX" not in all_scenarios())
    check("flap scenarios are flagged", _timing_sensitive("F8") and not _timing_sensitive("F3"))
    flap = {"F8": {"errors": [], "residual": [], "measured_valid_fixes": ["DO_NOTHING"], "timing_sensitive": True,
                   "actions": {"REROUTE": {"routes_restored": False}}}}
    check("a flap's route race is not a failure", run_failures({"scenarios": flap, "rows": []}, {}, full_run=False) == [])
    flap["F8"]["residual"] = ["r1 routes: + x"]
    check("but a flap's residual is", run_failures({"scenarios": flap, "rows": []}, {}, full_run=False) != [])
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
