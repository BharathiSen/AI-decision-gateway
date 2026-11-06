"""Audit measured recovery labels and build a consensus label file.

    python -m bench.label_audit                       # audit results/ground_truth.json
    python -m bench.label_audit results/labels/*      # combine repeated label runs
    python -m bench.label_audit results/labels/* --write results/labels/consensus.json

A label is TRUSTED only when, within the system's action space
(REROUTE, DO_NOTHING, ESCALATE):
  * it was measured at least MIN_REPEATS times at every traffic level in
    REQUIRED_NOISE (N0, N1, N2),
  * every one of those runs gave the same expected action set,
  * no run had a reset error or residual state for that scenario.
A fault whose measured signature equals the healthy (H0) signature had no
observable effect; it is flagged NO_EFFECT, because its label (DO_NOTHING)
says the injection failed, not that doing nothing fixes a fault.

The consensus file has the same shape as ground_truth.json (`scenarios` with
`expected_actions`), so `bench/labels.py` can score against it. Untrusted
scenarios are written with `trusted: false`; scoring code must report them.
Nothing under results/ is modified unless --write names a new file.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTION_SPACE = ("REROUTE", "DO_NOTHING", "ESCALATE")
REQUIRED_NOISE = ("N0", "N1", "N2")
MIN_REPEATS = 3


def _scoped(actions):
    usable = sorted(set(actions or []) & set(ACTION_SPACE))
    return usable or ["ESCALATE"]


def _load_run(path):
    path = Path(path)
    gt_path = path / "ground_truth.json" if path.is_dir() else path
    raw = gt_path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    return {
        "path": str(gt_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "noise_id": doc.get("noise_id") or "N0",
        "seed": doc.get("seed"),
        "scenarios": doc.get("scenarios") or {},
    }


def audit(paths, min_repeats=MIN_REPEATS, required_noise=REQUIRED_NOISE):
    runs = [_load_run(p) for p in paths]
    per = defaultdict(lambda: {"by_noise": defaultdict(list), "dirty": 0, "split": None, "no_effect": 0, "runs": 0})
    for run in runs:
        healthy = (run["scenarios"].get("H0") or {}).get("signature")
        for sid, row in run["scenarios"].items():
            entry = per[sid]
            entry["runs"] += 1
            entry["split"] = row.get("split")
            entry["by_noise"][run["noise_id"]].append(tuple(_scoped(row.get("expected_actions"))))
            if row.get("errors") or row.get("residual"):
                entry["dirty"] += 1
            if sid != "H0" and healthy and row.get("signature") == healthy:
                entry["no_effect"] += 1
    table = {}
    for sid in sorted(per):
        entry = per[sid]
        labels = {label for values in entry["by_noise"].values() for label in values}
        counts = {noise: len(entry["by_noise"].get(noise, [])) for noise in required_noise}
        problems = []
        short = [f"{noise} {counts[noise]}/{min_repeats}" for noise in required_noise if counts[noise] < min_repeats]
        if short:
            problems.append("too few repeats: " + ", ".join(short))
        if len(labels) > 1:
            problems.append("runs disagree: " + " vs ".join("/".join(label) for label in sorted(labels)))
        if entry["dirty"]:
            problems.append(f"{entry['dirty']} run(s) with reset errors or residual state")
        if entry["no_effect"]:
            problems.append(f"NO_EFFECT in {entry['no_effect']} run(s): signature equals healthy H0")
        consensus = sorted(labels)[0] if len(labels) == 1 else ("ESCALATE",)
        table[sid] = {
            "split": entry["split"],
            "expected_actions": list(consensus),
            "observed_labels": sorted("/".join(label) for label in labels),
            "repeats": counts,
            "trusted": not problems,
            "problems": problems,
        }
    return {"runs": [{k: r[k] for k in ("path", "sha256", "noise_id", "seed")} for r in runs], "scenarios": table}


def print_audit(result):
    rows = result["scenarios"]
    print(f"{len(result['runs'])} label run(s): " + ", ".join(f"{r['noise_id']}/seed {r['seed']}" for r in result["runs"]))
    print(f"{'id':5} {'split':11} {'label':22} trusted  problems")
    for sid, row in rows.items():
        print(f"{sid:5} {str(row['split']):11} {'/'.join(row['expected_actions']):22} {str(row['trusted']):7}  {'; '.join(row['problems'])}")
    trusted = sum(1 for row in rows.values() if row["trusted"])
    print(f"\n{trusted}/{len(rows)} labels trusted (>= {MIN_REPEATS} agreeing clean runs at each of {', '.join(REQUIRED_NOISE)})")


def write_consensus(result, path):
    from bench.contracts import refuse_protected_path

    path = Path(path)
    refuse_protected_path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; consensus files are never overwritten")
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "label_consensus",
        "noise_id": "+".join(REQUIRED_NOISE),
        "seed": None,
        "rules": {
            "valid_fixes": f"agreeing expected_actions across >= {MIN_REPEATS} clean runs at each of {', '.join(REQUIRED_NOISE)}",
        },
        "sources": result["runs"],
        "scenarios": result["scenarios"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _self_check():
    import tempfile

    def run(tmp, name, noise, seed, scenarios):
        d = Path(tmp) / name
        d.mkdir()
        (d / "ground_truth.json").write_text(json.dumps({"noise_id": noise, "seed": seed, "scenarios": scenarios}), encoding="utf-8")
        return d

    h0 = {"split": "baseline", "signature": {"T1_e2e_ping": "ok"}, "expected_actions": ["DO_NOTHING"]}
    f1 = {"split": "development", "signature": {"T1_e2e_ping": "down"}, "expected_actions": ["REROUTE", "REBALANCE"]}
    f8 = {"split": "development", "signature": {"T1_e2e_ping": "ok"}, "expected_actions": ["DO_NOTHING"]}
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for noise in REQUIRED_NOISE:
            for seed in range(MIN_REPEATS):
                f1_row = dict(f1)
                if noise == "N2" and seed == 2:
                    f1_row = dict(f1, expected_actions=["ESCALATE"])
                dirs.append(run(tmp, f"{noise}-{seed}", noise, seed, {"H0": h0, "F1": f1_row, "F8": f8}))
        result = audit(dirs)
        rows = result["scenarios"]
        assert rows["H0"]["trusted"] and rows["H0"]["expected_actions"] == ["DO_NOTHING"]
        assert not rows["F1"]["trusted"] and "disagree" in rows["F1"]["problems"][0]
        assert rows["F1"]["expected_actions"] == ["ESCALATE"], "disagreement must fall back to ESCALATE"
        assert not rows["F8"]["trusted"] and any("NO_EFFECT" in p for p in rows["F8"]["problems"])
        short = audit(dirs[:2])
        assert not short["scenarios"]["H0"]["trusted"] and "too few repeats" in short["scenarios"]["H0"]["problems"][0]
        out = write_consensus(result, Path(tmp) / "consensus.json")
        try:
            write_consensus(result, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("consensus overwrite allowed")
        from bench.labels import correct_actions

        assert correct_actions("H0", out) == {"DO_NOTHING"} and correct_actions("F1", out) == {"ESCALATE"}
    try:
        write_consensus(result, ROOT / "results" / "ground_truth.json")
    except ValueError:
        pass
    else:
        raise AssertionError("protected ground truth was writable")
    print("label audit: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="label run directories or ground_truth.json files")
    parser.add_argument("--write", help="new consensus file to write")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    paths = args.paths or [ROOT / "results" / "ground_truth.json"]
    result = audit(paths)
    print_audit(result)
    if args.write:
        print(f"wrote {write_consensus(result, args.write)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
