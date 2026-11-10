"""Evaluation labels from measured recovery, not from the gate's catalog.

`config/correct_fixes.yaml` is what the gate is allowed to know. It is not
ground truth: on the committed measurements it disagrees with measured
recovery for 18 of 23 scenarios. Scoring an arm against the same table the
gate reasons with would grade the gate on its own assumptions.

Labels come from `results/ground_truth.json`, written by
`bench/verify_fault_observability.py` on Mininet: each catalog scenario is
injected, every remediation is executed, and `expected_actions` lists the
actions after which traffic recovered (or ESCALATE when none did, or when the
scenario cannot be told apart from one that needs a different fix).

The agent and the controller can only take REROUTE, DO_NOTHING or ESCALATE.
A scenario whose only measured fix is outside that set (F5 recovers only
with REBALANCE) is labelled ESCALATE: no action the system can take fixes
it, so escalation is the only safe outcome.

The file is read, never written. Scenarios missing from it have no label;
`correct_actions` returns an empty set, so every action counts as wrong.
That overstates false allows rather than hiding them.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = ROOT / "results" / "ground_truth.json"
ACTION_SPACE = ("REROUTE", "DO_NOTHING", "ESCALATE")

_CACHE = {}


def default_source():
    """results/ground_truth.json, or the consensus file named by VERIFIX_LABELS.

    The chosen file and its SHA-256 are in every summary's `labels` block,
    so the label source is never a hidden experiment factor.
    """
    import os

    override = os.environ.get("VERIFIX_LABELS")
    if override:
        path = Path(override)
        return path if path.is_absolute() else ROOT / path
    return GROUND_TRUTH


def _load(path=None):
    path = Path(path or default_source())
    key = str(path.resolve())
    if key not in _CACHE:
        raw = path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        scenarios = doc.get("scenarios")
        if not isinstance(scenarios, dict) or not scenarios:
            raise ValueError(f"{path}: no measured scenarios")
        labels = {}
        for sid, row in scenarios.items():
            expected = row.get("expected_actions")
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"{path}: scenario {sid} has no expected_actions")
            labels[sid] = {
                "measured": sorted(set(expected)),
                "split": row.get("split"),
                "kind": row.get("kind"),
                # Single-run ground truth carries no trust flag; label_audit
                # consensus files do. Absent means "not established".
                "trusted": bool(row.get("trusted", False)),
            }
        try:
            source = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            source = str(path)
        _CACHE[key] = {
            "labels": labels,
            "provenance": {
                "source": source,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "generated_at": doc.get("generated_at"),
                "noise_id": doc.get("noise_id"),
                "seed": doc.get("seed"),
                "scenarios": len(labels),
                "rule": (doc.get("rules") or {}).get("valid_fixes"),
                "action_space": list(ACTION_SPACE),
                "trusted_scenarios": sorted(sid for sid, row in labels.items() if row["trusted"]),
                "untrusted_scenarios": sorted(sid for sid, row in labels.items() if not row["trusted"]),
            },
        }
    return _CACHE[key]


def correct_actions(scenario_id, path=None):
    """Measured correct actions within the system's action space."""
    row = _load(path)["labels"].get(scenario_id)
    if row is None:
        return set()
    usable = set(row["measured"]) & set(ACTION_SPACE)
    return usable or {"ESCALATE"}


def is_labelled(scenario_id, path=None):
    return scenario_id in _load(path)["labels"]


def provenance(path=None):
    """Where the labels came from, for manifests and summaries."""
    return dict(_load(path)["provenance"])


def disagreements(fixes_doc, path=None):
    """Scenario ids whose catalog `correct` set differs from the measured label."""
    out = {}
    catalog = dict((fixes_doc or {}).get("faults") or {})
    catalog.update((fixes_doc or {}).get("compounds") or {})
    for sid in sorted(_load(path)["labels"]):
        claimed = set((catalog.get(sid) or {}).get("correct") or [])
        measured = correct_actions(sid, path)
        if claimed != measured:
            out[sid] = {"catalog": sorted(claimed), "measured": sorted(measured)}
    return out


def _self_check():
    import tempfile

    import yaml

    labels = _load()["labels"]
    assert {"H0", "F1", "F11", "F12", "C1", "U1"} <= set(labels)
    assert correct_actions("H0") == {"DO_NOTHING"}
    assert correct_actions("F11") == {"ESCALATE"}
    assert correct_actions("F12") == {"DO_NOTHING"}
    assert correct_actions("F1") == {"REROUTE"}, "REBALANCE is outside the action space"
    assert correct_actions("F5") == {"ESCALATE"}, "only REBALANCE recovered F5"
    assert correct_actions("C1") == {"ESCALATE"}, "measured C1 is ESCALATE, not DO_NOTHING"
    assert "DO_NOTHING" not in correct_actions("C2")
    assert correct_actions("NOPE") == set() and not is_labelled("NOPE")
    prov = provenance()
    assert prov["source"] == "results/ground_truth.json" and len(prov["sha256"]) == 64
    assert prov["noise_id"] == "N0"
    fixes = yaml.safe_load((ROOT / "config" / "correct_fixes.yaml").read_text(encoding="utf-8"))
    diff = disagreements(fixes)
    # C1/C2 catalog entries were aligned to measured ESCALATE in Sprint 1.
    assert "C1" not in diff and "C2" not in diff and "H0" not in diff
    assert "F5" in diff, "catalog still disagrees with measured labels on some scenarios"

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "gt.json"
        bad.write_text(json.dumps({"scenarios": {"H0": {"expected_actions": []}}}), encoding="utf-8")
        try:
            _load(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("empty expected_actions accepted")
        alt = Path(tmp) / "gt2.json"
        alt.write_text(json.dumps({"scenarios": {"X": {"expected_actions": ["REBALANCE"]}}}), encoding="utf-8")
        assert correct_actions("X", alt) == {"ESCALATE"}
    print("labels: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
