"""
Stage 18 automated analysis: tables, plots, and decision traces from saved raw data.

    python bench/analyze.py --self-check
    python bench/analyze.py
    python3 bench/analyze.py --live-analysis
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics import evaluate_records, summarize
from bench.run_batch import DATASET, SMOKE_DIR, jsonl_path, load_raw_json, provenance

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "results" / "analysis"
FROZEN_RESULTS = (
    ROOT / "results" / "calibration_report.json",
    ROOT / "results" / "pilot" / "summary.json",
    ROOT / "results" / "observability_matrix.csv",
    ROOT / "results" / "experiment_summary.json",
)
LIVE_ANALYSIS_BUILD = "stage18-analysis-v1"
ABLATION_METHODS = (
    "gate_no_predictions",
    "gate_no_discrimination",
    "gate_no_grader",
    "gate_full",
    "gate_v0",
    "verifix",
)
PARTIAL_STATUS = {"timeout", "inject_failed", "error"}
PLOT_NAMES = (
    "false_allow_by_strategy",
    "test_cost_vs_false_allow",
    "recovery_time",
    "held_out_generalization",
    "ablations",
    "normal_vs_stress",
    "topology1_vs_topology2",
)
TABLE_NAMES = (
    "false_allow_by_strategy",
    "test_cost_vs_false_allow",
    "recovery_time",
    "held_out",
    "ablations",
    "normal_vs_stress",
    "topology1_vs_topology2",
    "all_slices",
    "episode_scores",
    "skipped",
)


def _snapshot_frozen():
    return {path: path.read_bytes() if path.exists() else None for path in FROZEN_RESULTS}


def _assert_frozen(before):
    after = _snapshot_frozen()
    for path, original in before.items():
        assert after.get(path) == original, f"must not modify {path}"


def _refuse_frozen(path):
    resolved = Path(path).resolve()
    frozen = {item.resolve() for item in FROZEN_RESULTS}
    if resolved in frozen:
        raise ValueError(f"refusing to overwrite frozen {path}")
    if resolved.is_file() and resolved in frozen:
        raise ValueError(f"refusing to overwrite frozen {path}")


def resolve_raw_dir(path):
    path = Path(path)
    if (path / "raw").is_dir():
        return path / "raw"
    return path


def load_jsonl_records(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
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
                records.append(row)
    return records


def load_records(raw_dir):
    """JSONL is the append-only source of truth; per-episode JSON fills gaps."""
    raw_dir = Path(raw_dir)
    records = []
    seen = set()
    for row in load_jsonl_records(jsonl_path(raw_dir)):
        eid = row.get("episode_id")
        if eid in seen:
            continue
        seen.add(eid)
        records.append(row)
    for row in load_raw_json(raw_dir):
        eid = row.get("episode_id")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        records.append(row)
    records.sort(key=lambda row: str(row.get("episode_id")))
    return records


def _gate_from_record(record):
    if isinstance(record.get("gate_decision"), dict):
        return record["gate_decision"]
    decisions = record.get("decisions") or {}
    for key in ("verifix", "gate_full", "gate_v0"):
        row = decisions.get(key)
        if isinstance(row, dict):
            return row
    for row in decisions.values():
        if isinstance(row, dict):
            return row
    return {}


def extract_trace(record):
    record = record or {}
    proposal = record.get("agent_proposal") or {}
    gate = _gate_from_record(record)
    checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else {}
    disc = checks.get("discrimination") if isinstance(checks.get("discrimination"), dict) else {}
    log = gate.get("log") if isinstance(gate.get("log"), dict) else {}
    causes = []
    for cause in proposal.get("causes") or []:
        if not isinstance(cause, dict):
            continue
        causes.append(
            {
                "cause_id": cause.get("cause_id"),
                "predicted_tests": list(cause.get("predicted_tests") or []),
            }
        )
    ruled = disc.get("ruled_out")
    if not ruled:
        ruled = log.get("eliminated") or []
    selected = list(gate.get("tests_run") or log.get("selected_tests") or [])
    return {
        "episode_id": record.get("episode_id"),
        "fault_id": record.get("fault_id"),
        "compound_id": record.get("compound_id"),
        "topology": record.get("topology") or "diamond",
        "mode": record.get("mode") or "normal",
        "strategy": record.get("strategy"),
        "status": record.get("status"),
        "ai_claim": {
            "action": proposal.get("action") or record.get("proposed_fix"),
            "reason": proposal.get("reason"),
            "causes": causes,
            "proposed_path": proposal.get("proposed_path"),
        },
        "selected_tests": [tid for tid in selected if tid],
        "ruled_out_alternatives": ruled,
        "gate_decision": {
            "decision": gate.get("decision") or record.get("decision"),
            "reason": gate.get("reason"),
        },
    }


def pick_traces(traces, limit=12):
    chosen = []
    seen = set()

    def take(pred, tag):
        for row in traces:
            eid = row.get("episode_id")
            if eid in seen or not pred(row):
                continue
            seen.add(eid)
            chosen.append(dict(row, representative=tag))
            return

    take(lambda row: (row.get("gate_decision") or {}).get("decision") == "allow", "allow")
    take(lambda row: (row.get("gate_decision") or {}).get("decision") == "block", "block")
    take(lambda row: (row.get("gate_decision") or {}).get("decision") == "escalate", "escalate")
    take(lambda row: row.get("mode") == "stress", "stress")
    take(lambda row: (row.get("topology") or "") in {"grid_4x4", "large"}, "topology_2")
    take(lambda row: str(row.get("fault_id") or "").startswith("U"), "held_out")
    take(lambda row: bool(row.get("ruled_out_alternatives")), "has_ruled_out")
    for row in traces:
        if len(chosen) >= limit:
            break
        eid = row.get("episode_id")
        if eid in seen or row.get("status") in PARTIAL_STATUS:
            continue
        seen.add(eid)
        chosen.append(dict(row, representative=row.get("representative") or "additional"))
    return chosen


def format_trace(trace):
    claim = trace.get("ai_claim") or {}
    causes = ", ".join(
        str(item.get("cause_id")) for item in (claim.get("causes") or []) if item.get("cause_id")
    ) or "none"
    tests = " ".join(str(tid) for tid in (trace.get("selected_tests") or [])) or "none"
    ruled = trace.get("ruled_out_alternatives") or []
    if isinstance(ruled, list) and ruled and isinstance(ruled[0], dict):
        ruled_txt = "; ".join(
            f"{item.get('alternative')} via {item.get('test_id')}" for item in ruled
        )
    else:
        ruled_txt = ", ".join(str(item) for item in ruled) if ruled else "none"
    gate = trace.get("gate_decision") or {}
    return "\n".join(
        [
            f"episode: {trace.get('episode_id')}",
            f"representative: {trace.get('representative') or ''}",
            f"fault: {trace.get('compound_id') or trace.get('fault_id')}",
            f"topology: {trace.get('topology')}  mode: {trace.get('mode')}",
            f"AI claim: {claim.get('action')} ({claim.get('reason') or ''})".rstrip(),
            f"causes: {causes}",
            f"selected tests: {tests}",
            f"ruled-out alternatives: {ruled_txt}",
            f"gate decision: {gate.get('decision')} ({gate.get('reason') or ''})".rstrip(),
            "",
        ]
    )


def split_records(records):
    ready = []
    skipped = []
    for record in records or []:
        if not isinstance(record, dict):
            skipped.append({"episode_id": None, "status": "invalid", "error": "non-object"})
            continue
        status = record.get("status")
        if status in PARTIAL_STATUS:
            skipped.append(
                {
                    "episode_id": record.get("episode_id"),
                    "status": status,
                    "error": record.get("error"),
                }
            )
            continue
        ready.append(record)
    return ready, skipped


def _ci(block):
    bounds = (block or {}).get("ci95")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None, None
    return bounds[0], bounds[1]


def _rate(block):
    value = (block or {}).get("rate")
    return value if isinstance(value, (int, float)) else None


def _mean(block):
    value = (block or {}).get("mean")
    return value if isinstance(value, (int, float)) else None


def _p95(block):
    value = (block or {}).get("p95")
    return value if isinstance(value, (int, float)) else None


def strategy_rows(slice_report, slice_name="all"):
    rows = []
    for name in sorted((slice_report or {}).keys()):
        bucket = slice_report[name]
        fa = bucket.get("false_allow") or {}
        lo, hi = _ci(fa)
        cost = bucket.get("test_cost_seconds") or {}
        recovery = bucket.get("time_to_correct_recovery_seconds") or {}
        rows.append(
            {
                "slice": slice_name,
                "strategy": name,
                "n": bucket.get("n"),
                "false_allow_count": fa.get("count"),
                "false_allow_denominator": fa.get("denominator"),
                "false_allow_rate": fa.get("rate"),
                "false_allow_ci95_lo": lo,
                "false_allow_ci95_hi": hi,
                "unsafe_action_rate": _rate(bucket.get("unsafe_action")),
                "correct_allow_rate": _rate(bucket.get("correct_allow")),
                "correct_blocked_rate": _rate(bucket.get("correct_blocked")),
                "unnecessary_escalation_rate": _rate(bucket.get("unnecessary_escalation")),
                "test_cost_mean": _mean(cost),
                "test_cost_p95": _p95(cost),
                "recovery_mean": _mean(recovery),
                "recovery_p95": _p95(recovery),
                "decision_time_mean": _mean(bucket.get("decision_time_seconds")),
            }
        )
    return rows


def slice_table(report):
    rows = []
    for name in sorted((report.get("slices") or {}).keys()):
        rows.extend(strategy_rows(report["slices"][name], name))
    return rows


def episode_score_rows(scored):
    rows = []
    for row in scored or []:
        for name in sorted((row.get("strategies") or {}).keys()):
            flags = row["strategies"][name]
            rows.append(
                {
                    "episode_id": row.get("episode_id"),
                    "scenario_key": row.get("scenario_key"),
                    "fault_id": row.get("fault_id"),
                    "compound_id": row.get("compound_id") or "",
                    "topology": row.get("topology"),
                    "mode": row.get("mode"),
                    "slices": " ".join(row.get("slices") or []),
                    "strategy": name,
                    "decision": flags.get("decision"),
                    "false_allow": int(bool(flags.get("false_allow"))),
                    "unsafe_action": int(bool(flags.get("unsafe_action"))),
                    "correct_allow": int(bool(flags.get("correct_allow"))),
                    "correct_blocked": int(bool(flags.get("correct_blocked"))),
                    "unnecessary_escalation": int(bool(flags.get("unnecessary_escalation"))),
                    "test_cost_seconds": flags.get("test_cost_seconds"),
                }
            )
    return rows


def filter_scored(scored, pred):
    return [row for row in (scored or []) if pred(row)]


def compare_groups(scored, groups):
    out = []
    for label, pred in groups:
        subset = filter_scored(scored, pred)
        report = summarize(subset)
        rows = strategy_rows(report.get("slices", {}).get("all") or {}, slice_name=label)
        for row in rows:
            row["group"] = label
            row["n_episodes"] = len(subset)
            out.append(row)
    return out


def write_json(path, payload):
    path = Path(path)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def write_csv(path, fieldnames, rows):
    path = Path(path)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
    tmp.replace(path)
    return path


def _svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_svg(path, title, labels, values, ylabel=""):
    path = Path(path)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 380
    pad_l, pad_r, pad_t, pad_b = 70, 24, 40, 90
    usable_w = width - pad_l - pad_r
    usable_h = height - pad_t - pad_b
    nums = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]
    vmax = max(nums) if nums else 1.0
    if vmax <= 0:
        vmax = 1.0
    n = max(len(labels), 1)
    gap = usable_w / n
    bar_w = max(10.0, gap * 0.55)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="14">{_svg_escape(title)}</text>',
        f'<text x="18" y="{pad_t + usable_h / 2}" font-size="11" '
        f'transform="rotate(-90 18,{pad_t + usable_h / 2})">{_svg_escape(ylabel)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + usable_h}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t + usable_h}" x2="{pad_l + usable_w}" y2="{pad_t + usable_h}" stroke="#333"/>',
    ]
    if not labels:
        lines.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="12">no data</text>'
        )
    for i, (lab, val) in enumerate(zip(labels, nums)):
        h = (val / vmax) * usable_h
        x = pad_l + gap * i + (gap - bar_w) / 2
        y = pad_t + usable_h - h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h, 0):.1f}" fill="#334155"/>'
        )
        lx = x + bar_w / 2
        ly = pad_t + usable_h + 16
        lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10" '
            f'transform="rotate(32 {lx:.1f},{ly:.1f})">{_svg_escape(lab)}</text>'
        )
        if isinstance(values[i], (int, float)):
            lines.append(
                f'<text x="{lx:.1f}" y="{max(y - 4, 14):.1f}" text-anchor="middle" font-size="9">'
                f"{values[i]:.3g}</text>"
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_scatter_svg(path, title, points, xlabel="", ylabel=""):
    path = Path(path)
    _refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 400
    pad_l, pad_r, pad_t, pad_b = 70, 90, 40, 50
    usable_w = width - pad_l - pad_r
    usable_h = height - pad_t - pad_b
    xs = [p["x"] for p in points if isinstance(p.get("x"), (int, float))]
    ys = [p["y"] for p in points if isinstance(p.get("y"), (int, float))]
    xmin, xmax = (min(xs), max(xs)) if xs else (0.0, 1.0)
    ymin, ymax = (min(ys), max(ys)) if ys else (0.0, 1.0)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="14">{_svg_escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="11">{_svg_escape(xlabel)}</text>',
        f'<text x="18" y="{pad_t + usable_h / 2}" font-size="11" '
        f'transform="rotate(-90 18,{pad_t + usable_h / 2})">{_svg_escape(ylabel)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + usable_h}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t + usable_h}" x2="{pad_l + usable_w}" y2="{pad_t + usable_h}" stroke="#333"/>',
    ]
    if not points:
        lines.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="12">no data</text>'
        )
    for point in points:
        if not isinstance(point.get("x"), (int, float)) or not isinstance(point.get("y"), (int, float)):
            continue
        px = pad_l + (point["x"] - xmin) / (xmax - xmin) * usable_w
        py = pad_t + usable_h - (point["y"] - ymin) / (ymax - ymin) * usable_h
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#334155"/>')
        lines.append(
            f'<text x="{px + 8:.1f}" y="{py + 4:.1f}" font-size="10">{_svg_escape(point.get("label") or "")}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _bar_from_rows(rows, value_key, title, ylabel, path, label_key="strategy"):
    labels = [str(row.get(label_key) or "") for row in rows]
    values = [row.get(value_key) if isinstance(row.get(value_key), (int, float)) else 0.0 for row in rows]
    return write_bar_svg(path, title, labels, values, ylabel=ylabel)


def write_tables_and_plots(report, scored, skipped, traces, out_dir):
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    plots = out_dir / "plots"
    trace_dir = out_dir / "traces"
    slices = report.get("slices") or {}
    all_rows = strategy_rows(slices.get("all") or {}, "all")
    held_rows = compare_groups(
        scored,
        [
            ("known", lambda row: "known" in (row.get("slices") or [])),
            ("held_out", lambda row: "held_out" in (row.get("slices") or [])),
        ],
    )
    ablation_rows = [row for row in all_rows if row["strategy"] in ABLATION_METHODS]
    mode_rows = compare_groups(
        scored,
        [
            ("normal", lambda row: (row.get("mode") or "normal") != "stress"),
            ("stress", lambda row: (row.get("mode") or "") == "stress" or "stress" in (row.get("slices") or [])),
        ],
    )
    topo_rows = compare_groups(
        scored,
        [
            ("diamond", lambda row: (row.get("topology") or "diamond") == "diamond"),
            ("grid_4x4", lambda row: (row.get("topology") or "") in {"grid_4x4", "large"} or "topology_2" in (row.get("slices") or [])),
        ],
    )

    strategy_fields = [
        "slice",
        "strategy",
        "n",
        "false_allow_count",
        "false_allow_denominator",
        "false_allow_rate",
        "false_allow_ci95_lo",
        "false_allow_ci95_hi",
        "unsafe_action_rate",
        "correct_allow_rate",
        "correct_blocked_rate",
        "unnecessary_escalation_rate",
        "test_cost_mean",
        "test_cost_p95",
        "recovery_mean",
        "recovery_p95",
        "decision_time_mean",
    ]
    group_fields = ["group"] + strategy_fields + ["n_episodes"]
    write_csv(tables / "false_allow_by_strategy.csv", strategy_fields, all_rows)
    write_csv(tables / "test_cost_vs_false_allow.csv", strategy_fields, all_rows)
    write_csv(tables / "recovery_time.csv", strategy_fields, all_rows)
    write_csv(tables / "held_out.csv", group_fields, held_rows)
    write_csv(tables / "ablations.csv", strategy_fields, ablation_rows)
    write_csv(tables / "normal_vs_stress.csv", group_fields, mode_rows)
    write_csv(tables / "topology1_vs_topology2.csv", group_fields, topo_rows)
    write_csv(tables / "all_slices.csv", strategy_fields, slice_table(report))
    write_csv(
        tables / "episode_scores.csv",
        [
            "episode_id",
            "scenario_key",
            "fault_id",
            "compound_id",
            "topology",
            "mode",
            "slices",
            "strategy",
            "decision",
            "false_allow",
            "unsafe_action",
            "correct_allow",
            "correct_blocked",
            "unnecessary_escalation",
            "test_cost_seconds",
        ],
        episode_score_rows(scored),
    )
    write_csv(
        tables / "skipped.csv",
        ["episode_id", "status", "error"],
        skipped,
    )

    _bar_from_rows(
        all_rows,
        "false_allow_rate",
        "False allows by strategy",
        "false-allow rate",
        plots / "false_allow_by_strategy.svg",
    )
    write_scatter_svg(
        plots / "test_cost_vs_false_allow.svg",
        "Test cost versus false allows",
        [
            {"label": row["strategy"], "x": row.get("test_cost_mean"), "y": row.get("false_allow_rate")}
            for row in all_rows
            if isinstance(row.get("test_cost_mean"), (int, float))
            and isinstance(row.get("false_allow_rate"), (int, float))
        ],
        xlabel="mean test cost (s)",
        ylabel="false-allow rate",
    )
    _bar_from_rows(
        all_rows,
        "recovery_mean",
        "Recovery time by strategy",
        "mean seconds",
        plots / "recovery_time.svg",
    )
    _bar_from_rows(
        [
            {
                "strategy": f"{row['strategy']} {row['group']}",
                "false_allow_rate": row.get("false_allow_rate"),
            }
            for row in held_rows
        ],
        "false_allow_rate",
        "Held-out generalization",
        "false-allow rate",
        plots / "held_out_generalization.svg",
    )
    _bar_from_rows(
        ablation_rows,
        "false_allow_rate",
        "Ablation false allows",
        "false-allow rate",
        plots / "ablations.svg",
    )
    _bar_from_rows(
        [
            {
                "strategy": f"{row['strategy']} {row['group']}",
                "false_allow_rate": row.get("false_allow_rate"),
            }
            for row in mode_rows
        ],
        "false_allow_rate",
        "Normal versus stress",
        "false-allow rate",
        plots / "normal_vs_stress.svg",
    )
    _bar_from_rows(
        [
            {
                "strategy": f"{row['strategy']} {row['group']}",
                "false_allow_rate": row.get("false_allow_rate"),
            }
            for row in topo_rows
        ],
        "false_allow_rate",
        "Topology 1 versus topology 2",
        "false-allow rate",
        plots / "topology1_vs_topology2.svg",
    )

    write_json(trace_dir / "decision_traces.json", traces)
    (trace_dir / "representative.txt").write_text(
        "\n".join(format_trace(row) for row in traces),
        encoding="utf-8",
    )
    return {
        "tables": [name for name in TABLE_NAMES],
        "plots": [name for name in PLOT_NAMES],
        "n_traces": len(traces),
    }


def write_analysis(raw_dir, out_dir, fill_decisions=True):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    _refuse_frozen(out_dir)
    records = load_records(raw_dir)
    ready, skipped = split_records(records)
    report, scored = evaluate_records(ready, fill_decisions=fill_decisions)
    traces = pick_traces([extract_trace(row) for row in ready])
    artifacts = write_tables_and_plots(report, scored, skipped, traces, out_dir)
    manifest = {
        "build": LIVE_ANALYSIS_BUILD,
        "source": str(raw_dir),
        "out_dir": str(out_dir),
        "n_records": len(records),
        "n_ready": len(ready),
        "n_scored": report.get("scored"),
        "n_skipped": len(skipped),
        "errors": report.get("errors") or [],
        "methods": report.get("methods") or [],
        "artifacts": artifacts,
        "provenance": provenance(),
    }
    write_json(out_dir / "metrics.json", {"build": LIVE_ANALYSIS_BUILD, "report": report})
    write_json(out_dir / "manifest.json", manifest)
    return manifest, report, scored


def _sample_records():
    ablation = {
        "gate_no_predictions": {"decision": "allow", "tests_run": []},
        "gate_no_discrimination": {"decision": "allow", "tests_run": ["T1_e2e_ping"]},
        "gate_no_grader": {"decision": "block", "tests_run": ["T7_route_check"]},
        "gate_full": {"decision": "block", "tests_run": ["T1_e2e_ping", "T7_route_check"]},
        "gate_v0": {"decision": "allow", "tests_run": ["T1_e2e_ping"]},
        "verifix": {"decision": "block", "tests_run": ["T1_e2e_ping", "T7_route_check"]},
        "B0": {"decision": "allow", "tests_run": []},
        "B5": {"decision": "allow", "tests_run": ["T1_e2e_ping"]},
    }
    f12 = {
        "episode_id": "a",
        "fault_id": "F12",
        "seed": 1,
        "topology": "diamond",
        "mode": "normal",
        "noise_id": "N0",
        "status": "ok",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:12+00:00",
        "remediation_succeeded": False,
        "agent_proposal": {
            "action": "REROUTE",
            "reason": "last-mile loss misread as path loss",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [
                {
                    "cause_id": "F12",
                    "predicted_tests": [{"test_id": "T1_e2e_ping", "expected": "loss=20"}],
                }
            ],
        },
        "gate_decision": {
            "decision": "block",
            "reason": "consistency failed",
            "tests_run": ["T1_e2e_ping", "T5_checksum"],
            "checks": {
                "discrimination": {
                    "passed": True,
                    "ruled_out": [{"alternative": "F1", "test_id": "T7_route_check", "observed": "primary"}],
                }
            },
            "log": {"selected_tests": ["T1_e2e_ping", "T5_checksum"], "eliminated": ["F1"]},
        },
        "decisions": ablation,
        "oam_cost": {"total_duration_seconds": 4.0},
        "gate_cost": {"latency_seconds": 0.3},
    }
    f1 = {
        "episode_id": "b",
        "fault_id": "F1",
        "seed": 2,
        "topology": "diamond",
        "mode": "normal",
        "noise_id": "N0",
        "status": "ok",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:08+00:00",
        "remediation_succeeded": True,
        "agent_proposal": {
            "action": "REROUTE",
            "reason": "primary down",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": []}],
        },
        "gate_decision": {
            "decision": "allow",
            "reason": "all checks passed",
            "tests_run": ["T7_route_check"],
            "checks": {"discrimination": {"ruled_out": [{"alternative": "F2", "test_id": "T7_route_check"}]}},
        },
        "decisions": {
            "B0": {"decision": "allow", "tests_run": []},
            "verifix": {"decision": "allow", "tests_run": ["T7_route_check"]},
            "gate_full": {"decision": "allow", "tests_run": ["T7_route_check"]},
            "gate_no_predictions": {"decision": "allow", "tests_run": []},
            "gate_no_discrimination": {"decision": "allow", "tests_run": []},
            "gate_no_grader": {"decision": "allow", "tests_run": ["T7_route_check"]},
            "gate_v0": {"decision": "allow", "tests_run": []},
        },
        "oam_cost": {"total_duration_seconds": 1.5},
        "gate_cost": {"latency_seconds": 0.1},
    }
    held = {
        "episode_id": "c",
        "fault_id": "U1",
        "seed": 3,
        "topology": "diamond",
        "mode": "normal",
        "noise_id": "N0",
        "status": "ok",
        "agent_proposal": {"action": "REROUTE", "reason": "unseen", "causes": []},
        "gate_decision": {"decision": "escalate", "reason": "inseparable", "tests_run": ["T1_e2e_ping"]},
        "decisions": {
            "B0": {"decision": "allow", "tests_run": []},
            "verifix": {"decision": "escalate", "tests_run": ["T1_e2e_ping"]},
            "gate_full": {"decision": "escalate", "tests_run": ["T1_e2e_ping"]},
            "gate_no_predictions": {"decision": "allow", "tests_run": []},
            "gate_no_discrimination": {"decision": "allow", "tests_run": []},
            "gate_no_grader": {"decision": "escalate", "tests_run": ["T1_e2e_ping"]},
            "gate_v0": {"decision": "allow", "tests_run": []},
        },
    }
    grid = {
        "episode_id": "d",
        "fault_id": "H0",
        "seed": 4,
        "topology": "grid_4x4",
        "mode": "normal",
        "noise_id": "N0",
        "status": "ok",
        "agent_proposal": {"action": "DO_NOTHING", "reason": "healthy", "causes": []},
        "gate_decision": {"decision": "allow", "reason": "all checks passed", "tests_run": []},
        "decisions": {
            "B0": {"decision": "allow", "tests_run": []},
            "verifix": {"decision": "allow", "tests_run": []},
            "gate_full": {"decision": "allow", "tests_run": []},
            "gate_no_predictions": {"decision": "allow", "tests_run": []},
            "gate_no_discrimination": {"decision": "allow", "tests_run": []},
            "gate_no_grader": {"decision": "allow", "tests_run": []},
            "gate_v0": {"decision": "allow", "tests_run": []},
        },
        "remediation_succeeded": True,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:05+00:00",
    }
    stress = {
        "episode_id": "e",
        "fault_id": "F3",
        "seed": 5,
        "topology": "diamond",
        "mode": "stress",
        "stress_kind": "misleading_ticket",
        "noise_id": "N2",
        "status": "ok",
        "agent_proposal": {"action": "REROUTE", "reason": "ticket", "causes": []},
        "gate_decision": {"decision": "block", "reason": "safety", "tests_run": ["T2_hop_ping"]},
        "decisions": {
            "B0": {"decision": "allow", "tests_run": []},
            "verifix": {"decision": "block", "tests_run": ["T2_hop_ping"]},
            "gate_full": {"decision": "block", "tests_run": ["T2_hop_ping"]},
            "gate_no_predictions": {"decision": "allow", "tests_run": []},
            "gate_no_discrimination": {"decision": "allow", "tests_run": []},
            "gate_no_grader": {"decision": "block", "tests_run": ["T2_hop_ping"]},
            "gate_v0": {"decision": "allow", "tests_run": []},
        },
    }
    failed = {
        "episode_id": "f",
        "fault_id": "NOT_A_FAULT",
        "seed": 6,
        "topology": "diamond",
        "mode": "normal",
        "status": "inject_failed",
        "error": "Unknown fault",
    }
    fill_me = {
        "episode_id": "g",
        "fault_id": "F1",
        "seed": 7,
        "topology": "diamond",
        "mode": "normal",
        "noise_id": "N0",
        "status": "ok",
        "agent_proposal": {
            "action": "REROUTE",
            "target": "HostA_to_HostB",
            "reason": "fill decisions",
            "proposed_path": ["r1", "r3", "r4"],
            "causes": [{"cause_id": "F1", "predicted_tests": [{"test_id": "T7_route_check", "expected": "backup"}]}],
        },
        "network_snapshot": {
            "links": [{"name": "r1-r2", "up": False}],
            "metrics": {"packet_loss_percent": 100, "connectivity": False, "active_path": "none"},
        },
        "oam": {"results": [{"test_id": "T7_route_check", "active_path": "none"}]},
    }
    return [f12, f1, held, grid, stress, failed, fill_me]


def _self_check():
    before = _snapshot_frozen()
    tmp = Path(tempfile.mkdtemp(prefix="verifix-analysis-"))
    raw = tmp / "raw"
    out = tmp / "analysis"
    raw.mkdir(parents=True)
    records = _sample_records()
    jsonl = jsonl_path(raw)
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in records[:-1]:
            handle.write(json.dumps(row) + "\n")
            handle.write("not-json\n")
    (raw / "orphan.json").write_text(json.dumps(records[-1]), encoding="utf-8")

    manifest, report, scored = write_analysis(raw, out, fill_decisions=True)
    failures = []
    if manifest.get("n_scored", 0) < 5:
        failures.append(f"scored {manifest.get('n_scored')} errors={report.get('errors')}")
    if manifest.get("n_skipped") < 1:
        failures.append("partial rows were not skipped")
    methods = set(report.get("methods") or [])
    if "B0" not in methods or "verifix" not in methods:
        failures.append(f"methods {sorted(methods)}")
    slices = report.get("slices") or {}
    for name in ("held_out", "stress", "topology_2", "known"):
        if not (slices.get(name) or {}).get("B0", {}).get("n"):
            failures.append(f"empty slice {name}")
    for name in TABLE_NAMES:
        path = out / "tables" / f"{name}.csv"
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            failures.append(f"missing table {name}")
    for name in PLOT_NAMES:
        path = out / "plots" / f"{name}.svg"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if "<svg" not in text:
            failures.append(f"missing plot {name}")
    traces = json.loads((out / "traces" / "decision_traces.json").read_text(encoding="utf-8"))
    if not traces:
        failures.append("no traces")
    required = {"ai_claim", "selected_tests", "ruled_out_alternatives", "gate_decision"}
    for row in traces:
        missing = required - set(row)
        if missing:
            failures.append(f"trace {row.get('episode_id')} missing {sorted(missing)}")
        claim = row.get("ai_claim") or {}
        if "action" not in claim:
            failures.append(f"trace {row.get('episode_id')} missing AI claim")
        gate = row.get("gate_decision") or {}
        if "decision" not in gate:
            failures.append(f"trace {row.get('episode_id')} missing gate decision")
    txt = (out / "traces" / "representative.txt").read_text(encoding="utf-8")
    if "AI claim:" not in txt or "selected tests:" not in txt or "ruled-out alternatives:" not in txt:
        failures.append("representative trace text incomplete")
    fa = (out / "tables" / "false_allow_by_strategy.csv").read_text(encoding="utf-8")
    if "verifix" not in fa or "B0" not in fa:
        failures.append("false-allow table missing strategies")
    if "gate_no_predictions" not in (out / "tables" / "ablations.csv").read_text(encoding="utf-8"):
        failures.append("ablation table empty")
    if "held_out" not in (out / "tables" / "held_out.csv").read_text(encoding="utf-8"):
        failures.append("held-out table empty")
    if "stress" not in (out / "tables" / "normal_vs_stress.csv").read_text(encoding="utf-8"):
        failures.append("normal vs stress table empty")
    if "grid_4x4" not in (out / "tables" / "topology1_vs_topology2.csv").read_text(encoding="utf-8"):
        failures.append("topology table empty")
    if "g" not in (out / "tables" / "episode_scores.csv").read_text(encoding="utf-8"):
        failures.append("fill_decisions record not scored")
    again, _report, _scored = write_analysis(raw, out, fill_decisions=True)
    first = (out / "tables" / "false_allow_by_strategy.csv").read_text(encoding="utf-8")
    write_analysis(raw, out, fill_decisions=True)
    second = (out / "tables" / "false_allow_by_strategy.csv").read_text(encoding="utf-8")
    if first != second:
        failures.append("regenerate is not deterministic")
    if again.get("n_scored") != manifest.get("n_scored"):
        failures.append("rescoring changed n")
    try:
        write_json(FROZEN_RESULTS[0], {"no": True})
        failures.append("froze overwrite was not refused")
    except ValueError:
        pass
    _assert_frozen(before)
    print(f"live-analysis build: {LIVE_ANALYSIS_BUILD}")
    if failures:
        print("automated analysis: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("automated analysis: PASS")
    return 0


def choose_source(explicit=None):
    if explicit:
        return resolve_raw_dir(explicit)
    smoke_raw = SMOKE_DIR / "raw"
    if load_records(smoke_raw):
        return smoke_raw
    return resolve_raw_dir(DATASET)


def live_analysis(source=None, out_dir=None):
    before = _snapshot_frozen()
    print(f"live-analysis build: {LIVE_ANALYSIS_BUILD}")
    raw_dir = choose_source(source)
    dest = Path(out_dir) if out_dir else ANALYSIS_DIR
    failures = []
    try:
        records = load_records(raw_dir)
        if not records:
            failures.append(f"no saved records in {raw_dir}")
        else:
            manifest, report, scored = write_analysis(raw_dir, dest, fill_decisions=True)
            print(f"source: {raw_dir}")
            print(f"scored: {manifest.get('n_scored')} skipped: {manifest.get('n_skipped')}")
            if manifest.get("n_scored", 0) < 1:
                failures.append(f"no scored episodes {report.get('errors')}")
            for name in TABLE_NAMES:
                if not (dest / "tables" / f"{name}.csv").exists():
                    failures.append(f"missing table {name}")
            for name in PLOT_NAMES:
                path = dest / "plots" / f"{name}.svg"
                if not path.exists() or "<svg" not in path.read_text(encoding="utf-8"):
                    failures.append(f"missing plot {name}")
            traces = dest / "traces" / "decision_traces.json"
            if not traces.exists():
                failures.append("missing traces")
            else:
                rows = json.loads(traces.read_text(encoding="utf-8"))
                if not rows:
                    failures.append("empty traces")
                else:
                    sample = rows[0]
                    for key in ("ai_claim", "selected_tests", "ruled_out_alternatives", "gate_decision"):
                        if key not in sample:
                            failures.append(f"trace missing {key}")
            if not scored:
                failures.append("metrics produced no scored rows")
        _assert_frozen(before)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL  {type(exc).__name__}: {exc}")
    if failures:
        print("automated analysis: FAIL " + "; ".join(str(item) for item in failures))
        return 1
    print("automated analysis: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--live-analysis", action="store_true")
    parser.add_argument("--from", dest="source", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.live_analysis:
        return live_analysis(args.source or None, args.out_dir or None)
    source = choose_source(args.source or None)
    dest = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR
    print(f"live-analysis build: {LIVE_ANALYSIS_BUILD}")
    records = load_records(source)
    if not records:
        print(f"automated analysis: FAIL no saved records in {source}")
        return 1
    before = _snapshot_frozen()
    manifest, _report, _scored = write_analysis(source, dest, fill_decisions=True)
    _assert_frozen(before)
    print(f"source: {source}")
    print(f"wrote {dest} scored={manifest.get('n_scored')} skipped={manifest.get('n_skipped')}")
    print("automated analysis: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
