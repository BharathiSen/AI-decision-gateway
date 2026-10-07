"""
Generate paper figures from results/*.json (matplotlib optional).

    python3 paper/generate_figures.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(path, default=None):
    p = ROOT / path
    if not p.exists():
        return default or {}
    return json.loads(p.read_text(encoding="utf-8"))


def safety_vs_cost_svg(summary):
    """Write a minimal SVG safety vs cost plot without matplotlib."""
    methods = summary.get("methods", {})
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320">']
    lines.append('<text x="20" y="20" font-size="14">Safety vs test cost (placeholder)</text>')
    x = 60
    for name, data in methods.items():
        rate = data.get("false_allow_rate", 0) * 100
        lines.append(f'<circle cx="{x}" cy="{280 - rate * 2}" r="6" fill="#333"/>')
        lines.append(f'<text x="{x - 20}" y="300" font-size="10">{name}</text>')
        x += 100
    lines.append("</svg>")
    (FIG / "safety_vs_cost.svg").write_text("\n".join(lines), encoding="utf-8")


def main():
    pilot = _load("results/pilot/summary.json")
    exp = _load("results/experiment_summary.json")
    combined = {"pilot": pilot, "experiments": exp}
    safety_vs_cost_svg(exp if exp.get("methods") else {"methods": {"B5": {}, "gate_v0": {}}})
    (FIG / "manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
