"""Fixed fault catalog for gate alternative hypotheses."""

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def load_fault_catalog():
    with (_CONFIG / "faults.yaml").open() as f:
        faults = yaml.safe_load(f)["faults"]
    return {f["id"]: f for f in faults}


def plausible_alternatives(symptoms, exclude=None):
    """
    Return fault ids whose description keywords loosely match symptoms.
    symptoms: dict with keys like high_loss, link_down, high_latency.
    """
    catalog = load_fault_catalog()
    exclude = set(exclude or [])
    scores = []
    for fid, spec in catalog.items():
        if fid in exclude or fid == "H0":
            continue
        desc = spec.get("description", "").lower()
        score = 0
        if symptoms.get("high_loss") and "loss" in desc:
            score += 2
        if symptoms.get("link_down") and ("failure" in desc or "down" in desc):
            score += 2
        if symptoms.get("high_latency") and "latency" in desc:
            score += 2
        if symptoms.get("congestion") and "congestion" in desc:
            score += 2
        if score > 0:
            scores.append((score, fid))
    scores.sort(reverse=True)
    return [fid for _, fid in scores[:5]]
