"""
Small learned grader for noisy OAM tests (train on episodes; optional).

Default: rule-based pass-through until training data exists.
"""

import json
from pathlib import Path

_MODEL = Path(__file__).resolve().parent.parent / "results" / "grader_model.json"


def grade_test(test_id, observation, predicted_expected):
    """Return confidence 0..1 that observation supports prediction."""
    obs = str(observation).lower()
    exp = (predicted_expected or "").lower()
    if exp in obs or obs in exp:
        return 1.0
    if _MODEL.exists():
        data = json.loads(_MODEL.read_text(encoding="utf-8"))
        key = f"{test_id}|{exp}"
        return float(data.get(key, 0.5))
    return 0.5


def train_grader(samples):
    """
    samples: list of {test_id, expected, observation, label} label in {0,1}
    Writes simple lookup table to results/grader_model.json
    """
    table = {}
    for s in samples:
        key = f"{s['test_id']}|{(s.get('expected') or '').lower()}"
        table[key] = float(s.get("label", 0))
    _MODEL.parent.mkdir(parents=True, exist_ok=True)
    _MODEL.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table
