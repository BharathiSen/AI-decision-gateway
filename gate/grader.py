"""
Small learned grader for noisy OAM tests (train on episodes; optional).

Default: rule-based pass-through until training data exists.
"""

import json
from pathlib import Path

_MODEL = Path(__file__).resolve().parent.parent / "results" / "grader_model.json"


def grade_test(test_id, observation, predicted_expected, use_model=True, model_path=None):
    """Return confidence 0..1 that observation supports prediction."""
    from bench.contracts import prediction_matches

    if prediction_matches(test_id, predicted_expected, observation):
        return 1.0
    path = Path(model_path) if model_path is not None else _MODEL
    if use_model and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        key = f"{test_id}|{(predicted_expected or '').lower()}"
        return float(data.get(key, 0.5))
    return 0.5


def train_grader(samples):
    """
    samples: list of {test_id, expected, observation, label} label in {0,1}
    Writes simple lookup table to results/grader_model.json
    Samples from held-out faults or compounds (fault_id) are refused.
    """
    from gate.selector import held_out_ids

    hidden = held_out_ids()
    leaked = sorted({s["fault_id"] for s in samples if s.get("fault_id") in hidden})
    if leaked:
        raise ValueError(f"held-out scenarios cannot train the grader: {leaked}")
    table = {}
    for s in samples:
        key = f"{s['test_id']}|{(s.get('expected') or '').lower()}"
        table[key] = float(s.get("label", 0))
    _MODEL.parent.mkdir(parents=True, exist_ok=True)
    _MODEL.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table
