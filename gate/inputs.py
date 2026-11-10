"""Formal gate inputs: frozen context, proposal, and live network state.

Incomplete or inconsistent values are returned as errors. Callers must not
invent catalog rows, thresholds, or snapshot fields to paper over them.
"""

from pathlib import Path

import yaml

from bench.contracts import TEST_IDS
from bench.outcomes import load_thresholds, threshold_set_errors
from gate import catalog, checks, selector
from gate.grader import grade_test

_ROOT = Path(__file__).resolve().parent.parent
_NOISE = _ROOT / "config" / "noise.yaml"
_GATE = _ROOT / "config" / "gate.yaml"
CONSISTENCY_SCOPES = ("discriminating", "all")
REQUIRED_LINKS = ("r1-r2", "r2-r4", "r1-r3", "r3-r4")
REQUIRED_ROUTES = (("r1", "10.0.4.0/24"), ("r4", "10.0.1.0/24"))
KNOWN_PATHS = {"primary", "backup", "none"}
MODES = ("full", "pilot_v0")


def pipeline_interfaces():
    """Named callables for each gate stage. Decision logic stays in decide()."""
    from bench.outcomes import grade_result

    return {
        "candidate_generation": catalog.plausible_alternatives,
        "test_selection": selector.pick_cheapest_separating_test,
        "outcome_grading": grade_result,
        "reliability_grading": grade_test,
        "safety_checking": checks.check_safety,
        "consistency_checking": checks.check_consistency,
        "discrimination_checking": checks.check_discrimination,
    }


def _noise_ids():
    """Traffic levels defined in config/noise.yaml (thresholds may still be unaccepted)."""
    from bench.traffic import load_noise_levels

    return set(load_noise_levels().keys())


def load_gate_config():
    """Frozen VeriFix gate settings. Missing or unfrozen files fail closed."""
    errors = []
    if not _GATE.exists():
        return {}, ["gate config missing"]
    try:
        doc = yaml.safe_load(_GATE.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, [f"gate config: {type(exc).__name__}: {exc}"]
    if not isinstance(doc, dict):
        return {}, ["gate config is not a mapping"]
    if doc.get("frozen") is not True:
        errors.append("gate config is not frozen")
    if "learned_grader" not in doc:
        errors.append("gate config missing learned_grader")
    budget = doc.get("budget_seconds")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        errors.append("gate config budget_seconds must be a positive number")
    age = doc.get("reuse_max_age_seconds", 120)
    if not isinstance(age, (int, float)) or isinstance(age, bool) or age < 0:
        errors.append("gate config reuse_max_age_seconds must be a non-negative number")
    conf = doc.get("min_grade_confidence", 0.5)
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
        errors.append("gate config min_grade_confidence must be between 0 and 1")
    scope = doc.get("consistency_scope")
    if scope not in CONSISTENCY_SCOPES:
        errors.append(f"gate config consistency_scope must be one of {list(CONSISTENCY_SCOPES)}")
    cfg = {
        "frozen": doc.get("frozen") is True,
        "learned_grader": bool(doc.get("learned_grader")),
        "min_grade_confidence": float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.5,
        "budget_seconds": float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) else 8.0,
        "reuse_max_age_seconds": float(age) if isinstance(age, (int, float)) and not isinstance(age, bool) else 120.0,
        "require_signatures": bool(doc.get("require_signatures")),
        "consistency_scope": scope if scope in CONSISTENCY_SCOPES else None,
        "version": doc.get("version"),
    }
    return cfg, errors


def load_frozen_context(noise_id=None):
    """Load catalog, matrix, thresholds, and costs. Empty or missing files fail.

    noise_id is required (episode traffic level). There is no silent N0 default.
    """
    errors = []
    if noise_id is None:
        return False, ["noise_id required"], {
            "noise_id": None,
            "catalog": {},
            "matrix": [],
            "thresholds": {},
            "costs": {},
            "fixes": {},
            "held_out": set(),
            "gate": {},
        }
    known = _noise_ids()
    if noise_id not in known:
        errors.append(f"unknown noise_id {noise_id!r}")

    try:
        faults = catalog.load_fault_catalog()
    except Exception as exc:
        faults = {}
        errors.append(f"fault catalog: {type(exc).__name__}: {exc}")
    if not faults:
        errors.append("fault catalog is empty")
    for required in ("H0", "F1", "F12"):
        if required not in faults:
            errors.append(f"fault catalog missing {required}")

    try:
        costs = selector._load_costs()
    except Exception as exc:
        costs = {}
        errors.append(f"test costs: {type(exc).__name__}: {exc}")
    if not isinstance(costs, dict) or not costs:
        errors.append("test costs are empty")
    else:
        for tid in TEST_IDS:
            if tid not in costs:
                errors.append(f"test costs missing {tid}")
            elif not isinstance((costs.get(tid) or {}).get("time_seconds"), (int, float)):
                errors.append(f"test costs {tid} has no time_seconds")

    try:
        matrix = selector._load_matrix()
    except Exception as exc:
        matrix = []
        errors.append(f"observability matrix: {type(exc).__name__}: {exc}")
    if not matrix:
        errors.append("observability matrix is empty")
    else:
        for row in matrix[:1]:
            for key in ("fault_id", "test_id", "outcome"):
                if key not in row:
                    errors.append(f"observability matrix missing column {key}")

    try:
        fixes = selector._load_fixes()
    except Exception as exc:
        fixes = {}
        errors.append(f"correct fixes: {type(exc).__name__}: {exc}")

    thresholds = None
    if noise_id in known:
        try:
            # Unaccepted calibration raises ThresholdsUnavailable: the gate
            # escalates on "frozen inputs" instead of grading.
            thresholds = load_thresholds(noise_id)
        except Exception as exc:
            errors.append(f"thresholds: {type(exc).__name__}: {exc}")
        else:
            errors.extend(f"thresholds: {gap}" for gap in threshold_set_errors(thresholds))

    gate_cfg, gate_errors = load_gate_config()
    errors.extend(gate_errors)

    context = {
        "noise_id": noise_id,
        "catalog": faults,
        "matrix": matrix,
        "thresholds": thresholds or {},
        "costs": costs if isinstance(costs, dict) else {},
        "fixes": fixes if isinstance(fixes, dict) else {},
        "held_out": selector.held_out_ids(),
        "gate": gate_cfg,
    }
    return (not errors), errors, context


def validate_proposal_inputs(proposal, costs=None):
    """Proposed fix, hypotheses, and predicted outcomes. No silent fill-in."""
    from agent.validate import evaluation_cause_ids, validate_proposal

    if not isinstance(proposal, dict):
        return ["proposal must be an object"]
    ok, errors = validate_proposal(proposal, cause_ids=evaluation_cause_ids())
    errors = list(errors)
    if not ok and not errors:
        errors.append("invalid proposal")
    allowed = set(TEST_IDS)
    if isinstance(costs, dict) and costs:
        allowed = set(costs)
    for i, cause in enumerate(proposal.get("causes") or []):
        if not isinstance(cause, dict):
            continue
        for j, pred in enumerate(cause.get("predicted_tests") or []):
            if not isinstance(pred, dict):
                continue
            expected = pred.get("expected")
            if "expected" in pred and isinstance(expected, str) and not expected.strip():
                errors.append(f"causes[{i}].predicted_tests[{j}].expected is empty")
            tid = pred.get("test_id")
            if tid and allowed and tid not in allowed:
                errors.append(f"causes[{i}].predicted_tests[{j}].test_id {tid!r} is not a permitted test")
    return errors


def validate_snapshot(snapshot):
    """Topology, route, and interface fields the gate is allowed to consume."""
    if not isinstance(snapshot, dict) or not snapshot:
        return ["network snapshot is missing"]
    errors = []
    links = snapshot.get("links")
    if not isinstance(links, list) or not links:
        errors.append("snapshot.links is missing")
        links = []
    by_name = {}
    for i, link in enumerate(links):
        if not isinstance(link, dict):
            errors.append(f"snapshot.links[{i}] must be an object")
            continue
        name = link.get("name")
        if not name:
            errors.append(f"snapshot.links[{i}] has no name")
            continue
        if "up" not in link:
            errors.append(f"snapshot.links[{name}].up is missing")
        elif not isinstance(link.get("up"), bool):
            errors.append(f"snapshot.links[{name}].up must be a boolean")
        by_name[name] = link
    for name in REQUIRED_LINKS:
        if name not in by_name:
            errors.append(f"snapshot missing link {name}")

    routes = snapshot.get("routes")
    if routes is not None:
        if not isinstance(routes, list):
            errors.append("snapshot.routes must be an array")
            routes = []
        found = set()
        for i, row in enumerate(routes):
            if not isinstance(row, dict):
                errors.append(f"snapshot.routes[{i}] must be an object")
                continue
            for key in ("node", "prefix"):
                if not isinstance(row.get(key), str) or not row.get(key):
                    errors.append(f"snapshot.routes[{i}].{key} is missing")
            # Empty route text is a live observation (link down / no matching prefix).
            if "route" not in row or not isinstance(row.get("route"), str):
                errors.append(f"snapshot.routes[{i}].route is missing")
            found.add((row.get("node"), row.get("prefix")))
        for pair in REQUIRED_ROUTES:
            if pair not in found and routes:
                errors.append(f"snapshot missing route {pair[0]} {pair[1]}")

    metrics = snapshot.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("snapshot.metrics is missing")
        metrics = {}
    path = metrics.get("active_path")
    has_routes = isinstance(routes, list) and bool(routes)
    if path is None and not has_routes:
        errors.append("snapshot has no active_path and no routes")
    elif path is not None and path not in KNOWN_PATHS:
        errors.append(f"snapshot.metrics.active_path {path!r} is not primary|backup|none")

    counters = snapshot.get("interface_counters")
    if counters is not None and not isinstance(counters, dict):
        errors.append("snapshot.interface_counters must be an object")
    return errors


def validate_test_results(test_results, already_ran=None):
    errors = []
    if test_results is None:
        rows = []
    elif not isinstance(test_results, list):
        return ["test_results must be an array"]
    else:
        rows = test_results
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("test_id"):
            errors.append(f"test_results[{i}] must be an object with test_id")
    if already_ran is not None and not isinstance(already_ran, (list, set, tuple)):
        errors.append("already_ran must be a list of test ids")
    return errors
