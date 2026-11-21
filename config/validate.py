from pathlib import Path
import hashlib
import json
import sys
import yaml

BASE = Path(__file__).parent
ROOT = BASE.parent

def load_yaml(name):
    with (BASE / name).open() as f:
        return yaml.safe_load(f)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inseparable_pairs_freshness(results_dir=None):
    """Fail if inseparable_pairs.json is missing provenance or lags correct_fixes/matrix."""
    results_dir = Path(results_dir) if results_dir else ROOT / "results"
    pairs_path = results_dir / "inseparable_pairs.json"
    meta_path = results_dir / "inseparable_pairs.meta.json"
    fixes_path = BASE / "correct_fixes.yaml"
    matrix_path = results_dir / "observability_matrix.csv"
    if not pairs_path.exists():
        return "missing results/inseparable_pairs.json"
    if not meta_path.exists():
        return "missing results/inseparable_pairs.meta.json (regenerate pairs on WSL Mininet (Ubuntu-22.04))"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable inseparable_pairs.meta.json: {exc}"
    live_fixes = _file_sha256(fixes_path)
    if meta.get("correct_fixes_sha256") != live_fixes:
        return (
            "inseparable_pairs.meta.json correct_fixes_sha256 mismatch; "
            "regenerate inseparable_pairs.json on WSL Mininet (Ubuntu-22.04)"
        )
    if matrix_path.exists():
        live_matrix = _file_sha256(matrix_path)
        recorded = meta.get("observability_matrix_sha256")
        if recorded and recorded != live_matrix:
            return (
                "inseparable_pairs.meta.json observability_matrix_sha256 mismatch; "
                "regenerate inseparable_pairs.json on WSL Mininet (Ubuntu-22.04)"
            )
    return None

faults_doc = load_yaml("faults.yaml")
compounds_doc = load_yaml("compounds.yaml")
noise_doc = load_yaml("noise.yaml")

faults = faults_doc["faults"]
compounds = compounds_doc["compounds"]
noise = noise_doc["noise_levels"]

allowed_injections = {
    "none", "link_down", "netem_loss", "netem_corrupt",
    "netem_delay", "tbf_congestion", "link_flap",
    "blackhole_route", "acl_drop", "mtu_mismatch",
    "netem_reorder_dup", "wrong_return_route",
}

allowed_cleanup = {
    "none", "link_up", "qdisc_reset",
    "traffic_and_qdisc_reset", "stop_flap_and_link_up",
    "delete_injected_route", "remove_injected_acl",
    "restore_mtu",
}

ids = [f["id"] for f in faults]
assert len(ids) == len(set(ids)), "Duplicate fault IDs"

expected = {"H0", *[f"F{i}" for i in range(1, 13)], "U1", "U2", "U3", "U4", "U5"}
assert set(ids) == expected, "Missing or unexpected fault IDs"

for fault in faults:
    assert fault.get("description"), f"{fault['id']}: missing description"
    assert fault.get("split") in {
        "baseline", "development", "held_out"
    }, f"{fault['id']}: invalid split"

    injection = fault.get("injection", {})
    cleanup = fault.get("cleanup", {})

    assert injection.get("type") in allowed_injections, (
        f"{fault['id']}: invalid injection type"
    )
    assert cleanup.get("type") in allowed_cleanup, (
        f"{fault['id']}: invalid cleanup type"
    )
    assert isinstance(fault.get("params"), dict), (
        f"{fault['id']}: params must be a mapping"
    )
    assert "initial_acceptable_actions" in fault, (
        f"{fault['id']}: missing acceptable actions"
    )
    allowed_actions = {"REROUTE", "DO_NOTHING", "ESCALATE"}
    extras = set(fault.get("initial_acceptable_actions") or []) - allowed_actions
    assert not extras, f"{fault['id']}: non-canonical actions {sorted(extras)}"

assert faults_doc["version"] == 1
assert compounds_doc["version"] == 1
assert noise_doc["version"] == 1

compound_ids = [c["id"] for c in compounds]
assert len(compound_ids) == len(set(compound_ids)), (
    "Duplicate compound IDs"
)

fault_id_set = set(ids)
for compound in compounds:
    assert len(compound["components"]) >= 2
    assert set(compound["components"]) <= fault_id_set
    assert compound["injection_order"] == compound["components"]
    assert compound["cleanup_order"] == list(
        reversed(compound["injection_order"])
    )

assert len(noise) == 3
assert len({n["id"] for n in noise}) == len(noise)

for uid in ("U1", "U2", "U3", "U4", "U5"):
    assert next(f for f in faults if f["id"] == uid)["split"] == "held_out"

for compound in compounds:
    assert compound.get("split") in {"development", "held_out"}, (
        f"{compound['id']}: invalid split"
    )

development_sets = {
    frozenset(c["components"]) for c in compounds if c["split"] == "development"
}
for compound in compounds:
    if compound["split"] == "held_out":
        assert frozenset(compound["components"]) not in development_sets, (
            f"{compound['id']}: same components as a development compound"
        )

print(f"Validated {len(faults)} faults")
print(f"Validated {len(compounds)} compound scenarios")
print(f"Validated {len(noise)} noise levels")
print("YAML structure and references: PASS")

_pairs_gap = check_inseparable_pairs_freshness()
if _pairs_gap:
    print(f"inseparable_pairs freshness: FAIL {_pairs_gap}", file=sys.stderr)
    raise SystemExit(1)
print("inseparable_pairs freshness: PASS")
