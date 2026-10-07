from pathlib import Path
import yaml

BASE = Path(__file__).parent

def load_yaml(name):
    with (BASE / name).open() as f:
        return yaml.safe_load(f)

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

print(f"Validated {len(faults)} faults")
print(f"Validated {len(compounds)} compound scenarios")
print(f"Validated {len(noise)} noise levels")
print("YAML structure and references: PASS")
