"""Apply background-traffic / jitter noise levels from config/noise.yaml."""

from pathlib import Path

import yaml

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "noise.yaml"


def load_noise_levels():
    with _CONFIG.open() as f:
        return {n["id"]: n for n in yaml.safe_load(f)["noise_levels"]}


def apply_noise(net, noise_id, links=None):
    """Best-effort apply N0/N1/N2 to the running network."""
    levels = load_noise_levels()
    if noise_id not in levels:
        raise ValueError(f"Unknown noise level {noise_id!r}")
    level = levels[noise_id]
    applied = {"noise_id": noise_id}

    if level["background_traffic"]["enabled"]:
        rate = level["background_traffic"]["rate_mbit"]
        pid = net.get("hostA").cmd(
            f"iperf3 -c {net.get('hostB').IP()} -t 3600 -b {rate}M "
            f"> /tmp/noise_bg.log 2>&1 & echo $!"
        ).strip()
        applied["background_pid"] = pid
        applied["background_rate_mbit"] = rate

    if level["jitter"]["enabled"]:
        delay_ms = level["jitter"]["delay_ms"]
        var_ms = level["jitter"]["variation_ms"]
        for link_name in ("r1-r2", "r2-r4"):
            link = ctl.get_link(net, link_name, links)
            ctl.configure_link(link, delay=f"{delay_ms}ms", loss=0)
        applied["jitter_ms"] = delay_ms
        applied["jitter_var_ms"] = var_ms

    return applied


def clear_noise(net, applied, links=None):
    """Undo apply_noise side effects."""
    pid = applied.get("background_pid")
    if pid:
        net.get("hostA").cmd(f"kill {pid} 2>/dev/null")
    for link_name in ("r1-r2", "r2-r4"):
        ctl.configure_link(ctl.get_link(net, link_name, links), **ctl.BASELINE_LINK_PARAMS)
