"""Apply background-traffic / jitter noise levels from config/noise.yaml."""

import subprocess
import time
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
        # UDP on 5202. T8 and T10 keep the TCP iperf3 server on 5201.
        from bench.mininet_util import ensure_iperf3_server

        rate = level["background_traffic"]["rate_mbit"]
        dest = net.get("hostB")
        server = ensure_iperf3_server(net, "hostB", port=5202)
        time.sleep(0.4)
        # popen, not the Mininet shell. A background client started with
        # node.cmd() leaves that shell waiting, and the speed test then hangs.
        client = net.get("hostA").popen(
            ["iperf3", "-c", dest.IP(), "-p", "5202", "-u", "-b", f"{rate}M", "-t", "3600"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        applied["background_proc"] = client
        applied["background_server"] = server
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
    try:
        from bench.mininet_util import recover_net
    except ImportError:
        from mininet_util import recover_net
    recover_net(net)
    client = applied.get("background_proc")
    if client is not None and client.poll() is None:
        client.terminate()
    pid = applied.get("background_pid")
    if pid:
        net.get("hostA").cmd(f"kill {pid} 2>/dev/null")
    server = applied.get("background_server")
    if server is not None and server.poll() is None:
        server.terminate()
    for link_name in ("r1-r2", "r2-r4"):
        ctl.configure_link(ctl.get_link(net, link_name, links), **ctl.BASELINE_LINK_PARAMS)
