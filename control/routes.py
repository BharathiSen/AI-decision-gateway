"""Exact route specs for r1/r4 and route commands whose outcome is checked.

Used by the gated executor (control/gated.py, control/actions.execute_gated)
and the executor service. Baselines keep their own paths (B0 restores through
bench.inject; B1 and ground-truth tooling use network.controller directly).

A "spec" is the route line as `ip route show <dest>` prints it, with kernel
state flags removed, so it can be given back to `ip route replace`.
"""

try:
    from network import controller as ctl
except ImportError:  # pragma: no cover - run from network/
    import controller as ctl

# Same flags as bench.inject._ROUTE_FLAGS (checked by `python -m control`).
ROUTE_FLAGS = frozenset({"linkdown", "dead", "offload", "trap", "onlink"})
ROUTERS = tuple(ctl.ROUTE_TARGETS)
_RC_MARK = "__VERIFIX_RC="


class RouteReadError(RuntimeError):
    """A route could not be read. Never means "no route"."""


class RouteCommandError(RuntimeError):
    """A route command exited non-zero or could not run."""

    def __init__(self, router, command, detail):
        super().__init__(f"{router}: `{command}` failed: {detail}")
        self.router = router
        self.command = command


def route_spec(line):
    return " ".join(token for token in (line or "").split() if token not in ROUTE_FLAGS)


def specs_from_route_text(text):
    """One spec from `ip route show <dest>` output; '' when there is no route."""
    specs = [route_spec(line) for line in (text or "").splitlines() if line.strip()]
    if len(specs) > 1:
        raise RouteReadError(f"more than one route: {specs}")
    return specs[0] if specs else ""


def specs_from_snapshot(snapshot):
    """{router: spec} from a read_state()/telemetry snapshot. Raises if a router is missing."""
    found = {}
    for item in (snapshot or {}).get("routes") or []:
        node = item.get("node")
        if node in ctl.ROUTE_TARGETS and item.get("prefix") == ctl.ROUTE_TARGETS[node]["dest"]:
            if node in found:
                raise RouteReadError(f"{node}: route listed twice")
            found[node] = specs_from_route_text(item.get("route"))
    missing = [router for router in ROUTERS if router not in found]
    if missing:
        raise RouteReadError(f"routes not read for {missing}")
    return found


def run_checked(node, command):
    """Run one command in the node's namespace. Returns (output, exit code).

    pexec (own process, real exit status) when the node has it; otherwise the
    exit status is echoed after the command and parsed, so it is never assumed.
    """
    if hasattr(node, "pexec"):
        out, err, code = node.pexec(["bash", "-c", command])
        return (out or "") + (err or ""), code
    raw = node.cmd(f"{command}; echo {_RC_MARK}$?") or ""
    head, sep, tail = raw.rpartition(_RC_MARK)
    if not sep:
        return raw, None
    try:
        return head, int(tail.strip().split()[0])
    except (ValueError, IndexError):
        return head, None


def _checked(net, router, command):
    try:
        out, code = run_checked(net.get(router), command)
    except Exception as exc:
        raise RouteCommandError(router, command, f"{type(exc).__name__}: {exc}") from exc
    if code != 0:
        raise RouteCommandError(router, command, f"exit {code}: {(out or '').strip()}")
    return out


def read_specs(net):
    """{router: spec} read from each router in its own process."""
    specs = {}
    for router, cfg in ctl.ROUTE_TARGETS.items():
        command = f"ip route show {cfg['dest']}"
        try:
            out = _checked(net, router, command)
        except RouteCommandError as exc:
            raise RouteReadError(str(exc)) from exc
        specs[router] = specs_from_route_text(out)
    return specs


def apply_path_checked(net, path):
    """Point r1 then r4 at path. Raises RouteCommandError at the first failure."""
    if path not in ("primary", "backup"):
        raise ValueError("path must be 'primary' or 'backup'")
    for router, cfg in ctl.ROUTE_TARGETS.items():
        hop = cfg[path]
        _checked(net, router, f"ip route replace {cfg['dest']} via {hop['via']} dev {hop['dev']}")
    return path


def restore_spec_checked(net, router, spec):
    """Put one router's exact captured spec back. A gated start always has a route."""
    if router not in ctl.ROUTE_TARGETS:
        raise ValueError(f"unknown router {router!r}")
    if not spec or not spec.startswith(ctl.ROUTE_TARGETS[router]["dest"] + " "):
        raise ValueError(f"{router}: refusing to restore {spec!r}: not a captured route to {ctl.ROUTE_TARGETS[router]['dest']}")
    _checked(net, router, f"ip route replace {spec}")


class RouteControl:
    """Checked restore for one network, handed to the execution boundary."""

    def __init__(self, net):
        self._net = net

    def restore(self, router, spec):
        restore_spec_checked(self._net, router, spec)
