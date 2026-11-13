"""Gated execution lifecycle: capability, single flight, alarm masking, compensation.

Shared by the in-process boundary (bench.run_episode.apply_authorized_action)
and the executor service (control/executor_service.py). It never decides
whether an action is authorized: callers verify the gate-issued authorization
first (gate/authority.py) and only then mint a Capability.

Lifecycle of one authorized action:
  1. routes_before: each router's exact spec from a fresh read inside the
     boundary (never a caller-supplied snapshot).
  2. The action runs with checked per-router commands.
  3. Routes are ALWAYS read afterwards, also when the action raised. A read
     that shows no change after a route command is confirmed by a second
     read; a failed read is retried once.
  4. Not fully applied and something changed (or the state is unknown):
     the research harness measures post-action OAM first (measure_partial),
     then exactly one restore pass puts each CHANGED router's captured spec
     back, verified by a read-back and a confirming read.
  5. Unverified restore -> CompensationFailed (the boundary turns it into
     TopologyContaminated). Compensation carries the same decision_id.
SIGALRM is blocked from step 2 to the end of step 4 and delivered afterwards.
BLOCK and ESCALATE never reach this module.
"""

import contextlib
import copy
import signal as _signal
import threading
import time
from dataclasses import dataclass, field

try:
    from network import controller as ctl
except ImportError:  # pragma: no cover
    import controller as ctl

ROUTE_CHANGING = ("REROUTE", "REBALANCE")
_MINT = object()
_EXECUTION_LOCK = threading.Lock()


class ExecutionBusy(RuntimeError):
    """Another authorized execution is in progress in this process."""


class CompensationFailed(RuntimeError):
    """A partial action could not be restored and verified. outcome is attached."""

    def __init__(self, message, outcome):
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True)
class Capability:
    """Permission to run one verified authorization. Minted only by mint().

    In one process this is an accident guard, not a security boundary: code
    here can reach _MINT. The process boundary is the executor service.
    """

    decision_id: str
    action: str
    reroute_path: object
    target: object
    expires_at: float
    proposal: dict
    _mint: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._mint is not _MINT:
            raise PermissionError("a capability is minted only after the gate authorization is verified")


def mint(authorization, proposal):
    """Capability for a VERIFIED authorization. Callers must verify first."""
    return Capability(
        decision_id=authorization["decision_id"],
        action=authorization["action"],
        reroute_path=authorization.get("reroute_path"),
        target=authorization.get("target"),
        expires_at=float(authorization["expires_at"]),
        proposal=copy.deepcopy(proposal),
        _mint=_MINT,
    )


def capability_errors(capability, now=None):
    if not isinstance(capability, Capability) or capability._mint is not _MINT:
        return ["gated execution requires a verified authorization capability"]
    if (time.time() if now is None else now) >= capability.expires_at:
        return ["authorization has expired"]
    return []


@contextlib.contextmanager
def single_flight():
    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise ExecutionBusy("another authorized execution is in progress")
    try:
        yield
    finally:
        _EXECUTION_LOCK.release()


_ALARM = {"depth": 0, "pending": False}


def alarm_should_defer():
    """For SIGALRM handlers: True (and remembered) while a gated section runs.

    Python runs signal handlers in the main thread whichever thread the
    kernel delivered the signal to (a flap thread, for example), so a handler
    that calls this is deferred reliably; a thread signal mask alone is not.
    """
    if _ALARM["depth"] > 0:
        _ALARM["pending"] = True
        return True
    return False


@contextlib.contextmanager
def deferred_alarm(sig=_signal):
    """Hold SIGALRM for the critical section and deliver it when the section ends.

    Two layers: the thread signal mask (main thread), and alarm_should_defer()
    for handlers that run while the section is active. A deferred alarm is
    re-raised on exit, except when the section ends in CompensationFailed:
    then it is discarded so a timeout cannot replace the contamination stop.
    """
    masked = hasattr(sig, "pthread_sigmask") and hasattr(sig, "SIGALRM")
    previous = sig.pthread_sigmask(sig.SIG_BLOCK, {sig.SIGALRM}) if masked else None
    _ALARM["depth"] += 1
    stopped = False
    try:
        yield
    except CompensationFailed:
        stopped = True
        if masked and sig.SIGALRM in sig.sigpending():
            sig.sigtimedwait({sig.SIGALRM}, 0)
        raise
    finally:
        _ALARM["depth"] -= 1
        redeliver = _ALARM["pending"] and _ALARM["depth"] == 0 and not stopped
        if _ALARM["depth"] == 0:
            _ALARM["pending"] = False
        if masked:
            sig.pthread_sigmask(sig.SIG_SETMASK, previous)
        if redeliver and hasattr(sig, "raise_signal") and hasattr(sig, "SIGALRM"):
            sig.raise_signal(sig.SIGALRM)


def _read(read_routes):
    try:
        return read_routes(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_lifecycle(
    capability,
    routes_before,
    *,
    execute,
    read_routes,
    restore_route,
    measure_partial=None,
    mask=deferred_alarm,
):
    """Run one authorized action and compensate a partial one. See module docstring.

    routes_before: {router: spec} from the boundary's own fresh read.
    execute(capability) -> result dict; read_routes() -> {router: spec};
    restore_route(router, spec) raises on failure; measure_partial(view) is
    called before any compensation. Returns the outcome dict.
    """
    errors = capability_errors(capability)
    if errors:
        raise PermissionError(errors[0])
    if set(routes_before) != set(ctl.ROUTE_TARGETS) or not all(routes_before.values()):
        raise ValueError(f"routes before the action are incomplete: {routes_before}")
    route_changing = capability.action in ROUTE_CHANGING
    outcome = {
        "decision_id": capability.decision_id,
        "execution_result": None,
        "execution_error": None,
        "routes_before": dict(routes_before),
        "routes_after": None,
        "routes_changed": None,
        "route_read_error": None,
        "partial_application": None,
        "fully_applied": None,
        "compensation": None,
    }
    try:
        _lifecycle_section(
            capability, routes_before, outcome, route_changing,
            execute=execute, read_routes=read_routes, restore_route=restore_route,
            measure_partial=measure_partial, mask=mask,
        )
    except BaseException as exc:
        # A deferred timeout delivered when the section ends must not lose the
        # outcome: it travels on the exception (CompensationFailed has its own).
        if not isinstance(exc, CompensationFailed):
            try:
                exc.gated_outcome = outcome
            except Exception:
                pass
        raise
    return outcome


def _lifecycle_section(
    capability, routes_before, outcome, route_changing, *, execute, read_routes, restore_route, measure_partial, mask
):
    with mask():
        try:
            outcome["execution_result"] = execute(capability)
        except Exception as exc:
            outcome["execution_error"] = f"{type(exc).__name__}: {exc}"
        after, read_error = _read(read_routes)
        if after is None:
            after, read_error = _read(read_routes)
        elif route_changing and after == routes_before:
            confirm, confirm_error = _read(read_routes)
            if confirm is not None and confirm != routes_before:
                after = confirm
        outcome["routes_after"] = after
        outcome["route_read_error"] = read_error if after is None else None
        result = outcome["execution_result"]
        executed = isinstance(result, dict) and result.get("executed") is True
        if after is None:
            changed = None
        else:
            changed = sorted(router for router in routes_before if after.get(router) != routes_before[router])
        outcome["routes_changed"] = changed
        if route_changing:
            outcome["fully_applied"] = bool(
                after is not None
                and outcome["execution_error"] is None
                and executed
                and all(ctl.path_from_route(router, after.get(router)) == capability.reroute_path for router in routes_before)
            )
        else:
            outcome["fully_applied"] = outcome["execution_error"] is None
        if after is not None:
            outcome["partial_application"] = bool(route_changing and changed and not outcome["fully_applied"])
        needs_restore = route_changing and not outcome["fully_applied"] and (after is None or bool(changed))
        if needs_restore:
            if measure_partial is not None:
                try:
                    measure_partial(dict(outcome))
                except Exception as exc:
                    outcome["partial_measure_error"] = f"{type(exc).__name__}: {exc}"
            routers = list(changed) if after is not None else list(routes_before)
            compensation = {
                "decision_id": capability.decision_id,
                "attempted": True,
                "routers": routers,
                "errors": [],
                "verified": False,
                "routes_restored": None,
            }
            outcome["compensation"] = compensation
            for router in routers:
                try:
                    restore_route(router, routes_before[router])
                except Exception as exc:
                    compensation["errors"].append(f"{router}: {type(exc).__name__}: {exc}")
            first, first_error = _read(read_routes)
            second, second_error = _read(read_routes)
            compensation["routes_restored"] = second if second is not None else first
            compensation["verified"] = first == routes_before and second == routes_before
            if not compensation["verified"]:
                detail = "; ".join(compensation["errors"]) or first_error or second_error or (
                    f"routers show {compensation['routes_restored']}, expected {routes_before}"
                )
                raise CompensationFailed(
                    f"compensation for decision {capability.decision_id} not verified: {detail}", outcome
                )
