"""B0: execute the agent proposal without a gate.

`decide()` always allows. Harm is measured on the live network in `run()`,
then the previous path is restored. Decision-only callers never change
the network.
"""

from control.actions import execute, proposal_to_action

try:
    from network import controller as ctl
except ImportError:
    import controller as ctl


def decide(proposal, snapshot, test_results):
    """Decision only. Ungated execution stays in run()."""
    return {
        "method": "B0",
        "decision": "allow",
        "action": (proposal or {}).get("action"),
        "tests_run": [],
        "reason": "ungated",
    }


def _harm(before, after):
    """True when the post-action T1 reading is worse than the pre-action one."""
    if not before or not after:
        return False
    if after.get("error") and not before.get("error"):
        return True
    before_loss = before.get("packet_loss_percent")
    after_loss = after.get("packet_loss_percent")
    if before_loss is None:
        before_loss = 0.0
    if after_loss is None:
        after_loss = 100.0
    if after_loss >= 100.0 and before_loss < 100.0:
        return True
    if after_loss > before_loss + 5.0:
        return True
    if before.get("connectivity") is True and after.get("connectivity") is False:
        return True
    before_lat = before.get("latency_ms")
    after_lat = after.get("latency_ms")
    if (
        isinstance(before_lat, (int, float))
        and isinstance(after_lat, (int, float))
        and after_lat > max(100.0, before_lat * 2.0)
    ):
        return True
    return False


def run(net, proposal, measure=True, noise_id=None):
    """Execute, measure post-action recovery (same battery as VeriFix), then restore.

    Harm remains a T1 before/after comparison. Recovery status comes from the
    independent post-remediation OAM battery and is never inferred from
    execution.executed. noise_id grades recovery with the episode traffic level.
    """
    from bench.tests import T1_e2e_ping
    from bench.run_episode import measure_traffic_recovery, traffic_conditions

    if noise_id is None:
        raise ValueError("B0 run requires noise_id (episode traffic level)")

    before_path = None
    try:
        before_path = ctl.get_active_path(net)
    except Exception as exc:
        before_path = None
        path_error = f"{type(exc).__name__}: {exc}"
    else:
        path_error = None

    before_t1 = None
    after_t1 = None
    post_oam = None
    if measure:
        try:
            before_t1 = T1_e2e_ping(net)
        except Exception as exc:
            before_t1 = {"test_id": "T1_e2e_ping", "error": f"{type(exc).__name__}: {exc}"}

    try:
        action, payload = proposal_to_action(proposal)
        execution = execute(net, action, payload)
    except Exception as exc:
        execution = {"executed": False, "error": f"{type(exc).__name__}: {exc}"}

    # Equivalent post-action opportunity: measure whenever asked, including
    # DO_NOTHING (executed=False). Command failure does not skip measurement.
    # Harm still uses T1; if the recovery battery lacks a usable T1 row, fall
    # back to a dedicated T1 so infra gaps are not scored as harm.
    if measure:
        try:
            post_oam = measure_traffic_recovery(net, noise_id=noise_id)
        except Exception as exc:
            post_oam = {
                "recovery_status": "measurement_unavailable",
                "traffic_recovered": None,
                "error": f"{type(exc).__name__}: {exc}",
                "infrastructure_errors": {"measure": f"{type(exc).__name__}: {exc}"},
                "results": [],
            }
        after_t1 = None
        for row in post_oam.get("results") or []:
            if isinstance(row, dict) and row.get("test_id") == "T1_e2e_ping":
                after_t1 = row
                break
        if after_t1 is None or after_t1.get("error"):
            try:
                after_t1 = T1_e2e_ping(net)
            except Exception as exc:
                after_t1 = {"test_id": "T1_e2e_ping", "error": f"{type(exc).__name__}: {exc}"}

    harm = _harm(before_t1, after_t1)
    rolled_back = False
    rollback_error = path_error
    if execution.get("executed") and before_path in ("primary", "backup"):
        try:
            ctl.set_active_path(net, before_path)
            rolled_back = True
        except Exception as exc:
            rollback_error = f"{type(exc).__name__}: {exc}"

    tests_run = []
    if isinstance(before_t1, dict) and before_t1.get("test_id"):
        tests_run.append(before_t1["test_id"])
    for row in (post_oam or {}).get("results") or []:
        tid = row.get("test_id") if isinstance(row, dict) else None
        if tid and tid not in tests_run:
            tests_run.append(tid)
    post_conditions = traffic_conditions((post_oam or {}).get("results") or [])

    return {
        "method": "B0",
        "decision": "allow",
        "action": (proposal or {}).get("action"),
        "execution": execution,
        "harm": harm,
        "rolled_back": rolled_back,
        "before_path": before_path,
        "before_t1": before_t1,
        "after_t1": after_t1,
        "post_oam": post_oam,
        "traffic_recovered": (post_oam or {}).get("traffic_recovered"),
        "recovery_status": (post_oam or {}).get("recovery_status"),
        "post_action": (post_oam or {}).get("post_action") or {
            "connectivity": post_conditions.get("connectivity"),
            "latency_ms": post_conditions.get("latency_ms"),
            "packet_loss_percent": post_conditions.get("packet_loss_percent"),
            "throughput_mbit": post_conditions.get("throughput_mbit"),
        },
        "rollback_error": rollback_error,
        "tests_run": tests_run,
    }
