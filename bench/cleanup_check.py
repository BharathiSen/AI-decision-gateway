"""Cleanup-halt regressions. Offline, no Mininet.

python -m bench.cleanup_check

A failed reset or cleanup must stop the experiment. No further episode may
run on a topology that can still carry the previous fault.
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Also runnable as `python bench/cleanup_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fakes(events, pre=None, post=None, clear_raises=False):
    from gate.evidence_check import _snapshot

    snap = _snapshot()

    def noise(net, noise_id, links=None):
        events.append("noise")
        return {}

    def clear(net, applied, links=None):
        events.append("clear")
        if clear_raises:
            raise RuntimeError("tc qdisc del failed")

    def reset(net, baseline=False):
        events.append("reset+baseline" if baseline else "reset")
        return post if baseline else pre

    def inject(net, fault_id, seed=None):
        events.append("inject")
        return "corr-1"

    return {
        "apply_noise": noise,
        "clear_noise": clear,
        "reset_all": reset,
        "inject": inject,
        "telemetry": lambda net, links=None: snap,
        "tests": lambda net: {"results": []},
        "read_state": lambda net, links=None: snap,
    }


CLEAN = {"cleared": [{"correlation_id": "corr-1"}], "errors": [], "residual": {}, "healthy": True, "loss_percent": 0.0}


def _agent(state):
    return {"action": "DO_NOTHING", "target": "HostA_to_HostB", "reason": "x", "proposed_path": []}


def main():
    from bench import inject as inj
    from bench.run_episode import TopologyContaminated, run_episode, teardown_problem

    # Clean episode: teardown recorded as clean.
    events = []
    rec = run_episode(None, None, noise_id="N0", fault_id="F1", agent_fn=_agent, gate_fn=None,
                      _components=_fakes(events, pre=CLEAN, post=CLEAN), run_oam=False)
    assert rec["teardown"]["clean"] is True and rec["teardown"]["cleared"] == 1, rec.get("teardown")

    # Dirty pre-episode reset: stop before injecting anything.
    events = []
    dirty = dict(CLEAN, residual={"corr-0": {"qdisc": "netem loss 20%"}})
    try:
        run_episode(None, None, noise_id="N0", fault_id="F1", agent_fn=_agent, _components=_fakes(events, pre=dirty, post=CLEAN), run_oam=False)
    except TopologyContaminated as exc:
        assert "pre-episode reset" in str(exc) and "residual" in str(exc), exc
    else:
        raise AssertionError("dirty pre-reset did not stop the episode")
    assert "inject" not in events, events
    assert events[-1] == "reset+baseline", "teardown must still run"

    # Dirty teardown: errors, residual, unhealthy, clear_noise failure.
    for post, clear_raises, needle in (
        (dict(CLEAN, errors=[{"correlation_id": "corr-1", "error": "RTNETLINK busy"}]), False, "reset errors"),
        (dict(CLEAN, residual={"corr-1": {"route": "r1 via r3"}}), False, "residual"),
        (dict(CLEAN, healthy=False, loss_percent=40.0), False, "loss 40.0%"),
        (CLEAN, True, "clear_noise"),
    ):
        events = []
        try:
            run_episode(None, None, noise_id="N0", fault_id="F1", agent_fn=_agent,
                        _components=_fakes(events, pre=CLEAN, post=post, clear_raises=clear_raises), run_oam=False)
        except TopologyContaminated as exc:
            assert needle in str(exc), (needle, str(exc))
            assert exc.record is not None and exc.record["teardown"]["clean"] is False
        else:
            raise AssertionError(f"teardown problem not raised: {needle}")

    # Pre-reset health is not judged (background traffic is still on).
    assert teardown_problem(dict(CLEAN, healthy=False), True, check_health=False) is None
    # The real reset must return a report; a fake may not.
    assert teardown_problem(None, real_reset=True) == "reset returned no report"
    assert teardown_problem(None, real_reset=False) is None
    assert teardown_problem({"errors": [], "residual": {}}, real_reset=True) == "reset report has no health check"
    assert callable(inj.reset_all)

    # The pilot loop stops (rc 4), does not count the episode, writes a halt file.
    import bench.run_pilot as rp

    out_dir = Path(tempfile.mkdtemp())
    out = out_dir / "halt_probe.json"
    real_rev, real_run = rp._git_revision, rp.run_episode
    calls = []

    def contaminated(*args, **kwargs):
        calls.append(kwargs.get("fault_id"))
        raise TopologyContaminated("episode teardown: residual changes", {"residual": {"x": 1}}, record={"episode_id": "e1"})

    rp._git_revision = lambda cwd=None: {"commit": "a" * 40, "tree_sha256": "d" * 40, "untracked_source": []}
    rp.run_episode = contaminated
    err = io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = rp.run_loop(rp._parser().parse_args(["--stub", "--faults", "H0,F1,F3", "--out", str(out)]), None, None)
    finally:
        rp._git_revision, rp.run_episode = real_rev, real_run
    assert rc == 4, (rc, err.getvalue())
    assert calls == ["H0"], f"episodes ran after contamination: {calls}"
    halt = json.loads(rp.halt_path(out).read_text(encoding="utf-8"))
    assert halt["episode_index"] == 0 and halt["episode_record"]["episode_id"] == "e1"
    assert not rp._state_path(out).exists(), "a halted episode must not advance resume state"
    assert "sudo mn -c" in err.getvalue()
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)

    # E5: reset_all / clear_noise exceptions surface as TopologyContaminated.
    events = []

    def boom_reset(net, baseline=False):
        events.append("reset+baseline" if baseline else "reset")
        if baseline:
            raise RuntimeError("stop_all_flaps exploded")
        return CLEAN

    try:
        run_episode(
            None,
            None,
            noise_id="N0", fault_id="F1",
            agent_fn=_agent,
            run_oam=False,
            _components=_fakes(events, pre=CLEAN, post=CLEAN) | {"reset_all": boom_reset},
        )
        raise AssertionError("reset exception must become TopologyContaminated")
    except TopologyContaminated as exc:
        assert "stop_all_flaps exploded" in str(exc), exc
        assert exc.__cause__ is not None
        assert exc.record is not None

    # E5: noise applied before a body failure still reaches cleanup.
    events = []
    noise_states = []

    def tracking_noise(net, noise_id, links=None):
        events.append("noise")
        noise_states.append("applied")
        return {"noise_id": noise_id}

    def tracking_clear(net, applied, links=None):
        events.append("clear")
        noise_states.append("cleared")

    def fail_pre(net, baseline=False):
        events.append("reset+baseline" if baseline else "reset")
        if baseline:
            return CLEAN
        raise RuntimeError("pre-reset hard fail")

    try:
        run_episode(
            None,
            None,
            noise_id="N0", fault_id="F1",
            agent_fn=_agent,
            run_oam=False,
            _components={
                **_fakes(events, pre=CLEAN, post=CLEAN),
                "apply_noise": tracking_noise,
                "clear_noise": tracking_clear,
                "reset_all": fail_pre,
            },
        )
        raise AssertionError("pre-reset exception must contaminate")
    except TopologyContaminated as exc:
        assert "pre-reset hard fail" in str(exc) or "episode teardown" in str(exc), exc
    assert events[0] == "noise" and "clear" in events and events[-1] == "reset+baseline", events
    assert noise_states == ["applied", "cleared"], noise_states

    # E6: timeout during teardown must halt as contamination, not a partial episode.
    import signal

    from bench.run_batch import run_one

    if hasattr(signal, "SIGALRM"):
        alarms_cleared = []

        def teardown_ok(net, baseline=False):
            if not baseline:
                return CLEAN
            alarms_cleared.append(True)
            return CLEAN

        comps = _fakes([], pre=CLEAN, post=CLEAN)
        comps["reset_all"] = teardown_ok
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(60)
        try:
            run_episode(
                None,
                None,
                noise_id="N0", fault_id="H0",
                agent_fn=_agent,
                run_oam=False,
                _components=comps,
            )
        finally:
            remaining = signal.alarm(0)
        assert remaining == 0, "teardown must cancel pending SIGALRM"
        assert alarms_cleared, "baseline teardown must still run"

    def batch_boom(net, baseline=False):
        if baseline:
            raise TimeoutError("teardown timed out")
        return CLEAN

    import bench.run_batch as rb

    real_run = rb.run_episode

    def wrapped_episode(*args, **kwargs):
        kwargs = dict(kwargs)
        comps2 = _fakes([], pre=CLEAN, post=CLEAN)
        comps2["reset_all"] = batch_boom
        kwargs["_components"] = comps2
        kwargs["agent_fn"] = _agent
        kwargs["gate_fn"] = None
        kwargs["run_oam"] = False
        return real_run(*args, **kwargs)

    saved = rb.run_episode
    rb.run_episode = wrapped_episode
    try:
        try:
            run_one(
                None,
                None,
                {"fault_id": "H0", "noise_id": "N0", "seed": 1, "strategy": "verifix"},
                timeout_seconds=0,
            )
            raise AssertionError("teardown TimeoutError must halt as TopologyContaminated")
        except TopologyContaminated as exc:
            assert "timeout" in str(exc).lower() or "teardown" in str(exc).lower(), exc
    finally:
        rb.run_episode = saved

    print("cleanup halt: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
