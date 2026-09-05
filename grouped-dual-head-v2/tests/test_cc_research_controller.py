"""Tests for scripts/cc_research_controller.py.

Uses only stdlib (pytest + subprocess shim).  No real research subprocesses are
run: every launch goes through the module's own ``fake-runner`` entry point,
which writes configurable outputs and exits with a configurable code.  No GPU,
no validation/test_id access, no network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "cc_research_controller.py"

sys.path.insert(0, str(PROJECT_ROOT))
import scripts.cc_research_controller as cc  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_plan(base_dir, tasks, plan_hash=None):
    """Build and write a plan.json; returns (plan, computed_sha)."""
    computed = cc.compute_plan_hash({"tasks": tasks})
    sha = plan_hash or computed
    plan = {
        "schema": cc.PLAN_SCHEMA,
        "plan_hash": sha,
        "plan_ref": "frozen-20260905-t1",
        "tasks": tasks,
    }
    cc.write_atomic(base_dir / "plan.json", cc.canonical_json(plan) + "\n")
    return plan, sha


def write_receipt(base_dir, task_id, plan_sha):
    rcpt_dir = base_dir / "receipts"
    rcpt_dir.mkdir(parents=True, exist_ok=True)
    cc.write_atomic(
        rcpt_dir / f"{task_id}.json",
        cc.canonical_json({"id": task_id, "plan_hash": plan_sha}) + "\n",
    )


def fake_runner() -> list:
    return [sys.executable, str(SCRIPT), "fake-runner"]


def task_command(config_path):
    """Task command argv that points the fake runner at a config file."""
    return {"argv": [str(config_path)], "prompt_file": ""}


def write_fake_cfg(base_dir, task_id, write_outputs=None, exit_code=0,
                   output_content="ok"):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "base_dir": str(base_dir),
        "task_id": task_id,
        "exit_code": exit_code,
        "write_outputs": write_outputs,
        "output_content": output_content,
    }
    p = base_dir / f".fakecfg-{task_id}.json"
    p.write_text(cc.canonical_json(cfg), encoding="utf-8")
    return p


def wait_for_exit(proc, timeout=10.0):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise


# --------------------------------------------------------------------------- #
# 1. Idempotency / duplicate submission
# --------------------------------------------------------------------------- #
def test_duplicate_submission_does_not_launch_twice(tmp_path):
    base = tmp_path / "state"
    cfg = write_fake_cfg(base, "t1", write_outputs=["out/t1.txt"])
    tasks = [
        {
            "id": "t1",
            "plan_hash": "",  # filled by build_plan authoring below
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/t1.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg),
        }
    ]
    # Author the plan so the task's plan_hash matches the frozen hash.
    _, sha = build_plan(base, tasks, plan_hash=None)
    # Rewrite tasks with the correct per-task plan_hash.
    author_sha = cc.compute_plan_hash({"tasks": tasks})
    for t in tasks:
        t["plan_hash"] = author_sha
    build_plan(base, tasks, plan_hash=author_sha)
    write_receipt(base, "t1", author_sha)

    ctrl = cc.ResearchController(base, runner=fake_runner())
    run1 = ctrl.run()
    assert run1["task_states"]["t1"] == "completed"
    assert run1["launch_count"] == 1

    # Second submission: same task + same plan_hash -> no relaunch.
    ctrl2 = cc.ResearchController(base, runner=fake_runner())
    run2 = ctrl2.run()
    assert run2["task_states"]["t1"] == "completed"
    assert run2["launch_count"] == 0, "duplicate submission must not relaunch"

    state = cc.read_state(base)
    assert state["tasks"]["t1"]["state"] == "completed"


# --------------------------------------------------------------------------- #
# 2. Hash drift rejection
# --------------------------------------------------------------------------- #
def test_hash_drift_rejected(tmp_path):
    base = tmp_path / "state"
    cfg = write_fake_cfg(base, "t1", write_outputs=["out/t1.txt"])
    tasks = [
        {
            "id": "t1",
            "plan_hash": "deadbeef" * 8,  # wrong hash, drift vs frozen plan
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/t1.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg),
        }
    ]
    frozen = cc.compute_plan_hash({"tasks": tasks})
    build_plan(base, tasks, plan_hash=frozen)
    write_receipt(base, "t1", frozen)

    ctrl = cc.ResearchController(base, runner=fake_runner())
    run = ctrl.run()
    assert run["task_states"]["t1"] == "rejected"
    state = cc.read_state(base)
    assert state["tasks"]["t1"]["result"] == "hash_drift"
    assert not (base / "out" / "t1.txt").exists(), "rejected task must not run"


# --------------------------------------------------------------------------- #
# 3. Dependency gate
# --------------------------------------------------------------------------- #
def test_dependency_gate_blocks_until_dep_resolves(tmp_path):
    base = tmp_path / "state"
    cfg_a = write_fake_cfg(base, "a", write_outputs=["out/a.txt"])
    cfg_b = write_fake_cfg(base, "b", write_outputs=["out/b.txt"])
    tasks = [
        {
            "id": "a",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/a.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg_a),
        },
        {
            "id": "b",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": ["a"],
            "expected_outputs": ["out/b.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg_b),
        },
    ]
    sha = cc.compute_plan_hash({"tasks": tasks})
    for t in tasks:
        t["plan_hash"] = sha
    build_plan(base, tasks, plan_hash=sha)
    # No receipt for "a": it stays awaiting_codex_decision, so b is gated.
    write_receipt(base, "b", sha)

    ctrl = cc.ResearchController(base, runner=fake_runner())
    run = ctrl.run()
    # a awaits codex; b is gated on an unresolved dependency => never launched.
    assert run["task_states"]["a"] == "awaiting_codex_decision"
    assert run["task_states"]["b"] in ("queued", "blocked", "awaiting")
    state = cc.read_state(base)
    assert state["tasks"]["b"]["result"] in ("awaiting_dependency", "dep_blocked")
    assert not (base / "out" / "b.txt").exists(), "gated dependency must not run"


# --------------------------------------------------------------------------- #
# 4. exit-0-but-missing-artifacts => rejected, dependents NOT triggered
# --------------------------------------------------------------------------- #
def test_exit0_missing_artifacts_rejected_and_no_dependents(tmp_path):
    base = tmp_path / "state"
    # Task "up" exits 0 but writes NOTHING (write_outputs=[]), so its expected
    # output is absent -> rejected.  Task "down" depends on it.
    cfg_up = write_fake_cfg(base, "up", write_outputs=[], exit_code=0)
    cfg_down = write_fake_cfg(base, "down", write_outputs=["out/down.txt"])
    tasks = [
        {
            "id": "up",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/up.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg_up),
        },
        {
            "id": "down",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": ["up"],
            "expected_outputs": ["out/down.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg_down),
        },
    ]
    sha = cc.compute_plan_hash({"tasks": tasks})
    for t in tasks:
        t["plan_hash"] = sha
    build_plan(base, tasks, plan_hash=sha)
    write_receipt(base, "up", sha)
    write_receipt(base, "down", sha)

    ctrl = cc.ResearchController(base, runner=fake_runner())
    run = ctrl.run()
    state = cc.read_state(base)
    assert state["tasks"]["up"]["state"] == "rejected"
    assert state["tasks"]["up"]["result"] == "missing_outputs"
    # Dependent must NOT be triggered.
    assert state["tasks"]["down"]["state"] in ("blocked", "queued")
    assert not (base / "out" / "down.txt").exists(), "dependent must not run"


# --------------------------------------------------------------------------- #
# 5. Lock contention: live owner is never preempted
# --------------------------------------------------------------------------- #
def test_lock_live_owner_not_preempted(tmp_path):
    base = tmp_path / "state"
    lock1 = cc.ControllerLock(base)
    assert lock1.acquire(block=True)
    # Second constructor must fail/block while the owner is alive.
    lock2 = cc.ControllerLock(base)
    assert lock2.acquire(block=False) is False, "live owner must not be preempted"

    # recover_stale_lock must NOT steal from a live owner.
    record = cc.read_lock_record(base)
    assert record is not None
    report = cc.ResearchController(base).recover_stale_lock()
    assert report["status"] == "not_reclamable"
    assert report["reason"] in ("alive", "ambiguous", "unknown")

    lock1.release()
    # After release, a new acquire succeeds.
    assert lock2.acquire(block=False) is True
    lock2.release()


# --------------------------------------------------------------------------- #
# 6. PID reuse: stale record with reused PID is NOT reclaimed blindly
# --------------------------------------------------------------------------- #
def test_pid_reuse_not_reclaimed(tmp_path):
    base = tmp_path / "state"
    base.mkdir(parents=True, exist_ok=True)
    # Spawn a live process, then write a lock record whose start_time is WRONG
    # for that PID (simulating a reused PID / stale start time identity).
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pid = proc.pid
        actual_start = cc.process_start_time(pid)
        assert actual_start is not None, "need a real start time for the live PID"
        stale_start = (actual_start or 0) - 5  # wrong identity for this PID
        cc.write_atomic(
            base / "controller.lock",
            cc.canonical_json(
                {"pid": pid, "start_time": stale_start, "acquired_utc": cc.utc_now(),
                 "candidate": cc.CANDIDATE}
            )
            + "\n",
        )
        report = cc.ResearchController(base).recover_stale_lock()
        # PID is alive but start time mismatch (reuse ambiguity) -> must NOT reclaim.
        assert report["status"] == "not_reclamable"
        assert report["reason"] == "ambiguous"
    finally:
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------- #
# 7. Incomplete write: atomic tmp+rename never yields torn state
# --------------------------------------------------------------------------- #
def test_atomic_write_never_torn(tmp_path):
    base = tmp_path / "state"
    base.mkdir(parents=True, exist_ok=True)

    # A large-ish state so a trivially small write is unlikely to be torn.
    payload = {
        "schema": cc.STATE_SCHEMA,
        "tasks": {
            f"t{i}": {
                "id": f"t{i}",
                "state": "running",
                "heartbeat_utc": cc.utc_now(),
                "payload": "x" * 5000,
            }
            for i in range(50)
        },
    }

    errors = []

    def writer():
        for _ in range(300):
            cc.write_state(base, payload)
            cc.write_state(base, {**payload, "seq": "y" * 100})
            time.sleep(0.001)

    def reader():
        for _ in range(300):
            s = cc.read_state(base)
            if s is None:
                continue
            # Torn/half state would be invalid JSON (json.load already guards),
            # or would be missing required top-level keys / truncated task keys.
            if "schema" not in s:
                errors.append("missing schema")
                return
            if not all(f"t{i}" in s["tasks"] for i in range(50)):
                errors.append("torn task dict")
                return

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    wt.start()
    rt.start()
    wt.join()
    rt.join()
    assert not errors, f"read torn/half state: {errors}"


# --------------------------------------------------------------------------- #
# 8. UTC / heartbeat timestamps: program-generated, not future, fields split
# --------------------------------------------------------------------------- #
def test_utc_heartbeat_fields(tmp_path):
    base = tmp_path / "state"
    cfg = write_fake_cfg(base, "t1", write_outputs=["out/t1.txt"])
    tasks = [
        {
            "id": "t1",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/t1.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg),
        }
    ]
    sha = cc.compute_plan_hash({"tasks": tasks})
    for t in tasks:
        t["plan_hash"] = sha
    build_plan(base, tasks, plan_hash=sha)
    write_receipt(base, "t1", sha)

    ctrl = cc.ResearchController(base, runner=fake_runner())
    ctrl.run()
    state = cc.read_state(base)

    ref = datetime.now(timezone.utc)
    # Parse and require timestamps are not in the future and are valid ISO.
    ts_fields = [
        state["heartbeat_utc"],
        state["last_substantive_event_utc"],
        state["tasks"]["t1"]["heartbeat_utc"],
        state["tasks"]["t1"]["last_substantive_event_utc"],
    ]
    for field in ts_fields:
        parsed = datetime.fromisoformat(field)
        assert parsed.tzinfo is not None, "timestamps must carry tz"
        assert (parsed - ref).total_seconds() <= 5, f"future timestamp: {field}"
    # The two fields must be independent keys (split), as required.
    assert "heartbeat_utc" in state["tasks"]["t1"]
    assert "last_substantive_event_utc" in state["tasks"]["t1"]
    assert state["tasks"]["t1"]["heartbeat_utc"] != state["plan_hash"]  # sanity


# --------------------------------------------------------------------------- #
# 9. awaiting_codex_decision: no approved task => clear awaiting, no fake wake
# --------------------------------------------------------------------------- #
def test_awaiting_codex_decision_no_fake_wake(tmp_path):
    base = tmp_path / "state"
    cfg = write_fake_cfg(base, "t1", write_outputs=["out/t1.txt"])
    tasks = [
        {
            "id": "t1",
            "plan_hash": "",
            "allowed_writes": ["out/*"],
            "deps": [],
            "expected_outputs": ["out/t1.txt"],
            "budget": {"timeout_s": 30, "max_output_bytes": 1000},
            "command": task_command(cfg),
        }
    ]
    sha = cc.compute_plan_hash({"tasks": tasks})
    for t in tasks:
        t["plan_hash"] = sha
    build_plan(base, tasks, plan_hash=sha)
    # NO receipt written -> consumer has not approved.

    ctrl = cc.ResearchController(base, runner=fake_runner())
    run = ctrl.run()
    assert run["status"] == "awaiting_codex_decision"
    assert run["any_awaiting"] is True
    state = cc.read_state(base)
    assert state["tasks"]["t1"]["state"] == "awaiting_codex_decision"
    assert state["tasks"]["t1"]["result"] == "awaiting_receipt"
    assert not (base / "out" / "t1.txt").exists(), "must NOT fake-wake / run"


# --------------------------------------------------------------------------- #
# Helpers: summarize_stream_json + resume argv
# --------------------------------------------------------------------------- #
def test_summarize_stream_json(tmp_path):
    log = tmp_path / "stream.jsonl"
    lines = [
        {"type": "system", "subtype": "api_retry", "message": "retry 1"},
        {"type": "stream_event", "partial_json": '{"text":"hi"}'},
        {"type": "stream_event", "delta": "x", "partial_json": '{"text":"hi"}'},
        {"type": "error", "error": "boom"},
        "not json at all",
        {"type": "system", "subtype": "init"},
    ]
    log.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    stats = cc.summarize_stream_json(str(log))
    assert stats["event_count"] == 6
    assert stats["api_retry_count"] == 1
    assert stats["partial_count"] == 2
    assert stats["error_count"] == 2  # one 'error' event + one malformed line
    assert stats["last_event_type"] in ("system::init", "error")


def test_resume_argv_and_already_running(tmp_path):
    base = tmp_path / "state"
    base.mkdir(parents=True, exist_ok=True)
    ctrl = cc.ResearchController(base, runner=["claude", "-p"])
    argv = ctrl.build_resume_argv("sess-123")
    assert argv == ["claude", "-p", "--resume", "sess-123"]

    # Mark a task as running with the same resume id -> report already_running.
    state = {
        "schema": cc.STATE_SCHEMA,
        "tasks": {"t1": {"state": "running", "resume_id": "sess-123"}},
    }
    cc.write_state(base, state)
    rep = ctrl.report_resume("sess-123")
    assert rep["status"] == "already_running"
    rep2 = ctrl.report_resume("sess-999")
    assert rep2["status"] == "not_running"
    assert rep2["argv"] == ["claude", "-p", "--resume", "sess-999"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
