"""Minimal file-based research task controller.

Deliberately small.  It models a *plan-driven* research task runner over the
filesystem: it reads a frozen ``plan.json``, accepts a task only when a consumer
(the "Codex" side of the loop) has written a matching **receipt**, launches the
task's ``argv`` with the prompt fed on stdin, and reconciles the result into an
explicit state machine.

Design intent (see ``docs/cc_research_system_20260905.md``):

* Everything the controller does is persisted in a small, atomic state file.
* A task is never accepted on PID liveness alone; acceptance requires a consumer
  receipt carrying the matching ``task_id`` + ``plan_hash``.
* A task that exits 0 but whose expected outputs are missing is ``rejected``
  (never ``completed``) and does NOT unblock its dependents.
* Duplicate submission is idempotent; recovery never re-launches a task that is
  already ``dispatched``/``running``/terminal unless ``--force`` resets it.
* A live lock owner is never preempted by expiry; stale-lock reclamation first
  verifies the owner PID and its start time (PID-reuse ambiguity blocks reclaim).

Only stdlib is used.  Tests inject a tiny fake runner through ``--runner``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# Frozen identity / schema
# --------------------------------------------------------------------------- #
CANDIDATE = "cc_research_controller_v1_20260905"
PLAN_SCHEMA = "cc_plan_v1"
STATE_SCHEMA = "cc_state_v1"


# --------------------------------------------------------------------------- #
# Time: program-generated UTC isoformat + monotonic seconds
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """Program-generated UTC ISO-8601 timestamp (never hand-filled or future)."""
    return datetime.now(timezone.utc).astimezone(timezone.utc).isoformat()


def monotonic_now() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------- #
# Canonical hashing
# --------------------------------------------------------------------------- #
def canonical_json(payload, indent=None) -> str:
    if indent is None:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return json.dumps(payload, sort_keys=True, indent=indent, ensure_ascii=True)


def _canonical_plan_dict(tasks) -> dict:
    """Canonical view of a task list (plan_hash field intentionally omitted, so
    a mutated task hash is detectable as drift rather than folded into the sha)."""
    return {
        "schema": PLAN_SCHEMA,
        "tasks": [
            {
                "id": t["id"],
                "allowed_writes": list(t.get("allowed_writes", [])),
                "deps": list(t.get("deps", [])),
                "expected_outputs": list(t.get("expected_outputs", [])),
                "budget": {
                    "timeout_s": int(t.get("budget", {}).get("timeout_s", 0)),
                    "max_output_bytes": int(t.get("budget", {}).get("max_output_bytes", 0)),
                },
                "command": {
                    "argv": list(t.get("command", {}).get("argv", [])),
                    "prompt_file": t.get("command", {}).get("prompt_file", ""),
                },
            }
            for t in tasks
        ],
    }


def compute_plan_hash(plan: dict) -> str:
    """sha256 of the canonical task list.  This is the authoring-side hash a
    plan author writes into ``plan.plan_hash`` so the controller can detect
    hash drift."""
    canon = canonical_json(_canonical_plan_dict(plan.get("tasks", [])))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def plan_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_plan_sha(plan: dict) -> str:
    """The frozen plan hash: prefer the stored ``plan_hash`` (the real frozen
    ref), else recompute from the canonical task list."""
    if isinstance(plan.get("plan_hash"), str) and plan["plan_hash"]:
        return plan["plan_hash"]
    return compute_plan_hash(plan)


# --------------------------------------------------------------------------- #
# stream-json condensation (never hold the raw log in memory)
# --------------------------------------------------------------------------- #
def _event_type(obj: dict) -> str:
    t = obj.get("type", "")
    subtype = obj.get("subtype", "")
    if t == "system":
        return f"system::{subtype}" if subtype else "system"
    if t == "stream_event":
        return "stream_event"
    return t or "unknown"


def summarize_stream_json(path: str) -> dict:
    """Read a stream-json log line by line and return a tiny stats dict."""
    event_count = 0
    api_retry_count = 0
    partial_count = 0
    error_count = 0
    last_event_type = None
    last_error = None

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event_count += 1
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                error_count += 1
                last_error = "malformed-json-line"
                last_event_type = "malformed"
                continue
            if not isinstance(obj, dict):
                error_count += 1
                last_error = "non-object-line"
                last_event_type = "malformed"
                continue

            et = _event_type(obj)
            last_event_type = et

            if et.startswith("system::api_retry") or obj.get("is_retry", False):
                api_retry_count += 1
            if et == "stream_event":
                partial_count += 1
            if et == "error" or obj.get("is_error", False):
                error_count += 1
                last_error = obj.get("error") or obj.get("message") or str(obj)

    return {
        "event_count": event_count,
        "api_retry_count": api_retry_count,
        "partial_count": partial_count,
        "last_event_type": last_event_type,
        "error_count": error_count,
        "last_error": last_error,
    }


# --------------------------------------------------------------------------- #
# Atomic filesystem I/O
# --------------------------------------------------------------------------- #
_TMP_COUNTER = 0


def _tmp_name(name: str) -> str:
    global _TMP_COUNTER
    _TMP_COUNTER += 1
    return f".{name}.tmp-{os.getpid()}-{time.time_ns()}-{_TMP_COUNTER}"


def write_atomic(path: Path, data: str) -> None:
    """Write via temp + os.replace so readers always see a full file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / _tmp_name(path.name)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_state(base_dir: Path):
    path = base_dir / "state.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_state(base_dir: Path, state: dict) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(base_dir / "state.json", canonical_json(state) + "\n")


def read_plan(base_dir: Path):
    path = base_dir / "plan.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_head(text: str, limit: int = 400) -> str:
    return " ".join(str(text).split())[:limit]


# --------------------------------------------------------------------------- #
# Process identity
# --------------------------------------------------------------------------- #
def process_start_time(pid: int):
    """Return proc starttime (clock ticks) or None if PID absent/unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
        right = stat.rfind(")")
        if right < 0:
            return None
        fields = stat[right + 2 :].split()
        # After ') ' fields[0] == state(3) ... fields[19] == starttime(22).
        if len(fields) >= 20:
            return int(fields[19])
        return None
    except (OSError, ValueError, IndexError):
        return None


def lock_owner_status(record: dict) -> str:
    """Classify a lock record's owner.

    Returns one of:
      "alive"     - (pid, start_time) matches a live process -> do not reclaim.
      "ambiguous" - pid is live but start_time differs (PID reuse) -> do not reclaim.
      "gone"      - pid is not present in /proc (owner exited; pid free) -> reclaim safe.
      "unknown"   - record lacks pid/start_time -> cannot confirm identity -> do not reclaim.
    """
    if not record or record.get("pid") is None or record.get("start_time") is None:
        return "unknown"
    pid = int(record["pid"])
    start = int(record["start_time"])
    actual = process_start_time(pid)
    if actual is None:
        return "gone"
    if actual == start:
        return "alive"
    return "ambiguous"


def lock_owner_alive(record: dict) -> bool:
    """True if the owner must NOT be preempted (alive, ambiguous, or unknown)."""
    return lock_owner_status(record) != "gone"


# --------------------------------------------------------------------------- #
# Locking: flock() with PID + start-time bound into the lock file
# --------------------------------------------------------------------------- #
class ControllerLock:
    """A single-writer lock.  Owner PID + start time are persisted in the lock
    file.  flock() guarantees a live owner is never preempted; there is no timer
    that could steal a still-alive owner's lock."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lock_path = base_dir / "controller.lock"
        self._fd = None
        self.record = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, block: bool = True, timeout_s: float = 0.0) -> bool:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        flags = fcntl.LOCK_EX
        if not block:
            flags |= fcntl.LOCK_NB
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, flags)
                break
            except (BlockingIOError, OSError):
                if not block:
                    os.close(fd)
                    return False
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.05)

        self._fd = fd
        pid = os.getpid()
        start = process_start_time(pid)
        self.record = {
            "pid": pid,
            "start_time": start,
            "acquired_utc": utc_now(),
            "candidate": CANDIDATE,
        }
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, canonical_json(self.record).encode("utf-8"))
        os.fsync(fd)
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None
        self.record = None

    def __enter__(self):
        self.acquire(block=True)
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def read_lock_record(base_dir: Path):
    lock_path = base_dir / "controller.lock"
    if not lock_path.exists():
        return None
    try:
        with open(lock_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Task state machine
# --------------------------------------------------------------------------- #
TERMINAL_STATES = {"completed", "rejected", "failed"}
ACTIVE_STATES = {"dispatched", "accepted", "running"}
NO_RELAUNCH_STATES = ACTIVE_STATES | TERMINAL_STATES
RETRYABLE_STATES = {"queued", "awaiting", "blocked", "awaiting_codex_decision"}


def _expand_runner(runner):
    if runner is None:
        return []
    if isinstance(runner, (list, tuple)):
        return [str(x) for x in runner]
    return str(runner).split()


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class ResearchController:
    def __init__(self, base_dir, runner=None, lock_wait_s: float = 0.0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runner = _expand_runner(runner)
        self.lock_wait_s = lock_wait_s
        self.lock = ControllerLock(self.base_dir)
        self._launch_count = 0

    # ---- state primitives --------------------------------------------------- #
    def _init_state(self, plan: dict) -> dict:
        sha = load_plan_sha(plan)
        tasks = {}
        now = utc_now()
        for t in plan.get("tasks", []):
            tasks[t["id"]] = {
                "id": t["id"],
                "plan_hash": t.get("plan_hash", sha),
                "state": "queued",
                "deps": list(t.get("deps", [])),
                "allowed_writes": list(t.get("allowed_writes", [])),
                "expected_outputs": list(t.get("expected_outputs", [])),
                "budget": dict(t.get("budget", {}) or {}),
                "command": dict(t.get("command", {}) or {}),
                "exit_code": None,
                "pid": None,
                "start_time": None,
                "resume_id": None,
                "result": None,
                "heartbeat_utc": now,
                "last_substantive_event_utc": now,
            }
        return {
            "schema": STATE_SCHEMA,
            "candidate": CANDIDATE,
            "plan_hash": sha,
            "task_count": len(tasks),
            "tasks": tasks,
            "locked_by": None,
            "heartbeat_utc": now,
            "last_substantive_event_utc": now,
        }

    def load_or_init_state(self, plan: dict) -> dict:
        state = read_state(self.base_dir)
        if state is None:
            state = self._init_state(plan)
            write_state(self.base_dir, state)
            return state
        state.setdefault("plan_hash", load_plan_sha(plan))
        state.setdefault("candidate", CANDIDATE)
        state.setdefault("schema", STATE_SCHEMA)
        state.setdefault("tasks", {})
        state.setdefault("heartbeat_utc", utc_now())
        state.setdefault("last_substantive_event_utc", utc_now())
        return state

    def save_state(self, state: dict) -> None:
        write_state(self.base_dir, state)

    # ---- receipts ----------------------------------------------------------- #
    def _receipt_path(self, task_id: str) -> Path:
        return self.base_dir / "receipts" / f"{task_id}.json"

    def read_receipt(self, task_id: str):
        p = self._receipt_path(task_id)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            return None

    def _accept_from_receipt(self, task: dict, plan_sha: str):
        """Accept only if a consumer receipt confirms id + plan_hash.

        Never accepts on PID liveness.  A receipt for the wrong id, or whose
        plan_hash does not match the frozen plan, is a rejection.
        """
        receipt = self.read_receipt(task["id"])
        if receipt is None:
            return None
        rid = receipt.get("id")
        rsha = receipt.get("plan_hash")
        if rid != task["id"]:
            return "rejected"
        if rsha is None or rsha != plan_sha:
            return "rejected"
        if task.get("plan_hash") != plan_sha:
            return "rejected"
        return "accepted"

    # ---- dependency gating --------------------------------------------------- #
    def _deps_state(self, task: dict, state: dict) -> str:
        deps = task.get("deps", [])
        if not deps:
            return "ready"
        for d in deps:
            dep = state["tasks"].get(d)
            if dep is None:
                return "blocked"
            ds = dep.get("state")
            if ds == "completed":
                continue
            if ds in ("rejected", "failed", "blocked"):
                return "blocked"
            return "awaiting"
        return "ready"

    # ---- output presence ------------------------------------------------------ #
    def _outputs_present(self, task: dict, base_dir: Path) -> bool:
        expected = task.get("expected_outputs", [])
        if not expected:
            return True
        for pat in expected:
            if not (base_dir / pat).exists():
                return False
        return True

    # ---- single-task launch --------------------------------------------------- #
    def launch_task(self, task: dict, plan: dict, state: dict) -> dict:
        tid = task["id"]
        sha = load_plan_sha(plan)

        # Idempotency: do not re-launch active/terminal tasks (unless --force
        # elsewhere reset them to queued).
        if task.get("state") in NO_RELAUNCH_STATES:
            return {"status": "already_launched", "id": tid, "state": task.get("state")}

        # Frozen-plan hash drift rejection.
        if task.get("plan_hash") != sha:
            task["state"] = "rejected"
            task["result"] = "hash_drift"
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "rejected", "id": tid, "reason": "hash_drift"}

        # Dependency gate (fires before approval so an unapproved dep cannot be
        # bypassed by a present receipt).
        dep_state = self._deps_state(task, state)
        if dep_state == "awaiting":
            task["state"] = "queued"
            task["result"] = "awaiting_dependency"
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "awaiting", "id": tid, "reason": "dependency_gate"}
        if dep_state == "blocked":
            task["state"] = "blocked"
            task["result"] = "dep_blocked"
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "blocked", "id": tid, "reason": "dependency_blocked"}

        # Consumer approval via receipt.
        accept = self._accept_from_receipt(task, sha)
        if accept is None:
            task["state"] = "awaiting_codex_decision"
            task["result"] = "awaiting_receipt"
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "awaiting_codex_decision", "id": tid}
        if accept == "rejected":
            task["state"] = "rejected"
            task["result"] = "receipt_hash_drift"
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "rejected", "id": tid, "reason": "receipt_hash_drift"}

        # Accepted -> dispatch + run.
        task["state"] = "dispatched"
        task["result"] = "dispatched"
        task["pid"] = os.getpid()
        task["start_time"] = process_start_time(task["pid"])
        task["last_substantive_event_utc"] = utc_now()
        self.save_state(state)

        command = task.get("command", {})
        task_argv = list(command.get("argv", []))
        argv = self.runner + task_argv  # ALWAYS a list; never a shell string.
        self._launch_count += 1

        prompt = None
        prompt_file = command.get("prompt_file")
        if prompt_file:
            ppath = Path(prompt_file)
            prompt = ppath.read_text(encoding="utf-8", errors="replace") if ppath.exists() else ""

        timeout_s = int(task.get("budget", {}).get("timeout_s", 0)) or None
        max_out = int(task.get("budget", {}).get("max_output_bytes", 0)) or 0

        try:
            result = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(self.base_dir),
            )
            exit_code = result.returncode
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            out_bytes = len(stdout.encode("utf-8")) + len(stderr.encode("utf-8"))
            if max_out and out_bytes > max_out:
                task["state"] = "failed"
                task["result"] = "output_budget_exceeded"
                task["exit_code"] = exit_code
                task["last_substantive_event_utc"] = utc_now()
                self.save_state(state)
                return {"status": "failed", "id": tid, "reason": "output_budget_exceeded",
                        "exit_code": exit_code}
        except subprocess.TimeoutExpired:
            task["state"] = "failed"
            task["result"] = "timeout"
            task["exit_code"] = None
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "failed", "id": tid, "reason": "timeout"}
        except FileNotFoundError:
            task["state"] = "failed"
            task["result"] = "runner_not_found"
            task["exit_code"] = None
            task["last_substantive_event_utc"] = utc_now()
            self.save_state(state)
            return {"status": "failed", "id": tid, "reason": "runner_not_found"}

        task["exit_code"] = exit_code
        task["last_substantive_event_utc"] = utc_now()
        outputs_ok = self._outputs_present(task, base_dir=self.base_dir)
        if exit_code == 0 and outputs_ok:
            task["state"] = "completed"
            task["result"] = "outputs_present"
        elif exit_code == 0 and not outputs_ok:
            # exit 0 but missing artifacts -> never completed, no dependents.
            task["state"] = "rejected"
            task["result"] = "missing_outputs"
        else:
            task["state"] = "failed"
            task["result"] = "nonzero_exit"
        self.save_state(state)

        return {"status": task["state"], "id": tid, "exit_code": exit_code,
                "outputs_ok": outputs_ok}

    # ---- run loop ------------------------------------------------------------- #
    def run(self) -> dict:
        plan = read_plan(self.base_dir)
        if plan is None:
            return {"status": "no_plan", "detail": "plan.json missing"}
        state = self.load_or_init_state(plan)
        state["heartbeat_utc"] = utc_now()

        outcomes = []
        for tid in list(state["tasks"].keys()):
            task = state["tasks"][tid]
            if task.get("state") in NO_RELAUNCH_STATES:
                continue
            # Re-run any retryable task: this lets a late receipt be picked up
            # and lets a blocked/awaiting task advance once its dep resolves.
            outcomes.append(self.launch_task(task, plan, state))

        self.save_state(state)

        tstates = {tid: t.get("state") for tid, t in state["tasks"].items()}
        any_awaiting = any(s == "awaiting_codex_decision" for s in tstates.values())
        any_queued = any(s in ("queued", "blocked", "awaiting") for s in tstates.values())
        overall = "awaiting_codex_decision" if any_awaiting else ("awaiting" if any_queued else "ok")
        return {
            "status": overall,
            "outcomes": outcomes,
            "task_states": tstates,
            "launch_count": self._launch_count,
            "any_awaiting": any_awaiting,
            "any_queued": any_queued,
        }

    # ---- resume helper --------------------------------------------------------- #
    def build_resume_argv(self, session_id: str) -> list:
        """argv for resuming a known session: ``<runner> --resume <id>``."""
        if not session_id:
            return []
        return list(self.runner) + ["--resume", session_id]

    def report_resume(self, session_id: str) -> dict:
        state = read_state(self.base_dir)
        if state:
            for t in state.get("tasks", {}).values():
                if t.get("state") in ACTIVE_STATES and t.get("resume_id") == session_id:
                    return {"status": "already_running", "session_id": session_id}
        return {"status": "not_running", "session_id": session_id,
                "argv": self.build_resume_argv(session_id)}

    # ---- stale-lock recovery --------------------------------------------------- #
    def recover_stale_lock(self) -> dict:
        record = read_lock_record(self.base_dir)
        if record is None:
            return {"status": "no_lock"}
        status = lock_owner_status(record)
        if status == "gone":
            # Owner confirmed exited, PID free, no reuse ambiguity -> reclaim.
            ok = self.lock.acquire(block=True, timeout_s=self.lock_wait_s)
            if ok:
                return {"status": "reclaimed", "pid": record.get("pid")}
            return {"status": "reclaim_failed"}
        # alive, ambiguous, or unknown -> never steal. Report, do not kill.
        return {"status": "not_reclamable", "reason": status,
                "pid": record.get("pid"), "start_time": record.get("start_time")}

    # ---- status ---------------------------------------------------------------- #
    def status(self) -> dict:
        state = read_state(self.base_dir)
        return state or {"schema": STATE_SCHEMA, "candidate": CANDIDATE}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Minimal CC research controller")
    parser.add_argument("--base-dir", default="cc_state", help="State dir")
    parser.add_argument(
        "--runner",
        default="python3 -m scripts.cc_research_controller",
        help="Runner prefix (space-split argv; never passed through a shell)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp_run = sub.add_parser("run", help="Run the controller once")
    sp_run.add_argument("--force", action="store_true",
                        help="Reset retryable/terminal tasks to queued before run")
    sp_run.add_argument("--resume", dest="resume_id", default=None,
                        help="Report resume status for a session id (no auto-run)")

    sub.add_parser("status", help="Print current state")
    sub.add_parser("recover", help="Attempt stale-lock reclamation")
    p_sum = sub.add_parser("summarize", help="Condense a stream-json log")
    p_sum.add_argument("file")
    sub.add_parser("usage", help="Print help")

    args = parser.parse_args(argv)

    if args.command == "usage":
        parser.print_help()
        return 0

    if args.command == "status":
        ctrl = ResearchController(args.base_dir, runner=args.runner)
        print(canonical_json(ctrl.status(), indent=2, sort_keys=True))
        return 0

    if args.command == "recover":
        ctrl = ResearchController(args.base_dir, runner=args.runner)
        print(canonical_json(ctrl.recover_stale_lock(), indent=2, sort_keys=True))
        return 0

    if args.command == "summarize":
        print(canonical_json(summarize_stream_json(args.file), indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        ctrl = ResearchController(args.base_dir, runner=args.runner)
        if args.force:
            state = read_state(ctrl.base_dir)
            if state:
                for t in state["tasks"].values():
                    if t.get("state") in NO_RELAUNCH_STATES:
                        t["state"] = "queued"
                        t["result"] = "force_reset"
                write_state(ctrl.base_dir, state)
        if args.resume_id:
            print(canonical_json(ctrl.report_resume(args.resume_id), indent=2, sort_keys=True))
            return 0
        summary = ctrl.run()
        print(canonical_json(summary, indent=2, sort_keys=True))
        # Exit 0 means "stage processed", not "research succeeded".
        return 0 if summary.get("status") != "no_plan" else 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fake-runner":
        # Test shim (stdlib only).  argv[2] is a JSON config:
        #   {"base_dir": "...", "task_id": "...", "exit_code": 0,
        #    "write_outputs": ["rel/glob", ...]}
        # It writes the requested outputs (defaulting to the plan's
        # expected_outputs for that task), then exits with the given code.
        # It never touches real research work.
        cfg_path = sys.argv[2]
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        base = Path(cfg["base_dir"])
        tid = cfg.get("task_id")
        outputs = cfg.get("write_outputs")
        if outputs is None and tid:
            # Default to the plan's expected_outputs for this task.
            plan = read_plan(base)
            if plan:
                for t in plan.get("tasks", []):
                    if t.get("id") == tid:
                        outputs = t.get("expected_outputs", [])
                        break
        for rel in outputs or []:
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cfg.get("output_content", "ok"), encoding="utf-8")
        sys.exit(int(cfg.get("exit_code", 0)))
    sys.exit(main())
