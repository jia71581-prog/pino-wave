# CC Research Controller System — 2026-09-05

Candidate: `cc_research_controller_v1_20260905`

## Purpose

A deliberately small, file-based controller that models a plan-driven research
task runner over the filesystem. It does **not** run research; it governs the
ordering, approval, and reconciliation of a task plan authored in a frozen
`plan.json`, and persists every decision in an atomic state file. The consumer
("Codex" side of the loop) approves tasks by writing a receipt; the controller
never *pretends* to wake Codex.

Scope is intentionally limited. It is not a scheduler, not a distributed runner,
and not a GPU orchestrator. It has no notion of training, checkpoints, or
validation/test_id wavefields (those remain sealed during algorithm development).

## Files

| Path | Role |
| --- | --- |
| `scripts/cc_research_controller.py` | Module + CLI, stdlib only |
| `tests/test_cc_research_controller.py` | pytest suite, fake-runner based, no real subprocesses |
| `docs/cc_research_system_20260905.md` | This document |

## Data model

`plan.json`:

```json
{
  "schema": "cc_plan_v1",
  "plan_hash": "<sha256 of canonical task list>",
  "plan_ref": "<frozen plan ref id>",
  "tasks": [ ... ]
}
```

Each task:

```json
{
  "id": "<task_id>",
  "plan_hash": "<must match plan.plan_hash>",
  "allowed_writes": ["rel/globs", "..."],
  "deps": ["<task_id>"],
  "expected_outputs": ["rel/globs", "..."],
  "budget": {"timeout_s": 0, "max_output_bytes": 0},
  "command": {"argv": ["prog", "--flag"], "prompt_file": "rel/path"}
}
```

Constraints enforced:

- `command.argv` is **always a list**, never a shell string. The controller
  builds `runner_prefix + task.argv` and calls `subprocess.run(..., shell=False)`.
- The prompt is read from `prompt_file` and fed via `stdin`; it is never
  concatenated into the argv or a shell string.

## Hash semantics

- `compute_plan_hash` = sha256 of the canonical (sorted, compact) task list,
  excluding each task's own `plan_hash` field, so an author can mint a frozen
  hash and then stamp every task with it without changing the hash.
- The frozen ref is `plan.plan_hash`. A task whose `plan_hash != plan.plan_hash`
  is **hash drift** → rejected, and not run.
- A receipt must carry both `id` and `plan_hash` equal to the frozen hash.

## State machine

`queued` → `dispatched` → `accepted` → `running` → `completed`
plus `rejected`, `failed`, `blocked`, `awaiting_codex_decision`.

Semantic rules:

- **Acceptance requires a consumer receipt** carrying matching `id` + `plan_hash`.
  Never accepted on PID liveness.
- **exit 0 but missing expected outputs** → classified `rejected`
  (`result: missing_outputs`), never `completed`. Its dependents are **not**
  unblocked (`_deps_state` returns `blocked`).
- **Non-zero exit** → `failed`.
- **Duplicate submission is idempotent.** Same task id + same plan_hash does not
  launch twice; recovery never re-launches a task that is
  `dispatched`/`accepted`/`running`/terminal unless `--force` resets it to
  `queued`.
- **No consumer approval** → `awaiting_codex_decision`, and the controller does
  not auto-run and does not "wake" Codex.

## Filesystem layout (managed under a base dir, default `cc_state/`)

- `state.json` — current task states, written atomically (temp + `os.replace`).
- `plan.json` — the frozen plan.
- `receipts/<task_id>.json` — consumer confirmations.
- `inbox/`, `outbox/` — reserved mailboxes for Codex decisions / new routes
  (fully created on first use; not otherwise used by the controller).
- `controller.lock` — owner identity record for the writer lock.

## Locking

- `flock()` (fcntl) on `controller.lock`. A second `ControllerLock` constructor
  blocks (or fails in non-blocking mode) until release; a live owner is never
  preempted. There is no expiry timer that could steal a still-alive owner's
  lock.
- PID **and** process start time (from `/proc/<pid>/stat`, field 22) are bound
  to the lock record and to each launched task.
- Stale-lock recovery (`recover_stale_lock`) classifies the owner:
  - `gone` — PID absent from `/proc` and start time consistent → reclaim safe.
  - `alive` — (pid, start_time) matches a live process → **do not reclaim**.
  - `ambiguous` — PID is live but start time differs (PID reuse) →
    **do not reclaim**, report.
  - `unknown` — record lacks pid/start_time → **do not reclaim**, report.
  In no case does recovery kill or forcibly steal from a live/ambiguous owner.

## Time

- All timestamps are program-generated: UTC ISO-8601 (`utc_now`) plus monotonic
  seconds. No hand-filled or future values.
- Two distinct fields: `heartbeat_utc` (liveness) and
  `last_substantive_event_utc` (real progress), on both the state and each task.
- Quiet logs do not imply death; the controller never auto-kills or auto-restarts
  a GPU.

## Subprocess safety

- No shell string construction for argv.
- Prompt via stdin, never in argv.
- Secrets and full environment are never logged; `safe_head` truncates any
  captured text.
- `summarize_stream_json(path)` condenses a stream-json log into a small dict
  (`event_count`, `api_retry_count`, `partial_count`, `last_event_type`,
  `error_count`, `last_error`) so callers never hold the raw log in memory.
- Output byte budget (`budget.max_output_bytes`) is enforced post-capture; an
  over-budget task is `failed` with `output_budget_exceeded`.

## Resume handling

- `build_resume_argv(session_id)` → `<runner> --resume <id>`, always an argv
  list.
- `report_resume(session_id)` reports `already_running` if an active task already
  holds that session id; otherwise `not_running` with the constructed argv. It
  never resumes a live session and never fabricates approval. No global kill
  logic exists.

## Gates

- The controller only proceeds when a task's dependency results and plan gates
  are satisfied (`_deps_state`). New routes, budget changes, or threshold changes
  are written to the `outbox` for Codex decision, not auto-adopted.

## Exit-code note

`controller run` returns exit code 0 to mean *the stage was processed* (the
controller ran and reconciled the plan), **not** that research succeeded. Callers
must inspect the reported task states to judge outcome.
