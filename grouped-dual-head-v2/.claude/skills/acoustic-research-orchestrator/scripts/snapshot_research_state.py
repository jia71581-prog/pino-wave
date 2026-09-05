#!/usr/bin/env python3
"""Print a read-only JSON snapshot of the acoustic research workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


DEFAULT_REPO = Path(
    "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
)


def command(args: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"unreadable": str(path)}


def last_jsonl(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return {"unreadable": str(path)}


def compact_report(value: object | None) -> object | None:
    if not isinstance(value, dict):
        return value
    metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
    return {
        key: value.get(key)
        for key in (
            "status", "error", "event", "epoch", "attempt", "update",
            "updates_per_epoch", "global_step", "score", "train_loss",
            "checkpoint", "run_digest", "manifest_digest", "schedule_digest",
        )
        if key in value
    } | (
        {
            "metrics": {
                key: metrics.get(key)
                for key in (
                    "aggregate_relative_l2", "family_relative_l2",
                    "time_bin_relative_l2", "spectrum_relative_l2",
                )
                if key in metrics
            }
        }
        if metrics
        else {}
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    artifact = None if args.artifact is None else args.artifact.expanduser().resolve()
    payload: dict[str, object] = {
        "schema": "acoustic_research_state_snapshot_v1",
        "timestamp_unix_s": time.time(),
        "repo": str(repo),
        "processes": command(
            [
                "ps", "-eo", "pid,ppid,sid,stat,etime,cmd", "--sort=pid"
            ],
            repo,
        ),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            repo,
        ),
        "disk": command(["df", "-h", "/root/autodl-tmp"], repo),
    }
    process_output = str(payload["processes"].get("stdout", ""))
    payload["processes"]["stdout"] = "\n".join(
        line
        for line in process_output.splitlines()
        if any(token in line for token in ("torchrun", "train_", "adaptation"))
    )
    if artifact is not None:
        run = artifact / "run" if (artifact / "run").is_dir() else artifact
        payload["artifact"] = {
            "path": str(artifact),
            "run_identity": compact_report(read_json(run / "run_identity.json")),
            "terminal": compact_report(read_json(run / "terminal.json")),
            "best": compact_report(read_json(run / "best.json")),
            "latest_metric": compact_report(last_jsonl(run / "metrics.jsonl")),
            "latest_update": compact_report(last_jsonl(run / "updates.jsonl")),
            "epoch_control": compact_report(
                read_json(run / "epoch_validation_control.json")
            ),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
