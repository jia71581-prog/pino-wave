#!/usr/bin/env python
"""Advance pilot -> production -> sealed evaluation only through explicit gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_saved_time_v4_probe import _atomic_json


def pilot_allows_production(terminal: Mapping[str, object]) -> bool:
    gate = terminal.get("pilot_gate")
    return bool(
        terminal.get("status") == "complete"
        and isinstance(gate, Mapping)
        and gate.get("passed") is True
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_terminal(path: Path, *, pid: int, poll_seconds: int):
    while True:
        if path.is_file():
            payload = json.loads(path.read_text())
            if payload.get("status") in {"complete", "pilot_gate_failed", "failed"}:
                return payload
        if not _pid_alive(pid):
            raise RuntimeError(f"process {pid} exited without a terminal record")
        time.sleep(poll_seconds)


def _run_logged(command: list[str], *, cwd: Path, log: Path, environment):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"event": "launch", "command": command}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    artifact = Path(config["artifact_dir"])
    project = Path(__file__).resolve().parents[1]
    status_path = artifact / "supervisor_status.json"
    _atomic_json(
        {"status": "waiting_for_pilot", "pilot_pid": args.pilot_pid}, status_path
    )
    pilot_terminal = _wait_for_terminal(
        artifact / "pilot" / "terminal.json",
        pid=args.pilot_pid,
        poll_seconds=args.poll_seconds,
    )
    if not pilot_allows_production(pilot_terminal):
        _atomic_json(
            {"status": "stopped_at_pilot_gate", "pilot_terminal": pilot_terminal},
            status_path,
        )
        return 2

    environment = dict(os.environ)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_command = [
        sys.executable,
        "-u",
        "scripts/train_saved_time_v4_full_support.py",
        "--config",
        str(config_path),
    ]
    _atomic_json(
        {"status": "launching_production", "pilot_terminal": pilot_terminal},
        status_path,
    )
    train_code = _run_logged(
        train_command,
        cwd=project,
        log=artifact / "launcher.log",
        environment=environment,
    )
    run_terminal_path = artifact / "run" / "terminal.json"
    run_terminal = (
        json.loads(run_terminal_path.read_text())
        if run_terminal_path.is_file()
        else {"status": "missing", "return_code": train_code}
    )
    if train_code != 0 or run_terminal.get("status") != "complete":
        _atomic_json(
            {"status": "stopped_at_production", "run_terminal": run_terminal},
            status_path,
        )
        return 3

    evaluation_command = [
        sys.executable,
        "-u",
        "scripts/evaluate_saved_time_v4_full_support.py",
        "--config",
        str(config_path),
        "--time-block",
        "16",
    ]
    _atomic_json(
        {"status": "launching_sealed_evaluation", "run_terminal": run_terminal},
        status_path,
    )
    evaluation_code = _run_logged(
        evaluation_command,
        cwd=project,
        log=artifact / "evaluation_launcher.log",
        environment=environment,
    )
    report_path = artifact / "run" / "sealed_evaluation" / "evaluation_report.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.is_file()
        else {"status": "missing", "return_code": evaluation_code}
    )
    status = "complete" if evaluation_code == 0 and report.get("status") == "complete" else "evaluation_failed"
    _atomic_json(
        {"status": status, "run_terminal": run_terminal, "evaluation": report},
        status_path,
    )
    return 0 if status == "complete" else 4


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["pilot_allows_production"]
