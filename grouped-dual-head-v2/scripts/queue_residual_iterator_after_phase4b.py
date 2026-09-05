#!/usr/bin/env python3
"""Wait for Phase4b, smoke-test P0/P1, then launch four-GPU iterator training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_parent(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.jsonl"
    baseline = None
    candidates: list[tuple[float, dict[str, object], Path, int]] = []
    for line in metrics_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "baseline":
            baseline = event["metrics"]
            checkpoint = run_dir / "checkpoints" / "update_0000.pt"
            candidates.append(
                (float(baseline["aggregate_relative_l2"]), baseline, checkpoint, 0)
            )
        elif event.get("event") == "evaluation":
            candidates.append(
                (
                    float(event["metrics"]["aggregate_relative_l2"]),
                    event["metrics"],
                    Path(event["checkpoint"]),
                    int(event["update"]),
                )
            )
    if baseline is None or not candidates:
        raise RuntimeError("Phase4b metrics contain no baseline/evaluation candidates")
    baseline_family = {
        key: float(value) for key, value in baseline["family_relative_l2"].items()
    }
    eligible = []
    for aggregate, metrics, checkpoint, update in candidates:
        family = metrics["family_relative_l2"]
        no_regression = all(
            float(family[name]) <= baseline_family[name] + max(1.0e-12, 1.0e-6 * baseline_family[name])
            for name in baseline_family
        )
        if checkpoint.is_file() and math.isfinite(aggregate) and no_regression:
            eligible.append((aggregate, metrics, checkpoint, update))
    if not eligible:
        raise RuntimeError("no reproducible Phase4b checkpoint passed the family no-regression gate")
    aggregate, metrics, checkpoint, update = min(eligible, key=lambda item: (item[0], item[3]))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "update": update,
        "aggregate_relative_l2": aggregate,
        "family_relative_l2": metrics["family_relative_l2"],
    }


def _run(
    command: list[str], log_path: Path, *, environment=None,
    active_process_path: Path | None = None,
) -> int:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        stream.write(json.dumps({"pid": process.pid}) + "\n")
        stream.flush()
        if active_process_path is not None:
            _atomic_json(
                {"pid": process.pid, "command": command, "started_at_unix": time.time()},
                active_process_path,
            )
        return_code = process.wait()
        if active_process_path is not None:
            _atomic_json(
                {
                    "pid": process.pid,
                    "command": command,
                    "started_at_unix": json.loads(active_process_path.read_text())["started_at_unix"],
                    "finished_at_unix": time.time(),
                    "return_code": return_code,
                },
                active_process_path,
            )
        return return_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4b-run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    phase4b = Path(args.phase4b_run_dir).resolve()
    config = Path(args.config).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "controller_status.json"
    log_path = output / "controller.log"
    active_process_path = output / "active_process.json"
    phase_terminal = phase4b / "terminal.json"
    phase_pid_path = phase4b / "launcher.pid"
    status: dict[str, object] = {
        "status": "waiting_for_phase4b",
        "controller_pid": os.getpid(),
        "phase4b_run_dir": str(phase4b),
        "created_at_unix": time.time(),
    }
    _atomic_json(status, status_path)
    while not phase_terminal.is_file():
        if phase_pid_path.is_file():
            phase_pid = int(phase_pid_path.read_text().strip())
            try:
                os.kill(phase_pid, 0)
            except ProcessLookupError:
                status.update(
                    status="blocked",
                    reason="phase4b_process_exited_without_terminal_json",
                    finished_at_unix=time.time(),
                )
                _atomic_json(status, status_path)
                return 2
        time.sleep(max(5.0, float(args.poll_seconds)))

    terminal = json.loads(phase_terminal.read_text())
    if terminal.get("status") != "complete":
        status.update(
            status="blocked",
            reason="phase4b_terminal_status_not_complete",
            phase4b_terminal=terminal,
            finished_at_unix=time.time(),
        )
        _atomic_json(status, status_path)
        return 3
    selected = _select_parent(phase4b)
    selection_path = output / "selected_parent.json"
    _atomic_json(selected, selection_path)
    identity = phase4b / "run_identity.json"

    smoke_environment = os.environ.copy()
    smoke_environment["CUDA_VISIBLE_DEVICES"] = "0"
    smoke_base_command = [
        "/root/miniconda3/bin/python",
        "scripts/train_residual_iterator.py",
        "--config", str(config),
        "--parent-checkpoint", str(selected["checkpoint"]),
        "--parent-run-identity", str(identity),
        "--device", "cuda",
        "--epochs", "1",
        "--time-window", "16",
        "--patch-size", "96",
        "--iterations", "4",
        "--width", "24",
        "--learning-rate", "0.0002",
        "--smoke",
    ]
    selected_rank_batch = None
    smoke_checkpoint = None
    smoke_attempts: list[dict[str, object]] = []
    # Probe batch=2 first.  A 21.5 GiB cap leaves headroom for NCCL buffers in
    # the subsequent four-GPU run; failures fall back to the proven batch=1 path.
    for candidate_batch in (2, 1):
        smoke_dir = output / f"smoke_batch{candidate_batch}"
        smoke_dir.mkdir(exist_ok=True)
        status.update(
            status="smoke_running",
            selected_parent=selected,
            smoke_dir=str(smoke_dir),
            probing_per_rank_batch=candidate_batch,
            smoke_attempts=smoke_attempts,
        )
        _atomic_json(status, status_path)
        smoke_command = smoke_base_command + [
            "--output-dir", str(smoke_dir),
            "--per-rank-batch-size", str(candidate_batch),
        ]
        smoke_return = _run(
            smoke_command, log_path, environment=smoke_environment,
            active_process_path=active_process_path,
        )
        candidate_checkpoint = smoke_dir / "best_residual_iterator.pt"
        peak_cuda_bytes = None
        accepted_memory = False
        if smoke_return == 0 and candidate_checkpoint.is_file():
            payload = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
            peak_cuda_bytes = int(payload.get("peak_cuda_bytes", 0))
            accepted_memory = (
                candidate_batch == 1
                or peak_cuda_bytes <= int(21.5 * 1024**3)
            )
        smoke_attempts.append(
            {
                "per_rank_batch_size": candidate_batch,
                "return_code": smoke_return,
                "checkpoint": (
                    str(candidate_checkpoint) if candidate_checkpoint.is_file() else None
                ),
                "peak_cuda_bytes": peak_cuda_bytes,
                "accepted_memory": accepted_memory,
            }
        )
        if smoke_return == 0 and candidate_checkpoint.is_file() and accepted_memory:
            selected_rank_batch = candidate_batch
            smoke_checkpoint = candidate_checkpoint
            break
    if selected_rank_batch is None or smoke_checkpoint is None:
        status.update(
            status="blocked",
            reason="residual_iterator_smoke_failed",
            smoke_attempts=smoke_attempts,
            finished_at_unix=time.time(),
        )
        _atomic_json(status, status_path)
        return 4

    formal_dir = output / "formal_ddp4"
    formal_dir.mkdir(exist_ok=True)
    status.update(
        status="formal_training_running",
        formal_dir=str(formal_dir),
        selected_per_rank_batch_size=selected_rank_batch,
        global_batch_size=selected_rank_batch * 4,
        smoke_attempts=smoke_attempts,
    )
    _atomic_json(status, status_path)
    formal_command = [
        "/root/miniconda3/bin/torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/train_residual_iterator.py",
        "--config", str(config),
        "--output-dir", str(formal_dir),
        "--parent-checkpoint", str(selected["checkpoint"]),
        "--parent-run-identity", str(identity),
        "--device", "cuda",
        "--per-family", "8",
        "--epochs", "5",
        "--time-window", "16",
        "--patch-size", "96",
        "--iterations", "4",
        "--width", "24",
        "--learning-rate", "0.0002",
        "--per-rank-batch-size", str(selected_rank_batch),
    ]
    formal_return = _run(
        formal_command, log_path, active_process_path=active_process_path
    )
    formal_checkpoint = formal_dir / "best_residual_iterator.pt"
    if formal_return != 0 or not formal_checkpoint.is_file():
        status.update(
            status="blocked",
            reason="formal_residual_iterator_training_failed",
            formal_return_code=formal_return,
            finished_at_unix=time.time(),
        )
        _atomic_json(status, status_path)
        return 5

    evaluation_dir = output / "heldout3"
    evaluation_dir.mkdir(exist_ok=True)
    status.update(status="heldout3_running", heldout_dir=str(evaluation_dir))
    _atomic_json(status, status_path)
    evaluation_environment = os.environ.copy()
    evaluation_environment["CUDA_VISIBLE_DEVICES"] = "0"
    evaluation_command = [
        "/root/miniconda3/bin/python",
        "scripts/run_residual_iterator_adaptation.py",
        "--config", str(config),
        "--corrector-checkpoint", str(formal_checkpoint),
        "--parent-checkpoint", str(selected["checkpoint"]),
        "--parent-run-identity", str(identity),
        "--output-dir", str(evaluation_dir),
        "--device", "cuda",
        "--per-family", "1",
        "--no-fields",
    ]
    evaluation_return = _run(
        evaluation_command, log_path, environment=evaluation_environment,
        active_process_path=active_process_path,
    )
    summary = evaluation_dir / "summary.json"
    if evaluation_return != 0 or not summary.is_file():
        status.update(
            status="blocked",
            reason="sealed_heldout3_evaluation_failed",
            evaluation_return_code=evaluation_return,
            finished_at_unix=time.time(),
        )
        _atomic_json(status, status_path)
        return 6
    status.update(
        status="complete",
        formal_checkpoint=str(formal_checkpoint),
        formal_checkpoint_sha256=_sha256(formal_checkpoint),
        heldout_summary=str(summary),
        finished_at_unix=time.time(),
    )
    _atomic_json(status, status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
