#!/usr/bin/env python3
"""Supervise current CPADC panel workers and finalize frozen comparison outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.aggregate_cpadc_vs_pi_panel import aggregate
from scripts.evaluate_cpadc_on_pi_panel import _atomic_json, _sha256
from scripts.merge_causal_defect_evaluations import merge


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    return True


def supervise(args: argparse.Namespace) -> dict[str, object]:
    run_root = args.run_root.expanduser().resolve()
    terminal_path = run_root / "terminal.json"
    if terminal_path.exists():
        raise FileExistsError(f"refusing to overwrite supervisor terminal: {terminal_path}")
    workers = [run_root / f"worker_{index}.json" for index in range(4)]
    deadline = time.monotonic() + float(args.timeout_s)
    while not all(path.is_file() for path in workers):
        missing = [index for index, path in enumerate(workers) if not path.is_file()]
        failed = [index for index in missing if not _alive(args.worker_pid[index])]
        if failed:
            payload = {
                "status": "failed",
                "stage": "worker_evaluation",
                "failed_workers": failed,
                "worker_pids": list(args.worker_pid),
                "worker_outputs": [str(path) for path in workers],
                "updated_unix_s": time.time(),
            }
            _atomic_json(payload, terminal_path)
            raise RuntimeError(f"CPADC panel workers failed: {failed}")
        if time.monotonic() >= deadline:
            payload = {
                "status": "failed",
                "stage": "worker_timeout",
                "worker_pids": list(args.worker_pid),
                "worker_outputs": [str(path) for path in workers],
                "updated_unix_s": time.time(),
            }
            _atomic_json(payload, terminal_path)
            raise TimeoutError("CPADC panel workers exceeded supervisor timeout")
        time.sleep(float(args.poll_s))

    merged_dir = run_root / "merged_validation"
    merged_terminal = merge(
        args.config,
        output_dir=merged_dir,
        shard_dirs=tuple(run_root / f"shard_{index}" for index in range(4)),
        evaluation_split=args.split,
    )
    comparison = aggregate(
        workers,
        args.pi_complete,
        split=args.split,
        expected_records=int(args.expected_records),
    )
    comparison_path = run_root / "complete_validation_comparison.json"
    _atomic_json(comparison, comparison_path)
    payload = {
        "status": "complete",
        "stage": "complete",
        "split": args.split,
        "worker_pids": list(args.worker_pid),
        "worker_outputs": [str(path) for path in workers],
        "worker_sha256": {str(path): _sha256(path) for path in workers},
        "merged_validation_terminal": str(merged_dir / "terminal.json"),
        "merged_validation_terminal_sha256": _sha256(merged_dir / "terminal.json"),
        "same_protocol_validation_passed": bool(
            merged_terminal["same_protocol_validation_passed"]
        ),
        "comparison": str(comparison_path),
        "comparison_sha256": _sha256(comparison_path),
        "cpadc_checkpoint_sha256": comparison["bindings"][
            "cpadc_checkpoint_sha256"
        ],
        "pi_checkpoint_sha256": comparison["bindings"]["pi_checkpoint_sha256"],
        "record_count": int(comparison["record_count"]),
        "frames_per_record": int(comparison["frames_per_record_for_accuracy"]),
        "updated_unix_s": time.time(),
        "claim": (
            "frozen validation comparison complete"
            if merged_terminal["same_protocol_validation_passed"]
            else "frozen validation comparison complete; promotion not authorized"
        ),
    }
    _atomic_json(payload, terminal_path)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pi-complete", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), default="validation")
    parser.add_argument("--expected-records", type=int, default=480)
    parser.add_argument("--poll-s", type=float, default=15.0)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    args = parser.parse_args(argv)
    if len(args.worker_pid) != 4:
        parser.error("exactly four --worker-pid values are required")
    result = supervise(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
