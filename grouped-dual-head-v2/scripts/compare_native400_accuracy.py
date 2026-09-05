#!/usr/bin/env python3
"""Compare native 400x400x160 census CSVs with the preregistered gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fno_acoustic.native400_gate import (  # noqa: E402
    aggregate_candidate_rows_by_sample,
    aggregate_seed_gates,
    evaluate_native400_gate,
    validate_task9_numeric,
)
from fno_acoustic.query_census import REQUIRED_NUMERIC_COLUMNS, SAMPLE_COLUMNS  # noqa: E402


_TEXT_COLUMNS = set(SAMPLE_COLUMNS[:10]) - {"sample_id"}


def _parse_sample_id(value: str, *, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: sample_id must be an integer") from exc
    if str(parsed) != value.strip():
        raise ValueError(f"row {row_number}: sample_id must use canonical integer syntax")
    return parsed


def _parse_native400_bytes(path: Path, data: bytes) -> list[dict[str, object]]:
    try:
        text_handle = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", newline="")
    except (TypeError, UnicodeError) as exc:
        raise ValueError(f"CSV is not valid UTF-8: {path}") from exc
    with text_handle as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != SAMPLE_COLUMNS:
            raise ValueError("CSV columns must exactly match Task9 SAMPLE_COLUMNS in order")
        rows: list[dict[str, object]] = []
        for row_number, raw in enumerate(reader, 2):
            if None in raw or set(raw) != set(SAMPLE_COLUMNS):
                raise ValueError(f"row {row_number}: malformed CSV row")
            row: dict[str, object] = {}
            row["sample_id"] = _parse_sample_id(raw["sample_id"], row_number=row_number)
            for name in _TEXT_COLUMNS:
                value = raw[name]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"row {row_number}: {name} must be nonempty")
                row[name] = value
            for name in REQUIRED_NUMERIC_COLUMNS:
                try:
                    value = float(raw[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"row {row_number}: {name} must be numeric") from exc
                row[name] = validate_task9_numeric(name, value, f"row {row_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV has no sample rows: {path}")
    return rows


def read_native400_snapshot(path: Path) -> tuple[list[dict[str, object]], str]:
    if not path.is_file():
        raise ValueError(f"CSV does not exist: {path}")
    data = path.read_bytes()
    return _parse_native400_bytes(path, data), hashlib.sha256(data).hexdigest()


def read_native400_rows(path: Path) -> list[dict[str, object]]:
    return read_native400_snapshot(path)[0]


def _result_dict(result) -> dict[str, object]:
    return asdict(result)


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_output_dir_atomically(output_dir: Path, payload: dict[str, object]) -> Path:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        gate_json = staging / "gate.json"
        with gate_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.replace(staging, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return output_dir / "gate.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, action="append", required=True)
    parser.add_argument("--aggregate-seeds", action="store_true")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--output-json", type=Path, help="Legacy single-file compatibility output")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    candidate_count = len(args.candidate_csv)
    if args.aggregate_seeds and candidate_count != 3:
        parser.error("--aggregate-seeds requires exactly three --candidate-csv files")
    if not args.aggregate_seeds and candidate_count != 1:
        parser.error("without --aggregate-seeds exactly one --candidate-csv is required")
    try:
        if args.output_dir is not None and args.output_dir.exists():
            raise ValueError(f"output directory already exists: {args.output_dir}")
        baseline_rows, baseline_hash = read_native400_snapshot(args.baseline_csv)
        candidate_snapshots = [read_native400_snapshot(path) for path in args.candidate_csv]
        candidate_rows = [snapshot[0] for snapshot in candidate_snapshots]
        candidate_hashes = [snapshot[1] for snapshot in candidate_snapshots]
        if args.aggregate_seeds and len(set(candidate_hashes)) != 3:
            raise ValueError("three candidate CSVs require unique file content hashes")
        input_hashes = {
            "baseline": baseline_hash,
            "candidates": candidate_hashes,
        }
        metadata = {
            "threshold": args.threshold,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "input_sha256": input_hashes,
        }
        if not args.aggregate_seeds:
            result = evaluate_native400_gate(
                baseline_rows,
                candidate_rows[0],
                args.threshold,
                args.bootstrap_replicates,
                args.seed,
            )
            payload = {**_result_dict(result), **metadata}
            passed = result.passed
        else:
            seed_results = [
                evaluate_native400_gate(
                    baseline_rows,
                    rows,
                    args.threshold,
                    args.bootstrap_replicates,
                    args.seed + index,
                )
                for index, rows in enumerate(candidate_rows)
            ]
            aggregate_rows = aggregate_candidate_rows_by_sample(candidate_rows)
            aggregate_result = evaluate_native400_gate(
                baseline_rows,
                aggregate_rows,
                args.threshold,
                args.bootstrap_replicates,
                args.seed,
            )
            seed_gate = aggregate_seed_gates(seed_results, aggregate_result)
            payload = {
                "passed": seed_gate.passed,
                "passed_seed_count": seed_gate.passed_seed_count,
                "seed_results": [_result_dict(result) for result in seed_results],
                "aggregate": _result_dict(seed_gate.aggregate),
                "aggregation": "per_sample_median_then_paired_bootstrap",
                **metadata,
            }
            passed = seed_gate.passed
        if args.output_dir is not None:
            published_path = _write_output_dir_atomically(args.output_dir, payload)
        else:
            _write_json_atomically(args.output_json, payload)
            published_path = args.output_json
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(published_path)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
