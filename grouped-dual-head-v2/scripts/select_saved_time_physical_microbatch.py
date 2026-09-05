#!/usr/bin/env python3
"""Select the largest safe physical microbatch from fresh-process probes."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


# Family weighting requires every physical split to preserve the registered
# 12-record homogeneous macro boundary, so only its useful divisors are valid.
ATTEMPT_ORDER = (12, 6, 4, 3)


def select_largest_safe_microbatch(
    reports: Sequence[Mapping[str, object]], *, maximum_gib: float = 23.0
) -> int:
    """Validate all descending attempts and return the first safe result."""

    maximum = float(maximum_gib)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("microbatch maximum GiB is invalid")
    by_size: dict[int, list[Mapping[str, object]]] = {
        size: [] for size in ATTEMPT_ORDER
    }
    for report in reports:
        if not isinstance(report, Mapping):
            raise ValueError("microbatch probe evidence is malformed")
        try:
            size = int(report["physical_microbatch_records"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("microbatch probe evidence is malformed") from error
        if size not in by_size:
            raise ValueError("microbatch probe used an unregistered size")
        by_size[size].append(report)
    if any(len(by_size[size]) != 1 for size in ATTEMPT_ORDER):
        raise ValueError("microbatch selection requires exactly one report per size")

    safe: dict[int, bool] = {}
    for size in ATTEMPT_ORDER:
        report = by_size[size][0]
        try:
            oom = report["oom"]
            return_code = int(report["return_code"])
            peak_raw = report.get("peak_cuda_bytes")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("microbatch probe evidence is malformed") from error
        if not isinstance(oom, bool) or (oom and return_code == 0) or (
            not oom and return_code != 0
        ):
            raise ValueError("microbatch probe evidence is internally inconsistent")
        if oom:
            safe[size] = False
            continue
        try:
            peak = int(peak_raw)
        except (TypeError, ValueError) as error:
            raise ValueError("microbatch probe evidence lacks a CUDA peak") from error
        if peak < 0:
            raise ValueError("microbatch probe evidence has an invalid CUDA peak")
        safe[size] = peak < maximum * 1024**3
    for size in ATTEMPT_ORDER:
        if safe[size]:
            return size
    raise RuntimeError("no safe physical microbatch in the registered probe range")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--maximum-gib", type=float, default=23.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report_paths = tuple(Path(value).resolve() for value in args.report)
    reports = [json.loads(path.read_text()) for path in report_paths]
    selected = select_largest_safe_microbatch(
        reports, maximum_gib=float(args.maximum_gib)
    )
    value = {
        "schema": "saved_time_physical_microbatch_selection_v1",
        "selected_physical_microbatch_records": selected,
        "maximum_peak_cuda_gib": float(args.maximum_gib),
        "probe_reports": [str(path) for path in report_paths],
    }
    _atomic_json(Path(args.output), value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ATTEMPT_ORDER", "select_largest_safe_microbatch"]
