#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fno_acoustic.data_generation.hdf5_lwc84 import compute_train_only_dataset_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute exact train-only normalization moments from an LWC84 VDS."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-train-samples", type=int, default=None)
    parser.add_argument("--time-chunk-size", type=int, default=16)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    stats = compute_train_only_dataset_stats(
        args.dataset,
        time_chunk_size=int(args.time_chunk_size),
    )
    if (
        args.expected_train_samples is not None
        and int(stats["train_sample_count"]) != int(args.expected_train_samples)
    ):
        raise ValueError(
            f"expected {args.expected_train_samples} train samples, "
            f"got {stats['train_sample_count']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    payload = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, output)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
