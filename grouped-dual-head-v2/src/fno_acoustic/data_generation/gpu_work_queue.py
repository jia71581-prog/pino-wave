from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def sort_work_items(rows: Iterable[dict]) -> list[dict]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda r: (
            str(r.get("split", "")),
            str(r.get("truth_grid_tier", "5m")),
            int(r.get("base_model_id", -1)),
            int(r.get("truth_internal_n_substeps", 0)),
            str(r.get("solver_dtype", "float32")),
            int(r.get("frequency_pair_id", -1)),
            int(r.get("sample_id", -1)),
        ),
    )


def shard_rows(rows: Iterable[dict], *, samples_per_shard: int) -> list[list[dict]]:
    sorted_rows = sort_work_items(rows)
    return [sorted_rows[i : i + int(samples_per_shard)] for i in range(0, len(sorted_rows), int(samples_per_shard))]


def group_by_base_model(rows: Iterable[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["base_model_id"])].append(dict(row))
    return dict(groups)
