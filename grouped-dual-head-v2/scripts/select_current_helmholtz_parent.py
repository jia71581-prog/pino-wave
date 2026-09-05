#!/usr/bin/env python3
"""Select the newest completed full3203 lineage run, then its audited best checkpoint.

Each continuation evaluates and saves its parent at update 0, so selecting the newest
completed run and then that run's internal best is monotonic across the lineage. The
seed terminal is used only before the first full3203 run completes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_helmholtz_g3_heldout import resolve_best_warmstart


def select_current_parent(
    results_root: str | Path,
    fallback_terminal: str | Path,
    *,
    time_pool_count: int = 96,
) -> Path:
    root = Path(results_root).expanduser().resolve()
    fallback = Path(fallback_terminal).expanduser().resolve()
    candidates: list[tuple[float, Path]] = []
    for terminal in root.glob("helmholtz_full3203_time*_vrba_ep*/terminal.json"):
        try:
            payload = json.loads(terminal.read_text())
            if (
                payload.get("status") != "complete"
                or int(payload.get("train_record_count", -1)) != 3203
                or payload.get("training_splits")
                != ["train", "validation", "test_id", "ood_canonical"]
                or int(payload.get("training_time_pool_count", -1))
                != int(time_pool_count)
                or not bool(payload.get("anomaly_excluded", False))
            ):
                continue
            resolve_best_warmstart(terminal)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((terminal.stat().st_mtime, terminal.resolve()))
    selected = max(candidates)[1] if candidates else fallback
    resolve_best_warmstart(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--fallback-terminal", required=True)
    parser.add_argument("--time-pool-count", type=int, default=96)
    args = parser.parse_args()
    print(
        select_current_parent(
            args.results_root,
            args.fallback_terminal,
            time_pool_count=int(args.time_pool_count),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
