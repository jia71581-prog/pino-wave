#!/usr/bin/env python3
"""Print a compact, read-only status line for an R45 result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_jsonl(path: Path):
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    payload = {"output_dir": str(output)}
    identity_path = output / "run_identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "world_size": identity.get("world_size"),
                "parameter_count": identity.get("model", {}).get("parameter_count"),
                "paper_modified": identity.get("data_boundary", {}).get(
                    "paper_modified"
                ),
                "r29b_opened": identity.get("data_boundary", {}).get("r29b_opened"),
            }
        )
    metrics_path = output / "latest_holdout.json"
    if not metrics_path.is_file():
        metrics_path = output / "initial_holdout.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = metrics["selected"]
        scale = float(selected["correction_scale"])
        hard = metrics.get("fit_hard_scales", {}).get(str(scale), {})
        hard_candidates = list(metrics.get("fit_hard_scales", {}).values())
        hard_best = (
            min(
                hard_candidates,
                key=lambda item: float(item["aggregate"]["candidate_max"])
                + 0.1 * float(item["aggregate"]["candidate_mean"]),
            )
            if hard_candidates
            else None
        )
        nonzero_candidates = [
            item
            for item in metrics.get("scales", {}).values()
            if float(item["correction_scale"]) > 0.0
        ]
        nonzero_best = (
            min(
                nonzero_candidates,
                key=lambda item: float(item["aggregate"]["candidate_max"])
                + 0.1 * float(item["aggregate"]["candidate_mean"]),
            )
            if nonzero_candidates
            else None
        )

        def scale_curve(items):
            curve = []
            for item in sorted(
                items, key=lambda entry: float(entry["correction_scale"])
            ):
                aggregate = item["aggregate"]
                curve.append(
                    {
                        "scale": float(item["correction_scale"]),
                        "mean": aggregate["candidate_mean"],
                        "max": aggregate["candidate_max"],
                    }
                )
            return curve

        development_worst = None
        if nonzero_best is not None and nonzero_best.get("records"):
            development_worst = max(
                nonzero_best["records"],
                key=lambda record: float(record["candidate_rel_l2"]),
            )
        payload.update(
            {
                "metrics_file": metrics_path.name,
                "epoch": metrics.get("epoch"),
                "global_step": metrics.get("global_step"),
                "train_loss": metrics.get("train_loss"),
                "selected_scale": scale,
                "candidate_mean": selected["aggregate"]["candidate_mean"],
                "candidate_max": selected["aggregate"]["candidate_max"],
                "absolute_goal_passed": selected["absolute_goal"]["passed"],
                "fit_hard_max_at_selected_scale": hard.get("aggregate", {}).get(
                    "candidate_max"
                ),
                "fit_hard_best_scale": (
                    float(hard_best["correction_scale"])
                    if hard_best is not None
                    else None
                ),
                "fit_hard_best_mean": (
                    hard_best["aggregate"]["candidate_mean"]
                    if hard_best is not None
                    else None
                ),
                "fit_hard_best_max": (
                    hard_best["aggregate"]["candidate_max"]
                    if hard_best is not None
                    else None
                ),
                "development_best_nonzero_scale": (
                    float(nonzero_best["correction_scale"])
                    if nonzero_best is not None
                    else None
                ),
                "development_best_nonzero_mean": (
                    nonzero_best["aggregate"]["candidate_mean"]
                    if nonzero_best is not None
                    else None
                ),
                "development_best_nonzero_max": (
                    nonzero_best["aggregate"]["candidate_max"]
                    if nonzero_best is not None
                    else None
                ),
                "development_best_nonzero_worst_record": development_worst,
                "development_scale_curve": scale_curve(
                    list(metrics.get("scales", {}).values())
                ),
                "fit_hard_scale_curve": scale_curve(hard_candidates),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
            }
        )
    update = last_jsonl(output / "updates.jsonl")
    if update is not None:
        payload["latest_update"] = update
    summary_path = output / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["run_status"] = summary.get("status")
        payload["best_epoch"] = summary.get("best_epoch")
        payload["best_score"] = summary.get("best_score")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
