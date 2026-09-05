#!/usr/bin/env python3
"""Summarize immutable train-only r5e/r5f/r5g ablation artifacts.

The report deliberately separates the unchanged parent baseline from trained
candidates.  Treating the parent as the "best candidate" hides a failed
ablation behind a zero improvement, which is unsafe for research decisions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


DEFAULT_ROOT = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/local_field_w128_hicap"
)
RUNS = {
    "r5e_aligned_control": "temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag",
    "r5f_uniform_replay": "temporal_latent_a3_rank32_r5f_uniform_random_aligned_time_train_diag",
    "r5g_dropout005": "temporal_latent_a3_rank32_r5g_dropout005_aligned_time_train_diag",
}


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "median": median(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def _candidate_score(row: dict) -> float:
    value = row.get("score", row.get("candidate_score"))
    if value is None:
        raise ValueError("gate candidate is missing its score")
    return float(value)


def summarize_run(root: Path, directory: str) -> dict:
    artifact = root / directory
    run = artifact / "run"
    gate = _jsonl(run / "epoch_validation_control.jsonl")
    updates = [
        row for row in _jsonl(run / "updates.jsonl")
        if row.get("event") == "optimizer_update"
    ]
    baseline_rows = [row for row in gate if row.get("event") == "validation_baseline"]
    candidates = [
        row for row in gate
        if row.get("event") in {"epoch_accepted", "epoch_rejected"}
    ]
    scores = [_candidate_score(row) for row in candidates]
    accepted = [row for row in candidates if row.get("event") == "epoch_accepted"]
    terminal_path = artifact / "full_pipeline_terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else None
    baseline = (
        float(baseline_rows[0]["score"])
        if baseline_rows
        else None
    )
    best_candidate = min(scores, default=None)
    selected_score = min(
        ([baseline] if baseline is not None else []) + scores,
        default=None,
    )
    gradients = [float(row["gradient_norm_before_clip"]) for row in updates]
    temporal = [float(row.get("loss_components", {}).get("temporal", math.nan)) for row in updates]
    clipped = []
    for row in updates:
        prefixes = row.get("gradient_clipping", {}).get("prefixes", {})
        adapter = prefixes.get("dense_decoder")
        if adapter is not None:
            clipped.append(float(adapter["scale"]) < 0.999999)
    last_update = updates[-1] if updates else None
    current_attempt = None if last_update is None else int(last_update.get("attempt", 1))
    current_attempt_updates = (
        0
        if current_attempt is None
        else sum(int(row.get("attempt", 1)) == current_attempt for row in updates)
    )
    updates_per_attempt = (
        None
        if last_update is None or last_update.get("updates_per_epoch") is None
        else int(last_update["updates_per_epoch"])
    )
    if terminal is not None:
        execution_status = str(terminal.get("status"))
    elif baseline_rows or updates:
        execution_status = "running"
    else:
        execution_status = "not_started"
    if accepted:
        scientific_outcome = "accepted"
    elif candidates and candidates[-1].get("maximum_attempts_exhausted") is True:
        scientific_outcome = "rejected"
    elif execution_status == "failed":
        scientific_outcome = "failed"
    elif execution_status == "not_started":
        scientific_outcome = "not_started"
    else:
        scientific_outcome = "pending"
    gate_curve = [
        {
            "attempt": int(row.get("attempt", 0)),
            "learning_rate_multiplier": float(row.get("learning_rate_multiplier", 1.0)),
            "score": _candidate_score(row),
            "delta_candidate_minus_baseline": (
                None if baseline is None else _candidate_score(row) - baseline
            ),
            "event": str(row["event"]),
            "maximum_attempts_exhausted": bool(
                row.get("maximum_attempts_exhausted", False)
            ),
        }
        for row in candidates
    ]
    temporal_values = [
        float(row.get("loss_components", {}).get("temporal", math.nan))
        for row in updates
    ]
    temporal_outliers = [value for value in temporal_values if math.isfinite(value) and value > 10.0]
    gradient_spikes = [value for value in gradients if math.isfinite(value) and value > 1.0e4]
    return {
        "artifact": str(artifact),
        "execution_status": execution_status,
        "scientific_outcome": scientific_outcome,
        "baseline_aggregate_relative_l2": baseline,
        "best_candidate_aggregate_relative_l2": best_candidate,
        "candidate_absolute_improvement": (
            None if baseline is None or best_candidate is None else baseline - best_candidate
        ),
        "selected_score_after_gate": selected_score,
        "selected_source": (
            None
            if selected_score is None
            else "candidate"
            if best_candidate is not None and best_candidate < (baseline or math.inf)
            else "parent_baseline"
        ),
        "accepted_epoch_count": len(accepted),
        "gate_candidate_count": len(candidates),
        "gate_curve": gate_curve,
        "optimizer_update_count": len(updates),
        "current_progress": {
            "attempt": current_attempt,
            "updates_completed": current_attempt_updates,
            "updates_expected": updates_per_attempt,
            "fraction": (
                None
                if updates_per_attempt in {None, 0}
                else current_attempt_updates / updates_per_attempt
            ),
        },
        "gradient_norm_before_clip": _finite_summary(gradients),
        "gradient_spike_threshold": 1.0e4,
        "gradient_spike_count": len(gradient_spikes),
        "temporal_loss": _finite_summary(temporal_values),
        "temporal_outlier_threshold": 10.0,
        "temporal_outlier_count": len(temporal_outliers),
        "adapter_clip_fraction": (
            sum(clipped) / len(clipped) if clipped else None
        ),
    }


def paired_gate_comparisons(runs: dict[str, dict]) -> dict[str, list[dict]]:
    control = runs["r5e_aligned_control"]
    control_by_lr = {
        float(row["learning_rate_multiplier"]): row
        for row in control["gate_curve"]
    }
    output: dict[str, list[dict]] = {}
    for name in ("r5f_uniform_replay", "r5g_dropout005"):
        pairs = []
        for row in runs[name]["gate_curve"]:
            multiplier = float(row["learning_rate_multiplier"])
            reference = control_by_lr.get(multiplier)
            if reference is None:
                continue
            pairs.append(
                {
                    "learning_rate_multiplier": multiplier,
                    "control_score": float(reference["score"]),
                    "intervention_score": float(row["score"]),
                    "intervention_minus_control": float(row["score"])
                    - float(reference["score"]),
                }
            )
        output[name] = pairs
    return output


def render_markdown(payload: dict) -> str:
    lines = [
        "# Aligned-time train-only ablation status",
        "",
        "Validation and test_id truth remain sealed.",
        "",
        "| Run | Status | Attempt progress | Baseline | Best candidate | Candidate improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, run in payload["runs"].items():
        progress = run["current_progress"]
        progress_text = (
            "-"
            if progress["attempt"] is None
            else f"{progress['attempt']}:{progress['updates_completed']}/{progress['updates_expected']}"
        )
        def value(number):
            return "-" if number is None else f"{float(number):.12g}"
        lines.append(
            f"| {name} | {run['scientific_outcome']} | {progress_text} | "
            f"{value(run['baseline_aggregate_relative_l2'])} | "
            f"{value(run['best_candidate_aggregate_relative_l2'])} | "
            f"{value(run['candidate_absolute_improvement'])} |"
        )
    lines.extend(["", "## Paired candidate differences", ""])
    for name, pairs in payload["paired_gate_comparisons"].items():
        lines.append(f"- {name}: " + (
            ", ".join(
                f"LRx{row['learning_rate_multiplier']:g}={row['intervention_minus_control']:+.3e}"
                for row in pairs
            )
            if pairs else "pending"
        ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/r5efg_aligned_time_ablation_summary_20260813.json"),
    )
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a read-only snapshot without modifying report files.",
    )
    args = parser.parse_args()
    runs = {
        name: summarize_run(args.artifact_root, directory)
        for name, directory in RUNS.items()
    }
    baselines = [
        run["baseline_aggregate_relative_l2"]
        for run in runs.values()
        if run["baseline_aggregate_relative_l2"] is not None
    ]
    payload = {
        "schema": "r5efg_aligned_time_ablation_summary_v2",
        "selection_split": "train",
        "validation_truth_opened": False,
        "test_id_truth_opened": False,
        "baseline_consistency": {
            "observed_count": len(baselines),
            "maximum_absolute_difference": (
                max(baselines) - min(baselines) if baselines else None
            ),
            "exactly_consistent": len(set(baselines)) <= 1,
        },
        "runs": runs,
        "paired_gate_comparisons": paired_gate_comparisons(runs),
    }
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            markdown_temporary = args.markdown_output.with_name(
                f".{args.markdown_output.name}.tmp"
            )
            markdown_temporary.write_text(render_markdown(payload))
            markdown_temporary.replace(args.markdown_output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
