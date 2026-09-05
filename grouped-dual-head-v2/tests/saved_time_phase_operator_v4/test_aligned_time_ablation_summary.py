from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/summarize_r5efg_aligned_time_ablations.py")
SPEC = importlib.util.spec_from_file_location("aligned_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_failed_ablation_keeps_parent_separate_from_best_candidate(tmp_path):
    directory = "candidate"
    run = tmp_path / directory / "run"
    _write_jsonl(
        run / "epoch_validation_control.jsonl",
        [
            {"event": "validation_baseline", "score": 0.2},
            {
                "event": "epoch_rejected",
                "attempt": 1,
                "candidate_score": 0.21,
                "learning_rate_multiplier": 1.0,
                "maximum_attempts_exhausted": True,
            },
        ],
    )
    _write_jsonl(
        run / "updates.jsonl",
        [
            {
                "event": "optimizer_update",
                "attempt": 1,
                "update": 140,
                "updates_per_epoch": 140,
                "gradient_norm_before_clip": 2.0e4,
                "gradient_clipping": {"prefixes": {"dense_decoder": {"scale": 0.5}}},
                "loss_components": {"temporal": 20.0},
            }
        ],
    )
    result = MODULE.summarize_run(tmp_path, directory)

    assert result["best_candidate_aggregate_relative_l2"] == 0.21
    assert result["candidate_absolute_improvement"] == pytest.approx(-0.01)
    assert result["selected_score_after_gate"] == 0.2
    assert result["selected_source"] == "parent_baseline"
    assert result["scientific_outcome"] == "rejected"
    assert result["gradient_spike_count"] == 1
    assert result["temporal_outlier_count"] == 1


def test_progress_and_paired_differences_are_explicit(tmp_path):
    for directory, score in (("control", 0.21), ("intervention", 0.22)):
        run = tmp_path / directory / "run"
        _write_jsonl(
            run / "epoch_validation_control.jsonl",
            [
                {"event": "validation_baseline", "score": 0.2},
                {
                    "event": "epoch_rejected",
                    "attempt": 1,
                    "candidate_score": score,
                    "learning_rate_multiplier": 1.0,
                },
            ],
        )
        _write_jsonl(
            run / "updates.jsonl",
            [
                {
                    "event": "optimizer_update",
                    "attempt": 2,
                    "update": update,
                    "updates_per_epoch": 140,
                    "gradient_norm_before_clip": 1.0,
                    "loss_components": {"temporal": 1.0},
                }
                for update in range(1, 8)
            ],
        )
    control = MODULE.summarize_run(tmp_path, "control")
    intervention = MODULE.summarize_run(tmp_path, "intervention")
    runs = {
        "r5e_aligned_control": control,
        "r5f_uniform_replay": intervention,
        "r5g_dropout005": {**intervention, "gate_curve": []},
    }

    paired = MODULE.paired_gate_comparisons(runs)
    assert intervention["current_progress"] == {
        "attempt": 2,
        "updates_completed": 7,
        "updates_expected": 140,
        "fraction": 0.05,
    }
    assert paired["r5f_uniform_replay"][0][
        "intervention_minus_control"
    ] == pytest.approx(0.01)
