from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_marmousi_source_position_scores import compare_scores


METRICS = {
    "record_relative_l2": 0.2,
    "late_relative_l2": 0.3,
    "spectrum_high_relative_l2": 0.4,
    "receiver_phase_coherence": 0.8,
}


def _score(path: Path, *, candidate: bool, protocol: str = "a" * 64) -> Path:
    rows = []
    for rank in range(1, 31):
        for position in range(8):
            role = "interpolation" if position < 5 else "outside_train_position_range"
            metrics = dict(METRICS)
            if candidate:
                metrics["record_relative_l2"] -= 0.03
                metrics["late_relative_l2"] -= 0.04
                metrics["spectrum_high_relative_l2"] -= 0.05
                metrics["receiver_phase_coherence"] += 0.02
            rows.append(
                {
                    "record_id": f"r{rank:02d}_p{position}",
                    "slice_rank": rank,
                    "case_id": f"p{position}",
                    "role": role,
                    "source_parameters": [float(position), 100.0, 19.0, 1.5 / 19.0, 1.0],
                    "metrics": metrics,
                }
            )
    payload = {
        "schema": "marmousi_fixed_frequency_source_position_scores_v1",
        "status": "complete",
        "source_generalization_variable": "position_only",
        "fixed_source_frequency_hz": 19.0,
        "frequency_generalization_claim_permitted": False,
        "protocol_sha256": protocol,
        "reference_manifest_sha256": "b" * 64,
        "records": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf8")
    return path


def test_comparison_uses_slice_pairs_and_orients_improvement(tmp_path: Path) -> None:
    candidate = _score(tmp_path / "candidate.json", candidate=True)
    comparator = _score(tmp_path / "comparator.json", candidate=False)
    result = compare_scores(candidate, comparator, repetitions=500, seed=4)
    assert result["independent_velocity_slice_count"] == 30
    assert result["repeated_source_positions_per_slice"] == 8
    assert result["frequency_generalization_claim_permitted"] is False
    assert result["accuracy_superiority_gate_passed"] is True
    assert result["metrics"]["record_relative_l2"]["mean_candidate_improvement"] == pytest.approx(0.03)
    assert result["metrics"]["receiver_phase_coherence"]["mean_candidate_improvement"] == pytest.approx(0.02)


def test_comparison_fails_closed_on_protocol_mismatch(tmp_path: Path) -> None:
    candidate = _score(tmp_path / "candidate.json", candidate=True)
    comparator = _score(tmp_path / "comparator.json", candidate=False, protocol="c" * 64)
    with pytest.raises(ValueError, match="protocol_sha256"):
        compare_scores(candidate, comparator, repetitions=100)
