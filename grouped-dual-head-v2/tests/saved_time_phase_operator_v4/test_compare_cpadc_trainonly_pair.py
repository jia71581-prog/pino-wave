from __future__ import annotations

import pytest

from scripts.compare_cpadc_trainonly_pair import compare_trainonly_pair


def _summary(*, candidate: bool) -> dict:
    records = []
    for family_index, family in enumerate(("uniform", "layered", "marmousi")):
        for record_index in range(4):
            parent = 0.2 + 0.01 * family_index
            ablation = parent * 0.90
            adapted = parent * 0.88 if candidate else ablation
            records.append(
                {
                    "sample_id": f"{family}-{record_index}",
                    "medium_type": family,
                    "parent_future_fullfield_relative_l2": parent,
                    "future_fullfield_relative_l2": adapted,
                    "adaptation": {
                        "future_truth_used": False,
                        "adaptation_elapsed_s": 0.4 + 0.01 * record_index,
                        "effective_design_rank": 32,
                        "objective_before": 1.0,
                        "objective_after": 0.8,
                    },
                }
            )
    return {
        "selection_split": "train",
        "risk_calibration_requested": True,
        "basis": {
            "parent_checkpoint_sha256": "a" * 64,
            "basis_rank": 32,
            "phase_rank": 8,
            "basis_manifest_digest": "basis",
            "evaluation_manifest_digest": "evaluation",
        },
        "records": records,
    }


def test_trainonly_pair_gate_passes_strict_sample_paired_gain():
    result = compare_trainonly_pair(_summary(candidate=True), _summary(candidate=False))
    assert result["passed"] is True
    assert result["record_count"] == 12
    assert result["mean_paired_gain_percentage_points"] == pytest.approx(2.0)
    assert result["families"]["marmousi"][
        "mean_paired_gain_percentage_points"
    ] == pytest.approx(2.0)
    assert all(result["checks"].values())


def test_trainonly_pair_gate_fails_closed_on_parent_or_sample_drift():
    candidate = _summary(candidate=True)
    ablation = _summary(candidate=False)
    ablation["records"][0]["parent_future_fullfield_relative_l2"] += 1.0e-3
    with pytest.raises(ValueError, match="parent error mismatch"):
        compare_trainonly_pair(candidate, ablation)

    ablation = _summary(candidate=False)
    ablation["records"].pop()
    with pytest.raises(ValueError, match="sample sets differ"):
        compare_trainonly_pair(candidate, ablation)


def test_trainonly_pair_gate_rejects_unsealed_or_nontrain_input():
    candidate = _summary(candidate=True)
    candidate["records"][0]["adaptation"]["future_truth_used"] = True
    with pytest.raises(ValueError, match="not sealed"):
        compare_trainonly_pair(candidate, _summary(candidate=False))

    candidate = _summary(candidate=True)
    candidate["selection_split"] = "validation"
    with pytest.raises(ValueError, match="not train-only"):
        compare_trainonly_pair(candidate, _summary(candidate=False))
