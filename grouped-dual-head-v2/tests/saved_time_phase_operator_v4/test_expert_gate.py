import pytest
from types import SimpleNamespace

from saved_time_phase_operator_v4.expert_gate import family_expert_gate
from scripts.gate_saved_time_family_expert_candidate import (
    _bound_same_panel_metrics,
    _coarse_anchor_required,
    _minimum_physical_microbatch,
    _overfit_reduction,
    _selection_effective_batch,
    _selection_parent_checkpoint,
)
from scripts.evaluate_saved_time_family_expert_parent import (
    _band_adapter_parent_is_exact,
    _family_expert_parent_is_exact,
)


def _metrics(score: float, *, coarse: float = 0.50):
    return {
        "aggregate_relative_l2": score,
        "family_relative_l2": {
            "uniform": score - 0.01,
            "layered": score,
            "marmousi": score + 0.01,
        },
        "source_relative_l2": {
            "validation_uniform_00001": score - 0.01,
            "validation_layered_00001": score,
            "validation_marmousi_00001": score + 0.01,
        },
        "coarse_aggregate_relative_l2": coarse,
    }


def _gate(**overrides):
    arguments = {
        "direct_parent": _metrics(0.45),
        "global_parent": _metrics(0.44),
        "candidate": _metrics(0.43),
        "overfit_reduction": 0.25,
        "router_accuracy": 0.98,
        "route_probabilities": (0.30, 0.30, 0.40),
        "high_band_parent": 0.65,
        "high_band_candidate": 0.64,
        "peak_cuda_bytes": 22 * 1024**3,
        "physical_microbatch": 3,
        "minimum_physical_microbatch": 3,
        "effective_batch": 96,
        "coarse_anchor_required": True,
    }
    arguments.update(overrides)
    return family_expert_gate(**arguments)


def test_expert_gate_requires_material_improvement_over_both_parents():
    report = _gate()

    assert report["passes"] is True
    assert report["checks"]["direct_parent_improvement_1pct"] is True
    assert report["checks"]["global_parent_improvement_1pct"] is True

    report = _gate(candidate=_metrics(0.438))

    assert report["passes"] is False
    assert report["checks"]["global_parent_improvement_1pct"] is False


def test_expert_gate_rejects_cross_panel_parent_comparisons():
    direct_parent = _metrics(0.45)
    direct_parent["source_relative_l2"] = {
        "validation_uniform_00002": 0.44,
        "validation_layered_00002": 0.45,
        "validation_marmousi_00002": 0.46,
    }

    report = _gate(direct_parent=direct_parent)

    assert report["passes"] is False
    assert report["checks"]["direct_parent_same_panel"] is False
    assert report["checks"]["global_parent_same_panel"] is True


def test_same_panel_parent_report_is_bound_to_checkpoint_and_active_manifest(tmp_path):
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = {
        "schema": "saved_time_family_expert_same_panel_parent_v1",
        "status": "complete",
        "binding": {
            "parent_checkpoint": str(checkpoint),
            "active_manifest_digest": "active-manifest",
        },
        "metrics": _metrics(0.45),
    }

    assert _bound_same_panel_metrics(
        report,
        expected_checkpoint=checkpoint,
        expected_manifest_digest="active-manifest",
    ) == report["metrics"]

    with pytest.raises(ValueError, match="manifest"):
        _bound_same_panel_metrics(
            report,
            expected_checkpoint=checkpoint,
            expected_manifest_digest="different-manifest",
        )

    with pytest.raises(ValueError, match="checkpoint"):
        _bound_same_panel_metrics(
            report,
            expected_checkpoint=tmp_path / "different.pt",
            expected_manifest_digest="active-manifest",
        )


def test_existing_family_expert_parent_is_exact_without_expansion_report():
    existing = SimpleNamespace(
        dense_decoder=SimpleNamespace(family_experts=object()),
        family_expert_transfer_report=None,
    )
    expanded = SimpleNamespace(
        dense_decoder=SimpleNamespace(family_experts=object()),
        family_expert_transfer_report={"exact_parent_identity": True},
    )
    malformed = SimpleNamespace(
        dense_decoder=SimpleNamespace(family_experts=object()),
        family_expert_transfer_report={"exact_parent_identity": False},
    )
    missing = SimpleNamespace(dense_decoder=SimpleNamespace(family_experts=None))

    assert _family_expert_parent_is_exact(existing, {"family_experts": {}})
    assert _family_expert_parent_is_exact(expanded, {"family_experts": {}})
    assert not _family_expert_parent_is_exact(malformed, {"family_experts": {}})
    assert not _family_expert_parent_is_exact(missing, {"family_experts": {}})
    assert _family_expert_parent_is_exact(missing, {})


def test_band_adapter_parent_requires_registered_exact_identity():
    existing = SimpleNamespace(
        dense_decoder=SimpleNamespace(band_limited_adapter=object()),
        band_adapter_transfer_report=None,
    )
    expanded = SimpleNamespace(
        dense_decoder=SimpleNamespace(band_limited_adapter=object()),
        band_adapter_transfer_report={"exact_parent_identity": True},
    )
    malformed = SimpleNamespace(
        dense_decoder=SimpleNamespace(band_limited_adapter=object()),
        band_adapter_transfer_report={"exact_parent_identity": False},
    )
    missing = SimpleNamespace(
        dense_decoder=SimpleNamespace(band_limited_adapter=None),
    )
    enabled = {"variant_overrides": {"band_adapter_rank": 16}}

    assert _band_adapter_parent_is_exact(existing, enabled)
    assert _band_adapter_parent_is_exact(expanded, enabled)
    assert not _band_adapter_parent_is_exact(malformed, enabled)
    assert not _band_adapter_parent_is_exact(missing, enabled)
    assert _band_adapter_parent_is_exact(missing, {"variant_overrides": {}})

def test_expert_gate_rejects_router_collapse():
    report = _gate(route_probabilities=(0.05, 0.45, 0.50))

    assert report["passes"] is False
    assert report["checks"]["no_route_collapse"] is False


def test_expert_gate_requires_overfit_router_resource_and_batch_evidence():
    report = _gate(
        overfit_reduction=0.19,
        router_accuracy=0.94,
        peak_cuda_bytes=23 * 1024**3,
        physical_microbatch=2,
        effective_batch=192,
    )

    assert report["passes"] is False
    assert report["checks"]["overfit_reduction_20pct"] is False
    assert report["checks"]["router_accuracy_95pct"] is False
    assert report["checks"]["cuda_peak_safe"] is False
    assert report["checks"]["physical_microbatch_stage_safe"] is False
    assert report["checks"]["effective_batch_96"] is False


def test_gate_prefers_cumulative_anchor_reduction_for_continuations():
    report = {
        "anchor_relative_reduction": 0.25,
        "relative_reduction": 0.02,
        "overfit_reduction": 0.01,
    }

    assert _overfit_reduction(report) == pytest.approx(0.25)


def test_expert_gate_requires_family_spectrum_and_coarse_safety():
    candidate = _metrics(0.43, coarse=0.42)
    candidate["family_relative_l2"]["layered"] = 0.46

    report = _gate(candidate=candidate, high_band_candidate=0.66)

    assert report["passes"] is False
    assert report["checks"]["families_safe"] is False
    assert report["checks"]["high_band_safe"] is False
    assert report["checks"]["better_than_coarse"] is False


def test_expert_gate_allows_only_registered_fft_roundoff_in_high_band():
    assert _gate(high_band_candidate=0.650019)["checks"]["high_band_safe"] is True
    assert _gate(high_band_candidate=0.650021)["checks"]["high_band_safe"] is False


def test_gate_reads_nested_band_adapter_parent_selection():
    selection = {
        "parent": {"checkpoint": "/artifact/v49/epoch_0001.pt"},
        "candidate": {"effective_batch": 96},
    }

    assert _selection_parent_checkpoint(selection) == "/artifact/v49/epoch_0001.pt"
    assert _selection_effective_batch(selection) == 96

    with pytest.raises(ValueError, match="checkpoint"):
        _selection_parent_checkpoint({})
    with pytest.raises(ValueError, match="batch"):
        _selection_effective_batch({})


def test_expert_gate_skips_moving_coarse_anchor_for_geometry_stage():
    candidate = _metrics(0.43, coarse=0.42)

    report = _gate(
        candidate=candidate,
        physical_microbatch=2,
        minimum_physical_microbatch=2,
        coarse_anchor_required=False,
    )

    assert report["passes"] is True
    assert report["checks"]["better_than_coarse"] is True
    assert report["candidate"]["coarse_anchor_required"] is False
    assert report["thresholds"]["minimum_physical_microbatch"] == 2


@pytest.mark.parametrize(
    ("prefixes", "coarse_required", "minimum_microbatch"),
    (
        (("dense_decoder.family_experts",), True, 3),
        (
            ("dense_decoder.family_experts", "dense_decoder"),
            False,
            3,
        ),
        (
            (
                "dense_decoder.family_experts",
                "dense_decoder",
                "source_encoder",
                "fusion",
                "coordinate_encoder",
                "travel_branch",
            ),
            False,
            2,
        ),
        (("medium_encoder",), False, 2),
    ),
)
def test_gate_derives_anchor_and_memory_floor_from_trainable_stage(
    prefixes, coarse_required, minimum_microbatch
):
    row = {"trainable_stage": {"trainable_prefixes": list(prefixes)}}

    assert _coarse_anchor_required(row) is coarse_required
    assert _minimum_physical_microbatch(row) == minimum_microbatch


@pytest.mark.parametrize(
    "overrides",
    (
        {"router_accuracy": float("nan")},
        {"route_probabilities": (0.5, 0.5)},
        {"candidate": {"aggregate_relative_l2": 0.4}},
        {"physical_microbatch": 0},
    ),
)
def test_expert_gate_rejects_malformed_or_nonfinite_evidence(overrides):
    with pytest.raises(ValueError, match="family expert gate"):
        _gate(**overrides)
