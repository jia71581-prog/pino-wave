from pathlib import Path

import pytest
import torch
from torch import nn

from grouped_ufno_mionet_v3.model.operator import PhaseAlignedComplexFNOMIONet
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer, ScaleMetadata
from saved_time_phase_operator_v4.band_adapter import registered_high_band_mask
from saved_time_phase_operator_v4.decoder import QueryInvariantTemporalBasis
from saved_time_phase_operator_v4.operator import (
    SavedTimePhaseOperatorV4,
    load_v3_backbone,
)


def _normalizer() -> PhysicalNormalizer:
    return PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=2000.0,
            velocity_scale_mps=500.0,
            pressure_scale_pa=2.0e-8,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="test-train",
            allowed_medium_types=("uniform", "layered", "marmousi"),
            record_count=2,
            algorithm="test",
        )
    )


def _model(
    *,
    depth: int = 2,
    use_local_phase: bool = True,
    local_differential_residual: bool = False,
    temporal_basis_rank: int = 0,
    family_expert_rank: int = 0,
    band_adapter_rank: int = 0,
    band_adapter_architecture: str = "low_rank",
    band_adapter_spectral_rank: int = 4,
    band_adapter_modes: int = 4,
    band_adapter_full_depth: int = 2,
    band_adapter_coarse_depth: int = 1,
    band_adapter_activation_checkpointing: bool = False,
    band_adapter_dropout: float = 0.0,
    band_adapter_preserve_high_band: bool = True,
) -> SavedTimePhaseOperatorV4:
    return SavedTimePhaseOperatorV4(
        saved_time_s=torch.linspace(0.0, 1.0, 401, dtype=torch.float64),
        width=8,
        rank=6,
        spectral_rank=4,
        modes=(3, 2),
        dense_spectral_rank=6,
        dense_modes=4,
        dense_depth=depth,
        dense_time_block=2,
        dense_local_differential_residual=local_differential_residual,
        dense_temporal_basis_rank=temporal_basis_rank,
        dense_family_expert_rank=family_expert_rank,
        dense_band_adapter_rank=band_adapter_rank,
        dense_band_adapter_architecture=band_adapter_architecture,
        dense_band_adapter_spectral_rank=band_adapter_spectral_rank,
        dense_band_adapter_modes=band_adapter_modes,
        dense_band_adapter_full_depth=band_adapter_full_depth,
        dense_band_adapter_coarse_depth=band_adapter_coarse_depth,
        dense_band_adapter_activation_checkpointing=(
            band_adapter_activation_checkpointing
        ),
        dense_band_adapter_dropout=band_adapter_dropout,
        dense_band_adapter_preserve_high_band=band_adapter_preserve_high_band,
        use_local_phase=use_local_phase,
        heads=2,
        token_grid=2,
        position_bands=2,
        fourier_bands=2,
        gabor_scales_s=(0.03, 0.08),
        ray_samples=4,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
    )


def _prepared(model: SavedTimePhaseOperatorV4):
    velocity = torch.full((1, 1, 9, 11), 2000.0)
    velocity[:, :, 5:] += 200.0
    sources = torch.tensor(
        [[400.0, 300.0, 10.0, 0.1, 1.0], [1200.0, 350.0, 14.0, 0.12, 2.0]]
    )
    source_maps = torch.zeros(2, 1, 9, 11)
    source_maps[0, 0, 1, 2] = 1.0
    source_maps[1, 0, 1, 6] = 1.0
    normalizer = _normalizer()
    return model.prepare_sources(
        model.encode_medium(velocity, normalizer),
        sources,
        source_maps,
        normalizer,
        record_to_medium=torch.tensor([0, 0]),
    )


def _axes():
    return torch.linspace(0.0, 2000.0, 11), torch.linspace(0.0, 2000.0, 9)


def test_v4_reuses_one_medium_for_multiple_sources():
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()

    fields = model.dense_normalized(
        prepared, torch.tensor([[0.25, 0.5], [0.25, 0.5]]), x_m=x_m, z_m=z_m
    )

    assert fields.shape == (2, 2, 9, 11)
    assert torch.isfinite(fields).all()
    assert fields[:, :, 0].abs().max() == 0.0
    assert not torch.allclose(fields[0], fields[1])


def test_complete_field_route_can_disable_the_legacy_free_surface_taper():
    torch.manual_seed(23)
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25], [0.25]])
    historical = model.dense_normalized(
        prepared, times, x_m=x_m, z_m=z_m, apply_correction=False
    )
    assert model.dense_apply_free_surface_factor is True
    assert historical[..., 0, :].abs().max() == 0.0
    model.dense_apply_free_surface_factor = False
    complete_field = model.dense_normalized(
        prepared, times, x_m=x_m, z_m=z_m, apply_correction=False
    )
    assert complete_field[..., 0, :].abs().max() > 0.0


def test_dynamic_band_adapter_receives_grouped_medium_and_source_context():
    torch.manual_seed(20260721)
    model = _model(
        family_expert_rank=4,
        band_adapter_rank=6,
        band_adapter_architecture="dynamic_multiscale_spectral",
    ).eval()
    for expert in model.dense_decoder.band_limited_adapter.experts:
        torch.nn.init.normal_(expert.output.weight, std=1.0e-2)
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.20], [0.20]])

    prediction = model.dense_normalized(
        prepared,
        times,
        x_m=x_m,
        z_m=z_m,
    )

    assert prediction.shape == (2, 1, 9, 11)
    assert torch.isfinite(prediction).all()
    assert not torch.allclose(prediction[0], prediction[1])


def test_v4_starts_as_a_near_identity_correction_of_the_transferred_coarse_field():
    torch.manual_seed(41)
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25, 0.5], [0.25, 0.5]])

    corrected = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)
    coarse = model.dense_normalized(
        prepared, times, x_m=x_m, z_m=z_m, apply_correction=False
    )

    torch.testing.assert_close(corrected, coarse, rtol=1.0e-4, atol=1.0e-4)


def test_v4_returns_corrected_and_coarse_fields_from_one_public_call():
    torch.manual_seed(43)
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25, 0.5], [0.25, 0.5]])

    corrected, coarse = model.dense_normalized_with_coarse(
        prepared, times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(
        corrected,
        model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m),
    )
    torch.testing.assert_close(
        coarse,
        model.dense_normalized(
            prepared, times, x_m=x_m, z_m=z_m, apply_correction=False
        ),
    )


def test_recovery_activation_restores_a_nonvanishing_fixed_residual_gate():
    torch.manual_seed(47)
    model = _model()
    model.dense_decoder.correction_scale.data.fill_(0.02)
    model.dense_decoder.output.weight.data.fill_(1.0e-6)

    report = model.dense_decoder.activate_residual_correction(
        scale=1.0, output_std=1.0e-3
    )

    assert model.dense_decoder.correction_scale.item() == pytest.approx(1.0)
    assert not model.dense_decoder.correction_scale.requires_grad
    assert model.dense_decoder.output.weight.std().item() > 5.0e-4
    assert report["scale"] == pytest.approx(1.0)


def test_v4_rejects_interpolated_dense_time():
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()

    with pytest.raises(ValueError, match="stored HDF5 time"):
        model.dense_normalized(prepared, torch.tensor([[0.00125], [0.00125]]), x_m=x_m, z_m=z_m)


def test_v4_phase_and_spectral_parameters_receive_gradients():
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()

    output = model.dense_normalized(prepared, torch.tensor([0.25, 0.5]), x_m=x_m, z_m=z_m)
    output[:, :, 1:].square().mean().backward()

    groups = model.required_gradient_groups()
    for name in ("dense_spectral", "dense_phase", "dense_time", "dense_local"):
        assert name in groups and groups[name]
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in groups[name])


def test_deep_capacity_variant_has_target_parameter_count():
    model = SavedTimePhaseOperatorV4(
        saved_time_s=torch.linspace(0.0, 1.0, 401, dtype=torch.float64),
        dense_spectral_rank=112,
        dense_modes=32,
        dense_depth=8,
    )

    count = sum(parameter.numel() for parameter in model.parameters())
    assert 18_000_000 < count < 25_000_000


def test_v4_can_enable_local_differential_residual_without_changing_the_default():
    parent = _model(local_differential_residual=False)
    candidate = _model(local_differential_residual=True)

    assert all(block.local_differential is None for block in parent.dense_decoder.stack.blocks)
    assert all(
        block.local_differential is not None
        for block in candidate.dense_decoder.stack.blocks
    )


def test_temporal_basis_branch_is_an_exact_zero_gate_checkpoint_expansion():
    torch.manual_seed(59)
    parent = _model(temporal_basis_rank=0)
    candidate = _model(temporal_basis_rank=8)
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "dense_decoder.temporal_basis.gate",
        "dense_decoder.temporal_basis.coefficient.weight",
        "dense_decoder.temporal_basis.coefficient.bias",
        "dense_decoder.temporal_basis.time_trunk.0.weight",
        "dense_decoder.temporal_basis.time_trunk.0.bias",
        "dense_decoder.temporal_basis.time_trunk.2.weight",
        "dense_decoder.temporal_basis.time_trunk.2.bias",
    }
    prepared_parent = _prepared(parent)
    prepared_candidate = _prepared(candidate)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25, 0.50], [0.25, 0.50]])

    parent_field = parent.dense_normalized(
        prepared_parent, times, x_m=x_m, z_m=z_m
    )
    candidate_field = candidate.dense_normalized(
        prepared_candidate, times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(candidate_field, parent_field, rtol=0.0, atol=0.0)
    assert candidate.dense_decoder.temporal_basis.gate.item() == 0.0


def test_temporal_basis_prediction_is_invariant_to_companion_time_queries():
    torch.manual_seed(61)
    model = _model(temporal_basis_rank=8)
    model.dense_decoder.temporal_basis.gate.data.fill_(0.25)
    prepared = _prepared(model)
    x_m, z_m = _axes()

    single = model.dense_normalized(
        prepared, torch.tensor([[0.25], [0.25]]), x_m=x_m, z_m=z_m
    )
    block = model.dense_normalized(
        prepared,
        torch.tensor([[0.25, 0.50], [0.25, 0.50]]),
        x_m=x_m,
        z_m=z_m,
    )

    torch.testing.assert_close(single[:, 0], block[:, 0], rtol=2.0e-5, atol=2.0e-5)


def test_family_expert_expansion_is_exact_parent_identity():
    torch.manual_seed(73)
    parent = _model(family_expert_rank=0).eval()
    candidate = _model(family_expert_rank=4).eval()
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("dense_decoder.family_experts.")
        for key in incompatible.missing_keys
    )
    prepared_parent = _prepared(parent)
    prepared_candidate = _prepared(candidate)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    parent_value = parent.dense_normalized(
        prepared_parent, times, x_m=x_m, z_m=z_m
    )
    candidate_value = candidate.dense_normalized(
        prepared_candidate, times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(candidate_value, parent_value, rtol=0.0, atol=0.0)


def test_band_adapter_expansion_is_exact_parent_identity():
    torch.manual_seed(74)
    parent = _model(family_expert_rank=4, band_adapter_rank=0).eval()
    candidate = _model(family_expert_rank=4, band_adapter_rank=4).eval()
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("dense_decoder.band_limited_adapter.")
        for key in incompatible.missing_keys
    )
    prepared = _prepared(candidate)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    anchor, increment, logits = (
        candidate.dense_normalized_with_anchor_increment_and_routing(
            prepared, times, x_m=x_m, z_m=z_m
        )
    )
    prediction, _, routed_logits = candidate.dense_normalized_with_coarse_and_routing(
        prepared, times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(prediction, anchor, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(increment) == 0
    torch.testing.assert_close(routed_logits, logits, rtol=0.0, atol=0.0)


def test_shared_dynamic_band_adapter_needs_no_family_router():
    torch.manual_seed(20260813)
    parent = _model(family_expert_rank=0, band_adapter_rank=0).eval()
    candidate = _model(
        family_expert_rank=0,
        band_adapter_rank=4,
        band_adapter_architecture="shared_dynamic_multiscale_spectral",
        band_adapter_spectral_rank=4,
        band_adapter_modes=4,
        band_adapter_full_depth=2,
        band_adapter_coarse_depth=1,
    ).eval()
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith("dense_decoder.band_limited_adapter.")
        for key in incompatible.missing_keys
    )
    prepared = _prepared(candidate)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    parent_value = parent.dense_normalized(
        _prepared(parent), times, x_m=x_m, z_m=z_m
    )
    candidate_value = candidate.dense_normalized(
        prepared, times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(candidate_value, parent_value, rtol=0.0, atol=0.0)


def test_dropout_shared_dynamic_adapter_still_starts_at_exact_parent_in_train_mode():
    torch.manual_seed(20260813)
    parent = _model(family_expert_rank=0, band_adapter_rank=0).train()
    candidate = _model(
        family_expert_rank=0,
        band_adapter_rank=4,
        band_adapter_architecture="shared_dynamic_multiscale_spectral",
        band_adapter_spectral_rank=4,
        band_adapter_modes=4,
        band_adapter_full_depth=2,
        band_adapter_coarse_depth=1,
        band_adapter_dropout=0.5,
    ).train()
    candidate.load_state_dict(parent.state_dict(), strict=False)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    parent_value = parent.dense_normalized(
        _prepared(parent), times, x_m=x_m, z_m=z_m
    )
    candidate_value = candidate.dense_normalized(
        _prepared(candidate), times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(candidate_value, parent_value, rtol=0.0, atol=0.0)


def test_multiscale_band_adapter_preserves_parent_and_guarded_full_field():
    torch.manual_seed(20260720)
    parent = _model(family_expert_rank=4, band_adapter_rank=0).eval()
    candidate = _model(
        family_expert_rank=4,
        band_adapter_rank=6,
        band_adapter_architecture="multiscale_spectral",
        band_adapter_spectral_rank=4,
        band_adapter_modes=4,
        band_adapter_full_depth=2,
        band_adapter_coarse_depth=1,
    ).eval()
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("dense_decoder.band_limited_adapter.")
        for key in incompatible.missing_keys
    )
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])
    parent_wavefield = parent.dense_normalized(
        _prepared(parent), times, x_m=x_m, z_m=z_m
    )
    candidate_wavefield = candidate.dense_normalized(
        _prepared(candidate), times, x_m=x_m, z_m=z_m
    )

    torch.testing.assert_close(
        candidate_wavefield,
        parent_wavefield,
        rtol=0.0,
        atol=0.0,
    )
    assert candidate_wavefield.shape == (2, 2, 9, 11)
    with torch.no_grad():
        for expert in candidate.dense_decoder.band_limited_adapter.experts:
            expert.output.weight.normal_(std=1.0e-3)
    anchor, increment, _ = (
        candidate.dense_normalized_with_anchor_increment_and_routing(
            _prepared(candidate),
            times,
            x_m=x_m,
            z_m=z_m,
        )
    )
    adapted = candidate.dense_normalized(
        _prepared(candidate), times, x_m=x_m, z_m=z_m
    )
    high = registered_high_band_mask(9, 11, adapted.device)
    increment_fft = torch.fft.rfft2(increment.float(), norm="ortho")

    assert increment.abs().max().item() > 0.0
    assert increment_fft[..., high].abs().max().item() < 2.0e-5
    assert increment[..., 0, :].abs().max().item() < 2.0e-5
    assert adapted[..., 0, :].abs().max().item() == 0.0
    torch.testing.assert_close(adapted, anchor + increment)


def test_band_adapter_increment_preserves_high_band_and_free_surface():
    torch.manual_seed(75)
    model = _model(family_expert_rank=4, band_adapter_rank=4).eval()
    for index, expert in enumerate(model.dense_decoder.band_limited_adapter.experts):
        torch.nn.init.constant_(expert.output.bias, 0.1 * (index + 1))
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    anchor, increment, _ = model.dense_normalized_with_anchor_increment_and_routing(
        prepared, times, x_m=x_m, z_m=z_m
    )
    prediction = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)

    assert increment.abs().max().item() > 0.0
    spectrum = torch.fft.rfft2(increment.float(), norm="ortho")
    high = registered_high_band_mask(9, 11, spectrum.device)
    assert spectrum[..., high].abs().max().item() < 2.0e-5
    assert increment[..., 0, :].abs().max().item() < 2.0e-5
    assert prediction[..., 0, :].abs().max().item() == 0.0
    torch.testing.assert_close(prediction, anchor + increment)


def test_band_adapter_can_learn_full_spatial_band_when_explicitly_allowed():
    torch.manual_seed(775)
    model = _model(
        family_expert_rank=4,
        band_adapter_rank=4,
        band_adapter_preserve_high_band=False,
    ).eval()
    for index, expert in enumerate(model.dense_decoder.band_limited_adapter.experts):
        torch.nn.init.constant_(expert.output.bias, 0.1 * (index + 1))
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    anchor, increment, _ = model.dense_normalized_with_anchor_increment_and_routing(
        prepared, times, x_m=x_m, z_m=z_m
    )
    prediction = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)

    spectrum = torch.fft.rfft2(increment.float(), norm="ortho")
    high = registered_high_band_mask(9, 11, spectrum.device)
    assert spectrum[..., high].abs().max().item() > 2.0e-5
    assert increment[..., 0, :].abs().max().item() == 0.0
    torch.testing.assert_close(prediction, anchor + increment)


def test_band_adapter_output_heads_receive_gradients_while_parent_is_frozen():
    torch.manual_seed(76)
    model = _model(family_expert_rank=4, band_adapter_rank=4)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("dense_decoder.band_limited_adapter.")
        )
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.10, 0.25], [0.10, 0.25]])

    prediction = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)
    prediction[:, :, 1:].square().mean().backward()

    adapter_gradients = []
    for name, parameter in model.named_parameters():
        if name.startswith("dense_decoder.band_limited_adapter."):
            if parameter.grad is not None:
                adapter_gradients.append(float(parameter.grad.abs().sum()))
        else:
            assert parameter.grad is None
    assert adapter_gradients
    assert max(adapter_gradients) > 0.0


def test_family_expert_query_is_block_invariant():
    torch.manual_seed(79)
    model = _model(family_expert_rank=4).eval()
    for expert in model.dense_decoder.family_experts.experts:
        torch.nn.init.normal_(expert.output.weight, std=1.0e-2)
    prepared = _prepared(model)
    x_m, z_m = _axes()

    single = model.dense_normalized(
        prepared, torch.tensor([[0.25], [0.25]]), x_m=x_m, z_m=z_m
    )
    block = model.dense_normalized(
        prepared,
        torch.tensor([[0.10, 0.25, 0.40], [0.10, 0.25, 0.40]]),
        x_m=x_m,
        z_m=z_m,
    )

    torch.testing.assert_close(single[:, 0], block[:, 1], rtol=2.0e-5, atol=2.0e-5)


def test_family_route_override_reaches_dense_experts_but_not_public_inference():
    torch.manual_seed(83)
    model = _model(family_expert_rank=4).eval()
    for index, expert in enumerate(model.dense_decoder.family_experts.experts):
        torch.nn.init.zeros_(expert.output.weight)
        torch.nn.init.constant_(expert.output.bias, 0.1 * (index + 1))
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25], [0.25]])

    routed, _, logits = model.dense_normalized_with_coarse_and_routing(
        prepared,
        times,
        x_m=x_m,
        z_m=z_m,
        route_override=torch.tensor([2]),
    )
    inferred = model.dense_normalized(
        prepared, times, x_m=x_m, z_m=z_m
    )

    assert logits.shape == (1, 3)
    assert not torch.allclose(routed, inferred)
    assert torch.allclose(
        routed[:, :, 1:] - inferred[:, :, 1:],
        (0.3 - logits.softmax(-1).mul(torch.tensor([0.1, 0.2, 0.3])).sum())
        * torch.ones_like(routed[:, :, 1:]),
        atol=1.0e-5,
    )


def test_temporal_basis_gate_then_features_wake_on_two_updates():
    torch.manual_seed(67)
    model = _model(temporal_basis_rank=8)
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([[0.25, 0.50], [0.25, 0.50]])
    branch = model.dense_decoder.temporal_basis

    model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m).square().mean().backward()
    assert branch.gate.grad is not None and branch.gate.grad.abs().item() > 0.0
    first_feature_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in branch.feature_parameters()
        if parameter.grad is not None
    )
    assert first_feature_grad == 0.0

    with torch.no_grad():
        branch.gate.add_(-0.1 * branch.gate.grad)
    model.zero_grad(set_to_none=True)
    prepared = _prepared(model)
    model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m).square().mean().backward()
    second_feature_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in branch.feature_parameters()
        if parameter.grad is not None
    )
    assert second_feature_grad > 0.0
    groups = model.required_gradient_groups()
    assert groups["dense_temporal_basis_gate"] == (branch.gate,)
    assert groups["dense_temporal_basis_features"]


def test_temporal_basis_gate_absorption_preserves_predictions_and_removes_small_gate():
    torch.manual_seed(71)
    branch = QueryInvariantTemporalBasis(width=4, rank=6)
    branch.gate.data.fill_(2.79e-4)
    spatial = torch.randn(3, 4, 7, 9)
    time_features = torch.randn(3, 5, 5)
    old_weight = branch.coefficient.weight.detach().clone()
    old_bias = branch.coefficient.bias.detach().clone()

    before = branch(spatial, time_features, spatial_stack=nn.Identity())
    report = branch.absorb_gate_into_coefficient()
    after = branch(spatial, time_features, spatial_stack=nn.Identity())

    torch.testing.assert_close(after, before, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(
        branch.coefficient.weight, old_weight * 2.79e-4, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        branch.coefficient.bias, old_bias * 2.79e-4, rtol=0.0, atol=0.0
    )
    assert branch.gate.item() == pytest.approx(1.0)
    assert branch.gate.requires_grad
    assert report["pre_gate"] == pytest.approx(2.79e-4)
    assert report["post_gate"] == pytest.approx(1.0)


def test_v3_transfer_loads_every_nondense_parameter(tmp_path: Path):
    torch.manual_seed(23)
    parent = PhaseAlignedComplexFNOMIONet(
        width=8, rank=6, spectral_rank=4, modes=(3, 2), dense_modes=(3, 2),
        heads=2, token_grid=2, position_bands=2, fourier_bands=2,
        gabor_scales_s=(0.03, 0.08), ray_samples=4,
    )
    checkpoint = tmp_path / "v3.pt"
    torch.save({"model_state": parent.state_dict()}, checkpoint)
    child = _model()

    report = load_v3_backbone(child, checkpoint)

    assert report["loaded_parameter_tensors"] > 0
    torch.testing.assert_close(child.medium_encoder.lift.weight, parent.medium_encoder.lift.weight)
