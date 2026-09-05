from dataclasses import replace

import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.band_adapter import (
    BandLimitedFamilyAdapter,
    activate_band_adapter_output,
    build_band_adapter_adamw,
    configure_band_adapter_stage,
    project_low_mid_increment,
    registered_band_adapter_learning_rates,
    registered_high_band_mask,
)
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.train_saved_time_v4_full_support import (
    activate_band_adapter_output_if_requested,
    band_adapter_identity_report,
    band_adapter_missing_prefixes,
)


def _multiscale_adapter() -> BandLimitedFamilyAdapter:
    return BandLimitedFamilyAdapter(
        width=8,
        rank=6,
        architecture="multiscale_spectral",
        spectral_rank=4,
        modes=4,
        full_depth=2,
        coarse_depth=1,
        activation_checkpointing=False,
    )


def _dynamic_adapter() -> BandLimitedFamilyAdapter:
    return BandLimitedFamilyAdapter(
        width=8,
        rank=6,
        architecture="dynamic_multiscale_spectral",
        spectral_rank=4,
        modes=4,
        full_depth=2,
        coarse_depth=1,
        activation_checkpointing=False,
    )


def _shared_dynamic_adapter(*, dropout: float = 0.0) -> BandLimitedFamilyAdapter:
    return BandLimitedFamilyAdapter(
        width=8,
        rank=6,
        architecture="shared_dynamic_multiscale_spectral",
        spectral_rank=4,
        modes=4,
        full_depth=2,
        coarse_depth=1,
        activation_checkpointing=False,
        dropout=dropout,
    )


def _dynamic_inputs():
    generator = torch.Generator().manual_seed(20260721)
    shared = torch.randn(6, 8, 17, 19, generator=generator)
    time_features = torch.randn(2, 3, 5, generator=generator)
    mapping = torch.tensor([0, 0])
    routes = torch.tensor([[1.0, 0.0, 0.0]])
    velocity = torch.full((1, 1, 17, 19), 2000.0)
    source = torch.tensor(
        [[-0.6, -0.4, -0.7, -0.8, 0.0], [0.6, -0.3, -0.5, -0.7, 0.2]]
    )
    return shared, time_features, mapping, routes, velocity, source


def test_dynamic_adapter_is_exact_identity_and_encodes_each_medium_once():
    adapter = _dynamic_adapter()
    shared, times, mapping, routes, velocity, source = _dynamic_inputs()
    calls = []
    handle = adapter.medium_encoder.register_forward_hook(
        lambda _module, inputs, _output: calls.append(tuple(inputs[0].shape))
    )
    try:
        increment = adapter(
            shared,
            times,
            mapping,
            routes,
            velocity_mps=velocity,
            source_features=source,
        )
    finally:
        handle.remove()

    assert increment.shape == (2, 3, 17, 19)
    assert torch.count_nonzero(increment) == 0
    assert calls == [(1, 2, 17, 19)]


def test_shared_dynamic_adapter_is_route_free_exact_identity():
    adapter = _shared_dynamic_adapter()
    shared, times, mapping, _, velocity, source = _dynamic_inputs()

    increment = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )

    assert len(adapter.experts) == 1
    assert increment.shape == (2, 3, 17, 19)
    assert torch.count_nonzero(increment) == 0


def test_shared_dynamic_adapter_responds_without_family_router():
    adapter = _shared_dynamic_adapter()
    shared, times, mapping, _, velocity, source = _dynamic_inputs()
    with torch.no_grad():
        adapter.experts[0].output.weight.normal_(std=1.0e-2)

    baseline = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )
    changed = velocity.clone()
    changed[..., 8:, :] += 700.0
    conditioned = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=changed,
        source_features=source,
    )

    assert not torch.allclose(baseline, conditioned)


def test_shared_dynamic_adapter_zero_dropout_preserves_default_behavior():
    default = _shared_dynamic_adapter()
    explicit = _shared_dynamic_adapter(dropout=0.0)
    explicit.load_state_dict(default.state_dict())
    default.train()
    explicit.train()
    shared, times, mapping, _, velocity, source = _dynamic_inputs()
    with torch.no_grad():
        default.experts[0].output.weight.normal_(std=1.0e-2)
        explicit.load_state_dict(default.state_dict())

    default_output = default(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )
    explicit_output = explicit(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )

    torch.testing.assert_close(default_output, explicit_output, rtol=0.0, atol=0.0)


def test_shared_dynamic_adapter_dropout_is_train_only():
    adapter = _shared_dynamic_adapter(dropout=0.5)
    shared, times, mapping, _, velocity, source = _dynamic_inputs()
    with torch.no_grad():
        adapter.experts[0].output.weight.normal_(std=1.0e-2)

    adapter.train()
    first_train = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )
    second_train = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )
    adapter.eval()
    first_eval = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )
    second_eval = adapter(
        shared,
        times,
        mapping,
        shared.new_empty((0, 3)),
        velocity_mps=velocity,
        source_features=source,
    )

    assert not torch.allclose(first_train, second_train)
    torch.testing.assert_close(first_eval, second_eval, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dropout", (-0.1, 1.0, float("nan")))
def test_multiscale_adapter_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        _shared_dynamic_adapter(dropout=dropout)


def test_low_rank_adapter_rejects_nonzero_dropout():
    with pytest.raises(ValueError, match="multiscale"):
        BandLimitedFamilyAdapter(width=8, rank=6, dropout=0.05)


def test_dynamic_adapter_responds_to_velocity_and_source_and_backpropagates():
    adapter = _dynamic_adapter()
    shared, times, mapping, routes, velocity, source = _dynamic_inputs()
    with torch.no_grad():
        adapter.experts[0].output.weight.normal_(std=1.0e-2)

    baseline = adapter(
        shared,
        times,
        mapping,
        routes,
        velocity_mps=velocity,
        source_features=source,
    )
    changed_velocity = velocity.clone()
    changed_velocity[..., 8:, :] += 700.0
    velocity_output = adapter(
        shared,
        times,
        mapping,
        routes,
        velocity_mps=changed_velocity,
        source_features=source,
    )
    changed_source = source.clone()
    changed_source[1, 2] += 0.8
    source_output = adapter(
        shared,
        times,
        mapping,
        routes,
        velocity_mps=velocity,
        source_features=changed_source,
    )

    assert not torch.allclose(baseline, velocity_output)
    assert not torch.allclose(baseline[1], source_output[1])
    baseline.square().mean().backward()
    assert adapter.medium_encoder[0].weight.grad.norm().item() > 0.0
    assert adapter.source_encoder[0].weight.grad.norm().item() > 0.0


def test_dynamic_adapter_requires_physical_conditioning_inputs():
    adapter = _dynamic_adapter()
    shared, times, mapping, routes, _, _ = _dynamic_inputs()

    with pytest.raises(ValueError, match="velocity_mps and source_features"):
        adapter(shared, times, mapping, routes)


def test_dynamic_output_wakeup_is_deterministic_and_activates_feature_gradients():
    first = _dynamic_adapter()
    second = _dynamic_adapter()
    second.load_state_dict(first.state_dict())

    first_report = activate_band_adapter_output(first, std=1.0e-4, seed=73)
    second_report = activate_band_adapter_output(second, std=1.0e-4, seed=73)

    assert first_report == second_report
    assert first_report["architecture"] == "dynamic_multiscale_spectral"
    assert first_report["output_std"] > 0.0
    for left, right in zip(first.experts, second.experts, strict=True):
        torch.testing.assert_close(left.output.weight, right.output.weight)
        assert torch.count_nonzero(left.output.weight) > 0

    shared, times, mapping, routes, velocity, source = _dynamic_inputs()
    first(
        shared,
        times,
        mapping,
        routes,
        velocity_mps=velocity,
        source_features=source,
    ).square().mean().backward()
    assert first.medium_encoder[0].weight.grad.norm().item() > 0.0
    assert first.source_encoder[0].weight.grad.norm().item() > 0.0


@pytest.mark.parametrize("std", (0.0, -1.0e-4, float("nan")))
def test_dynamic_output_wakeup_rejects_invalid_scale(std):
    with pytest.raises(ValueError, match="output initialization"):
        activate_band_adapter_output(_dynamic_adapter(), std=std, seed=1)


def test_training_config_applies_dynamic_output_wakeup_after_identity_transfer():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _dynamic_adapter()
    config = {
        "seed": 372,
        "band_limited_adapter": {"output_initialization_std": 1.0e-4},
    }

    report = activate_band_adapter_output_if_requested(model, config)

    assert report["requested_output_std"] == pytest.approx(1.0e-4)
    assert all(
        torch.count_nonzero(expert.output.weight) > 0
        for expert in model.dense_decoder.band_limited_adapter.experts
    )


def test_low_mid_projection_zeros_registered_high_band():
    generator = torch.Generator().manual_seed(20260720)
    value = torch.randn(2, 3, 201, 201, generator=generator)

    projected = project_low_mid_increment(value)

    spectrum = torch.fft.rfft2(projected, norm="ortho")
    mask = registered_high_band_mask(201, 201, spectrum.device)
    assert spectrum[..., mask].abs().max().item() < 2.0e-5
    assert projected[..., 0, :].abs().max().item() < 2.0e-5
    assert projected.shape == value.shape
    assert projected.dtype == torch.float32


def test_low_mid_projection_requires_record_time_fields():
    with pytest.raises(ValueError, match="record,time,z,x"):
        project_low_mid_increment(torch.randn(2, 201, 201))


def test_zero_initialized_adapter_preserves_parent_and_output_heads_receive_gradients():
    generator = torch.Generator().manual_seed(372)
    adapter = BandLimitedFamilyAdapter(width=8, rank=4)
    shared = torch.randn(6, 8, 17, 19, generator=generator)
    time_features = torch.randn(2, 3, 5, generator=generator)
    record_to_medium = torch.tensor([0, 1])
    route_probabilities = torch.full((2, 3), 1.0 / 3.0)

    increment = adapter(
        shared,
        time_features,
        record_to_medium,
        route_probabilities,
    )

    assert increment.shape == (2, 3, 17, 19)
    assert torch.count_nonzero(increment) == 0
    increment.sum().backward()
    for expert in adapter.experts:
        assert expert.output.weight.grad is not None
        assert torch.isfinite(expert.output.weight.grad).all()
        assert expert.output.weight.grad.norm().item() > 0.0


def test_multiscale_adapter_is_exact_zero_then_wakes_internal_gradients():
    generator = torch.Generator().manual_seed(20260720)
    adapter = _multiscale_adapter()
    shared = torch.randn(6, 8, 17, 19, generator=generator)
    time_features = torch.randn(2, 3, 5, generator=generator)
    mapping = torch.tensor([0, 1])
    routes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    initial = adapter(shared, time_features, mapping, routes)

    assert initial.shape == (2, 3, 17, 19)
    assert torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert adapter.experts[0].output.weight.grad is not None
    assert adapter.experts[0].output.weight.grad.norm().item() > 0.0
    with torch.no_grad():
        adapter.experts[0].output.weight.normal_(std=1.0e-3)
    adapter.zero_grad(set_to_none=True)

    adapter(shared, time_features, mapping, routes).square().mean().backward()

    assert adapter.experts[0].input_projection.weight.grad is not None
    assert adapter.experts[0].input_projection.weight.grad.norm().item() > 0.0
    residual_scale = adapter.experts[0].full_stack.blocks[0].residual_scale
    assert residual_scale.grad is not None
    assert residual_scale.grad.norm().item() > 0.0


def test_multiscale_adapter_sparse_one_hot_dispatch_skips_unused_family():
    generator = torch.Generator().manual_seed(372)
    adapter = _multiscale_adapter()
    calls = [0, 0, 0]
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, expert in enumerate(adapter.experts)
    ]
    try:
        adapter(
            torch.randn(4, 8, 15, 17, generator=generator),
            torch.randn(2, 2, 5, generator=generator),
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [1, 0, 1]


def test_multiscale_adapter_soft_route_matches_explicit_weighted_sum():
    generator = torch.Generator().manual_seed(372)
    adapter = _multiscale_adapter().eval()
    with torch.no_grad():
        for expert in adapter.experts:
            expert.output.weight.normal_(std=1.0e-3)
    shared = torch.randn(2, 8, 15, 17, generator=generator)
    features = torch.randn(1, 2, 5, generator=generator)
    probabilities = torch.tensor([[0.2, 0.3, 0.5]])

    observed = adapter(shared, features, torch.tensor([0]), probabilities)
    expected = sum(
        probabilities[0, index] * expert(shared, features)
        for index, expert in enumerate(adapter.experts)
    )

    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("architecture", "unknown"),
        ("spectral_rank", 0),
        ("modes", 0),
        ("full_depth", 0),
        ("coarse_depth", 0),
    ),
)
def test_multiscale_adapter_rejects_invalid_capacity(name, value):
    arguments = {
        "width": 8,
        "rank": 6,
        "architecture": "multiscale_spectral",
        "spectral_rank": 4,
        "modes": 4,
        "full_depth": 2,
        "coarse_depth": 1,
        "activation_checkpointing": False,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match="band adapter"):
        BandLimitedFamilyAdapter(**arguments)


def test_one_hot_adapter_routing_executes_only_selected_experts():
    generator = torch.Generator().manual_seed(372)
    adapter = BandLimitedFamilyAdapter(width=8, rank=4)
    shared = torch.randn(6, 8, 17, 19, generator=generator)
    time_features = torch.randn(2, 3, 5, generator=generator)
    mapping = torch.tensor([0, 1])
    probabilities = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    calls = [0, 0, 0]
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, expert in enumerate(adapter.experts)
    ]
    try:
        increment = adapter(shared, time_features, mapping, probabilities)
        increment.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [1, 0, 1]
    assert adapter.experts[0].output.weight.grad is not None
    assert adapter.experts[1].output.weight.grad is None
    assert adapter.experts[2].output.weight.grad is not None


def test_adapter_rejects_malformed_router_probabilities():
    adapter = BandLimitedFamilyAdapter(width=8, rank=4)
    shared = torch.randn(2, 8, 9, 11)
    time_features = torch.randn(1, 2, 5)
    mapping = torch.tensor([0])

    with pytest.raises(ValueError, match="router probabilities"):
        adapter(shared, time_features, mapping, torch.ones(1, 2))
    with pytest.raises(ValueError, match="sum to one"):
        adapter(shared, time_features, mapping, torch.ones(1, 3))


def test_band_adapter_expansion_requires_explicit_zero_output_transfer():
    parent = ProbeVariant(
        depth=8,
        use_local_phase=True,
        temporal_basis_rank=96,
        family_expert_rank=16,
    )
    candidate = replace(parent, band_adapter_rank=16)
    config = {
        "checkpoint_transfer": {
            "allow_new_band_adapter_parameters": True,
            "parent_optimizer_state": False,
        }
    }

    assert band_adapter_missing_prefixes(
        config,
        parent_variant=parent,
        candidate_variant=candidate,
    ) == ("dense_decoder.band_limited_adapter.",)

    with pytest.raises(ValueError, match="explicit transfer permission"):
        band_adapter_missing_prefixes(
            {},
            parent_variant=parent,
            candidate_variant=candidate,
        )
    with pytest.raises(ValueError, match="optimizer"):
        band_adapter_missing_prefixes(
            {
                "checkpoint_transfer": {
                    "allow_new_band_adapter_parameters": True,
                    "parent_optimizer_state": True,
                }
            },
            parent_variant=parent,
            candidate_variant=candidate,
        )
    with pytest.raises(ValueError, match="staged"):
        band_adapter_missing_prefixes(
            config,
            parent_variant=parent,
            candidate_variant=replace(candidate, modes=101),
        )


def test_adapter_identity_report_requires_zero_output_heads():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = BandLimitedFamilyAdapter(
        width=8,
        rank=4,
    )

    report = band_adapter_identity_report(model, expected_rank=4)

    assert report == {
        "band_adapter_rank": 4,
        "band_adapter_architecture": "low_rank",
        "expert_count": 3,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.dense_decoder.band_limited_adapter.parameters()
        ),
        "exact_parent_identity": True,
    }
    model.dense_decoder.band_limited_adapter.experts[0].output.bias.data.fill_(1.0)
    with pytest.raises(ValueError, match="exact parent identity"):
        band_adapter_identity_report(model, expected_rank=4)


def test_multiscale_adapter_identity_report_records_architecture_and_parameters():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _multiscale_adapter()
    expected_parameters = sum(
        parameter.numel()
        for parameter in model.dense_decoder.band_limited_adapter.parameters()
    )

    report = band_adapter_identity_report(
        model,
        expected_rank=6,
        expected_architecture="multiscale_spectral",
    )

    assert report == {
        "band_adapter_rank": 6,
        "band_adapter_architecture": "multiscale_spectral",
        "expert_count": 3,
        "trainable_parameters": expected_parameters,
        "exact_parent_identity": True,
    }


def test_shared_dynamic_identity_report_records_one_expert():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _shared_dynamic_adapter()

    report = band_adapter_identity_report(
        model,
        expected_rank=6,
        expected_architecture="shared_dynamic_multiscale_spectral",
    )

    assert report["expert_count"] == 1
    assert report["exact_parent_identity"] is True


def test_band_adapter_optimizer_separates_output_head_and_features():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _multiscale_adapter()

    optimizer = build_band_adapter_adamw(
        model,
        feature_lr=3.0e-4,
        output_lr=1.0e-5,
        weight_decay=1.0e-6,
    )

    observed = {
        id(parameter): (str(group["group_name"]), float(group["lr"]))
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())
    assert set(observed) == {id(parameter) for parameter in named.values()}
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        named
    )
    for name, parameter in named.items():
        group, learning_rate = observed[id(parameter)]
        if ".output." in name:
            assert group.startswith("adapter_output_")
            assert learning_rate == pytest.approx(1.0e-5)
        else:
            assert group.startswith("adapter_feature_")
            assert learning_rate == pytest.approx(3.0e-4)


@pytest.mark.parametrize(
    ("feature_lr", "output_lr", "weight_decay"),
    (
        (0.0, 1.0e-5, 0.0),
        (3.0e-4, float("nan"), 0.0),
        (3.0e-4, 1.0e-5, -1.0e-6),
    ),
)
def test_band_adapter_optimizer_rejects_invalid_rates(
    feature_lr, output_lr, weight_decay
):
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _multiscale_adapter()

    with pytest.raises(ValueError, match="learning rates|weight decay"):
        build_band_adapter_adamw(
            model,
            feature_lr=feature_lr,
            output_lr=output_lr,
            weight_decay=weight_decay,
        )


def test_registered_band_adapter_learning_rates_are_all_or_nothing():
    config = {
        "band_limited_adapter": {"adapter_only": True},
        "optimizer": {
            "band_adapter_feature_learning_rate": 3.0e-4,
            "band_adapter_output_learning_rate": 1.0e-5,
        },
    }

    assert registered_band_adapter_learning_rates(config) == pytest.approx(
        (3.0e-4, 1.0e-5)
    )
    assert registered_band_adapter_learning_rates(
        {"optimizer": {"dense_learning_rate": 3.0e-4}}
    ) is None
    del config["optimizer"]["band_adapter_output_learning_rate"]
    with pytest.raises(ValueError, match="registered together"):
        registered_band_adapter_learning_rates(config)


def test_adapter_stage_freezes_every_parent_parameter():
    model = nn.Module()
    model.parent = nn.Linear(5, 5)
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = BandLimitedFamilyAdapter(
        width=8,
        rank=4,
    )

    stage = configure_band_adapter_stage(model, epoch=1)

    assert stage.trainable_prefixes == ("dense_decoder.band_limited_adapter",)
    assert stage.trainable_parameters > 0
    assert stage.frozen_parameters > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(
            "dense_decoder.band_limited_adapter."
        )
