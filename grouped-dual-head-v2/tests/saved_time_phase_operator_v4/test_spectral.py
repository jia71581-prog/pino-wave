import torch

import saved_time_phase_operator_v4.spectral as spectral
from saved_time_phase_operator_v4.spectral import (
    AxisFactorizedComplexSpectralConv2d,
    FactorizedComplexResidualStack,
)


def test_expanded_spectral_modes_preserve_parent_output_and_zero_new_modes():
    torch.manual_seed(83)
    parent = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=6,
        modes=3,
        depth=2,
        activation_checkpointing=False,
    )
    candidate = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=6,
        modes=9,
        depth=2,
        activation_checkpointing=False,
    )

    report = spectral.transfer_expanded_spectral_modes(parent, candidate)
    value = torch.randn(2, 8, 17, 19)

    torch.testing.assert_close(candidate(value), parent(value), rtol=0.0, atol=0.0)
    assert report == {
        "source_modes": 3,
        "target_modes": 9,
        "expanded_tensors": 4,
    }
    for block in candidate.blocks:
        assert torch.count_nonzero(block.spectral.weight_x[:, :, 3:]) == 0
        assert torch.count_nonzero(block.spectral.weight_z[:, :, 3:]) == 0


def test_spectral_mode_transfer_rejects_shrinking():
    parent = FactorizedComplexResidualStack(
        width=8, spectral_rank=6, modes=9, depth=1, activation_checkpointing=False
    )
    candidate = FactorizedComplexResidualStack(
        width=8, spectral_rank=6, modes=3, depth=1, activation_checkpointing=False
    )

    try:
        spectral.transfer_expanded_spectral_modes(parent, candidate)
    except ValueError as error:
        assert "strictly expand" in str(error)
    else:
        raise AssertionError("spectral mode shrinking was accepted")


def test_factorized_stack_preserves_shape_and_gradients():
    model = FactorizedComplexResidualStack(width=16, spectral_rank=8, modes=12, depth=8)
    value = torch.randn(2, 16, 33, 33, requires_grad=True)

    model(value).square().mean().backward()

    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_eight_factorized_blocks_fit_factorized_parameter_budget():
    model = FactorizedComplexResidualStack(width=64, spectral_rank=40, modes=32, depth=8)

    assert sum(parameter.numel() for parameter in model.parameters()) < 8_000_000


def test_high_rank_deep_stack_supplies_a_real_capacity_control():
    shallow = FactorizedComplexResidualStack(width=64, spectral_rank=112, modes=32, depth=2)
    deep = FactorizedComplexResidualStack(width=64, spectral_rank=112, modes=32, depth=8)

    shallow_count = sum(parameter.numel() for parameter in shallow.parameters())
    deep_count = sum(parameter.numel() for parameter in deep.parameters())
    assert deep_count > 4 * shallow_count - 100_000
    assert 12_000_000 < deep_count < 15_000_000


def test_factorized_convolution_clips_modes_on_small_grids():
    layer = AxisFactorizedComplexSpectralConv2d(4, 5, modes=32)
    value = torch.randn(2, 4, 9, 11, requires_grad=True)

    output = layer(value)
    output.mean().backward()

    assert output.shape == (2, 5, 9, 11)
    assert value.grad is not None


def test_boundary_halo_spectral_path_preserves_schema_shape_and_gradients():
    torch.manual_seed(97)
    parent = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=6,
        modes=5,
        depth=2,
        activation_checkpointing=False,
        boundary_halo_radius=0,
    )
    candidate = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=6,
        modes=5,
        depth=2,
        activation_checkpointing=False,
        boundary_halo_radius=3,
    )
    assert tuple(candidate.state_dict()) == tuple(parent.state_dict())
    candidate.load_state_dict(parent.state_dict(), strict=True)
    value = torch.randn(2, 8, 17, 19, requires_grad=True)

    output = candidate(value)
    output.square().mean().backward()

    assert output.shape == value.shape
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(parameter.grad is not None for parameter in candidate.parameters())
    assert (output.detach() - parent(value.detach())).abs().max() > 1.0e-7


def test_zero_boundary_halo_is_exact_legacy_path():
    torch.manual_seed(101)
    legacy = AxisFactorizedComplexSpectralConv2d(4, 5, modes=6)
    explicit_zero = AxisFactorizedComplexSpectralConv2d(
        4, 5, modes=6, boundary_halo_radius=0
    )
    explicit_zero.load_state_dict(legacy.state_dict(), strict=True)
    value = torch.randn(2, 4, 13, 15)

    torch.testing.assert_close(
        explicit_zero(value), legacy(value), rtol=0.0, atol=0.0
    )


def test_activation_checkpointing_matches_direct_forward():
    direct = FactorizedComplexResidualStack(
        width=8, spectral_rank=4, modes=6, depth=4, activation_checkpointing=False
    )
    checkpointed = FactorizedComplexResidualStack(
        width=8, spectral_rank=4, modes=6, depth=4, activation_checkpointing=True
    )
    checkpointed.load_state_dict(direct.state_dict())
    direct.train()
    checkpointed.train()
    left = torch.randn(1, 8, 17, 17, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)

    direct_output = direct(left)
    checkpointed_output = checkpointed(right)

    torch.testing.assert_close(checkpointed_output, direct_output)


def test_coupled_axis_path_changes_2d_response_without_changing_checkpoint_schema():
    torch.manual_seed(73)
    independent = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=2,
        activation_checkpointing=False,
        coupled_axes=False,
    )
    coupled = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=2,
        activation_checkpointing=False,
        coupled_axes=True,
    )

    assert tuple(coupled.state_dict()) == tuple(independent.state_dict())
    coupled.load_state_dict(independent.state_dict(), strict=True)
    value = torch.randn(1, 8, 17, 19, requires_grad=True)
    baseline = independent(value.detach())
    candidate = coupled(value)

    assert candidate.shape == baseline.shape
    assert (candidate - baseline).abs().max().item() > 1.0e-6
    candidate.square().mean().backward()
    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in coupled.parameters())


def test_local_differential_path_is_zero_gated_but_gate_can_learn():
    torch.manual_seed(79)
    parent = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=2,
        activation_checkpointing=False,
        local_differential_residual=False,
    )
    candidate = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=2,
        activation_checkpointing=False,
        local_differential_residual=True,
    )
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(".local_differential." in key for key in incompatible.missing_keys)
    value = torch.randn(1, 8, 17, 19)

    torch.testing.assert_close(candidate(value), parent(value), rtol=0.0, atol=0.0)

    candidate(value).square().mean().backward()
    paths = tuple(block.local_differential for block in candidate.blocks)
    assert all(path is not None for path in paths)
    assert all(
        path.scale.grad is not None
        and torch.isfinite(path.scale.grad)
        and torch.count_nonzero(path.scale.grad) > 0
        for path in paths
    )
    assert all(path.projection.weight.grad is not None for path in paths)
    assert all(
        torch.count_nonzero(path.projection.weight.grad) == 0 for path in paths
    )


def test_local_differential_features_receive_gradient_after_gate_opens():
    model = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        local_differential_residual=True,
    )
    model.blocks[0].local_differential.scale.data.fill_(1.0e-3)
    value = torch.randn(1, 8, 17, 19, requires_grad=True)

    model(value).square().mean().backward()

    gradient = model.blocks[0].local_differential.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
