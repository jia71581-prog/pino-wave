"""Contract tests for AIS-MQFNO local-encoder variants."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fno_acoustic.ais_normalization import AISNormalizationBinding
from fno_acoustic.ais_model_components import (
    DispersionResidualHead,
    LocalQueryEncoder,
    MultiScaleLocalQueryEncoder,
)
from fno_acoustic.model_ais_mqfno import (
    AISMQFNO,
    LocalQueryEncoder as LegacyLocalQueryEncoder,
)
from fno_acoustic.temporal_operator import _make_validated_time_grid
from scripts.evaluate_ais_mqfno import _model_patch_geometry


def _full160_time() -> torch.Tensor:
    return torch.linspace(0.0, 1.2, 160)


def _model_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "global_in_features": 2,
        "native_in_channels": 3,
        "spatial_width": 4,
        "spatial_modes": 2,
        "spatial_layers": 1,
        "temporal_modes": 4,
        "local_dim": 3,
        "fusion_dim": 5,
        "halo_size": 17,
    }
    kwargs.update(overrides)
    return kwargs


def test_single_halo_patch_geometry_comes_from_model() -> None:
    model = AISMQFNO(**_model_kwargs(halo_size=25))

    assert _model_patch_geometry(model, 10.0, 20.0) == {
        "source": "model",
        "branches": [25],
        "branch_physical_spans_m": [
            {"halo_size": 25, "x_m": 240.0, "z_m": 480.0}
        ],
    }


def test_multiscale_patch_geometry_records_both_fixed_branches() -> None:
    model = AISMQFNO(
        **_model_kwargs(
            halo_size=25,
            local_encoder_kind="multiscale_9_25",
            local_dim=24,
            fusion_dim=48,
        )
    )

    assert _model_patch_geometry(model, 5.0, 7.5) == {
        "source": "model",
        "branches": [9, 25],
        "branch_physical_spans_m": [
            {"halo_size": 9, "x_m": 40.0, "z_m": 60.0},
            {"halo_size": 25, "x_m": 120.0, "z_m": 180.0},
        ],
    }


def test_multiscale_encoder_concatenates_independent_9_and_25_halo_branches() -> None:
    encoder = MultiScaleLocalQueryEncoder(in_channels=3)
    native = torch.randn(2, 3, 11, 13)
    queries = torch.rand(2, 4, 2)

    encoded = encoder(native, queries)
    small = encoder.small(native, queries)
    large = encoder.large(native, queries)

    assert encoder.output_dim == 24
    assert encoder.small.halo_size == 9
    assert encoder.large.halo_size == 25
    assert encoded.shape == (2, 4, 24)
    assert torch.equal(encoded, torch.cat((small, large), dim=-1))

    small_ptrs = {parameter.data_ptr() for parameter in encoder.small.parameters()}
    large_ptrs = {parameter.data_ptr() for parameter in encoder.large.parameters()}
    assert small_ptrs.isdisjoint(large_ptrs)


def test_multiscale_encoder_zero_init_unblocks_both_branches_after_one_step() -> None:
    torch.manual_seed(7)
    encoder = MultiScaleLocalQueryEncoder(in_channels=2)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    native = torch.randn(1, 2, 15, 17)
    queries = torch.rand(1, 3, 2)
    target = torch.ones(1, 3, 24)

    (encoder(native, queries) - target).square().mean().backward()

    for branch in (encoder.small, encoder.large):
        assert torch.count_nonzero(branch.encoder[-1].weight) == 0
        assert branch.encoder[-1].weight.grad is not None
        assert branch.encoder[-1].weight.grad.abs().sum() > 0
        assert branch.encoder[0].weight.grad is not None
        assert torch.count_nonzero(branch.encoder[0].weight.grad) == 0

    optimizer.step()
    for branch in (encoder.small, encoder.large):
        assert torch.count_nonzero(branch.encoder[-1].weight) > 0
    optimizer.zero_grad()
    (encoder(native, queries) - target).square().mean().backward()

    for branch in (encoder.small, encoder.large):
        assert branch.encoder[0].weight.grad is not None
        assert branch.encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["in_channels", "local_dim"])
@pytest.mark.parametrize("value", [True, 1.5, "2", [2], 0, -1])
def test_local_encoder_rejects_invalid_channel_and_width_types(
    name: str, value: object
) -> None:
    kwargs: dict[str, object] = {"in_channels": 2, "local_dim": 3}
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        LocalQueryEncoder(**kwargs)


def test_multiscale_encoder_rejects_nonfixed_branch_width() -> None:
    with pytest.raises(ValueError, match="branch_dim.*12"):
        MultiScaleLocalQueryEncoder(in_channels=3, branch_dim=8)


def test_model_multiscale_variant_uses_fixed_widths_and_maximum_halo_contract() -> None:
    model = AISMQFNO(
        **_model_kwargs(
            local_encoder_kind="multiscale_9_25",
            local_dim=24,
            fusion_dim=48,
            halo_size=25,
        )
    )
    output = model(
        torch.randn(1, 5, 5, 160, 2),
        torch.randn(1, 3, 15, 17),
        torch.rand(1, 2, 2),
        _full160_time(),
    )

    assert isinstance(model.local_encoder, MultiScaleLocalQueryEncoder)
    assert model.local_encoder_kind == "multiscale_9_25"
    assert model.halo_size == 25
    assert model.fusion[0].in_features == 4 + 24
    assert model.fusion[0].out_features == 48
    assert model.last_native_query_shape == (25, 25)
    assert output.shape == (1, 2, 160)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"local_encoder_kind": "unknown"}, "local_encoder_kind"),
        ({"local_encoder_kind": ["single"]}, "local_encoder_kind"),
        (
            {
                "local_encoder_kind": "multiscale_9_25",
                "local_dim": 12,
                "fusion_dim": 48,
                "halo_size": 25,
            },
            "local_dim.*24",
        ),
        (
            {
                "local_encoder_kind": "multiscale_9_25",
                "local_dim": 24,
                "fusion_dim": 32,
                "halo_size": 25,
            },
            "fusion_dim.*48",
        ),
        (
            {
                "local_encoder_kind": "multiscale_9_25",
                "local_dim": 24,
                "fusion_dim": 48,
                "halo_size": 17,
            },
            "halo_size.*25",
        ),
    ],
)
def test_model_rejects_invalid_local_encoder_variant_combinations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AISMQFNO(**_model_kwargs(**overrides))


def test_model_rejects_unhashable_str_subclass_encoder_kind() -> None:
    class UnhashableStr(str):
        __hash__ = None

    with pytest.raises(ValueError, match="local_encoder_kind"):
        AISMQFNO(**_model_kwargs(local_encoder_kind=UnhashableStr("single")))


def test_explicit_single_variant_matches_default_state_and_forward_exactly() -> None:
    torch.manual_seed(19)
    default = AISMQFNO(**_model_kwargs()).eval()
    torch.manual_seed(19)
    explicit = AISMQFNO(**_model_kwargs(local_encoder_kind="single")).eval()

    default_state = default.state_dict()
    explicit_state = explicit.state_dict()
    assert default_state.keys() == explicit_state.keys()
    assert all(
        torch.equal(default_state[key], explicit_state[key]) for key in default_state
    )
    assert isinstance(default.local_encoder, LegacyLocalQueryEncoder)
    assert default.last_native_query_shape == (17, 17)

    inputs = (
        torch.randn(1, 5, 5, 160, 2),
        torch.randn(1, 3, 11, 13),
        torch.rand(1, 2, 2),
        _full160_time(),
    )
    with torch.no_grad():
        assert torch.equal(default(*inputs), explicit(*inputs))


def test_dispersion_head_is_zero_initialized_and_uses_registered_statistics() -> None:
    head = DispersionResidualHead(4, 48, 24, 3000.0, 500.0)
    residual = head(
        torch.randn(2, 11, 4),
        torch.zeros(2, 11),
        30.0,
        40.0,
        torch.linspace(0.2, 0.7, 160),
    )

    assert residual.shape == (2, 11, 160)
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(head.mlp[-1].weight) == 0
    assert torch.count_nonzero(head.mlp[-1].bias) == 0
    assert dict(head.named_buffers()).keys() >= {"velocity_mean", "velocity_std"}
    assert not any(name.startswith("velocity_") for name, _ in head.named_parameters())


def test_dispersion_head_forward_does_not_extract_tensor_scalars(monkeypatch) -> None:
    head = DispersionResidualHead(2, 48, 24, 3000.0, 500.0)

    with monkeypatch.context() as context:
        context.setattr(
            torch.Tensor,
            "item",
            lambda self: pytest.fail("dispersion forward must not call Tensor.item"),
        )
        residual = head(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            30.0,
            40.0,
            _full160_time(),
        )

    assert torch.equal(residual, torch.zeros_like(residual))


def test_prevalidated_dispersion_chunk_uses_one_dynamic_host_gate(
    monkeypatch,
) -> None:
    head = DispersionResidualHead(2, 48, 24, 3000.0, 500.0)
    calls = 0
    original = head._validate_condition

    def counted(condition: torch.Tensor, message: str) -> None:
        nonlocal calls
        calls += 1
        original(condition, message)

    monkeypatch.setattr(head, "_validate_condition", counted)
    time_s = _full160_time()
    head._forward_validated(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2),
        30.0,
        40.0,
        _make_validated_time_grid(time_s),
        spacing_host_validated=True,
    )

    assert calls == 1


def test_public_model_and_dispersion_apis_reject_validated_tensor_bypass() -> None:
    model = AISMQFNO(**_model_kwargs()).eval()
    time_s = _full160_time()
    global_inputs = torch.randn(1, 5, 5, 160, 2)
    native = torch.randn(1, 3, 11, 13)
    queries = torch.rand(1, 2, 2)
    context = model.encode_global(global_inputs, time_s)

    with pytest.raises(TypeError):
        model.encode_global(global_inputs, time_s, _validated_time_s=time_s)
    with pytest.raises(TypeError):
        model.decode_queries(
            context,
            native,
            queries,
            time_s,
            _validated_time_s=time_s,
        )
    with pytest.raises(TypeError):
        model.trajectory_head(
            torch.randn(1, 2, 160, model.trajectory_head.in_dim),
            time_s,
            _validated_time_s=time_s,
        )
    head = DispersionResidualHead(2, 48, 24, 3000.0, 500.0)
    with pytest.raises(TypeError):
        head(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            30.0,
            40.0,
            time_s,
            _validated_time_s=time_s,
        )


def test_dispersion_statistics_preserve_precision_until_compute_dtype() -> None:
    head = DispersionResidualHead(2, 48, 24, 1.0e300, 1.0)

    assert head.velocity_mean.dtype == torch.float64
    assert torch.isfinite(head.velocity_mean)
    with pytest.raises(ValueError, match="finite|representable"):
        head(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            30.0,
            40.0,
            _full160_time(),
        )


@pytest.mark.parametrize(
    ("head", "velocity_hat", "dx_m", "time_s"),
    [
        (
            DispersionResidualHead(2, 48, 24, 0.0, 1.0),
            torch.nextafter(torch.zeros(1, 1), torch.ones(1, 1)),
            30.0,
            _full160_time(),
        ),
        (
            DispersionResidualHead(2, 48, 24, 3000.0, 500.0),
            torch.zeros(1, 1),
            1.0e10,
            torch.linspace(0.0, 1.0e-35, 160),
        ),
    ],
)
def test_dispersion_head_rejects_finite_inputs_that_overflow_features(
    head: DispersionResidualHead,
    velocity_hat: torch.Tensor,
    dx_m: float,
    time_s: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="finite|representable"):
        head(
            torch.zeros(1, 1, 2), velocity_hat, dx_m, 30.0, time_s
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dispersion_cuda_validation_keeps_context_usable() -> None:
    script = r"""
import torch
from fno_acoustic.ais_model_components import DispersionResidualHead

head = DispersionResidualHead(2, 48, 24, 0.0, 1.0).cuda()
tiny = torch.nextafter(torch.zeros(1, 1, device="cuda"), torch.ones(1, 1, device="cuda"))
try:
    head(
        torch.zeros(1, 1, 2, device="cuda"),
        tiny,
        30.0,
        30.0,
        torch.linspace(0.0, 1.0, 160, device="cuda"),
    )
except ValueError:
    pass
else:
    raise AssertionError("invalid CUDA dispersion inputs must raise ValueError")
probe = torch.ones(4, device="cuda") + 1
torch.cuda.synchronize()
assert torch.equal(probe.cpu(), torch.full((4,), 2.0))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_dispersion_features_use_positive_basis_modes_not_source_frequency() -> None:
    head = DispersionResidualHead(2, 48, 24, 3000.0, 500.0)

    features = head.dispersion_features(
        torch.zeros(1, 1), 30.0, 40.0, duration_s=0.5
    )

    assert features.shape == (1, 1, 24, 3)
    assert features[0, 0, 0, 0].item() == pytest.approx(1 / 24)
    assert features[0, 0, 23, 0].item() == pytest.approx(1.0)
    assert features[0, 0, 0, 1].item() == pytest.approx((1 / 0.5) * 30 / 3000)
    assert features[0, 0, 23, 1].item() == pytest.approx((24 / 0.5) * 30 / 3000)
    assert features[0, 0, 0, 2].item() == pytest.approx((1 / 0.5) * 40 / 3000)


@pytest.mark.parametrize(
    ("velocity_hat", "dx_m", "dz_m", "time_s", "message"),
    [
        (torch.zeros(1, 1), 0.0, 30.0, torch.linspace(0, 1, 160), "dx_m"),
        (torch.zeros(1, 1), 30.0, float("nan"), torch.linspace(0, 1, 160), "dz_m"),
        (torch.full((1, 1), -7.0), 30.0, 30.0, torch.linspace(0, 1, 160), "velocity"),
        (torch.zeros(1, 1), 30.0, 30.0, torch.linspace(0, 1, 159), "160"),
        (torch.zeros(1, 1), 30.0, 30.0, torch.zeros(160), "increasing"),
    ],
)
def test_dispersion_head_rejects_invalid_physical_inputs(
    velocity_hat: torch.Tensor,
    dx_m: float,
    dz_m: float,
    time_s: torch.Tensor,
    message: str,
) -> None:
    head = DispersionResidualHead(2, 48, 24, 3000.0, 500.0)
    with pytest.raises(ValueError, match=message):
        head(torch.zeros(1, 1, 2), velocity_hat, dx_m, dz_m, time_s)


def test_dispersion_head_two_steps_activate_mlp_and_local_encoder_gradients() -> None:
    local_encoder = LocalQueryEncoder(2, 4, halo_size=3)
    head = DispersionResidualHead(4, 48, 24, 3000.0, 500.0)
    optimizer = torch.optim.SGD(
        [*local_encoder.parameters(), *head.parameters()], lr=0.1
    )
    native = torch.randn(1, 2, 5, 5)
    native[:, 0].zero_()
    queries = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]])
    target = torch.ones(1, 2, 160)
    time_s = torch.linspace(0.0, 1.0, 160)

    optimizer.zero_grad(set_to_none=True)
    local = local_encoder(native, queries)
    first = head(local, torch.zeros(1, 2), 30.0, 30.0, time_s)
    torch.nn.functional.mse_loss(first, target).backward()
    assert head.mlp[-1].weight.grad is not None
    assert head.mlp[-1].weight.grad.abs().sum() > 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    local = local_encoder(native, queries)
    second = head(local, torch.zeros(1, 2), 30.0, 30.0, time_s)
    torch.nn.functional.mse_loss(second, target).backward()
    assert head.mlp[0].weight.grad is not None
    assert head.mlp[0].weight.grad.abs().sum() > 0
    assert local_encoder.encoder[-1].weight.grad is not None
    assert local_encoder.encoder[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dispersion_head": "unknown"}, "dispersion_head"),
        ({"dispersion_head": ["none"]}, "dispersion_head"),
        ({"dispersion_head": "phase_residual_24"}, "velocity"),
    ],
)
def test_model_rejects_invalid_dispersion_configuration(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AISMQFNO(**_model_kwargs(**overrides))


@pytest.mark.parametrize(
    "invalid_kind",
    [
        True,
        1,
        ["none"],
        type("HashableStr", (str,), {})("none"),
        type("UnhashableStr", (str,), {"__hash__": None})("none"),
    ],
)
def test_model_rejects_non_exact_string_dispersion_head_kinds(
    invalid_kind: object,
) -> None:
    with pytest.raises(ValueError, match="dispersion_head"):
        AISMQFNO(**_model_kwargs(dispersion_head=invalid_kind))


def _dispersion_model() -> AISMQFNO:
    return AISMQFNO(
        **_model_kwargs(
            dispersion_head="phase_residual_24",
            velocity_mean=3000.0,
            velocity_std=500.0,
        )
    )


def test_model_dispersion_samples_normalized_velocity_at_query_centers() -> None:
    class RecordingHead(torch.nn.Module):
        def forward(self, local, velocity_hat, dx_m, dz_m, time_s):
            self.velocity_hat = velocity_hat.detach().clone()
            self.spacing = (dx_m, dz_m)
            return torch.zeros(*velocity_hat.shape, 160, device=velocity_hat.device)

    model = _dispersion_model()
    recorder = RecordingHead()
    model.dispersion_residual_head = recorder
    native = torch.zeros(1, 3, 3, 3)
    native[0, 0] = torch.arange(9).reshape(3, 3)
    query = torch.tensor([[[0.5, 0.5], [0.0, 1.0]]])

    model.decode_queries(
        torch.randn(1, 3, 3, 160, 4),
        native,
        query,
        _full160_time(),
        dx_m=30.0,
        dz_m=40.0,
    )

    assert torch.equal(recorder.velocity_hat, torch.tensor([[4.0, 2.0]]))
    assert recorder.spacing == (30.0, 40.0)


def test_model_dispersion_direct_forward_accepts_spacing_and_requires_it() -> None:
    model = _dispersion_model().eval()
    inputs = (
        torch.randn(1, 5, 5, 160, 2),
        torch.randn(1, 3, 11, 13),
        torch.rand(1, 2, 2),
        _full160_time(),
    )

    with torch.no_grad():
        output = model(*inputs, dx_m=30.0, dz_m=40.0)
    assert output.shape == (1, 2, 160)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="spacing"):
        model(*inputs)


def test_model_ordinary_direct_forward_remains_spacing_optional() -> None:
    model = AISMQFNO(**_model_kwargs()).eval()
    inputs = (
        torch.randn(1, 5, 5, 160, 2),
        torch.randn(1, 3, 11, 13),
        torch.rand(1, 2, 2),
        _full160_time(),
    )

    with torch.no_grad():
        without_spacing = model(*inputs)
        with_spacing = model(*inputs, dx_m=30.0, dz_m=40.0)
    assert torch.equal(without_spacing, with_spacing)


def test_dispersion_model_requires_spacing_but_none_model_accepts_omission() -> None:
    context = torch.randn(1, 3, 3, 160, 4)
    native = torch.zeros(1, 3, 5, 5)
    query = torch.rand(1, 2, 2)
    time_s = _full160_time()

    with pytest.raises(ValueError, match="dx_m|spacing"):
        _dispersion_model().decode_queries(context, native, query, time_s)

    ordinary = AISMQFNO(**_model_kwargs())
    assert ordinary.decode_queries(context, native, query, time_s).shape == (1, 2, 160)


def test_zero_initialized_dispersion_preserves_standardized_trajectory_exactly() -> None:
    torch.manual_seed(83)
    ordinary = AISMQFNO(**_model_kwargs()).eval()
    torch.manual_seed(83)
    dispersion = _dispersion_model().eval()
    context = torch.randn(1, 3, 3, 160, 4)
    native = torch.zeros(1, 3, 5, 5)
    query = torch.rand(1, 2, 2)
    time_s = _full160_time()

    with torch.no_grad():
        expected = ordinary.decode_queries(context, native, query, time_s)
        actual = dispersion.decode_queries(
            context, native, query, time_s, dx_m=30.0, dz_m=40.0
        )
    assert torch.equal(actual, expected)


def test_train_model_kwargs_injects_binding_statistics_and_rejects_yaml_override() -> None:
    from scripts.train_ais_mqfno import _model_kwargs as training_model_kwargs

    config = {
        "sampling": {"global_size": 3},
        "model": {
            **_model_kwargs(),
            "name": "ais_mqfno",
            "dispersion_head": "phase_residual_24",
        },
    }
    binding = AISNormalizationBinding(
        Path("stats.json"), "a" * 64, 3100.0, 450.0, 0.0, 1.0, 1.0e-6
    )

    kwargs = training_model_kwargs(config, normalization=binding)
    assert kwargs["velocity_mean"] == 3100.0
    assert kwargs["velocity_std"] == 450.0
    config["model"]["velocity_mean"] = 9999.0
    with pytest.raises(ValueError, match="unknown.*velocity_mean"):
        training_model_kwargs(config, normalization=binding)
