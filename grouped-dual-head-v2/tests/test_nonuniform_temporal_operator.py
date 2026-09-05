import copy
import pickle

import pytest
import torch

from fno_acoustic.temporal_operator import (
    NonUniformTemporalOperator,
    nonuniform_fourier_analysis,
    nonuniform_fourier_synthesis,
    trapezoid_weights,
)


def mixed_full160_time(device: torch.device | str = "cpu") -> torch.Tensor:
    early = torch.linspace(0.0, 0.3, 97, dtype=torch.float64, device=device)[:-1]
    late = torch.linspace(0.3, 1.2, 64, dtype=torch.float64, device=device)
    return torch.cat((early, late))


def circular_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.atan2(torch.sin(left - right), torch.cos(left - right)))


def test_trapezoid_weights_integrate_constant_over_physical_interval():
    time_s = mixed_full160_time()

    weights = trapezoid_weights(time_s)

    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), time_s[-1] - time_s[0], atol=1e-12)


def test_nutno_returns_real_complete_trace_and_finite_gradients():
    time_s = mixed_full160_time()
    layer = NonUniformTemporalOperator(channels=4, modes=24, residual_init="identity")
    x = torch.randn(2, 7, 160, 4, requires_grad=True)

    output = layer(x, time_s)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert not torch.is_complex(output)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_public_temporal_calls_validate_time_once_without_internal_rechecks(
    monkeypatch,
) -> None:
    import fno_acoustic.temporal_operator as temporal_module

    calls = 0
    original = temporal_module._validate_time_vector

    def counted(time_s: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        original(time_s)

    monkeypatch.setattr(temporal_module, "_validate_time_vector", counted)
    time_s = mixed_full160_time()
    x = torch.randn(1, 2, 160, 2)
    NonUniformTemporalOperator(2, 8)(x, time_s)
    assert calls == 1
    calls = 0
    coefficients = nonuniform_fourier_analysis(x, time_s, modes=8)
    assert calls == 1
    calls = 0
    nonuniform_fourier_synthesis(coefficients, time_s)
    assert calls == 1


def test_public_temporal_apis_do_not_accept_prevalidation_bypass_kwargs() -> None:
    time_s = mixed_full160_time()
    descending = time_s.flip(0)
    x = torch.randn(1, 2, 160, 2)
    coefficients = torch.randn(1, 2, 8, 2, dtype=torch.complex64)

    with pytest.raises(TypeError):
        trapezoid_weights(descending, _time_prevalidated=True)
    with pytest.raises(TypeError):
        nonuniform_fourier_analysis(
            x, descending, modes=8, _time_prevalidated=True
        )
    with pytest.raises(TypeError):
        nonuniform_fourier_synthesis(
            coefficients, descending, _time_prevalidated=True
        )
    with pytest.raises(TypeError):
        NonUniformTemporalOperator(2, 8)(
            x, descending, _time_prevalidated=True
        )


def test_validated_time_grid_owns_snapshot_and_rejects_internal_mutation() -> None:
    import fno_acoustic.temporal_operator as temporal_module

    original = mixed_full160_time()
    with pytest.raises(ValueError, match="strictly increasing"):
        temporal_module._ValidatedTimeGrid(original.flip(0))
    token = temporal_module._make_validated_time_grid(original)
    snapshot = token.time_s.clone()

    assert token.time_s.data_ptr() != original.data_ptr()
    original.add_(10.0)
    token.assert_current()
    assert torch.equal(token.time_s, snapshot)

    exposed = token.time_s
    exposed.data.copy_(torch.zeros_like(exposed))
    token.assert_current()
    assert torch.equal(token.time_s, snapshot)

    token._time_s.add_(1.0)
    with pytest.raises(ValueError, match="mutated|identity|version"):
        token.assert_current()


def test_validated_time_grid_pickle_and_deepcopy_refresh_bindings() -> None:
    import fno_acoustic.temporal_operator as temporal_module

    original = mixed_full160_time()
    token = temporal_module._ValidatedTimeGrid(original)

    for restored in (pickle.loads(pickle.dumps(token)), copy.deepcopy(token)):
        restored.assert_current()
        assert torch.equal(restored.time_s, original)
        assert restored._time_s.data_ptr() != token._time_s.data_ptr()


@pytest.mark.parametrize("api", ["analysis", "synthesis", "operator"])
def test_public_temporal_apis_preserve_time_gradients(api: str) -> None:
    time_s = mixed_full160_time().requires_grad_()
    x = torch.randn(1, 2, 160, 2)

    if api == "analysis":
        output = nonuniform_fourier_analysis(x, time_s, modes=8).abs().square().sum()
    elif api == "synthesis":
        coefficients = torch.randn(1, 2, 8, 2, dtype=torch.complex128)
        output = nonuniform_fourier_synthesis(coefficients, time_s).square().sum()
    else:
        output = NonUniformTemporalOperator(2, 8)(x, time_s).square().sum()

    output.backward()
    assert time_s.grad is not None
    assert torch.isfinite(time_s.grad).all()


def test_nutno_identity_initialization_preserves_constant_and_sinusoid():
    time_s = mixed_full160_time()
    layer = NonUniformTemporalOperator(channels=1, modes=32, residual_init="identity")
    x = (1.0 + torch.sin(2 * torch.pi * 8.0 * time_s))[None, None, :, None].float()

    assert torch.equal(layer(x, time_s), x)


def test_nonuniform_projection_recovers_known_mode_phase():
    time_s = mixed_full160_time()
    phase = torch.tensor(0.37, dtype=torch.float64)
    tau = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    signal = torch.cos(2 * torch.pi * 6 * tau + phase).float()

    coefficients = nonuniform_fourier_analysis(
        signal[None, None, :, None], time_s, modes=16
    )

    error = circular_distance(torch.angle(coefficients[0, 0, 6, 0]), phase)
    assert error < 0.03


def test_analysis_synthesis_recovers_single_positive_frequency_amplitude():
    time_s = mixed_full160_time()
    tau = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    signal = torch.cos(2 * torch.pi * 5 * tau + 0.41).float()

    coefficients = nonuniform_fourier_analysis(
        signal[None, None, :, None], time_s, modes=12
    )
    reconstructed = nonuniform_fourier_synthesis(coefficients, time_s)[0, 0, :, 0]

    assert torch.allclose(coefficients[0, 0, 5, 0].abs(), torch.tensor(0.5), atol=0.015)
    amplitude_ratio = torch.linalg.vector_norm(reconstructed) / torch.linalg.vector_norm(signal)
    assert torch.allclose(amplitude_ratio, torch.tensor(1.0), atol=0.02)
    relative_error = torch.linalg.vector_norm(reconstructed - signal) / torch.linalg.vector_norm(
        signal
    )
    assert relative_error < 0.02


def test_analysis_and_synthesis_have_documented_shapes_and_real_output():
    time_s = mixed_full160_time()
    x = torch.randn(2, 3, 160, 4)

    coefficients = nonuniform_fourier_analysis(x, time_s, modes=9)
    reconstructed = nonuniform_fourier_synthesis(coefficients, time_s)

    assert coefficients.shape == (2, 3, 9, 4)
    assert coefficients.dtype == torch.complex64
    assert reconstructed.shape == (2, 3, 160, 4)
    assert reconstructed.dtype == torch.float32
    assert torch.isfinite(reconstructed).all()


def test_identity_amplitude_does_not_depend_on_mode_count():
    time_s = mixed_full160_time()
    x = torch.randn(1, 3, 160, 2)

    assert torch.equal(NonUniformTemporalOperator(2, 8)(x, time_s), x)
    assert torch.equal(NonUniformTemporalOperator(2, 48)(x, time_s), x)


def test_complex_spectral_weight_receives_finite_nonzero_gradient():
    time_s = mixed_full160_time()
    layer = NonUniformTemporalOperator(2, 12)
    x = torch.randn(1, 2, 160, 2)

    layer(x, time_s).square().mean().backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad.real).all()
    assert torch.isfinite(layer.weight.grad.imag).all()
    assert layer.weight.grad.real.abs().sum() > 0
    assert layer.weight.grad.imag.abs().sum() > 0
    assert layer.dc_weight.grad is not None
    assert torch.isfinite(layer.dc_weight.grad).all()
    assert layer.dc_weight.grad.abs().sum() > 0


def test_only_non_dc_modes_are_exposed_as_complex_trainable_weights():
    layer = NonUniformTemporalOperator(2, 12)

    assert layer.weight.shape == (2, 2, 11)
    assert layer.weight.dtype == torch.complex64
    assert layer.dc_weight.shape == (2, 2)
    assert layer.dc_weight.dtype == torch.float32
    assert not torch.is_complex(layer.dc_weight)


def test_nutno_internal_complex_dtype_is_consistent_on_cpu_and_cuda_if_available():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        time_s = mixed_full160_time(device)
        layer = NonUniformTemporalOperator(2, 12).to(device)
        output = layer(torch.randn(1, 2, 160, 2, device=device), time_s)

        assert output.dtype == torch.float32
        assert layer.weight.dtype == torch.complex64


def test_forward_accepts_only_shared_strictly_increasing_full160_time():
    time_s = mixed_full160_time()
    x = torch.randn(2, 3, 160, 2)
    layer = NonUniformTemporalOperator(2, 8)

    assert torch.equal(layer(x, time_s.expand(2, -1).clone()), x)

    inconsistent = time_s.expand(2, -1).clone()
    inconsistent[1, 80] += 1e-4
    with pytest.raises(ValueError, match="shared"):
        layer(x, inconsistent)

    nonmonotone = time_s.clone()
    nonmonotone[80] = nonmonotone[79]
    with pytest.raises(ValueError, match="strictly increasing"):
        layer(x, nonmonotone)


@pytest.mark.parametrize(
    "shape",
    [(2, 160, 2), (2, 3, 159, 2), (2, 3, 160, 3)],
)
def test_forward_rejects_bad_input_shape(shape):
    layer = NonUniformTemporalOperator(2, 8)

    with pytest.raises(ValueError, match=r"\[B,Q,160,C\]"):
        layer(torch.randn(*shape), mixed_full160_time())


@pytest.mark.parametrize("modes", [0, -1, 161])
def test_invalid_mode_count_is_rejected(modes):
    with pytest.raises(ValueError, match="modes"):
        NonUniformTemporalOperator(channels=2, modes=modes)


@pytest.mark.parametrize("channels", [0, -2])
def test_invalid_channel_count_is_rejected(channels):
    with pytest.raises(ValueError, match="channels"):
        NonUniformTemporalOperator(channels=channels, modes=8)


def test_quadrature_rejects_bad_time_vectors():
    with pytest.raises(ValueError, match="one-dimensional"):
        trapezoid_weights(torch.ones(1, 3))
    with pytest.raises(ValueError, match="at least two"):
        trapezoid_weights(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        trapezoid_weights(torch.tensor([0.0, 0.2, 0.1]))
    with pytest.raises(ValueError, match="floating"):
        trapezoid_weights(torch.arange(3))


def test_analysis_rejects_bad_shape_time_and_modes():
    time_s = mixed_full160_time()
    x = torch.randn(1, 2, 160, 3)

    with pytest.raises(ValueError, match=r"\[B,Q,T,C\]"):
        nonuniform_fourier_analysis(x[0], time_s, modes=8)
    with pytest.raises(ValueError, match="time"):
        nonuniform_fourier_analysis(x, time_s[:-1], modes=8)
    with pytest.raises(ValueError, match="modes"):
        nonuniform_fourier_analysis(x, time_s, modes=0)


@pytest.mark.parametrize(
    ("dtype", "coefficient_dtype"),
    [
        (torch.float16, torch.complex64),
        (torch.bfloat16, torch.complex64),
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ],
)
def test_analysis_supports_real_signal_dtypes_at_safe_spectral_precision(
    dtype, coefficient_dtype
):
    x = torch.ones(1, 2, 160, 3, dtype=dtype)

    coefficients = nonuniform_fourier_analysis(x, mixed_full160_time(), modes=8)

    assert coefficients.dtype == coefficient_dtype


def test_analysis_rejects_complex_signal():
    x = torch.ones(1, 2, 160, 3, dtype=torch.complex64)

    with pytest.raises(ValueError, match="real floating"):
        nonuniform_fourier_analysis(x, mixed_full160_time(), modes=8)


def test_synthesis_rejects_bad_shape_and_time():
    coefficients = torch.randn(1, 2, 8, 3, dtype=torch.complex64)
    time_s = mixed_full160_time()

    with pytest.raises(ValueError, match=r"\[B,Q,M,C\]"):
        nonuniform_fourier_synthesis(coefficients[0], time_s)
    with pytest.raises(ValueError, match="time"):
        nonuniform_fourier_synthesis(coefficients, time_s[None, :])


def test_synthesis_supports_complex128_coefficients():
    coefficients = torch.ones(1, 2, 8, 3, dtype=torch.complex128)

    output = nonuniform_fourier_synthesis(coefficients, mixed_full160_time())

    assert output.dtype == torch.float64


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_forward_preserves_real_input_dtype(dtype):
    layer = NonUniformTemporalOperator(2, 8)
    x = torch.ones(1, 2, 160, 2, dtype=dtype)

    output = layer(x, mixed_full160_time())

    assert output.dtype == dtype
    assert torch.equal(output, x)


def test_forward_rejects_complex_input():
    layer = NonUniformTemporalOperator(2, 8)

    with pytest.raises(ValueError, match="real floating"):
        layer(torch.ones(1, 2, 160, 2, dtype=torch.complex64), mixed_full160_time())


def test_forward_rejects_integer_time():
    layer = NonUniformTemporalOperator(2, 8)

    with pytest.raises(ValueError, match="floating"):
        layer(torch.ones(1, 2, 160, 2), torch.arange(160))


class ParentTemporalModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.temporal = NonUniformTemporalOperator(2, 8)

    def forward(self, x, time_s):
        return self.temporal(self.linear(x), time_s)


@pytest.mark.parametrize(
    ("convert", "real_dtype", "complex_dtype"),
    [
        pytest.param(
            lambda parent: parent.double(),
            torch.float64,
            torch.complex128,
            id="parent-double",
        ),
        pytest.param(
            lambda parent: parent.to(dtype=torch.float64),
            torch.float64,
            torch.complex128,
            id="parent-to-float64",
        ),
    ],
)
def test_parent_float64_conversion_is_complete_and_forward_works(
    convert, real_dtype, complex_dtype
):
    parent = convert(ParentTemporalModule())
    x = torch.randn(1, 2, 160, 2, dtype=real_dtype)

    output = parent(x, mixed_full160_time())

    assert parent.linear.weight.dtype == real_dtype
    assert parent.temporal.dc_weight.dtype == real_dtype
    assert parent.temporal.weight.dtype == complex_dtype
    assert output.dtype == real_dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("convert", "input_dtype"),
    [
        pytest.param(lambda parent: parent.half(), torch.float16, id="parent-half"),
        pytest.param(
            lambda parent: parent.bfloat16(), torch.bfloat16, id="parent-bfloat16"
        ),
    ],
)
def test_parent_low_precision_conversion_keeps_temporal_spectral_precision(
    convert, input_dtype
):
    parent = convert(ParentTemporalModule())
    x = torch.randn(1, 2, 160, 2, dtype=input_dtype)

    output = parent(x, mixed_full160_time())

    assert parent.linear.weight.dtype == input_dtype
    assert parent.temporal.dc_weight.dtype == torch.float32
    assert parent.temporal.weight.dtype == torch.complex64
    assert output.dtype == input_dtype
    assert torch.isfinite(output.float()).all()


def test_parent_module_allows_pure_device_migration():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        layer = NonUniformTemporalOperator(2, 8)
        parent = torch.nn.Sequential(layer).to(device)
        time_s = mixed_full160_time(device)

        output = parent[0](torch.ones(1, 2, 160, 2, device=device), time_s)

        assert output.device.type == device.type
        assert parent[0].weight.dtype == torch.complex64
        assert parent[0].dc_weight.dtype == torch.float32


def test_double_operator_pure_cpu_migration_preserves_double_precision():
    layer = NonUniformTemporalOperator(2, 8).double().to("cpu")

    assert layer.dc_weight.dtype == torch.float64
    assert layer.weight.dtype == torch.complex128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_double_operator_device_roundtrip_preserves_double_precision():
    layer = NonUniformTemporalOperator(2, 8).double()

    layer = layer.to("cuda").to("cpu")

    assert layer.dc_weight.dtype == torch.float64
    assert layer.weight.dtype == torch.complex128


def test_residual_initialization_must_be_identity():
    with pytest.raises(ValueError, match="identity"):
        NonUniformTemporalOperator(2, 8, residual_init="random")
