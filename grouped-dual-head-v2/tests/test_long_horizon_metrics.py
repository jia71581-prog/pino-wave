from __future__ import annotations

import math

import pytest
import torch

from fno_acoustic.long_horizon_metrics import (
    ReceiverSiteManifest,
    estimate_komega_workspace_bytes,
    gather_native_receivers,
    komega_metrics,
    long_horizon_metrics,
    quarter_slices,
    receiver_arrival_metrics,
    receiver_energy_metrics,
    receiver_lag_phase_metrics,
    spatial_high_k_mask,
)
from fno_acoustic.query_losses import normalized_spatial_frequency_radius


def _time() -> torch.Tensor:
    increments = torch.linspace(0.002, 0.004, 159, dtype=torch.float64)
    return torch.cat((torch.zeros(1, dtype=torch.float64), increments.cumsum(0)))


def _manifest(height: int = 9, width: int = 7) -> ReceiverSiteManifest:
    indices = torch.tensor([0, width + 2, height * width - 1])
    xz = torch.tensor([[0.0, 0.0], [1.0, 2.0], [8.0, 6.0]], dtype=torch.float64)
    return ReceiverSiteManifest((height, width), indices, xz, "a" * 64)


def test_quarter_windows_are_exact_for_160_samples() -> None:
    assert quarter_slices(160) == {
        "q1": slice(0, 40), "q2": slice(40, 80),
        "q3": slice(80, 120), "q4": slice(120, 160),
    }
    with pytest.raises(ValueError, match="160"):
        quarter_slices(159)


def test_target_referenced_arrival_reports_prediction_miss() -> None:
    target = torch.zeros(1, 2, 160)
    target[..., 30] = 1.0
    result = receiver_arrival_metrics(torch.zeros_like(target), target, _time())
    assert result["arrival_miss_rate"] == 1.0
    assert result["arrival_target_coverage"] == 1.0


def test_nonuniform_energy_uses_physical_trapezoid_weights() -> None:
    target = torch.ones(1, 2, 160)
    result = receiver_energy_metrics(target.clone(), target, _time())
    assert result["energy_log_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_lag_tie_is_deterministic_and_phase_is_exact_for_identity() -> None:
    target = torch.zeros(1, 1, 160)
    target[..., 40:80] = 1.0
    result = receiver_lag_phase_metrics(target, target, _time())
    assert result["receiver_lag_abs_s"] == 0.0
    assert result["receiver_xcorr_peak"] == pytest.approx(1.0)
    assert result["receiver_phase_error"] == pytest.approx(0.0, abs=1e-6)
    assert result["receiver_phase_coherence"] == pytest.approx(1.0, abs=1e-6)


def test_random_independent_receiver_traces_do_not_get_extreme_overlap_peak() -> None:
    generator = torch.Generator().manual_seed(41)
    target = torch.randn(1, 3, 160, generator=generator)
    prediction = torch.randn(1, 3, 160, generator=generator)
    result = receiver_lag_phase_metrics(
        prediction, target, _time(), max_lag_fraction=0.25
    )
    assert result["receiver_xcorr_peak"] < 0.35
    duration = float(_time()[-1] - _time()[0])
    assert result["receiver_lag_abs_s"] <= 0.25 * duration + 1e-12


def test_strongly_nonuniform_receiver_lag_recovers_physical_delay() -> None:
    parameter = torch.linspace(0, 1, 160, dtype=torch.float64)
    time = 0.5 * parameter.square() + 0.1 * parameter
    delay = 0.06
    target = torch.exp(-((time - 0.30) / 0.025).square()).float()[None, None]
    prediction = torch.exp(-((time - delay - 0.30) / 0.025).square()).float()[None, None]
    result = receiver_lag_phase_metrics(
        prediction, target, time, max_lag_fraction=0.25
    )
    uniform_dt = float((time[-1] - time[0]) / 159)
    assert result["receiver_lag_abs_s"] == pytest.approx(
        delay, abs=1.5 * uniform_dt
    )


def test_komega_accepts_q4_window_and_detects_high_band_error() -> None:
    time = _time()[-40:]
    target = torch.zeros(1, 9, 7, 40)
    target[:, ::2, ::2] = torch.sin(torch.linspace(0, 4 * torch.pi, 40))
    prediction = target + 0.1 * torch.randn_like(target)
    result = komega_metrics(prediction, target, time, temporal_modes=32)
    assert set(result) == {"komega_relative_l2", "komega_high"}
    assert all(math.isfinite(value) and value > 0 for value in result.values())


def _direct_komega(
    prediction: torch.Tensor, target: torch.Tensor, time: torch.Tensor
) -> dict[str, float]:
    from fno_acoustic.temporal_operator import nonuniform_fourier_analysis

    batch, height, width, length = prediction.shape
    modes = min(7, length)

    def transform(field: torch.Tensor) -> torch.Tensor:
        coefficients = nonuniform_fourier_analysis(
            field.reshape(batch, height * width, length, 1), time, modes
        )
        return torch.fft.fft2(
            coefficients.reshape(batch, height, width, modes, 1),
            dim=(1, 2), norm="ortho",
        )

    prediction_k, target_k = transform(prediction), transform(target)
    high = spatial_high_k_mask(height, width, prediction.device)

    def relative(prediction_value: torch.Tensor, target_value: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(prediction_value - target_value)
            / torch.linalg.vector_norm(target_value).clamp_min(1e-12)
        )

    return {
        "komega_relative_l2": relative(prediction_k, target_k),
        "komega_high": relative(prediction_k[:, high], target_k[:, high]),
    }


def test_blocked_komega_matches_direct_tiny_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(1, 7, 9, 40, generator=generator)
    prediction = target + 0.03 * torch.randn(1, 7, 9, 40, generator=generator)
    expected = _direct_komega(prediction, target, _time()[:40])
    actual = komega_metrics(
        prediction, target, _time()[:40], temporal_modes=7, mode_block_size=3
    )
    assert actual == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_native400_blocked_workspace_is_audited_below_three_gib() -> None:
    audit = estimate_komega_workspace_bytes(
        1, 400, 400, 160, 32, 4, torch.float32
    )
    assert audit["legacy_dense_bytes"] > 12 * 1024**3
    assert audit["blocked_peak_bytes"] < 3 * 1024**3


@pytest.mark.parametrize("height,width", [(8, 10), (7, 9)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_komega_full_high_mask_matches_task4_positive_half(
    height: int, width: int, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    full = spatial_high_k_mask(height, width, torch.device(device))
    task4 = normalized_spatial_frequency_radius(height, width, device) >= 0.5
    assert torch.equal(full[:, : width // 2 + 1], task4)


def test_all_zero_target_produces_only_finite_metrics() -> None:
    target = torch.zeros(1, 9, 7, 160)
    result = long_horizon_metrics(target.clone(), target, _time(), _manifest())
    assert all(math.isfinite(float(value)) for value in result.values())
    assert result["arrival_mae_s"] == result["arrival_miss_rate"] == 0.0


def test_manifest_and_gather_are_strict() -> None:
    field = torch.arange(9 * 7 * 160, dtype=torch.float32).reshape(1, 9, 7, 160)
    gathered = gather_native_receivers(field, _manifest())
    assert gathered.shape == (1, 3, 160)
    with pytest.raises(ValueError, match="unique"):
        ReceiverSiteManifest((9, 7), torch.tensor([1, 1]), torch.zeros(2, 2), "b" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        ReceiverSiteManifest((9, 7), torch.tensor([1]), torch.zeros(1, 2), "bad")
    with pytest.raises(ValueError, match="manifest grid"):
        gather_native_receivers(torch.zeros(1, 7, 9, 160), _manifest())


@pytest.mark.parametrize("bad", [torch.zeros(9, 7, 160), torch.zeros(1, 9, 7, 159)])
def test_metric_shape_and_time_contract_is_strict(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        long_horizon_metrics(bad, bad, _time(), _manifest())


def test_direct_receiver_and_komega_metrics_reject_nonfinite_or_integer_inputs() -> None:
    integer = torch.zeros(1, 1, 160, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        receiver_arrival_metrics(integer, integer, _time())
    bad = torch.zeros(1, 9, 7, 40)
    bad[..., 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        komega_metrics(bad, bad, _time()[:40])
