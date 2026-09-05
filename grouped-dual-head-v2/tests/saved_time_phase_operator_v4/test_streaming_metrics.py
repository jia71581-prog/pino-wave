import pytest
import torch

from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)


def _metadata():
    return {
        "families": ("uniform", "uniform", "layered", "layered", "layered", "layered"),
        "group_ids": ("u0", "u1", "l0", "l0", "l0", "l0"),
        "sample_ids": ("a", "b", "c", "d", "e", "f"),
    }


def test_streaming_metrics_match_direct_record_relative_l2():
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(6, 16, 9, 9, generator=generator)
    prediction = target + 0.1 * torch.randn(6, 16, 9, 9, generator=generator)
    time_indices = torch.arange(16).repeat(6, 1)
    metadata = _metadata()
    accumulator = ExactWavefieldMetricAccumulator(energy_floor_fraction=0.01)

    accumulator.update(
        prediction[:2],
        target[:2],
        families=metadata["families"][:2],
        group_ids=metadata["group_ids"][:2],
        sample_ids=metadata["sample_ids"][:2],
        time_indices=time_indices[:2],
    )
    accumulator.update(
        prediction[2:],
        target[2:],
        families=metadata["families"][2:],
        group_ids=metadata["group_ids"][2:],
        sample_ids=metadata["sample_ids"][2:],
        time_indices=time_indices[2:],
    )
    result = accumulator.finalize()
    expected = (
        (prediction - target).flatten(1).norm(dim=1)
        / target.flatten(1).norm(dim=1)
    ).mean()

    assert result["aggregate_relative_l2"] == pytest.approx(float(expected), rel=1e-6)
    assert result["record_count"] == 6
    assert result["frame_count"] == 96
    assert result["unique_time_index_count"] == 16
    assert set(result["family_relative_l2"]) == {"uniform", "layered"}
    assert set(result["medium_relative_l2"]) == {"u0", "u1", "l0"}


def test_streaming_relative_l2_is_invariant_to_per_record_linear_normalization():
    generator = torch.Generator().manual_seed(29)
    target = torch.randn(3, 5, 4, 4, generator=generator)
    prediction = target + 0.2 * torch.randn(3, 5, 4, 4, generator=generator)
    record_scales = torch.tensor([0.25, 3.0, 17.0])[:, None, None, None]
    metadata = {
        "families": ("uniform", "layered", "marmousi"),
        "group_ids": ("u0", "l0", "m0"),
        "sample_ids": ("a", "b", "c"),
        "time_indices": torch.arange(5).repeat(3, 1),
    }
    physical = ExactWavefieldMetricAccumulator()
    normalized = ExactWavefieldMetricAccumulator()

    physical.update(prediction, target, **metadata)
    normalized.update(prediction / record_scales, target / record_scales, **metadata)

    assert normalized.finalize()["aggregate_relative_l2"] == pytest.approx(
        physical.finalize()["aggregate_relative_l2"], rel=1.0e-6
    )


def test_streaming_relative_l2_zero_target_uses_finite_squared_energy_epsilon():
    accumulator = ExactWavefieldMetricAccumulator()
    prediction = torch.ones(1, 2, 2, 2)
    target = torch.zeros_like(prediction)
    accumulator.update(
        prediction,
        target,
        families=("uniform",),
        group_ids=("u0",),
        sample_ids=("s0",),
        time_indices=torch.tensor([[0, 1]]),
    )

    result = accumulator.finalize()
    expected = prediction.double().square().sum().sqrt().item() / 1.0e-8
    assert result["aggregate_relative_l2"] == pytest.approx(expected)
    assert result["near_zero_frame_count"] == 2


def test_family_time_bin_cross_metric_resolves_per_family_temporal_concentration():
    """The (family x time_bin) cross metric must (a) equal the exact subset
    relative-L2 and (b) reveal that a family's error is late-concentrated -- the
    diagnostic that separates a temporal coarse-field limit from a spatial one."""
    torch.manual_seed(11)
    records, times, h, w = 4, 16, 6, 6
    stored = 16  # so 4*t//16 spreads t over pre_onset/early/middle/late (4 each)
    target = torch.randn(records, times, h, w)
    prediction = target.clone()
    # uniform (records 0,1): inject large error ONLY in late frames (t>=12).
    prediction[0:2, 12:, :, :] += 3.0 * torch.randn(2, 4, h, w)
    # layered (records 2,3): inject small, time-FLAT error everywhere.
    prediction[2:4, :, :, :] += 0.05 * torch.randn(2, times, h, w)

    families = ("uniform", "uniform", "layered", "layered")
    acc = ExactWavefieldMetricAccumulator(stored_time_count=stored)
    acc.update(
        prediction, target,
        families=families, group_ids=("u", "u", "l", "l"),
        sample_ids=("a", "b", "c", "d"),
        time_indices=torch.arange(times).repeat(records, 1),
    )
    result = acc.finalize()
    ftb = result["family_time_bin_relative_l2"]

    assert set(ftb) == {"uniform", "layered"}
    # (a) exact-value check against a direct subset computation.
    bins = {"pre_onset": slice(0, 4), "early": slice(4, 8),
            "middle": slice(8, 12), "late": slice(12, 16)}
    fam_rows = {"uniform": slice(0, 2), "layered": slice(2, 4)}
    for fam, rsl in fam_rows.items():
        for name, tsl in bins.items():
            diff = (prediction[rsl, tsl] - target[rsl, tsl]).double()
            err = float(diff.square().sum())
            tgt = float(target[rsl, tsl].double().square().sum())
            expected = (err ** 0.5) / (tgt ** 0.5)
            assert ftb[fam][name] == pytest.approx(expected, rel=1e-6)
    # (b) diagnostic property: uniform residual is LATE-concentrated (temporal),
    # layered residual is time-FLAT.  This is exactly the read that decides whether
    # a family's floor is a temporal lever (A3/space-time) or width-saturated.
    assert ftb["uniform"]["late"] > 5.0 * ftb["uniform"]["early"]
    u = ftb["layered"]
    assert max(u.values()) < 2.0 * min(u.values())


def test_streaming_metrics_accumulate_time_chunks_for_the_same_records():
    generator = torch.Generator().manual_seed(23)
    target = torch.randn(2, 12, 7, 7, generator=generator)
    prediction = target + 0.05 * torch.randn(2, 12, 7, 7, generator=generator)
    full = ExactWavefieldMetricAccumulator()
    chunked = ExactWavefieldMetricAccumulator()
    metadata = {
        "families": ("marmousi", "marmousi"),
        "group_ids": ("m0", "m0"),
        "sample_ids": ("s0", "s1"),
    }

    full.update(
        prediction,
        target,
        time_indices=torch.arange(12).repeat(2, 1),
        **metadata,
    )
    for start in (0, 4, 8):
        chunked.update(
            prediction[:, start : start + 4],
            target[:, start : start + 4],
            time_indices=torch.arange(start, start + 4).repeat(2, 1),
            **metadata,
        )

    assert chunked.finalize()["aggregate_relative_l2"] == pytest.approx(
        full.finalize()["aggregate_relative_l2"], rel=1e-6
    )


def test_streaming_metrics_reject_duplicate_sample_time_pairs():
    accumulator = ExactWavefieldMetricAccumulator(require_unique=True)
    prediction = torch.zeros(1, 2, 3, 3)
    target = torch.ones_like(prediction)
    arguments = {
        "families": ("uniform",),
        "group_ids": ("u0",),
        "sample_ids": ("s0",),
        "time_indices": torch.tensor([[1, 2]]),
    }

    accumulator.update(prediction, target, **arguments)
    with pytest.raises(ValueError, match="duplicate"):
        accumulator.update(prediction, target, **arguments)


def test_streaming_metrics_reject_complex_fields_instead_of_dropping_imaginary_part():
    accumulator = ExactWavefieldMetricAccumulator()
    prediction = torch.ones(1, 2, 3, 3, dtype=torch.complex64)
    target = torch.ones_like(prediction)

    with pytest.raises(ValueError, match="real scalar"):
        accumulator.update(
            prediction,
            target,
            families=("uniform",),
            group_ids=("u0",),
            sample_ids=("s0",),
            time_indices=torch.tensor([[1, 2]]),
        )


# --- displacement / transport diagnostics (Codex co-primary gate for r3/r4) ---


def _gaussian_bump(height, width, cz, cx, sigma=2.0):
    z = torch.arange(height).float()[:, None]
    x = torch.arange(width).float()[None, :]
    return torch.exp(-((z - cz) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))


def test_displacement_zero_when_prediction_equals_target():
    target = torch.stack([_gaussian_bump(24, 24, 12, 12) for _ in range(4)])[None]  # [1,4,24,24]
    time_indices = torch.arange(4)[None]
    acc = ExactWavefieldMetricAccumulator()
    acc.update(
        target.clone(), target.clone(),
        families=("uniform",), group_ids=("u0",), sample_ids=("a",),
        time_indices=time_indices,
    )
    result = acc.finalize()
    # perfect prediction -> no centroid shift, no xcorr peak offset
    assert result["centroid_shift_cells"] == pytest.approx(0.0, abs=1e-6)
    assert result["xcorr_peak_shift_cells"] == pytest.approx(0.0, abs=1e-9)


def test_centroid_shift_detects_known_translation():
    # reference bump at (12,12); prediction shifted +4 cells in x -> centroid shift ~4
    ref = _gaussian_bump(32, 32, 12, 12)[None, None]      # [1,1,32,32]
    pred = _gaussian_bump(32, 32, 12, 16)[None, None]
    acc = ExactWavefieldMetricAccumulator()
    acc.update(
        pred, ref,
        families=("uniform",), group_ids=("u0",), sample_ids=("a",),
        time_indices=torch.tensor([[0]]),
    )
    result = acc.finalize()
    assert result["centroid_shift_cells"] == pytest.approx(4.0, abs=0.2)
    # circular xcorr recovers the integer translation magnitude (4 cells)
    assert result["xcorr_peak_shift_cells"] == pytest.approx(4.0, abs=1e-6)


def test_displacement_keys_present_and_backward_compatible():
    target = torch.randn(2, 5, 8, 8)
    pred = target + 0.05 * torch.randn(2, 5, 8, 8)
    acc = ExactWavefieldMetricAccumulator()
    acc.update(
        pred, target,
        families=("uniform", "layered"), group_ids=("u0", "l0"),
        sample_ids=("a", "b"), time_indices=torch.arange(5).repeat(2, 1),
    )
    result = acc.finalize()
    for key in ("centroid_shift_cells", "xcorr_peak_shift_cells"):
        assert key in result and result[key] >= 0.0
    # existing keys still present (additive change)
    assert "aggregate_relative_l2" in result and "phase_correlation" in result
