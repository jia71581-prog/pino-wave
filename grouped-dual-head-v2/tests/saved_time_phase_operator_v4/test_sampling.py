import numpy as np
import pytest

from saved_time_phase_operator_v4.sampling import (
    appearance_coverage_ledger,
    appearance_time_indices,
    coverage_summary,
    stored_time_indices,
)


def test_sampler_covers_pre_onset_early_middle_and_late():
    axis = np.linspace(0.0, 1.0, 401)

    indices = stored_time_indices(axis, source_t0_s=0.1, phase_offset=7)

    assert len(indices) == 4
    assert indices[0] < 40
    assert 40 <= indices[1] < 160
    assert 160 <= indices[2] < 280
    assert 280 <= indices[3] <= 400
    assert len(set(indices.tolist())) == 4


def test_sampler_is_deterministic_and_changes_with_offset():
    axis = np.linspace(0.0, 1.0, 401)

    first = stored_time_indices(axis, source_t0_s=0.08, phase_offset=13)
    repeated = stored_time_indices(axis, source_t0_s=0.08, phase_offset=13)
    next_offset = stored_time_indices(axis, source_t0_s=0.08, phase_offset=14)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, next_offset)


def test_planned_coverage_has_no_midpoints_and_reaches_late_axis():
    report = coverage_summary(record_count=48, appearances=140, seed=17)

    assert report["minimum_unique_indices"] >= 120
    assert report["minimum_index"] == 0
    assert report["maximum_index"] == 400
    assert report["interpolated_requests"] == 0
    assert report["requested_time_count"] == 401


def test_sampler_rejects_invalid_axes_and_onsets():
    with pytest.raises(ValueError, match="strictly increasing"):
        stored_time_indices([0.0, 0.1, 0.2, 0.2, 0.3], source_t0_s=0.05, phase_offset=0)
    with pytest.raises(ValueError, match="inside the stored time axis"):
        stored_time_indices(np.linspace(0.0, 1.0, 401), source_t0_s=0.0, phase_offset=0)


def test_appearance_sampler_reaches_120_unique_times_at_epoch_30():
    axis = np.linspace(0.0, 1.0, 401)
    seen: set[int] = set()
    pre_onset = 0
    for appearance in range(30):
        indices = appearance_time_indices(
            axis,
            source_t0_s=0.05,
            sample_id="sample-1",
            appearance=appearance,
            seed=307,
        )
        seen.update(indices.tolist())
        pre_onset += int((indices < 20).sum())

    assert len(seen) >= 120
    assert pre_onset / (30 * 4) <= 0.05


def test_appearance_sampler_exceeds_180_unique_times_at_epoch_50():
    axis = np.linspace(0.0, 1.0, 401)
    seen = {
        int(index)
        for appearance in range(50)
        for index in appearance_time_indices(
            axis,
            source_t0_s=0.1874,
            sample_id="late-onset",
            appearance=appearance,
            seed=307,
        )
    }

    assert len(seen) > 180


def test_appearance_sampler_is_deterministic_and_source_specific():
    axis = np.linspace(0.0, 1.0, 401)
    first = appearance_time_indices(
        axis, source_t0_s=0.1, sample_id="a", appearance=4, seed=9
    )
    repeated = appearance_time_indices(
        axis, source_t0_s=0.1, sample_id="a", appearance=4, seed=9
    )
    other = appearance_time_indices(
        axis, source_t0_s=0.1, sample_id="b", appearance=4, seed=9
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert len(set(first.tolist())) == 4


@pytest.mark.parametrize("count", (24, 32, 64))
def test_recovery_appearance_sampler_supports_larger_exact_frame_panels(count):
    axis = np.linspace(0.0, 1.0, 401)
    onset = 40

    first = appearance_time_indices(
        axis,
        source_t0_s=float(axis[onset]),
        sample_id="large-panel",
        appearance=3,
        seed=307,
        count=count,
    )
    repeated = appearance_time_indices(
        axis,
        source_t0_s=float(axis[onset]),
        sample_id="large-panel",
        appearance=3,
        seed=307,
        count=count,
    )
    next_panel = appearance_time_indices(
        axis,
        source_t0_s=float(axis[onset]),
        sample_id="large-panel",
        appearance=4,
        seed=307,
        count=count,
    )

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (count,)
    assert len(np.unique(first)) == count
    assert (first[1:] > first[:-1]).all()
    assert first.min() >= 0 and first.max() < len(axis)
    assert onset in first and onset + 1 in first
    assert not np.array_equal(first, next_panel)


def test_large_recovery_panels_stagger_additional_frames_across_appearances():
    axis = np.linspace(0.0, 1.0, 401)
    panels = [
        set(
            appearance_time_indices(
                axis,
                source_t0_s=float(axis[40]),
                sample_id="staggered-large-panel",
                appearance=appearance,
                seed=307,
                count=64,
            ).tolist()
        )
        for appearance in range(8)
    ]

    assert len(set().union(*panels)) >= 300
    adjacent_overlap = [
        len(left & right) / 64.0
        for left, right in zip(panels, panels[1:])
    ]
    assert max(adjacent_overlap) <= 0.5


def test_appearance_coverage_ledger_is_bit_packed_and_auditable():
    axis = np.linspace(0.0, 1.0, 401)
    packed, report = appearance_coverage_ledger(
        axis,
        source_t0_s=(0.05, 0.1874),
        sample_ids=("early", "late"),
        appearance_counts=(30, 30),
        seed=307,
    )

    assert packed.shape == (2, 51)
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, :401]
    assert unpacked.sum(axis=1).min() >= 120
    assert report["minimum_unique_indices"] >= 120
    assert report["interpolated_requests"] == 0


def test_padding_appearance_51_does_not_exceed_pre_onset_budget():
    axis = np.linspace(0.0, 1.0, 401)
    onset = 20
    pre_onset = sum(
        int(
            (
                appearance_time_indices(
                    axis,
                    source_t0_s=0.05,
                    sample_id="padded-record",
                    appearance=appearance,
                    seed=307,
                )
                < onset
            ).sum()
        )
        for appearance in range(51)
    )

    assert pre_onset / (51 * 4) <= 0.05


from saved_time_phase_operator_v4.sampling import rad_bin_budget


def test_rad_bin_budget_shifts_toward_high_error_bin():
    # late error dominates -> late gets the largest share; total conserved
    budget = rad_bin_budget([0.14, 0.26, 0.43], total_active=13, k=1.0, c=1.0, floor_per_bin=1)
    assert sum(budget) == 13
    assert budget[2] >= budget[1] >= budget[0]
    assert all(b >= 1 for b in budget)


def test_rad_bin_budget_uniform_when_errors_equal():
    assert rad_bin_budget([0.3, 0.3, 0.3], total_active=12, floor_per_bin=1) == (4, 4, 4)


def test_rad_bin_budget_is_deterministic():
    a = rad_bin_budget([0.1, 0.5, 0.9], 13, k=2.0, c=0.5)
    b = rad_bin_budget([0.1, 0.5, 0.9], 13, k=2.0, c=0.5)
    assert a == b


def test_appearance_budget_preserves_count_and_onset_and_defaults_bitwise():
    axis = np.linspace(0.0, 1.0, 401)
    kw = dict(source_t0_s=0.1, source_f0_hz=12.0, sample_id="uniform_00069", seed=372, count=16)
    for appearance in range(6):
        default = appearance_time_indices(axis, appearance=appearance, **kw)
        budgeted = appearance_time_indices(
            axis, appearance=appearance, bin_frame_budget=(4, 4, 5), **kw
        )
        # exactly 16 unique frames in both, onset pair always present
        assert len(default) == 16 and len(set(default.tolist())) == 16
        assert len(budgeted) == 16 and len(set(budgeted.tolist())) == 16
        onset = int(np.searchsorted(axis, 0.1 - 1.0 / 12.0))
        assert onset in budgeted.tolist() and onset + 1 in budgeted.tolist()


def test_appearance_budget_none_is_identical_to_legacy():
    axis = np.linspace(0.0, 1.0, 401)
    kw = dict(source_t0_s=0.1, source_f0_hz=12.0, sample_id="marmousi_00109", seed=7, count=16)
    for appearance in range(8):
        legacy = appearance_time_indices(axis, appearance=appearance, **kw)
        explicit_none = appearance_time_indices(
            axis, appearance=appearance, bin_frame_budget=None, **kw
        )
        assert np.array_equal(legacy, explicit_none)
