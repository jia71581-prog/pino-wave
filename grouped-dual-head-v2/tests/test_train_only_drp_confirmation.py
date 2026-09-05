from __future__ import annotations

from scripts.confirm_train_only_drp_lwc84 import _hashes, _indices, _paired_summary


def _rows(candidate: list[float], standard: list[float], frequency: list[float]):
    rows = []
    for index, (cand, base, f0) in enumerate(zip(candidate, standard, frequency, strict=True)):
        common = {"source_index": index, "f0_hz": f0}
        rows.append({**common, "variant": "standard_taylor8", "relative_l2": base})
        rows.append({**common, "variant": "drp_candidate", "relative_l2": cand})
    return rows


def test_indices_require_nonempty_unique_values() -> None:
    assert _indices("3,5,9") == (3, 5, 9)


def test_hashes_require_sha256_values() -> None:
    assert _hashes("a" * 64 + "," + "0" * 64) == ("a" * 64, "0" * 64)


def test_paired_summary_applies_all_promotion_gates() -> None:
    summary = _paired_summary(
        _rows(
            [0.08, 0.085, 0.09, 0.095],
            [0.10, 0.10, 0.10, 0.10],
            [10.0, 20.0, 28.0, 30.0],
        ),
        high_frequency_cutoff_hz=28.0,
        minimum_relative_improvement=0.02,
        minimum_win_fraction=0.625,
        maximum_absolute_regression=0.01,
    )
    assert summary["paired_win_count"] == 4
    assert summary["promotion_passed"] is True
    assert all(summary["gates"].values())


def test_paired_summary_rejects_outlier_driven_mean_gain() -> None:
    summary = _paired_summary(
        _rows(
            [0.01, 0.102, 0.102, 0.102],
            [0.10, 0.10, 0.10, 0.10],
            [10.0, 20.0, 28.0, 30.0],
        ),
        high_frequency_cutoff_hz=28.0,
        minimum_relative_improvement=0.02,
        minimum_win_fraction=0.625,
        maximum_absolute_regression=0.01,
    )
    assert summary["relative_improvement"] > 0.02
    assert summary["gates"]["paired_win_fraction"] is False
    assert summary["gates"]["high_frequency_relative_improvement"] is False
    assert summary["promotion_passed"] is False
