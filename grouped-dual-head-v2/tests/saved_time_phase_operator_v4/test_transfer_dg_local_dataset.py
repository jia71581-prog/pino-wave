from __future__ import annotations

import numpy as np

from scripts.build_transfer_dg_local_frequency_dataset import (
    final_cosine_taper,
    non_cpml_element_origins,
    select_frequency_bins,
)


def test_terminal_taper_preserves_start_and_reaches_zero():
    taper = final_cosine_taper(401, fraction=0.10)
    assert np.all(taper[:361] == 1.0)
    assert taper[-1] == 0.0
    assert np.all(np.diff(taper[-40:]) <= 0.0)


def test_source_relative_frequency_bins_are_unique_and_bounded():
    frequencies = np.fft.rfftfreq(401, d=0.0025)
    for f0 in (8.0, 15.0, 22.0, 30.0):
        bins = select_frequency_bins(frequencies, f0)
        assert len(np.unique(bins)) == 4
        assert np.all(frequencies[bins] >= 4.0)
        assert np.all(frequencies[bins] <= 40.0)


def test_cpml_aligned_partition_keeps_72_physical_elements():
    origins = non_cpml_element_origins(
        201, 201, element_intervals=20, cpml_margin=20
    )
    assert origins.shape == (72, 2)
    assert origins[:, 1].min() == 20
    assert origins[:, 1].max() == 160
    assert origins[:, 0].min() == 0
    assert origins[:, 0].max() == 160
