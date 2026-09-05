from __future__ import annotations

import numpy as np

from fno_acoustic.schema import canonicalize_velocity, canonicalize_wavefield, inspect_hdf5


def test_schema_inspector_lists_keys(tiny_hdf5):
    schema = inspect_hdf5(tiny_hdf5)
    assert "/nu" in schema["datasets"]
    assert "/tensor" in schema["datasets"]
    assert schema["datasets"]["/tensor"]["shape"] == [4, 10, 16, 12]


def test_axis_canonicalization_common_layouts():
    arr_thw = np.zeros((5, 4, 3))
    assert canonicalize_wavefield(arr_thw, ["time", "x", "z"]).shape == (4, 3, 5)
    arr_hwt = np.zeros((4, 3, 5))
    assert canonicalize_wavefield(arr_hwt, ["x", "z", "time"]).shape == (4, 3, 5)
    vel_zh = np.zeros((3, 4))
    assert canonicalize_velocity(vel_zh, ["z", "x"]).shape == (4, 3)
