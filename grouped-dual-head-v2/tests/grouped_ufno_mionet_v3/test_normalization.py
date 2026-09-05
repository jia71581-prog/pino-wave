from __future__ import annotations

import pytest
import torch

from grouped_ufno_mionet_v3.normalization import (
    PhysicalNormalizer,
    ScaleMetadata,
    fit_scale_metadata,
)


def test_physical_pressure_round_trip_applies_amplitude_once():
    metadata = ScaleMetadata(
        velocity_center_mps=2200.0,
        velocity_scale_mps=400.0,
        pressure_scale_pa=2.0e-8,
        source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
        train_manifest_sha256="train-only",
        allowed_medium_types=("uniform", "layered", "marmousi"),
        record_count=2,
        algorithm="test",
    )
    normalizer = PhysicalNormalizer(metadata)
    pressure = torch.tensor([2.0e-8, -6.0e-8])
    amplitude = torch.tensor(2.0)
    encoded = normalizer.encode_pressure(pressure, amplitude)
    torch.testing.assert_close(encoded, torch.tensor([0.5, -1.5]))
    torch.testing.assert_close(normalizer.decode_pressure(encoded, amplitude), pressure)


def test_fit_scale_metadata_is_bound_to_explicit_train_values_only():
    train_velocity = torch.tensor([1800.0, 2000.0, 2200.0, 2400.0])
    train_pressure = torch.tensor([0.0, 1.0e-8, -2.0e-8, 3.0e-8])
    fitted = fit_scale_metadata(
        train_velocity,
        train_pressure,
        torch.tensor([[100.0, 200.0, 10.0, 0.1, 1.0]]),
        train_manifest_sha256="train-digest",
        record_count=4,
        pressure_percentile=100.0,
    )
    assert fitted.train_manifest_sha256 == "train-digest"
    assert fitted.record_count == 4
    assert fitted.velocity_center_mps == pytest.approx(2100.0)
    assert fitted.pressure_scale_pa == pytest.approx(3.0e-8)
    assert fitted.velocity_center_mps != 9999.0


def test_normalizer_rejects_wrong_manifest_and_family_contract():
    values = {
        "velocity_center_mps": 2000.0,
        "velocity_scale_mps": 100.0,
        "pressure_scale_pa": 1.0e-8,
        "source_scales": [2000.0, 2000.0, 50.0, 1.2, 1.0],
        "train_manifest_sha256": "actual",
        "allowed_medium_types": ["uniform", "layered", "marmousi"],
        "record_count": 1,
        "algorithm": "test",
    }
    with pytest.raises(ValueError, match="manifest"):
        PhysicalNormalizer.from_dict(values, expected_manifest="different")
    values["allowed_medium_types"] = ["uniform", "anomaly"]
    with pytest.raises(ValueError, match="family"):
        PhysicalNormalizer.from_dict(values)
