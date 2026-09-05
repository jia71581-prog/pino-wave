import pytest
import torch

from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer, ScaleMetadata


def metadata():
    return ScaleMetadata(
        velocity_center_mps=2500.0,
        velocity_scale_mps=1000.0,
        pressure_scale_pa=1.0e-8,
        source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
        train_manifest_sha256="abc",
    )


def test_pressure_round_trip_is_order_one_and_fp32():
    normalizer = PhysicalNormalizer(metadata())
    pressure = torch.tensor([1.0e-9, -2.0e-8])
    encoded = normalizer.encode_pressure(pressure, torch.tensor(2.0))
    assert encoded.dtype == torch.float32
    assert encoded.abs().max() >= 0.5
    torch.testing.assert_close(normalizer.decode_pressure(encoded, torch.tensor(2.0)), pressure)


def test_velocity_and_source_round_trip():
    normalizer = PhysicalNormalizer(metadata())
    velocity = torch.tensor([1500.0, 2500.0, 3500.0])
    source = torch.tensor([[1000.0, 500.0, 10.0, 0.12, 2.0]])
    torch.testing.assert_close(normalizer.decode_velocity(normalizer.encode_velocity(velocity)), velocity)
    torch.testing.assert_close(normalizer.decode_source(normalizer.encode_source(source)), source)


def test_normalizer_rejects_wrong_manifest():
    with pytest.raises(ValueError, match="manifest"):
        PhysicalNormalizer.from_dict(metadata().to_dict(), expected_manifest="wrong")


def test_metadata_rejects_nonpositive_scales():
    values = metadata().to_dict()
    values["pressure_scale_pa"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        PhysicalNormalizer.from_dict(values, expected_manifest="abc")


def test_fitted_statistics_metadata_can_include_audit_fields():
    values = metadata().to_dict() | {"split": "train", "record_count": 12, "algorithm": "stream-v1"}
    assert PhysicalNormalizer.from_dict(values).metadata.train_manifest_sha256 == "abc"
