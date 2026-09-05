from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import fno_acoustic.ais_dataset_binding as binding_module

from fno_acoustic.ais_dataset_binding import (
    canonical_array_bytes,
    canonical_json_bytes,
    canonical_root_attributes_bytes,
    compute_dataset_manifest,
    dataset_content_root,
    load_dataset_content_binding,
    merkle_root,
    validate_process_verification,
    verified_ids_sha256,
)


def _tiny_dataset(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "tensor", data=np.arange(2 * 160 * 2 * 3, dtype=np.float32).reshape(2, 160, 2, 3)
        )
        h5.create_dataset("nu", data=np.ones((2, 2, 3), dtype=np.float32))
        h5.create_dataset("source_mask", data=np.eye(2, 3, dtype=np.float32)[None].repeat(2, 0))
        h5.create_dataset("model_type", data=np.asarray([b"uniform", b"layered"]))
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.2, 160))
        h5.create_dataset("x-coordinate", data=np.asarray([0.1, 0.2]))
        h5.create_dataset("y-coordinate", data=np.asarray([0.3, 0.4, 0.5]))
        h5.create_dataset("wavelet", data=np.arange(7, dtype=np.float32))
        h5.attrs["sample_count"] = 2
        h5.attrs["producer"] = "tiny"
        h5.attrs["flag"] = True
        h5.attrs["scale"] = 1.25
        h5.attrs["count"] = np.int64(2)
        h5.attrs["raw"] = np.bytes_(b"abc")
        h5.attrs["save_indices"] = np.asarray([0, 3, 7], dtype=np.int64)
    return path


def _logical(manifest: dict[str, object]) -> tuple[object, ...]:
    return (
        manifest["global_sha256"],
        tuple(manifest["sample_digests"]),
        manifest["sample_merkle_root"],
        manifest["dataset_content_root"],
    )


def test_array_framing_is_big_endian_metadata_and_little_endian_values() -> None:
    framed = canonical_array_bytes(
        "nu", np.asarray([1.0, 2.0], dtype=">f4"), "<f4"
    )

    assert framed.startswith(
        b"\x00\x00\x00\x00\x00\x00\x00\x08array-v1"
        b"\x00\x00\x00\x00\x00\x00\x00\x02nu"
        b"\x00\x00\x00\x00\x00\x00\x00\x03<f4"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x00\x00\x00\x00\x02"
        b"\x00\x00\x00\x00\x00\x00\x00\x08"
    )
    assert framed[-8:] == np.asarray([1.0, 2.0], dtype="<f4").tobytes()


@pytest.mark.parametrize("field", ["tensor", "nu", "source_mask", "model_type"])
def test_consumed_sample_field_mutation_changes_leaf_and_content_root(
    tmp_path: Path, field: str,
) -> None:
    path = _tiny_dataset(tmp_path / "tiny.h5")
    before = compute_dataset_manifest(path)
    with h5py.File(path, "r+") as h5:
        if field == "model_type":
            h5[field][0] = b"marmousi"
        else:
            values = h5[field][0]
            values.flat[0] += 1.0
            h5[field][0] = values
    after = compute_dataset_manifest(path)

    assert before["sample_digests"][0] != after["sample_digests"][0]
    assert before["dataset_content_root"] != after["dataset_content_root"]


@pytest.mark.parametrize("field", ["t-coordinate", "x-coordinate", "y-coordinate"])
def test_shared_coordinate_mutation_changes_global_and_content_root(
    tmp_path: Path, field: str,
) -> None:
    path = _tiny_dataset(tmp_path / "tiny.h5")
    before = compute_dataset_manifest(path)
    with h5py.File(path, "r+") as h5:
        values = h5[field][...]
        values[-1] += 0.01
        h5[field][...] = values
    after = compute_dataset_manifest(path)

    assert before["global_sha256"] != after["global_sha256"]
    assert before["dataset_content_root"] != after["dataset_content_root"]


def test_root_attribute_changes_global_root_and_wrong_required_schema_is_fatal(
    tmp_path: Path,
) -> None:
    original = _tiny_dataset(tmp_path / "tiny.h5")
    before = compute_dataset_manifest(original)
    with h5py.File(original, "r+") as h5:
        h5.attrs["producer"] = "changed"
    attr_changed = compute_dataset_manifest(original)

    recreated = _tiny_dataset(tmp_path / "schema.h5")
    with h5py.File(recreated, "r+") as h5:
        values = h5["nu"][...]
        del h5["nu"]
        h5.create_dataset("nu", data=values.astype(np.float64))
    assert attr_changed["global_sha256"] != before["global_sha256"]
    with pytest.raises(ValueError, match="nu.*dtype"):
        compute_dataset_manifest(recreated)


def test_unconsumed_wavelet_does_not_change_logical_binding(tmp_path: Path) -> None:
    path = _tiny_dataset(tmp_path / "tiny.h5")
    before = compute_dataset_manifest(path)
    with h5py.File(path, "r+") as h5:
        h5["wavelet"][...] = -99.0
    after = compute_dataset_manifest(path)

    assert _logical(after) == _logical(before)


def test_merkle_and_content_roots_are_domain_separated_and_tamper_sensitive() -> None:
    leaves = ["00" * 32, "11" * 32, "22" * 32]
    root = merkle_root(leaves)

    assert root == merkle_root(copy.deepcopy(leaves))
    assert root != merkle_root([*leaves[:2], "23" * 32])
    assert dataset_content_root("33" * 32, root, 3) != dataset_content_root(
        "33" * 32, root, 4
    )


def test_root_attribute_type_domains_are_unambiguous_and_int64_array_is_supported() -> None:
    values = {
        "none": None,
        "false": False,
        "true": True,
        "int": 1,
        "float": 1.0,
        "text": "1",
        "bytes": b"1",
        "array": np.asarray([1], dtype=np.int64),
    }

    encoded = [canonical_root_attributes_bytes({name: value}) for name, value in values.items()]

    assert len(set(encoded)) == len(encoded)
    assert b"ndarray-v1" in encoded[-1]
    assert b"<i8" in encoded[-1]


def test_zero_dimensional_ndarray_root_attributes_use_known_scalar_domains() -> None:
    for scalar in (np.int64(7), np.float64(1.25), np.bool_(True), np.str_("x")):
        assert canonical_root_attributes_bytes(
            {"value": np.asarray(scalar)}
        ) == canonical_root_attributes_bytes({"value": scalar})


def test_manifest_loader_requires_canonical_v2_and_recomputes_internal_roots(
    tmp_path: Path,
) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    manifest = compute_dataset_manifest(dataset)
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))

    binding = load_dataset_content_binding(path, dataset)

    assert binding.dataset_content_root == manifest["dataset_content_root"]
    assert binding.sample_digests == tuple(manifest["sample_digests"])

    path.write_bytes(json.dumps(manifest, indent=2).encode())
    with pytest.raises(ValueError, match="canonical JSON"):
        load_dataset_content_binding(path, dataset)

    tampered = copy.deepcopy(manifest)
    tampered["sample_merkle_root"] = "0" * 64
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="Merkle"):
        load_dataset_content_binding(path, dataset)


def test_manifest_is_size_checked_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"x" * 32)
    monkeypatch.setattr(binding_module, "MAX_MANIFEST_BYTES", 16)

    with pytest.raises(ValueError, match="too large"):
        load_dataset_content_binding(manifest, dataset)


def test_schema_sample_ceiling_model_text_and_root_attribute_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    monkeypatch.setattr(binding_module, "MAX_SAMPLE_COUNT", 1)
    with pytest.raises(ValueError, match="sample count"):
        compute_dataset_manifest(dataset)

    monkeypatch.setattr(binding_module, "MAX_SAMPLE_COUNT", 1_000_000)
    monkeypatch.setattr(binding_module, "MAX_MODEL_TYPE_BYTES", 4)
    with pytest.raises(ValueError, match="model_type.*too long"):
        compute_dataset_manifest(dataset)

    monkeypatch.setattr(binding_module, "MAX_MODEL_TYPE_BYTES", 4096)
    monkeypatch.setattr(binding_module, "MAX_ROOT_ATTRIBUTES_BYTES", 8)
    with pytest.raises(ValueError, match="root attributes.*too large"):
        compute_dataset_manifest(dataset)


def test_manifest_loader_converts_parser_resource_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_json_bytes(compute_dataset_manifest(dataset)))
    monkeypatch.setattr(
        binding_module.json,
        "loads",
        lambda _raw: (_ for _ in ()).throw(RecursionError()),
    )

    with pytest.raises(ValueError, match="resource limit"):
        load_dataset_content_binding(manifest, dataset)


def test_manifest_loader_rejects_v1_extra_keys_and_global_drift(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    manifest = compute_dataset_manifest(dataset)
    path = tmp_path / "manifest.json"
    for mutation, message in (
        ({**manifest, "version": 1}, "version 2"),
        ({**manifest, "extra": True}, "key set"),
    ):
        path.write_bytes(canonical_json_bytes(mutation))
        with pytest.raises(ValueError, match=message):
            load_dataset_content_binding(path, dataset)

    path.write_bytes(canonical_json_bytes(manifest))
    with h5py.File(dataset, "r+") as h5:
        h5.attrs["producer"] = "drift"
    with pytest.raises(ValueError, match="stat|global"):
        load_dataset_content_binding(path, dataset)


def test_verified_ids_hash_sorts_deduplicates_and_binds_count() -> None:
    assert verified_ids_sha256([7, 2, 7]) == verified_ids_sha256([2, 7])
    assert verified_ids_sha256([2, 7]) != verified_ids_sha256([2, 7, 8])


def test_process_verification_summary_can_require_exact_consumed_ids(
    tmp_path: Path,
) -> None:
    dataset = _tiny_dataset(tmp_path / "tiny.h5")
    manifest = compute_dataset_manifest(dataset)
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    binding = load_dataset_content_binding(path, dataset)
    summary = {
        "dataset_content_root": binding.dataset_content_root,
        "verified_sample_count": 2,
        "verified_sample_ids_sha256": verified_ids_sha256([0, 1]),
        "verified_sample_scope": "process_unique",
    }

    validate_process_verification(summary, binding, expected_ids=[1, 0])
    with pytest.raises(ValueError, match="verified sample"):
        validate_process_verification(summary, binding, expected_ids=[0])
