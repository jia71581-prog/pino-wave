from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

import h5py
import numpy as np


MANIFEST_KEYS = frozenset(
    {
        "schema",
        "version",
        "hash_contract",
        "hash_algorithm",
        "dataset_path",
        "dataset_stat",
        "sample_count",
        "required_datasets",
        "global_sha256",
        "sample_digests",
        "sample_merkle_root",
        "dataset_content_root",
    }
)
DATASET_STAT_KEYS = frozenset({"device", "inode", "size_bytes", "mtime_ns"})
REQUIRED_DATASET_KEYS = frozenset({"name", "shape", "dtype", "axis_contract"})
HEADER_KEYS = frozenset(
    {
        "schema",
        "version",
        "hash_contract",
        "hash_algorithm",
        "dataset_path",
        "dataset_stat",
        "sample_count",
        "required_datasets",
        "global_sha256",
    }
)
PARTIAL_KEYS = frozenset(
    {"schema", "version", "header", "committed_count", "sample_digests"}
)
REQUIRED_LAYOUT = (
    ("tensor", "<f4", ("sample", "time", "x", "y")),
    ("nu", "<f4", ("sample", "x", "y")),
    ("source_mask", "<f4", ("sample", "x", "y")),
    ("model_type", "utf8-decoded", ("sample",)),
)
_HEX = frozenset("0123456789abcdef")
_HASH_CHUNK_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SAMPLE_COUNT = 1_000_000
MAX_MODEL_TYPE_BYTES = 4096
MAX_ROOT_ATTRIBUTES_BYTES = 16 * 1024 * 1024


def reject_symlink_components(path: str | Path, label: str) -> None:
    candidate = Path(path).absolute()
    for component in reversed((candidate, *candidate.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} path contains a symlink component")


def _uint32(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("uint32 value is out of range")
    return struct.pack(">I", value)


def _uint64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError("uint64 value is out of range")
    return struct.pack(">Q", value)


def _blob(value: bytes) -> bytes:
    return _uint64(len(value)) + value


def _text(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("canonical text value must be a string")
    return _blob(value.encode("utf-8"))


def _decoded_text(value: object, name: str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        try:
            decoded = bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} is not valid UTF-8") from error
    if isinstance(value, (str, np.str_)):
        decoded = str(value)
    elif not isinstance(value, (bytes, np.bytes_)):
        raise ValueError(f"{name} must be UTF-8 text")
    if name == "model_type" and len(decoded.encode("utf-8")) > MAX_MODEL_TYPE_BYTES:
        raise ValueError("model_type UTF-8 text is too long")
    return decoded


def _canonical_numeric_array(
    value: object, dtype: str, name: str
) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype.kind != expected.kind or array.dtype.itemsize != expected.itemsize:
        raise ValueError(f"{name} dtype must be {dtype}")
    if array.dtype != expected or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=expected)
    return array


def _array_header(name: str, dtype: str, shape: Sequence[int], nbytes: int) -> bytes:
    return b"".join(
        (
            _text("array-v1"),
            _text(name),
            _text(dtype),
            _uint32(len(shape)),
            *(_uint64(int(size)) for size in shape),
            _uint64(nbytes),
        )
    )


def _update_memoryview(hasher: object, array: np.ndarray) -> None:
    view = memoryview(array).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        hasher.update(view[start : start + _HASH_CHUNK_BYTES])


def _update_array(
    hasher: object, name: str, value: object, dtype: str
) -> None:
    array = _canonical_numeric_array(value, dtype, name)
    hasher.update(_array_header(name, dtype, array.shape, array.nbytes))
    _update_memoryview(hasher, array)


def canonical_array_bytes(name: str, value: object, dtype: str) -> bytes:
    array = _canonical_numeric_array(value, dtype, name)
    return _array_header(name, dtype, array.shape, array.nbytes) + array.tobytes(order="C")


def sample_digest(
    sample_id: int,
    tensor: object,
    nu: object,
    source_mask: object,
    model_type: object,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_text("AIS-DENSE-SAMPLE-v1"))
    hasher.update(_uint64(sample_id))
    _update_array(hasher, "tensor", tensor, "<f4")
    _update_array(hasher, "nu", nu, "<f4")
    _update_array(hasher, "source_mask", source_mask, "<f4")
    hasher.update(_text(_decoded_text(model_type, "model_type")))
    return hasher.hexdigest()


def sample_digest_from_h5(
    h5: h5py.File, sample_id: int, *, time_block: int | None = None
) -> str:
    schema = required_dataset_schema(h5)
    sample_count = int(schema[0]["shape"][0])
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or not 0 <= sample_id < sample_count:
        raise ValueError("sample_id is outside the dataset")
    if time_block is None:
        return sample_digest(
            sample_id,
            h5["tensor"][sample_id],
            h5["nu"][sample_id],
            h5["source_mask"][sample_id],
            h5["model_type"][sample_id],
        )
    if isinstance(time_block, bool) or not isinstance(time_block, int) or time_block <= 0:
        raise ValueError("time_block must be a positive integer")
    tensor = h5["tensor"]
    chunks = tensor.chunks
    if chunks is None:
        raise ValueError("time_block requires chunked tensor; use whole-sample mode")
    chunk_time = int(chunks[1])
    if time_block < chunk_time or time_block % chunk_time:
        raise ValueError(
            "time_block must align with the natural time chunk; use whole-sample mode"
        )
    sample_shape = tuple(int(size) for size in tensor.shape[1:])
    hasher = hashlib.sha256()
    hasher.update(_text("AIS-DENSE-SAMPLE-v1"))
    hasher.update(_uint64(sample_id))
    hasher.update(
        _array_header("tensor", "<f4", sample_shape, int(np.prod(sample_shape)) * 4)
    )
    for start in range(0, sample_shape[0], time_block):
        block = _canonical_numeric_array(
            tensor[sample_id, start : start + time_block], "<f4", "tensor"
        )
        _update_memoryview(hasher, block)
    _update_array(hasher, "nu", h5["nu"][sample_id], "<f4")
    _update_array(hasher, "source_mask", h5["source_mask"][sample_id], "<f4")
    hasher.update(_text(_decoded_text(h5["model_type"][sample_id], "model_type")))
    return hasher.hexdigest()


def _canonical_storage_dtype(dataset: h5py.Dataset, name: str) -> str:
    if name == "model_type":
        if h5py.check_string_dtype(dataset.dtype) is None and dataset.dtype.kind not in {
            "S",
            "U",
        }:
            raise ValueError("model_type dtype must be UTF-8 text")
        return "utf8-decoded"
    dtype = dataset.dtype
    if dtype.kind != "f" or dtype.itemsize != 4:
        raise ValueError(f"{name} dtype must be <f4")
    return "<f4"


def required_dataset_schema(h5: h5py.File) -> list[dict[str, object]]:
    missing = [name for name, _, _ in REQUIRED_LAYOUT if name not in h5]
    if missing:
        raise ValueError(f"AIS dataset is missing required datasets: {missing}")
    result: list[dict[str, object]] = []
    for name, expected_dtype, axes in REQUIRED_LAYOUT:
        dataset = h5[name]
        dtype = _canonical_storage_dtype(dataset, name)
        if dtype != expected_dtype or dataset.ndim != len(axes):
            raise ValueError(f"{name} schema differs from required dtype or axes")
        result.append(
            {
                "name": name,
                "shape": [int(size) for size in dataset.shape],
                "dtype": dtype,
                "axis_contract": list(axes),
            }
        )
    sample_count = result[0]["shape"][0]
    if not 0 < sample_count <= MAX_SAMPLE_COUNT:
        raise ValueError("required dataset sample count exceeds the supported limit")
    if any(item["shape"][0] != sample_count for item in result):
        raise ValueError("required dataset sample dimensions differ")
    tensor_shape = result[0]["shape"]
    if tensor_shape[1] != 160 or any(size <= 0 for size in tensor_shape[2:]):
        raise ValueError("tensor shape must have exactly 160 times and positive space")
    if result[1]["shape"] != [sample_count, tensor_shape[2], tensor_shape[3]] or result[
        2
    ]["shape"] != [sample_count, tensor_shape[2], tensor_shape[3]]:
        raise ValueError("required dataset spatial shapes differ")
    return result


def _schema_bytes(schema: Sequence[Mapping[str, object]]) -> bytes:
    parts = [_text("dataset-schema-v1"), _uint32(len(schema))]
    for item in schema:
        shape = item["shape"]
        axes = item["axis_contract"]
        parts.extend(
            (
                _text(str(item["name"])),
                _text(str(item["dtype"])),
                _uint32(len(shape)),
                *(_uint64(int(size)) for size in shape),
                _uint32(len(axes)),
                *(_text(str(axis)) for axis in axes),
            )
        )
    return b"".join(parts)


def _canonical_ndarray(value: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    dtype = value.dtype
    if dtype.kind == "b" and dtype.itemsize == 1:
        canonical = "|b1"
    elif dtype.kind in {"i", "u"} and dtype.itemsize == 8:
        canonical = f"<{dtype.kind}8"
    elif dtype.kind == "f" and dtype.itemsize in {4, 8}:
        canonical = f"<f{dtype.itemsize}"
    else:
        raise ValueError(f"root attribute {name} ndarray dtype is unsupported")
    array = np.asarray(value, dtype=np.dtype(canonical), order="C")
    return array, canonical


def _attribute_bytes(key: str, value: object) -> bytes:
    prefix = _text(key)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _attribute_bytes(key, value.item())
    if value is None:
        return prefix + b"\x00"
    if isinstance(value, (bool, np.bool_)):
        return prefix + (b"\x02" if bool(value) else b"\x01")
    if isinstance(value, np.ndarray) and value.ndim > 0:
        array, dtype = _canonical_ndarray(value, key)
        return b"".join(
            (
                prefix,
                b"\x07",
                _text("ndarray-v1"),
                _text(dtype),
                _uint32(array.ndim),
                *(_uint64(int(size)) for size in array.shape),
                _uint64(array.nbytes),
                array.tobytes(order="C"),
            )
        )
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        integer = int(value)
        if not -(2**63) <= integer < 2**63:
            raise ValueError(f"root attribute {key} int64 is out of range")
        return prefix + b"\x03" + struct.pack(">q", integer)
    if isinstance(value, (float, np.floating)):
        return prefix + b"\x04" + struct.pack(">d", float(value))
    if isinstance(value, (str, np.str_)):
        return prefix + b"\x05" + _text(str(value))
    if isinstance(value, (bytes, np.bytes_)):
        return prefix + b"\x06" + _blob(bytes(value))
    raise ValueError(f"root attribute {key} type is unsupported")


def _root_attributes_bytes(attributes: Mapping[str, object]) -> bytes:
    keys = sorted(attributes)
    encoded = b"".join(
        (
            _text("root-attrs-v1"),
            _uint32(len(keys)),
            *(_attribute_bytes(key, attributes[key]) for key in keys),
        )
    )
    if len(encoded) > MAX_ROOT_ATTRIBUTES_BYTES:
        raise ValueError("canonical root attributes payload is too large")
    return encoded


def canonical_root_attributes_bytes(attributes: Mapping[str, object]) -> bytes:
    return _root_attributes_bytes(attributes)


def global_digest(h5: h5py.File) -> tuple[str, list[dict[str, object]]]:
    for name in ("t-coordinate", "x-coordinate", "y-coordinate"):
        if name not in h5:
            raise ValueError(f"AIS dataset is missing required data: {name}")
    schema = required_dataset_schema(h5)
    tensor_shape = schema[0]["shape"]
    coordinates = (
        ("t-coordinate", h5["t-coordinate"][...], tensor_shape[1]),
        ("x-coordinate", 1000.0 * h5["x-coordinate"][...], tensor_shape[2]),
        ("y-coordinate", 1000.0 * h5["y-coordinate"][...], tensor_shape[3]),
    )
    hasher = hashlib.sha256()
    hasher.update(_text("AIS-DENSE-GLOBAL-v1"))
    for name, values, expected_size in coordinates:
        if np.asarray(values).shape != (expected_size,):
            raise ValueError(f"{name} shape differs from required schema")
        _update_array(hasher, name, values, "<f8")
    hasher.update(_schema_bytes(schema))
    hasher.update(_root_attributes_bytes(dict(h5.attrs)))
    return hasher.hexdigest(), schema


def _digest_bytes(value: str, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return bytes.fromhex(value)


def merkle_root(sample_digests: Sequence[str]) -> str:
    if not sample_digests:
        raise ValueError("sample digest list must be nonempty")
    nodes = [hashlib.sha256(b"\x00" + _digest_bytes(item, "sample digest")).digest() for item in sample_digests]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def dataset_content_root(global_sha256: str, merkle_sha256: str, sample_count: int) -> str:
    return hashlib.sha256(
        b"\x02"
        + _digest_bytes(global_sha256, "global digest")
        + _digest_bytes(merkle_sha256, "Merkle digest")
        + _uint64(sample_count)
    ).hexdigest()


def dataset_stat(stat_result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def manifest_header(
    dataset_path: Path, stat_result: os.stat_result, h5: h5py.File
) -> dict[str, object]:
    global_sha256, schema = global_digest(h5)
    return {
        "schema": "ais_dataset_content_manifest",
        "version": 2,
        "hash_contract": "ais_dense_logical_v1",
        "hash_algorithm": "sha256",
        "dataset_path": str(dataset_path),
        "dataset_stat": dataset_stat(stat_result),
        "sample_count": int(schema[0]["shape"][0]),
        "required_datasets": schema,
        "global_sha256": global_sha256,
    }


def manifest_from_digests(
    header: Mapping[str, object], sample_digests: Sequence[str]
) -> dict[str, object]:
    if set(header) != HEADER_KEYS:
        raise ValueError("manifest header key set is invalid")
    count = int(header["sample_count"])
    if len(sample_digests) != count:
        raise ValueError("sample digest count differs from manifest header")
    root = merkle_root(sample_digests)
    return {
        **dict(header),
        "sample_digests": list(sample_digests),
        "sample_merkle_root": root,
        "dataset_content_root": dataset_content_root(
            str(header["global_sha256"]), root, count
        ),
    }


def partial_checkpoint(
    header: Mapping[str, object], sample_digests: Sequence[str]
) -> dict[str, object]:
    if set(header) != HEADER_KEYS:
        raise ValueError("partial header key set is invalid")
    if len(sample_digests) > int(header["sample_count"]):
        raise ValueError("partial digest prefix exceeds sample count")
    for index, digest in enumerate(sample_digests):
        _digest_bytes(digest, f"partial sample_digests[{index}]")
    return {
        "schema": "ais_dataset_content_manifest_partial",
        "version": 1,
        "header": dict(header),
        "committed_count": len(sample_digests),
        "sample_digests": list(sample_digests),
    }


def validate_partial_checkpoint(
    payload: object, expected_header: Mapping[str, object]
) -> tuple[str, ...]:
    if not isinstance(payload, dict) or set(payload) != PARTIAL_KEYS:
        raise ValueError("partial checkpoint key set is invalid")
    if (
        payload["schema"] != "ais_dataset_content_manifest_partial"
        or payload["version"] != 1
        or payload["header"] != dict(expected_header)
    ):
        raise ValueError("partial checkpoint header differs from current dataset")
    count = _nonnegative_integer(payload["committed_count"], "committed_count")
    digests = payload["sample_digests"]
    if not isinstance(digests, list) or len(digests) != count:
        raise ValueError("partial checkpoint digest prefix length mismatch")
    if count > int(expected_header["sample_count"]):
        raise ValueError("partial checkpoint exceeds sample count")
    for index, digest in enumerate(digests):
        _digest_bytes(digest, f"partial sample_digests[{index}]")
    return tuple(digests)


def verified_ids_sha256(sample_ids: Sequence[int]) -> str:
    unique = sorted(set(sample_ids))
    hasher = hashlib.sha256()
    hasher.update(_text("AIS-VERIFIED-IDS-v1"))
    hasher.update(_uint64(len(unique)))
    for sample_id in unique:
        hasher.update(_uint64(sample_id))
    return hasher.hexdigest()


@dataclass(frozen=True)
class DatasetContentBinding:
    manifest_path: Path
    manifest_sha256: str
    dataset_path: Path
    dataset_stat: dict[str, int]
    sample_count: int
    required_datasets: tuple[dict[str, object], ...]
    global_sha256: str
    sample_digests: tuple[str, ...]
    sample_merkle_root: str
    dataset_content_root: str


def validate_process_verification(
    summary: Mapping[str, object],
    binding: DatasetContentBinding | Mapping[str, object],
    *,
    expected_ids: Sequence[int] | None = None,
) -> None:
    content_root = (
        binding.dataset_content_root
        if isinstance(binding, DatasetContentBinding)
        else binding.get("dataset_content_root")
    )
    sample_count = (
        binding.sample_count
        if isinstance(binding, DatasetContentBinding)
        else binding.get("sample_count")
    )
    if (
        summary.get("dataset_content_root") != content_root
        or summary.get("verified_sample_scope") != "process_unique"
    ):
        raise ValueError("dataset content root or verification scope mismatch")
    count = summary.get("verified_sample_count")
    digest = summary.get("verified_sample_ids_sha256")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 0 <= count <= sample_count
    ):
        raise ValueError("verified sample count is invalid")
    _digest_bytes(digest, "verified sample IDs digest")
    if expected_ids is not None:
        ids = sorted(set(expected_ids))
        if count != len(ids) or digest != verified_ids_sha256(ids):
            raise ValueError("verified sample IDs differ from expected process inputs")


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validated_manifest_structure(payload: object) -> dict[str, object]:
    if isinstance(payload, dict) and payload.get("schema_version") == 1:
        raise ValueError("formal dataset content manifest requires version 2")
    if not isinstance(payload, dict) or set(payload) != MANIFEST_KEYS:
        raise ValueError("dataset content manifest key set is invalid")
    if (
        payload["schema"] != "ais_dataset_content_manifest"
        or payload["version"] != 2
        or payload["hash_contract"] != "ais_dense_logical_v1"
        or payload["hash_algorithm"] != "sha256"
    ):
        raise ValueError("formal dataset content manifest requires version 2")
    path = payload["dataset_path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("manifest dataset_path must be canonical and absolute")
    raw_stat = payload["dataset_stat"]
    if not isinstance(raw_stat, dict) or set(raw_stat) != DATASET_STAT_KEYS:
        raise ValueError("manifest dataset_stat key set is invalid")
    payload["dataset_stat"] = {
        name: _nonnegative_integer(raw_stat[name], f"dataset_stat.{name}")
        for name in sorted(DATASET_STAT_KEYS)
    }
    count = _nonnegative_integer(payload["sample_count"], "sample_count")
    if not 0 < count <= MAX_SAMPLE_COUNT:
        raise ValueError("sample_count must be positive and within the supported limit")
    raw_schema = payload["required_datasets"]
    if not isinstance(raw_schema, list) or len(raw_schema) != len(REQUIRED_LAYOUT):
        raise ValueError("required_datasets must contain the exact four entries")
    schema: list[dict[str, object]] = []
    for index, ((name, dtype, axes), item) in enumerate(zip(REQUIRED_LAYOUT, raw_schema)):
        if not isinstance(item, dict) or set(item) != REQUIRED_DATASET_KEYS:
            raise ValueError("required dataset object key set is invalid")
        shape = item["shape"]
        if not isinstance(shape, list) or len(shape) != len(axes):
            raise ValueError(f"required dataset {name} shape is invalid")
        validated_shape = [
            _nonnegative_integer(size, f"required_datasets[{index}].shape")
            for size in shape
        ]
        if (
            item["name"] != name
            or item["dtype"] != dtype
            or item["axis_contract"] != list(axes)
            or validated_shape[0] != count
        ):
            raise ValueError(f"required dataset {name} contract mismatch")
        schema.append({**item, "shape": validated_shape})
    tensor_shape = schema[0]["shape"]
    if tensor_shape[1] != 160 or any(size <= 0 for size in tensor_shape[2:]):
        raise ValueError("manifest tensor shape contract mismatch")
    if schema[1]["shape"] != [count, tensor_shape[2], tensor_shape[3]] or schema[2][
        "shape"
    ] != [count, tensor_shape[2], tensor_shape[3]]:
        raise ValueError("manifest required dataset spatial shapes differ")
    digests = payload["sample_digests"]
    if not isinstance(digests, list) or len(digests) != count:
        raise ValueError("sample_digests count differs from sample_count")
    for index, digest in enumerate(digests):
        _digest_bytes(digest, f"sample_digests[{index}]")
    for key in ("global_sha256", "sample_merkle_root", "dataset_content_root"):
        _digest_bytes(payload[key], key)
    computed_merkle = merkle_root(digests)
    if payload["sample_merkle_root"] != computed_merkle:
        raise ValueError("manifest Merkle root differs from sample digests")
    computed_root = dataset_content_root(payload["global_sha256"], computed_merkle, count)
    if payload["dataset_content_root"] != computed_root:
        raise ValueError("manifest dataset content root is internally inconsistent")
    payload["required_datasets"] = schema
    return payload


def load_dataset_content_binding(
    manifest_path: str | Path,
    dataset_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> DatasetContentBinding:
    manifest_file = Path(manifest_path)
    reject_symlink_components(manifest_file, "dataset content manifest")
    manifest_fd = os.open(manifest_file, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        manifest_stat = os.fstat(manifest_fd)
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise ValueError("dataset content manifest must be a regular file")
        if manifest_stat.st_size > MAX_MANIFEST_BYTES:
            raise ValueError("dataset content manifest is too large")
        with os.fdopen(os.dup(manifest_fd), "rb") as stream:
            raw = stream.read()
    finally:
        os.close(manifest_fd)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None:
        _digest_bytes(expected_manifest_sha256, "expected manifest digest")
        if raw_sha256 != expected_manifest_sha256:
            raise ValueError("dataset content manifest SHA-256 mismatch")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset content manifest is not valid JSON") from error
    except (RecursionError, MemoryError) as error:
        raise ValueError("dataset content manifest exceeded a resource limit") from error
    try:
        if not isinstance(parsed, dict) or raw != canonical_json_bytes(parsed):
            raise ValueError("dataset content manifest must use canonical JSON")
        payload = _validated_manifest_structure(parsed)
    except (RecursionError, MemoryError) as error:
        raise ValueError("dataset content manifest exceeded a resource limit") from error

    candidate = Path(dataset_path)
    reject_symlink_components(candidate, "dataset")
    dataset_fd = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current_stat = os.fstat(dataset_fd)
        if not stat.S_ISREG(current_stat.st_mode):
            raise ValueError("dataset path must be a regular file")
        resolved = candidate.resolve(strict=True)
        if str(resolved) != payload["dataset_path"]:
            raise ValueError("manifest dataset_path differs from opened dataset")
        audit = dataset_stat(current_stat)
        if audit != payload["dataset_stat"]:
            raise ValueError("dataset stat differs from content manifest")
        with os.fdopen(os.dup(dataset_fd), "rb") as stream:
            with h5py.File(stream, "r") as h5:
                current_global, current_schema = global_digest(h5)
        if dataset_stat(os.fstat(dataset_fd)) != audit:
            raise ValueError("dataset stat changed during global verification")
    finally:
        os.close(dataset_fd)
    if current_schema != payload["required_datasets"]:
        raise ValueError("dataset required schema differs from content manifest")
    if current_global != payload["global_sha256"]:
        raise ValueError("dataset global digest differs from content manifest")
    return DatasetContentBinding(
        manifest_path=manifest_file.resolve(strict=True),
        manifest_sha256=raw_sha256,
        dataset_path=resolved,
        dataset_stat=audit,
        sample_count=int(payload["sample_count"]),
        required_datasets=tuple(payload["required_datasets"]),
        global_sha256=str(payload["global_sha256"]),
        sample_digests=tuple(payload["sample_digests"]),
        sample_merkle_root=str(payload["sample_merkle_root"]),
        dataset_content_root=str(payload["dataset_content_root"]),
    )


def compute_dataset_manifest(path: str | Path) -> dict[str, object]:
    dataset_path = Path(path)
    reject_symlink_components(dataset_path, "dataset")
    resolved = dataset_path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("dataset path must be a regular file")
    with h5py.File(resolved, "r") as h5:
        header = manifest_header(resolved, before, h5)
        sample_count = int(header["sample_count"])
        digests = [
            sample_digest_from_h5(h5, sample_id)
            for sample_id in range(sample_count)
        ]
    after = resolved.stat()
    if dataset_stat(before) != dataset_stat(after):
        raise ValueError("dataset stat changed during hashing")
    return manifest_from_digests(header, digests)
