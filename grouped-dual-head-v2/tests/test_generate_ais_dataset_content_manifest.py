from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import scripts.generate_ais_dataset_content_manifest as generator
from fno_acoustic.ais_dataset_binding import (
    canonical_json_bytes,
    load_dataset_content_binding,
    sample_digest_from_h5,
)


def _dataset(path: Path, count: int = 10, *, chunk_time: int = 160) -> Path:
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "tensor",
            data=np.arange(count * 160 * 2 * 2, dtype=np.float32).reshape(
                count, 160, 2, 2
            ),
            chunks=(1, chunk_time, 2, 2),
        )
        h5.create_dataset("nu", data=np.ones((count, 2, 2), dtype=np.float32))
        h5.create_dataset(
            "source_mask", data=np.ones((count, 2, 2), dtype=np.float32)
        )
        h5.create_dataset(
            "model_type", data=np.asarray([b"uniform"] * count)
        )
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.2, 160))
        h5.create_dataset("x-coordinate", data=np.asarray([0.1, 0.2]))
        h5.create_dataset("y-coordinate", data=np.asarray([0.3, 0.4]))
        h5.attrs["save_indices"] = np.arange(160, dtype=np.int64)
    return path


def test_streamed_time_blocks_are_digest_identical_to_whole_sample(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "tiny.h5", count=1, chunk_time=16)
    with h5py.File(dataset, "r") as h5:
        whole = sample_digest_from_h5(h5, 0)
        streamed = sample_digest_from_h5(h5, 0, time_block=16)

    assert streamed == whole


def test_time_block_smaller_than_natural_chunk_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "tiny.h5", count=1, chunk_time=160)
    with h5py.File(dataset, "r") as h5:
        with pytest.raises(ValueError, match="whole-sample"):
            sample_digest_from_h5(h5, 0, time_block=16)


def test_generator_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    dataset = _dataset(real / "tiny.h5", count=1)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        generator.generate(alias / dataset.name, tmp_path / "manifest.json")


def test_partial_is_size_checked_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "manifest.json.partial"
    partial.write_bytes(b"x" * 32)
    monkeypatch.setattr(generator, "MAX_MANIFEST_BYTES", 16)

    with pytest.raises(ValueError, match="too large"):
        generator._read_canonical_partial(partial)


def test_generator_publishes_canonical_v2_and_identical_final_is_idempotent(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "tiny.h5", count=2)
    output = tmp_path / "manifest.json"

    assert generator.generate(dataset, output) == output
    first = output.read_bytes()
    assert first == canonical_json_bytes(json.loads(first))
    assert load_dataset_content_binding(output, dataset).sample_count == 2

    assert generator.generate(dataset, output) == output
    assert output.read_bytes() == first


def test_interrupted_resume_rehashes_prefix_and_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path / "tiny.h5")
    output = tmp_path / "resumed.json"
    uninterrupted = tmp_path / "uninterrupted.json"
    original = generator.sample_digest_from_h5
    calls = 0

    def interrupt_after_checkpoint(h5: h5py.File, sample_id: int, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 9:
            raise KeyboardInterrupt
        return original(h5, sample_id, **kwargs)

    monkeypatch.setattr(generator, "sample_digest_from_h5", interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        generator.generate(dataset, output)
    partial = Path(f"{output}.partial")
    assert json.loads(partial.read_bytes())["committed_count"] == 8

    monkeypatch.setattr(generator, "sample_digest_from_h5", original)
    generator.generate(dataset, output, resume=True)
    generator.generate(dataset, uninterrupted)

    assert output.read_bytes() == uninterrupted.read_bytes()
    assert not partial.exists()


def test_resume_rejects_tampered_partial_prefix(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "tiny.h5")
    output = tmp_path / "manifest.json"
    partial = Path(f"{output}.partial")
    generator.generate(dataset, output)
    payload = json.loads(output.read_bytes())
    header = {
        key: payload[key]
        for key in (
            "schema", "version", "hash_contract", "hash_algorithm",
            "dataset_path", "dataset_stat", "sample_count",
            "required_datasets", "global_sha256",
        )
    }
    partial.write_bytes(
        canonical_json_bytes(
            {
                "schema": "ais_dataset_content_manifest_partial",
                "version": 1,
                "header": header,
                "committed_count": 1,
                "sample_digests": ["0" * 64],
            }
        )
    )
    output.unlink()

    with pytest.raises(ValueError, match="prefix"):
        generator.generate(dataset, output, resume=True)


def test_generator_refuses_symlinks_and_different_existing_final(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "tiny.h5", count=1)
    output = tmp_path / "manifest.json"
    output.write_text("different\n")
    with pytest.raises(ValueError, match="different existing final"):
        generator.generate(dataset, output)

    output.unlink()
    link = tmp_path / "linked.h5"
    link.symlink_to(dataset)
    with pytest.raises(ValueError, match="symlink"):
        generator.generate(link, output)


def test_generator_flock_and_fstat_drift_are_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path / "tiny.h5", count=1)
    output = tmp_path / "manifest.json"
    lock = Path(f"{output}.lock")
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="locked"):
            generator.generate(dataset, output)

    original = generator.sample_digest_from_h5

    def mutate_stat(h5: h5py.File, sample_id: int, **kwargs: object) -> str:
        result = original(h5, sample_id, **kwargs)
        current = dataset.stat()
        os.utime(
            dataset,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(generator, "sample_digest_from_h5", mutate_stat)
    with pytest.raises(ValueError, match="stat changed"):
        generator.generate(dataset, output)
