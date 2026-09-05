#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import sys

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fno_acoustic.ais_dataset_binding import (  # noqa: E402
    MAX_MANIFEST_BYTES,
    canonical_json_bytes,
    dataset_stat,
    manifest_from_digests,
    manifest_header,
    partial_checkpoint,
    reject_symlink_components,
    sample_digest_from_h5,
    validate_partial_checkpoint,
)


CHECKPOINT_EVERY = 8


def _derived_path(output: Path, suffix: str) -> Path:
    return Path(f"{output}{suffix}")


def _reject_symlink(path: Path, label: str) -> None:
    reject_symlink_components(path, label)


def _atomic_write(path: Path, raw: bytes) -> None:
    _reject_symlink(path, "destination")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_canonical_partial(path: Path) -> object:
    _reject_symlink(path, "partial checkpoint")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError("partial checkpoint must be a regular file")
        if opened_stat.st_size > MAX_MANIFEST_BYTES:
            raise ValueError("partial checkpoint is too large")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read()
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("partial checkpoint is not valid JSON") from error
    except (RecursionError, MemoryError) as error:
        raise ValueError("partial checkpoint exceeded a resource limit") from error
    try:
        if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
            raise ValueError("partial checkpoint must use canonical JSON")
    except (RecursionError, MemoryError) as error:
        raise ValueError("partial checkpoint exceeded a resource limit") from error
    return payload


def _stable(fd: int, expected: dict[str, int]) -> None:
    if dataset_stat(os.fstat(fd)) != expected:
        raise ValueError("dataset stat changed during manifest generation")


def generate(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    resume: bool = False,
    time_block: int | None = None,
) -> Path:
    dataset = Path(dataset_path)
    output = Path(output_path)
    partial = _derived_path(output, ".partial")
    lock_path = _derived_path(output, ".lock")
    if not output.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    for path, label in (
        (dataset, "dataset"),
        (output, "output"),
        (partial, "partial checkpoint"),
        (lock_path, "lock"),
    ):
        _reject_symlink(path, label)

    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("dataset manifest output is locked") from error
        flags = os.O_RDONLY | os.O_NOFOLLOW
        dataset_fd = os.open(dataset, flags)
        try:
            opened_stat = os.fstat(dataset_fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError("dataset path must be a regular file")
            resolved = dataset.resolve(strict=True)
            audit = dataset_stat(opened_stat)
            with os.fdopen(os.dup(dataset_fd), "rb") as raw_stream:
                with h5py.File(raw_stream, "r") as h5:
                    header = manifest_header(resolved, opened_stat, h5)
                    _stable(dataset_fd, audit)
                    digests: list[str] = []
                    if resume:
                        print(
                            "warning: --resume rehashes the complete committed prefix; "
                            "it does not save dataset read I/O",
                            file=sys.stderr,
                        )
                        if not partial.is_file():
                            raise ValueError("--resume requires a partial checkpoint")
                        expected_prefix = validate_partial_checkpoint(
                            _read_canonical_partial(partial), header
                        )
                        for sample_id, expected in enumerate(expected_prefix):
                            actual = sample_digest_from_h5(
                                h5, sample_id, time_block=time_block
                            )
                            if actual != expected:
                                raise ValueError(
                                    "partial checkpoint digest prefix differs from HDF5"
                                )
                            digests.append(actual)
                            _stable(dataset_fd, audit)
                    elif partial.exists():
                        raise ValueError(
                            "partial checkpoint exists; use --resume or remove it"
                        )
                    sample_count = int(header["sample_count"])
                    for sample_id in range(len(digests), sample_count):
                        digests.append(
                            sample_digest_from_h5(
                                h5, sample_id, time_block=time_block
                            )
                        )
                        _stable(dataset_fd, audit)
                        if len(digests) % CHECKPOINT_EVERY == 0:
                            _atomic_write(
                                partial,
                                canonical_json_bytes(
                                    partial_checkpoint(header, digests)
                                ),
                            )
                    _atomic_write(
                        partial,
                        canonical_json_bytes(partial_checkpoint(header, digests)),
                    )
                    final_raw = canonical_json_bytes(
                        manifest_from_digests(header, digests)
                    )
                    _stable(dataset_fd, audit)
            if output.exists():
                _reject_symlink(output, "output")
                output_fd = os.open(output, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    output_stat = os.fstat(output_fd)
                    if not stat.S_ISREG(output_stat.st_mode):
                        raise ValueError("existing final manifest must be a regular file")
                    if output_stat.st_size > MAX_MANIFEST_BYTES:
                        raise ValueError("existing final manifest is too large")
                    with os.fdopen(os.dup(output_fd), "rb") as stream:
                        existing_raw = stream.read()
                finally:
                    os.close(output_fd)
                if existing_raw != final_raw:
                    raise ValueError("refusing different existing final manifest")
            else:
                _atomic_write(output, final_raw)
            partial.unlink()
            return output
        finally:
            os.close(dataset_fd)
    finally:
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-block", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(
        args.dataset,
        args.output,
        resume=args.resume,
        time_block=args.time_block,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
