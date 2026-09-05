"""Identity primitives for sealed, exact-stored-time V4 evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


IDENTITY_KEYS = (
    "checkpoint_sha256",
    "manifest_digest",
    "time_axis_sha256",
    "record_census",
    "model_config_digest",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_axis_sha256(values: Sequence[float]) -> str:
    encoded = json.dumps([float(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf8")).hexdigest()


def validate_evaluation_identity(
    candidate: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    for key in IDENTITY_KEYS:
        if candidate.get(key) != expected.get(key):
            raise ValueError(f"evaluation {key.replace('_', ' ')} mismatch")


__all__ = [
    "IDENTITY_KEYS",
    "sha256_file",
    "time_axis_sha256",
    "validate_evaluation_identity",
]
