"""V12 production candidate lock and sealed evaluation terminal chain."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v7 import validate_checkpoint_identity
from .r16_dscp_engine_v11 import V11ProductionBackend
from .r16_dscp_training_v2 import BindingRefusal, atomic_json_exclusive, sha256_file


class V12ChainRefusal(BindingRefusal):
    """A sealed evaluation-chain binding is absent or has drifted."""


class V12ProductionBackend(V11ProductionBackend):
    """V11 execution contracts with a V12-owned candidate spool."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spool = CandidateSpool(
            self.run_dir,
            run_digest=self.lineage.run_digest,
            rank=int(os.environ.get("RANK", "0")),
            owned_root="/dev/shm/r16_dscp_v12",
        )


def complete_identity_v12(**values: Any) -> dict[str, Any]:
    payload = {
        "schema": "r16_dscp_v12_checkpoint_identity_v1",
        "candidate": "r16_dscp_v12",
        **values,
    }
    payload["identity_digest"] = canonical_sha(payload)
    validate_checkpoint_identity(payload)
    return payload


def terminal_v12(mode: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(payload),
        "schema": f"r16_dscp_v12_{mode.replace('-', '_')}_terminal_v1",
        "candidate": "r16_dscp_v12",
        "mode": mode,
        "status": status,
        "decision": status,
    }


def _digest(payload: Mapping[str, Any]) -> str:
    return canonical_sha(dict(payload))


def _checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise V12ChainRefusal("checkpoint absent")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }


def build_final_chain(
    *,
    run_dir: str | Path,
    status: str,
    checkpoint_path: str | Path | None,
    thresholds: Mapping[str, Any],
    effective: Mapping[str, str],
    data_hashes: Mapping[str, str],
    gates: Mapping[str, Any],
    terminal_extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Atomically write final terminal, then a complete immutable candidate lock."""
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    lock_path = root / "candidate_lock.json"
    extra = dict(terminal_extra or {})
    if status != "passed":
        terminal = terminal_v12(
            "final-train-confirm",
            status,
            {
                **extra,
                "gates": dict(gates),
                "validation_opened": False,
                "test_opened": False,
                "rollback": "no_candidate_lock_no_validation_or_test",
            },
        )
        atomic_json_exclusive(terminal, terminal_path)
        return terminal, None
    if checkpoint_path is None:
        raise V12ChainRefusal("passed final requires checkpoint")
    checkpoint = _checkpoint(checkpoint_path)
    threshold_digest = _digest(thresholds)
    terminal = terminal_v12(
        "final-train-confirm",
        "passed",
        {
            **extra,
            **checkpoint,
            "best_checkpoint": {
                "path": checkpoint["checkpoint_path"],
                "sha256": checkpoint["checkpoint_sha256"],
                "size_bytes": checkpoint["checkpoint_size_bytes"],
            },
            "thresholds": dict(thresholds),
            "threshold_digest": threshold_digest,
            "effective": dict(effective),
            "data_hashes": dict(data_hashes),
            "gates": dict(gates),
            "validation_opened": False,
            "test_opened": False,
            "rollback": "lock_candidate_then_validation_once",
        },
    )
    atomic_json_exclusive(terminal, terminal_path)
    lock = {
        "schema": "r16_dscp_v12_candidate_lock_v1",
        "candidate": "r16_dscp_v12",
        **checkpoint,
        "thresholds": dict(thresholds),
        "threshold_digest": threshold_digest,
        "effective": dict(effective),
        "data_hashes": dict(data_hashes),
        "final_terminal_path": str(terminal_path),
        "final_terminal_sha256": sha256_file(terminal_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_before_validation": True,
        "validation_opened": False,
        "test_opened": False,
    }
    lock["lock_digest"] = _digest(lock)
    atomic_json_exclusive(lock, lock_path)
    return terminal, lock


def validate_final_chain(
    *,
    terminal_path: str | Path,
    lock_path: str | Path,
    expected_effective: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    terminal_file = Path(terminal_path).resolve()
    lock_file = Path(lock_path).resolve()
    if not terminal_file.is_file() or not lock_file.is_file():
        raise V12ChainRefusal("final chain file absent")
    terminal = json.loads(terminal_file.read_text())
    lock = json.loads(lock_file.read_text())
    if (
        terminal.get("status") != "passed"
        or terminal.get("schema") != "r16_dscp_v12_final_train_confirm_terminal_v1"
        or lock.get("schema") != "r16_dscp_v12_candidate_lock_v1"
        or terminal.get("candidate") != "r16_dscp_v12"
        or lock.get("candidate") != "r16_dscp_v12"
    ):
        raise V12ChainRefusal("final chain not passed V12")
    if lock.get("lock_digest") != _digest(
        {key: value for key, value in lock.items() if key != "lock_digest"}
    ):
        raise V12ChainRefusal("lock digest mismatch")
    if (
        lock.get("final_terminal_path") != str(terminal_file)
        or lock.get("final_terminal_sha256") != sha256_file(terminal_file)
    ):
        raise V12ChainRefusal("lock terminal binding mismatch")
    for key in (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "threshold_digest",
        "effective",
        "data_hashes",
    ):
        if terminal.get(key) != lock.get(key):
            raise V12ChainRefusal(f"terminal/lock mismatch: {key}")
    if dict(lock["effective"]) != dict(expected_effective):
        raise V12ChainRefusal("effective drift")
    checkpoint = Path(lock["checkpoint_path"]).resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != lock["checkpoint_size_bytes"]
        or sha256_file(checkpoint) != lock["checkpoint_sha256"]
    ):
        raise V12ChainRefusal("current checkpoint drift")
    return lock, lock["lock_digest"]


def validation_authorization(
    *, terminal_path: str | Path, lock_path: str | Path, effective: Mapping[str, str]
) -> dict[str, Any]:
    lock, digest = validate_final_chain(
        terminal_path=terminal_path,
        lock_path=lock_path,
        expected_effective=effective,
    )
    return {
        "mode": "validation-once",
        "final_terminal_path": str(Path(terminal_path).resolve()),
        "final_terminal_sha256": sha256_file(terminal_path),
        "candidate_lock_path": str(Path(lock_path).resolve()),
        "candidate_lock_sha256": sha256_file(lock_path),
        "candidate_lock_digest": digest,
        "candidate_checkpoint_path": lock["checkpoint_path"],
        "candidate_checkpoint_sha256": lock["checkpoint_sha256"],
        "threshold_digest": lock["threshold_digest"],
        "effective": dict(effective),
        "data_hashes": dict(lock["data_hashes"]),
    }


def validate_validation_authorization(
    authorization: Mapping[str, Any], expected_effective: Mapping[str, str]
) -> dict[str, Any]:
    expected = validation_authorization(
        terminal_path=authorization["final_terminal_path"],
        lock_path=authorization["candidate_lock_path"],
        effective=expected_effective,
    )
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise V12ChainRefusal(f"validation authorization mismatch: {key}")
    return json.loads(Path(authorization["candidate_lock_path"]).read_text())


def build_validation_terminal(
    *,
    path: str | Path,
    status: str,
    authorization: Mapping[str, Any],
    gates: Mapping[str, Any],
    terminal_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = terminal_v12(
        "validation-once",
        status,
        {
            **dict(terminal_extra or {}),
            "candidate_lock_path": authorization["candidate_lock_path"],
            "candidate_lock_sha256": authorization["candidate_lock_sha256"],
            "candidate_lock_digest": authorization["candidate_lock_digest"],
            "candidate_checkpoint_path": authorization["candidate_checkpoint_path"],
            "candidate_checkpoint_sha256": authorization["candidate_checkpoint_sha256"],
            "candidate_checkpoint": {
                "path": authorization["candidate_checkpoint_path"],
                "sha256": authorization["candidate_checkpoint_sha256"],
            },
            "threshold_digest": authorization["threshold_digest"],
            "effective": dict(authorization["effective"]),
            "data_hashes": dict(authorization["data_hashes"]),
            "gates": dict(gates),
            "rollback": "passed_validation_required_before_test_once",
        },
    )
    atomic_json_exclusive(payload, path)
    return payload


def test_authorization(
    *,
    validation_terminal_path: str | Path,
    lock_path: str | Path,
    effective: Mapping[str, str],
) -> dict[str, Any]:
    validation_path = Path(validation_terminal_path).resolve()
    lock_file = Path(lock_path).resolve()
    if not validation_path.is_file() or not lock_file.is_file():
        raise V12ChainRefusal("test chain absent")
    validation = json.loads(validation_path.read_text())
    lock = json.loads(lock_file.read_text())
    if "final_terminal_path" not in lock:
        raise V12ChainRefusal("lock missing final terminal")
    validate_final_chain(
        terminal_path=lock["final_terminal_path"],
        lock_path=lock_file,
        expected_effective=effective,
    )
    if (
        validation.get("status") != "passed"
        or validation.get("schema") != "r16_dscp_v12_validation_once_terminal_v1"
        or validation.get("candidate") != "r16_dscp_v12"
    ):
        raise V12ChainRefusal("validation not passed V12")
    lock_values = {
        "candidate_checkpoint_path": lock.get("checkpoint_path"),
        "candidate_checkpoint_sha256": lock.get("checkpoint_sha256"),
    }
    for key in (
        "candidate_lock_path",
        "candidate_lock_sha256",
        "candidate_lock_digest",
        "candidate_checkpoint_path",
        "candidate_checkpoint_sha256",
        "threshold_digest",
        "effective",
        "data_hashes",
    ):
        expected = (
            str(lock_file)
            if key == "candidate_lock_path"
            else sha256_file(lock_file)
            if key == "candidate_lock_sha256"
            else lock.get("lock_digest")
            if key == "candidate_lock_digest"
            else lock_values.get(key, lock.get(key))
        )
        if validation.get(key) != expected:
            raise V12ChainRefusal(f"validation/lock mismatch: {key}")
    return {
        "mode": "test-once",
        "validation_terminal_path": str(validation_path),
        "validation_terminal_sha256": sha256_file(validation_path),
        "candidate_lock_path": str(lock_file),
        "candidate_lock_sha256": sha256_file(lock_file),
        "candidate_lock_digest": lock["lock_digest"],
        "candidate_checkpoint_path": lock["checkpoint_path"],
        "candidate_checkpoint_sha256": lock["checkpoint_sha256"],
        "threshold_digest": lock["threshold_digest"],
        "effective": dict(effective),
        "data_hashes": dict(lock["data_hashes"]),
    }


def validate_test_authorization(
    authorization: Mapping[str, Any], expected_effective: Mapping[str, str]
) -> dict[str, Any]:
    expected = test_authorization(
        validation_terminal_path=authorization["validation_terminal_path"],
        lock_path=authorization["candidate_lock_path"],
        effective=expected_effective,
    )
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise V12ChainRefusal(f"test authorization mismatch: {key}")
    return expected


def build_test_terminal(
    *,
    path: str | Path,
    status: str,
    authorization: Mapping[str, Any],
    gates: Mapping[str, Any],
    terminal_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = terminal_v12(
        "test-once",
        status,
        {
            **dict(terminal_extra or {}),
            "validation_terminal_path": authorization["validation_terminal_path"],
            "validation_terminal_sha256": authorization["validation_terminal_sha256"],
            "candidate_lock_path": authorization["candidate_lock_path"],
            "candidate_lock_sha256": authorization["candidate_lock_sha256"],
            "candidate_lock_digest": authorization["candidate_lock_digest"],
            "candidate_checkpoint_path": authorization["candidate_checkpoint_path"],
            "candidate_checkpoint_sha256": authorization["candidate_checkpoint_sha256"],
            "threshold_digest": authorization["threshold_digest"],
            "effective": dict(authorization["effective"]),
            "data_hashes": dict(authorization["data_hashes"]),
            "gates": dict(gates),
            "rollback": "final_independent_test_no_retuning",
        },
    )
    atomic_json_exclusive(payload, path)
    return payload


__all__ = [
    "V12ChainRefusal",
    "V12ProductionBackend",
    "build_final_chain",
    "build_test_terminal",
    "build_validation_terminal",
    "complete_identity_v12",
    "terminal_v12",
    "test_authorization",
    "validate_final_chain",
    "validate_test_authorization",
    "validate_validation_authorization",
    "validation_authorization",
]
