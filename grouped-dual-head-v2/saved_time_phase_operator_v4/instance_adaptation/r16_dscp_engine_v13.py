"""V13 production candidate lock and sealed evaluation terminal chain."""
from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v7 import validate_checkpoint_identity
from .r16_dscp_engine_v12 import V12ProductionBackend
from .r16_dscp_training_v2 import (
    BindingRefusal,
    atomic_json_exclusive,
    sha256_file,
    validate_checkpoint_payload,
)


class V13ChainRefusal(BindingRefusal):
    """A sealed evaluation-chain binding is absent or has drifted."""


class RuntimeClosureRefusal(BindingRefusal):
    """The frozen repo-local runtime import closure is incomplete or unsafe."""


class DistributedContractRefusal(BindingRefusal):
    """The torchrun environment or distributed terminal protocol is invalid."""


class LongResumeRefusal(BindingRefusal):
    """The requested long resume is not the exact authorized interrupted run."""


class V13ProductionBackend(V12ProductionBackend):
    """V11 execution contracts with a V13-owned candidate spool."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spool = CandidateSpool(
            self.run_dir,
            run_digest=self.lineage.run_digest,
            rank=int(os.environ.get("RANK", "0")),
            owned_root="/dev/shm/r16_dscp_v13",
        )


def complete_identity_v13(**values: Any) -> dict[str, Any]:
    payload = {
        "schema": "r16_dscp_v13_checkpoint_identity_v1",
        "candidate": "r16_dscp_v13",
        **values,
    }
    payload["identity_digest"] = canonical_sha(payload)
    validate_checkpoint_identity(payload)
    return payload


def terminal_v13(mode: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(payload),
        "schema": f"r16_dscp_v13_{mode.replace('-', '_')}_terminal_v1",
        "candidate": "r16_dscp_v13",
        "mode": mode,
        "status": status,
        "decision": status,
    }


def _digest(payload: Mapping[str, Any]) -> str:
    return canonical_sha(dict(payload))


def _checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise V13ChainRefusal("checkpoint absent")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }


def _repo_module_map(root: Path) -> dict[str, Path]:
    """Map repo-local import names to real, non-symlink Python sources."""
    root = root.resolve(strict=True)
    result: dict[str, Path] = {}
    for search_root in (root, root / "src"):
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*.py")):
            relative = path.relative_to(search_root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            module_parts = list(relative.with_suffix("").parts)
            if module_parts[-1] == "__init__":
                module_parts.pop()
            if not module_parts:
                continue
            module = ".".join(module_parts)
            if module in result and result[module] != path:
                raise RuntimeClosureRefusal(f"ambiguous repo module: {module}")
            result[module] = path
    return result


def _safe_runtime_file(path: Path, root: Path) -> Path:
    root = root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise RuntimeClosureRefusal(f"runtime path escape: {path}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeClosureRefusal(f"runtime symlink rejected: {relative}")
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeClosureRefusal(f"runtime file absent: {relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeClosureRefusal(f"resolved runtime path escape: {relative}") from exc
    if not resolved.is_file() or resolved.suffix != ".py":
        raise RuntimeClosureRefusal(f"runtime helper is not Python: {relative}")
    return resolved


def _module_for_path(path: Path, root: Path) -> tuple[str, bool]:
    relative = path.relative_to(root)
    if relative.parts[:1] == ("src",):
        relative = Path(*relative.parts[1:])
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _resolve_from(module: str, is_package: bool, level: int, target: str | None) -> str:
    package = module.split(".") if is_package else module.split(".")[:-1]
    if level:
        keep = len(package) - (level - 1)
        if keep < 0:
            raise RuntimeClosureRefusal(f"relative import escapes package: {module}")
        prefix = package[:keep]
    else:
        prefix = []
    if target:
        prefix.extend(target.split("."))
    return ".".join(prefix)


def build_runtime_import_closure(
    *, root: str | Path, entrypoints: Sequence[str | Path]
) -> dict[str, Any]:
    """Build a deterministic recursive closure of all repo-local runtime imports."""
    repo = Path(root).resolve(strict=True)
    module_map = _repo_module_map(repo)
    pending = [_safe_runtime_file(Path(item), repo) for item in entrypoints]
    seen: set[Path] = set()
    dynamic: list[dict[str, Any]] = []

    def enqueue_module(name: str) -> None:
        if not name:
            return
        candidate = module_map.get(name)
        if candidate is not None:
            pending.append(_safe_runtime_file(candidate, repo))
        parts = name.split(".")
        for count in range(1, len(parts)):
            package = module_map.get(".".join(parts[:count]))
            if package is not None and package.name == "__init__.py":
                pending.append(_safe_runtime_file(package, repo))

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        module, is_package = _module_for_path(path, repo)
        try:
            tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise RuntimeClosureRefusal(f"runtime source unreadable: {path.relative_to(repo)}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(module, is_package, node.level, node.module)
                enqueue_module(base)
                for alias in node.names:
                    if alias.name != "*":
                        enqueue_module(".".join(part for part in (base, alias.name) if part))
            elif isinstance(node, ast.Call):
                target = node.func
                is_dynamic = (
                    isinstance(target, ast.Name) and target.id in {"__import__", "import_module"}
                ) or (
                    isinstance(target, ast.Attribute)
                    and target.attr == "import_module"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "importlib"
                )
                if not is_dynamic:
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise RuntimeClosureRefusal(
                        f"unbound dynamic import: {path.relative_to(repo)}:{getattr(node, 'lineno', 0)}"
                    )
                imported = str(node.args[0].value)
                if imported.startswith("."):
                    level = len(imported) - len(imported.lstrip("."))
                    imported = _resolve_from(module, is_package, level, imported.lstrip("."))
                enqueue_module(imported)
                dynamic.append(
                    {
                        "importer": path.relative_to(repo).as_posix(),
                        "line": int(getattr(node, "lineno", 0)),
                        "module": imported,
                    }
                )
    entries = [
        {
            "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(seen, key=lambda item: item.relative_to(repo).as_posix())
    ]
    dynamic.sort(key=lambda item: (item["importer"], item["line"], item["module"]))
    payload = {
        "schema": "r16_dscp_v13_runtime_import_closure_v1",
        "entrypoints": sorted(Path(item).resolve().relative_to(repo).as_posix() for item in entrypoints),
        "entries": entries,
        "dynamic_imports": dynamic,
    }
    payload["closure_sha256"] = canonical_sha(payload)
    return payload


def verify_runtime_import_closure(
    expected: Mapping[str, Any], *, root: str | Path
) -> dict[str, Any]:
    if expected.get("schema") != "r16_dscp_v13_runtime_import_closure_v1":
        raise RuntimeClosureRefusal("runtime closure schema mismatch")
    repo = Path(root).resolve(strict=True)
    observed = build_runtime_import_closure(
        root=repo,
        entrypoints=[repo / str(path) for path in expected.get("entrypoints", [])],
    )
    if dict(observed) != dict(expected):
        old = {row.get("path"): row.get("sha256") for row in expected.get("entries", [])}
        new = {row.get("path"): row.get("sha256") for row in observed.get("entries", [])}
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        drift = sorted(path for path in set(old) & set(new) if old[path] != new[path])
        raise RuntimeClosureRefusal(
            f"runtime closure mismatch added={added} removed={removed} drift={drift}"
        )
    return observed


@dataclass
class DistributedSession:
    world_size: int
    backend: str = "nccl"
    rank: int = 0
    local_rank: int = 0
    initialized: bool = False

    def initialize(self) -> "DistributedSession":
        expected = int(self.world_size)
        if expected not in {1, 4}:
            raise DistributedContractRefusal("world size must be one or four")
        if expected == 1:
            self.rank = self.local_rank = 0
            return self
        required = {name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        if any(value is None for value in required.values()):
            raise DistributedContractRefusal(f"torchrun environment incomplete: {required}")
        try:
            self.rank = int(required["RANK"] or "-1")
            self.local_rank = int(required["LOCAL_RANK"] or "-1")
            actual_world = int(required["WORLD_SIZE"] or "-1")
        except ValueError as exc:
            raise DistributedContractRefusal("torchrun rank environment is not integral") from exc
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if (
            actual_world != expected
            or not 0 <= self.rank < expected
            or not 0 <= self.local_rank < expected
            or len(visible) != expected
            or len(set(visible)) != expected
        ):
            raise DistributedContractRefusal(
                f"torchrun binding mismatch rank={self.rank} local_rank={self.local_rank} "
                f"world={actual_world} visible={visible}"
            )
        if torch.distributed.is_initialized():
            raise DistributedContractRefusal("foreign process group already initialized")
        if self.backend == "nccl":
            torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group(
            backend=self.backend, rank=self.rank, world_size=expected
        )
        self.initialized = True
        return self

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    def synchronize_failure(self, error: BaseException | None) -> list[dict[str, Any]]:
        local = None if error is None else {
            "rank": self.rank,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if not self.initialized:
            return [] if local is None else [local]
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        return [dict(item) for item in gathered if item is not None]

    def close(self) -> None:
        if self.initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.initialized = False


def inspect_long_resume(
    *,
    run_dir: str | Path,
    authorized_run_dir: str | Path,
    expected_identity: Mapping[str, Any],
    authorization_sha256: str,
) -> tuple[dict[str, Any], Any]:
    lexical = Path(os.path.abspath(run_dir))
    authorized = Path(os.path.abspath(authorized_run_dir))
    if lexical != authorized or lexical.is_symlink():
        raise LongResumeRefusal("resume directory is not the exact authorized run directory")
    if not lexical.is_dir() or not any(lexical.iterdir()):
        raise LongResumeRefusal("authorized resume directory is absent or empty")
    if (lexical / "terminal.json").exists():
        raise LongResumeRefusal("terminal long run cannot resume")
    identity_path = lexical / "run_identity.json"
    last_path = lexical / "last.pt"
    if identity_path.is_symlink() or last_path.is_symlink() or not identity_path.is_file() or not last_path.is_file():
        raise LongResumeRefusal("resume identity/last checkpoint is incomplete or symlinked")
    identity = json.loads(identity_path.read_text())
    if identity.get("authorization_sha256") != str(authorization_sha256):
        raise LongResumeRefusal("resume authorization hash mismatch")
    validate_checkpoint_identity(identity, expected_identity)
    try:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise LongResumeRefusal(f"last checkpoint unreadable: {exc}") from exc
    validate_checkpoint_payload(payload, expected_run_identity=expected_identity)
    validate_checkpoint_identity(payload["run_identity"], expected_identity)
    rng = payload.get("rng")
    if not isinstance(rng, Mapping) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise LongResumeRefusal("resume RNG state incomplete")
    progress = payload.get("progress")
    required = {"epoch", "global_step", "group_index", "best", "bad_epochs", "best_epoch"}
    if not isinstance(progress, Mapping) or set(progress) != required:
        raise LongResumeRefusal("resume progress state incomplete or extended")
    from .r16_dscp_engine_v4 import LongResumeState
    state = LongResumeState(**{key: progress[key] for key in required})
    if state.epoch < 0 or state.global_step < 0 or state.group_index < 0 or state.bad_epochs < 0:
        raise LongResumeRefusal("resume progress is outside its valid domain")
    sampler = payload.get("sampler", {}).get("order")
    if not isinstance(sampler, list) or not sampler or any(not isinstance(item, int) for item in sampler):
        raise LongResumeRefusal("resume sampler order invalid")
    return payload, state


class InclusiveBudgetClock:
    """Give V5LongRunner the pre-cache start on its first clock read."""

    def __init__(self, started_at: float, clock: Callable[[], float] = time.monotonic):
        self.started_at = float(started_at)
        self.clock = clock
        self.first = True

    def __call__(self) -> float:
        if self.first:
            self.first = False
            return self.started_at
        return float(self.clock())


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
        terminal = terminal_v13(
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
        raise V13ChainRefusal("passed final requires checkpoint")
    checkpoint = _checkpoint(checkpoint_path)
    threshold_digest = _digest(thresholds)
    terminal = terminal_v13(
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
        "schema": "r16_dscp_v13_candidate_lock_v1",
        "candidate": "r16_dscp_v13",
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
        raise V13ChainRefusal("final chain file absent")
    terminal = json.loads(terminal_file.read_text())
    lock = json.loads(lock_file.read_text())
    if (
        terminal.get("status") != "passed"
        or terminal.get("schema") != "r16_dscp_v13_final_train_confirm_terminal_v1"
        or lock.get("schema") != "r16_dscp_v13_candidate_lock_v1"
        or terminal.get("candidate") != "r16_dscp_v13"
        or lock.get("candidate") != "r16_dscp_v13"
    ):
        raise V13ChainRefusal("final chain not passed V13")
    if lock.get("lock_digest") != _digest(
        {key: value for key, value in lock.items() if key != "lock_digest"}
    ):
        raise V13ChainRefusal("lock digest mismatch")
    if (
        lock.get("final_terminal_path") != str(terminal_file)
        or lock.get("final_terminal_sha256") != sha256_file(terminal_file)
    ):
        raise V13ChainRefusal("lock terminal binding mismatch")
    for key in (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "threshold_digest",
        "effective",
        "data_hashes",
    ):
        if terminal.get(key) != lock.get(key):
            raise V13ChainRefusal(f"terminal/lock mismatch: {key}")
    if dict(lock["effective"]) != dict(expected_effective):
        raise V13ChainRefusal("effective drift")
    checkpoint = Path(lock["checkpoint_path"]).resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != lock["checkpoint_size_bytes"]
        or sha256_file(checkpoint) != lock["checkpoint_sha256"]
    ):
        raise V13ChainRefusal("current checkpoint drift")
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
            raise V13ChainRefusal(f"validation authorization mismatch: {key}")
    return json.loads(Path(authorization["candidate_lock_path"]).read_text())


def build_validation_terminal(
    *,
    path: str | Path,
    status: str,
    authorization: Mapping[str, Any],
    gates: Mapping[str, Any],
    terminal_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = terminal_v13(
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
        raise V13ChainRefusal("test chain absent")
    validation = json.loads(validation_path.read_text())
    lock = json.loads(lock_file.read_text())
    if "final_terminal_path" not in lock:
        raise V13ChainRefusal("lock missing final terminal")
    validate_final_chain(
        terminal_path=lock["final_terminal_path"],
        lock_path=lock_file,
        expected_effective=effective,
    )
    if (
        validation.get("status") != "passed"
        or validation.get("schema") != "r16_dscp_v13_validation_once_terminal_v1"
        or validation.get("candidate") != "r16_dscp_v13"
    ):
        raise V13ChainRefusal("validation not passed V13")
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
            raise V13ChainRefusal(f"validation/lock mismatch: {key}")
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
            raise V13ChainRefusal(f"test authorization mismatch: {key}")
    return expected


def build_test_terminal(
    *,
    path: str | Path,
    status: str,
    authorization: Mapping[str, Any],
    gates: Mapping[str, Any],
    terminal_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = terminal_v13(
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
    "DistributedContractRefusal",
    "DistributedSession",
    "InclusiveBudgetClock",
    "LongResumeRefusal",
    "RuntimeClosureRefusal",
    "V13ChainRefusal",
    "V13ProductionBackend",
    "build_final_chain",
    "build_test_terminal",
    "build_validation_terminal",
    "complete_identity_v13",
    "terminal_v13",
    "test_authorization",
    "validate_final_chain",
    "validate_test_authorization",
    "validate_validation_authorization",
    "validation_authorization",
    "build_runtime_import_closure",
    "inspect_long_resume",
    "verify_runtime_import_closure",
]
