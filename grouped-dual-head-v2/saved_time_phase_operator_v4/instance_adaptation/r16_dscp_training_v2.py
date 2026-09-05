"""Fail-closed training/checkpoint harness for the frozen R16-DSCP design."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .r16_dscp import RANK, frozen_loss


PARENT_PATH = "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/a3_alltrain_rel_l2_pde_pml_ic_current_vds_r4/run/checkpoints/epoch_0007.pt"
PARENT_SHA256 = "448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789"
BASIS_FILE_SHA256 = "8cab01344fc0a88f43fa232458127e87d2f8bbb63523b9389b927b39389c4786"
BASIS_TENSOR_SHA256 = "4d12d9985134e6b1f82127d269c1f111f1eb6f90b440872a8e05896d7126f7d5"
PANELS_SHA256 = "6d2f2facd86b449c058f9f824895f77977a23fbecf954c0de69144ec50162464"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class TrainingContractError(RuntimeError):
    pass


class BindingRefusal(TrainingContractError):
    pass


class StageGateRefusal(TrainingContractError):
    pass


@dataclass(frozen=True)
class FrozenOptimization:
    optimizer: str = "AdamW"
    learning_rate: float = 3.0e-3
    betas: tuple[float, float] = (0.9, 0.99)
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    batch_records: int = 1
    time_microblock: int = 32
    seed: int = 372
    bf16_autocast: bool = True
    deterministic: bool = True
    tf32: bool = False
    workers_per_gpu_maximum: int = 2


FROZEN_OPTIMIZATION = FrozenOptimization()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf8")
    ).hexdigest()


def require_cuda_environment(*, visible_devices: str) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise BindingRefusal("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before CUDA use")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(visible_devices):
        raise BindingRefusal("CUDA_VISIBLE_DEVICES does not match the frozen command")


def configure_determinism(seed: int = 372) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_optimizer(parameters: Sequence[nn.Parameter]) -> torch.optim.AdamW:
    cfg = FROZEN_OPTIMIZATION
    return torch.optim.AdamW(
        parameters,
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.epsilon,
        weight_decay=cfg.weight_decay,
    )


def cosine_learning_rate(epoch: int, maximum_epochs: int = 20) -> float:
    if not 0 <= int(epoch) < int(maximum_epochs):
        raise ValueError("epoch lies outside the frozen long schedule")
    fraction = int(epoch) / max(int(maximum_epochs) - 1, 1)
    return 3.0e-5 + 0.5 * (3.0e-3 - 3.0e-5) * (1.0 + math.cos(math.pi * fraction))


def exact_microblocked_loss(
    corrected_future: torch.Tensor,
    truth_future: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    time_microblock: int = 32,
) -> Mapping[str, torch.Tensor]:
    """Numerically equivalent full-future loss without future-frame sampling."""
    prediction = torch.as_tensor(corrected_future).float()
    truth = torch.as_tensor(truth_future, device=prediction.device).float()
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("future tensors must match [B,T,H,W]")
    block = int(time_microblock)
    if block != 32:
        raise TrainingContractError("time_microblock is frozen at 32")
    frame_sum = prediction.new_zeros(())
    late_sum = prediction.new_zeros(())
    temporal_num = prediction.new_zeros(())
    temporal_den = prediction.new_zeros(())
    count = prediction.shape[0] * prediction.shape[1]
    late_start = (2 * prediction.shape[1]) // 3
    late_count = prediction.shape[0] * (prediction.shape[1] - late_start)
    previous_error = previous_truth = None
    for start in range(0, prediction.shape[1], block):
        stop = min(start + block, prediction.shape[1])
        pred = prediction[:, start:stop]
        target = truth[:, start:stop]
        error = pred - target
        denominator = target.square().sum(dim=(2, 3)).clamp_min(1.0e-30)
        frame_values = error.square().sum(dim=(2, 3)) / denominator
        frame_sum = frame_sum + frame_values.sum()
        overlap = max(start, late_start)
        if overlap < stop:
            late_sum = late_sum + frame_values[:, overlap - start :].sum()
        if previous_error is not None:
            de = error[:, :1] - previous_error
            dt = target[:, :1] - previous_truth
            temporal_num = temporal_num + de.square().sum()
            temporal_den = temporal_den + dt.square().sum()
        if error.shape[1] > 1:
            temporal_num = temporal_num + (error[:, 1:] - error[:, :-1]).square().sum()
            temporal_den = temporal_den + (target[:, 1:] - target[:, :-1]).square().sum()
        previous_error, previous_truth = error[:, -1:], target[:, -1:]
    frame_term = frame_sum / count
    late_term = late_sum / late_count
    temporal_term = temporal_num / temporal_den.clamp_min(1.0e-30)
    coefficient_term = torch.as_tensor(coefficients).float().square().mean()
    total = frame_term + 0.25 * late_term + 0.10 * temporal_term + 1.0e-4 * coefficient_term
    output = {
        "total": total,
        "frame_relative_l2_squared": frame_term,
        "late_third_relative_l2_squared": late_term,
        "temporal_difference": temporal_term,
        "normalized_coefficient_energy": coefficient_term,
    }
    reference = frozen_loss(prediction, truth, coefficients)
    for key in output:
        if not torch.allclose(output[key], reference[key], rtol=2.0e-6, atol=1.0e-8):
            raise TrainingContractError(f"microblocked loss differs from direct frozen loss: {key}")
    return output


@torch.no_grad()
def weighted_coefficient_target(
    basis: torch.Tensor,
    parent: torch.Tensor,
    truth: torch.Tensor,
    *,
    k1: int,
    time_microblock: int = 32,
) -> torch.Tensor:
    """Offline train-only weighted LS target; caller must not serialize output."""
    temporal = torch.as_tensor(basis, dtype=torch.float64, device="cpu")
    predicted = torch.as_tensor(parent, dtype=torch.float32, device="cpu")
    target = torch.as_tensor(truth, dtype=torch.float32, device="cpu")
    if predicted.shape != target.shape or predicted.ndim != 3 or temporal.shape != (401, RANK):
        raise ValueError("basis and one full record must be [401,16] and [401,H,W]")
    start = int(k1) + 1
    gram = torch.zeros((RANK, RANK), dtype=torch.float64)
    rhs = torch.zeros((RANK, predicted.shape[1] * predicted.shape[2]), dtype=torch.float64)
    for left in range(start, 401, int(time_microblock)):
        right = min(left + int(time_microblock), 401)
        b = temporal[left:right]
        truth_block = target[left:right].reshape(right - left, -1).double()
        residual = (target[left:right] - predicted[left:right]).reshape(right - left, -1).double()
        weights = truth_block.square().sum(dim=1).clamp_min(1.0e-30).reciprocal()
        gram += b.T @ (weights[:, None] * b)
        rhs += b.T @ (weights[:, None] * residual)
    condition = float(torch.linalg.cond(gram))
    if not math.isfinite(condition) or condition > 1.0e6:
        raise TrainingContractError("weighted coefficient normal matrix is unstable")
    coefficients = torch.linalg.solve(gram, rhs).float().reshape(RANK, *predicted.shape[1:])
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("weighted coefficient target is non-finite")
    return coefficients


def selection_metric(predictions: Sequence[torch.Tensor], truths: Sequence[torch.Tensor]) -> float:
    """Arithmetic mean of per-record full-future relative L2, lower is better."""
    if len(predictions) != len(truths) or not predictions:
        raise ValueError("matched nonempty records are required")
    values = []
    for prediction, truth in zip(predictions, truths):
        p = torch.as_tensor(prediction).double()
        t = torch.as_tensor(truth).double()
        values.append(float(torch.linalg.vector_norm(p - t) / torch.linalg.vector_norm(t).clamp_min(1e-30)))
    return float(sum(values) / len(values))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def predictor_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    run_identity: Mapping[str, Any],
    sampler_order: Sequence[int],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "r16_dscp_v2_checkpoint_v1",
        "predictor_parameters": predictor_parameter_state(model),
        "basis": {"file_sha256": BASIS_FILE_SHA256, "tensor_sha256": BASIS_TENSOR_SHA256},
        "optimizer": optimizer.state_dict(),
        "bf16_scaler": {"enabled": False, "state": {}},
        "rng": capture_rng_state(),
        "sampler": {"order": [int(value) for value in sampler_order]},
        "progress": dict(progress),
        "run_identity": dict(run_identity),
        "parent": {"path": PARENT_PATH, "sha256": PARENT_SHA256},
    }
    validate_checkpoint_payload(payload, expected_run_identity=run_identity)
    return payload


def validate_checkpoint_payload(
    payload: Mapping[str, Any], *, expected_run_identity: Mapping[str, Any]
) -> None:
    if payload.get("schema") != "r16_dscp_v2_checkpoint_v1":
        raise BindingRefusal("checkpoint schema mismatch")
    if payload.get("basis") != {
        "file_sha256": BASIS_FILE_SHA256,
        "tensor_sha256": BASIS_TENSOR_SHA256,
    }:
        raise BindingRefusal("checkpoint basis binding mismatch")
    if payload.get("parent") != {"path": PARENT_PATH, "sha256": PARENT_SHA256}:
        raise BindingRefusal("checkpoint parent binding mismatch")
    if dict(payload.get("run_identity", {})) != dict(expected_run_identity):
        raise BindingRefusal("checkpoint run identity mismatch")
    if not isinstance(payload.get("predictor_parameters"), Mapping):
        raise BindingRefusal("predictor parameter state missing")
    if any(key in payload for key in ("parent_state", "parent_model", "basis_tensor", "fields")):
        raise BindingRefusal("checkpoint embeds forbidden parent/basis/field content")


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> int:
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _directory_fsync(path.parent)
    finally:
        partial.unlink(missing_ok=True)
    return int(path.stat().st_size)


def save_best_last(
    payload: Mapping[str, Any], run_dir: str | Path, *, is_best: bool
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    last = root / "last.pt"
    size = atomic_torch_save(payload, last)
    best = root / "best.pt"
    if is_best:
        partial = root / f".best.pt.partial-{os.getpid()}"
        try:
            os.link(last, partial)
            os.replace(partial, best)
            _directory_fsync(root)
        finally:
            partial.unlink(missing_ok=True)
    extra = [path.name for path in root.glob("*.pt") if path.name not in {"best.pt", "last.pt"}]
    if extra:
        raise TrainingContractError(f"checkpoint retention violation: {extra}")
    return {
        "last": str(last),
        "best": str(best) if best.exists() else None,
        "size_bytes": size,
        "hardlinked": bool(best.exists() and best.stat().st_ino == last.stat().st_ino),
    }


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_run_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise BindingRefusal(f"checkpoint unreadable/corrupt: {exc}") from exc
    validate_checkpoint_payload(payload, expected_run_identity=expected_run_identity)
    named = dict(model.named_parameters())
    if set(payload["predictor_parameters"]) != set(named):
        raise BindingRefusal("predictor parameter keys mismatch")
    with torch.no_grad():
        for name, value in payload["predictor_parameters"].items():
            if named[name].shape != value.shape:
                raise BindingRefusal(f"predictor parameter shape mismatch: {name}")
            named[name].copy_(value.to(named[name]))
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return payload


def atomic_json_exclusive(payload: Mapping[str, Any], destination: str | Path) -> int:
    path = Path(destination).resolve()
    if path.exists():
        raise FileExistsError(f"refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf8")
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _directory_fsync(path.parent)
    finally:
        partial.unlink(missing_ok=True)
    return len(encoded)


def write_failure_terminal(
    run_dir: str | Path, *, mode: str, reason: str, run_identity: Mapping[str, Any]
) -> Path:
    path = Path(run_dir) / "terminal.json"
    atomic_json_exclusive(
        {
            "schema": "r16_dscp_v2_terminal_v1",
            "status": "failed",
            "mode": str(mode),
            "reason": str(reason),
            "run_identity": dict(run_identity),
            "validation_authorized": False,
            "test_authorized": False,
            "rollback": "preserve minimal best/last/log; do not touch parent/protected files",
        },
        path,
    )
    return path


def require_stage_terminal(path: str | Path, *, required_status: str = "passed") -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise StageGateRefusal(f"required terminal absent: {target}")
    payload = json.loads(target.read_text(encoding="utf8"))
    if payload.get("status") != required_status:
        raise StageGateRefusal(f"required terminal did not pass: {target}")
    return payload


def claim_once_token(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_json_exclusive(payload, path)


def space_gate(checkpoint_bytes: int, path: str | Path) -> dict[str, Any]:
    free = int(shutil.disk_usage(Path(path)).free)
    required = 2 * 1024**3 + 3 * int(checkpoint_bytes) + 64 * 1024**2
    return {"free_bytes": free, "required_bytes": required, "passed": free >= required}


__all__ = [
    "BASIS_FILE_SHA256", "BASIS_TENSOR_SHA256", "BindingRefusal",
    "CUBLAS_WORKSPACE_CONFIG", "FROZEN_OPTIMIZATION", "FrozenOptimization",
    "PANELS_SHA256", "PARENT_PATH", "PARENT_SHA256", "StageGateRefusal",
    "TrainingContractError", "atomic_json_exclusive", "atomic_torch_save",
    "capture_rng_state", "checkpoint_payload", "claim_once_token",
    "configure_determinism", "cosine_learning_rate", "exact_microblocked_loss",
    "load_checkpoint", "make_optimizer", "predictor_parameter_state",
    "require_cuda_environment", "require_stage_terminal", "restore_rng_state",
    "save_best_last", "selection_metric", "sha256_file", "space_gate",
    "validate_checkpoint_payload", "weighted_coefficient_target", "write_failure_terminal",
]
