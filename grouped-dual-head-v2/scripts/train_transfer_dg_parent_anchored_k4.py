#!/usr/bin/env python3
"""Fresh-init train-only K4-window parent-anchored correction pilot.

The sole algorithmic change from r3 is four distinct windows per record update.
This module deliberately does not import the historical parent-anchored trainer,
cache, or loss wrapper.  It never accepts a checkpoint/model-state input.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.parent_anchored_block import (  # noqa: E402
    ParentAnchoredBlockCorrector,
    parameter_count,
    render_retained_rfft_frames,
)
from saved_time_phase_operator_v4.parent_anchored_relative_loss import (  # noqa: E402
    record_energy_squared_loss,
    unbiased_window_weights,
)


CANDIDATE = "transfer_dg_parent_anchored_k4_r4_20260904"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
STARTS = (2, 34, 66, 98, 130, 162, 194, 226, 258, 290, 322, 354, 386)
OFFSETS = (0, 3, 6, 9)
SEED = 372
EPOCHS = 3
UPDATES_PER_EPOCH = 68
TOTAL_UPDATES = 204
SCHEDULER_T_MAX = 816
EXPECTED_LR_UPDATE_204 = 0.00025650535700620246
PHYSICAL_SCALE = 1.0e-8
MODEL_CONFIG = {
    "condition_channels": 9,
    "block_size": 32,
    "width": 32,
    "spectral_rank": 16,
    "modes": 16,
    "depth": 4,
    "maximum_correction_ratio": 0.5,
    "boundary_blend_frames": 4,
    "hard_free_surface": True,
    "parameter_count": 91396,
}
LOSS_CONFIG = {
    "derivative_weight": 0.01,
    "spectral_weight": 0.02,
    "nonworse_weight": 0.30,
    "correction_weight": 0.01,
    "spectral_floor_fraction": 0.005,
}
EXPECTED_ROLE_DIGESTS = {
    "fit": {
        "sample_ids": "64e647a045d636eb57616a7b0fe51d396c30a2417264ce5c019836c146eb8706",
        "groups": "5d0038d28a752e9e0804189c44f7ab70441a300757ba2350cea4c558b6899235",
        "sample_hashes": "0e5eecac4fff7d09f2c999cfb39b76739b7683841bdb82f6aced2131ab586dd1",
    },
    "calibration": {
        "sample_ids": "25d738d33c666a1f71ec6bcf3cd8ed4a17d885c0f235cefc1ce589af6f0cbb64",
        "groups": "0cbb27953161ab8034d89d10e48484add4f348ae799e76a7e62d7a09cdbe92dc",
        "sample_hashes": "284e872f3f53a8c3de967bdf5a4e0c3d516ad1c0467df8bf81862d9d8da76929",
    },
    "confirmation": {
        "sample_ids": "3c7be9c6d100d1cc585acc20ae436cea19c8efa13e7e98cd05825eeb91c6acff",
        "groups": "62ee37373613957ca24a403c159be12513b7219429e464f738e681780235d9c3",
        "sample_hashes": "d1221b899aa2b6fb09629cf7881c853c0f8b60df33885032b3ebc6efcd2c29b3",
    },
}
EXPECTED_ROLE_COUNTS = {"fit": 68, "calibration": 34, "confirmation": 34}
EXPECTED_FAMILY_COUNTS = {
    "fit": {"uniform": 16, "layered": 16, "anomaly": 16, "marmousi": 20},
    "calibration": {"uniform": 8, "layered": 8, "anomaly": 8, "marmousi": 10},
}
MECHANISM_PANEL = {
    "train_uniform_00014",
    "train_layered_00696",
    "train_anomaly_00096",
    "train_marmousi_00230",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sorted_string_digest(values: Sequence[Any]) -> str:
    ordered = sorted(str(value) for value in values)
    encoded = json.dumps(
        ordered, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging path: {staging}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with staging.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def atomic_checkpoint(payload: Mapping[str, Any], path: Path, *, immutable: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable checkpoint exists: {path}")
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale checkpoint staging path: {staging}")
    try:
        torch.save(dict(payload), staging)
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def stable_block_index(sample_id: str, epoch: int, block_count: int = len(STARTS)) -> int:
    digest = hashlib.sha256(f"pab32:{sample_id}:{int(epoch)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % int(block_count)


def k4_indices(anchor: int) -> tuple[int, int, int, int]:
    indices = tuple((int(anchor) + offset) % len(STARTS) for offset in OFFSETS)
    if len(set(indices)) != 4:
        raise RuntimeError("K4 offsets did not produce four distinct windows")
    return indices  # type: ignore[return-value]


def k4_starts(sample_id: str, epoch: int) -> tuple[int, int, int, int]:
    return tuple(STARTS[index] for index in k4_indices(stable_block_index(sample_id, epoch)))


@dataclass(frozen=True)
class TruthRecord:
    purpose: str
    role: str
    sample_id: str
    group_id: str
    sample_sha256: str
    start: int
    stop: int


class TruthAccessGuard:
    """Fail-closed authorization and ledger for every wavefield read."""

    PURPOSE_ROLE = {
        "mechanism": "fit",
        "smoke": "fit",
        "training": "fit",
        "calibration": "calibration",
    }
    ALWAYS_FORBIDDEN_ROLES = {"confirmation", "validation", "test_id"}

    def __init__(self, *, mechanism_panel: set[str] | None = None) -> None:
        self.mechanism_panel = set(MECHANISM_PANEL if mechanism_panel is None else mechanism_panel)
        self._records: list[TruthRecord] = []

    def authorize(
        self,
        *,
        purpose: str,
        role: str,
        sample_id: str,
        group_id: str,
        sample_sha256: str,
        start: int,
        stop: int,
        time_count: int = 401,
    ) -> None:
        purpose = str(purpose)
        role = str(role)
        sample_id = str(sample_id)
        if role in self.ALWAYS_FORBIDDEN_ROLES:
            raise PermissionError(f"future truth is permanently forbidden for role={role}")
        expected_role = self.PURPOSE_ROLE.get(purpose)
        if expected_role is None or role != expected_role:
            raise PermissionError(f"truth purpose/role mismatch: {purpose}/{role}")
        if purpose == "mechanism" and sample_id not in self.mechanism_panel:
            raise PermissionError("mechanism truth is restricted to the frozen four-record panel")
        if not 0 <= int(start) < int(stop) <= int(time_count):
            raise ValueError("truth slice outside registered time axis")
        self._records.append(
            TruthRecord(
                purpose=purpose,
                role=role,
                sample_id=sample_id,
                group_id=str(group_id),
                sample_sha256=str(sample_sha256),
                start=int(start),
                stop=int(stop),
            )
        )

    def summary(self) -> dict[str, Any]:
        purposes = Counter(record.purpose for record in self._records)
        roles = Counter(record.role for record in self._records)
        sample_ids = {record.sample_id for record in self._records}
        groups = {record.group_id for record in self._records}
        hashes = {record.sample_sha256 for record in self._records}
        starts = [record.start for record in self._records]
        stops = [record.stop for record in self._records]
        return {
            "authorized_call_count": len(self._records),
            "purpose_call_counts": dict(sorted(purposes.items())),
            "role_call_counts": dict(sorted(roles.items())),
            "unique_sample_count": len(sample_ids),
            "unique_group_count": len(groups),
            "unique_sample_hash_count": len(hashes),
            "unique_sample_id_digest": sorted_string_digest(tuple(sample_ids)),
            "unique_group_digest": sorted_string_digest(tuple(groups)),
            "unique_sample_hash_digest": sorted_string_digest(tuple(hashes)),
            "slice_start_min": min(starts) if starts else None,
            "slice_stop_max_exclusive": max(stops) if stops else None,
            "confirmation_opened": False,
            "validation_opened": False,
            "test_id_opened": False,
        }


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class GuardedParentCache:
    """Independent parent cache with guarded train-only truth access."""

    def __init__(
        self,
        path: Path,
        source_h5: Path,
        *,
        purpose: str,
        guard: TruthAccessGuard,
    ) -> None:
        self.path = path.resolve()
        self.source_path = source_h5.resolve()
        self.purpose = str(purpose)
        self.guard = guard
        self.cache = h5py.File(self.path, "r", swmr=True)
        self.source = h5py.File(self.source_path, "r", swmr=True)
        self._energy_cache: dict[int, tuple[float, float]] = {}
        try:
            if self.cache.attrs.get("schema", "") != "transfer_dg_parent_anchored_block_cache_v1":
                raise RuntimeError("unexpected cache schema")
            if self.cache.attrs.get("status", "") != "complete" or self.cache.attrs.get("split", "") != "train":
                raise RuntimeError("cache is not complete train-only data")
            if self.cache.attrs.get("validation_opened") or self.cache.attrs.get("test_id_opened"):
                raise RuntimeError("cache sealed-split flag drift")
            self.sample_ids = self.cache["sample_id"].asstr()[:]
            self.families = self.cache["family"].asstr()[:]
            self.roles = self.cache["role"].asstr()[:]
            self.group_ids = self.cache["group_id"].asstr()[:]
            self.source_indices = np.asarray(self.cache["source_index"], dtype=np.int64)
            self.sample_hashes = np.asarray(
                [_decode(self.source["sample_sha256"][int(index)]) for index in self.source_indices],
                dtype=object,
            )
            self.time_count = int(self.cache.attrs["time_count"])
            if self.time_count != 401:
                raise RuntimeError("time-count drift")
            for local, source_index in enumerate(self.source_indices):
                if (
                    _decode(self.source["sample_id"][int(source_index)]) != self.sample_ids[local]
                    or _decode(self.source["split"][int(source_index)]) != "train"
                    or _decode(self.source["group_id"][int(source_index)]) != self.group_ids[local]
                    or _decode(self.source["medium_type"][int(source_index)]) != self.families[local]
                ):
                    raise RuntimeError("cache/source metadata binding drift")
            self.role_audit = audit_role_metadata(self)
        except Exception:
            self.close()
            raise

    def positions(self, role: str) -> list[int]:
        return np.flatnonzero(self.roles == str(role)).astype(np.int64).tolist()

    def parent_and_condition(
        self, position: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.from_numpy(
            np.asarray(self.cache["parent_coefficients"][position], dtype=np.float32)
        )[None].to(device)
        condition = torch.from_numpy(
            np.asarray(self.cache["condition"][position], dtype=np.float32)
        )[None].to(device)
        return coefficients, condition

    def _authorize(self, position: int, start: int, stop: int, purpose: str | None) -> None:
        self.guard.authorize(
            purpose=self.purpose if purpose is None else purpose,
            role=str(self.roles[position]),
            sample_id=str(self.sample_ids[position]),
            group_id=str(self.group_ids[position]),
            sample_sha256=str(self.sample_hashes[position]),
            start=start,
            stop=stop,
            time_count=self.time_count,
        )

    def truth(
        self,
        position: int,
        start: int,
        stop: int,
        device: torch.device,
        *,
        purpose: str | None = None,
    ) -> torch.Tensor:
        self._authorize(position, start, stop, purpose)
        values = np.asarray(
            self.source["wavefield"][int(self.source_indices[position]), int(start) : int(stop)],
            dtype=np.float32,
        )
        return torch.from_numpy(values / PHYSICAL_SCALE)[None].to(device)

    def full_future_energies(
        self, position: int, *, purpose: str | None = None
    ) -> tuple[float, float]:
        key = int(position)
        if key not in self._energy_cache:
            total_energy = 0.0
            delta_energy = 0.0
            previous = None
            for start in range(2, self.time_count, 64):
                stop = min(start + 64, self.time_count)
                self._authorize(position, start, stop, purpose)
                values = np.asarray(
                    self.source["wavefield"][int(self.source_indices[position]), start:stop],
                    dtype=np.float32,
                ) / PHYSICAL_SCALE
                tensor = torch.from_numpy(values).double()
                total_energy += float(tensor.square().sum())
                if previous is not None:
                    delta_energy += float((tensor[0] - previous).square().sum())
                if tensor.shape[0] > 1:
                    delta_energy += float((tensor[1:] - tensor[:-1]).square().sum())
                previous = tensor[-1]
            if not math.isfinite(total_energy + delta_energy) or min(total_energy, delta_energy) <= 0.0:
                raise FloatingPointError("invalid complete-record train energy")
            self._energy_cache[key] = (total_energy, delta_energy)
        return self._energy_cache[key]

    def close(self) -> None:
        for name in ("cache", "source"):
            handle = getattr(self, name, None)
            if handle is not None and handle.id.valid:
                handle.close()


def cross_role_overlap_counts(
    role_sets: Mapping[str, Mapping[str, set[str]]]
) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    for left, right in (
        ("fit", "calibration"),
        ("fit", "confirmation"),
        ("calibration", "confirmation"),
    ):
        for field in ("sample_ids", "groups", "sample_hashes"):
            overlaps[f"{left}_{right}_{field}"] = len(
                role_sets[left][field] & role_sets[right][field]
            )
    return overlaps


def audit_role_metadata(cache: GuardedParentCache) -> dict[str, Any]:
    audit: dict[str, Any] = {"roles": {}}
    role_sets: dict[str, dict[str, set[str]]] = {}
    for role in ("fit", "calibration", "confirmation"):
        positions = cache.positions(role)
        ids = [str(cache.sample_ids[index]) for index in positions]
        groups = [str(cache.group_ids[index]) for index in positions]
        hashes = [str(cache.sample_hashes[index]) for index in positions]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"duplicate sample_id within role: {role}")
        if len(set(hashes)) != len(hashes):
            raise RuntimeError(f"duplicate sample_sha256 within role: {role}")
        families = Counter(str(cache.families[index]) for index in positions)
        digests = {
            "sample_ids": sorted_string_digest(ids),
            "groups": sorted_string_digest(tuple(set(groups))),
            "sample_hashes": sorted_string_digest(hashes),
        }
        if len(positions) != EXPECTED_ROLE_COUNTS[role] or digests != EXPECTED_ROLE_DIGESTS[role]:
            raise RuntimeError(f"role metadata drift: {role}")
        if role in EXPECTED_FAMILY_COUNTS and dict(families) != EXPECTED_FAMILY_COUNTS[role]:
            raise RuntimeError(f"role family census drift: {role}")
        role_sets[role] = {"sample_ids": set(ids), "groups": set(groups), "sample_hashes": set(hashes)}
        audit["roles"][role] = {
            "record_count": len(positions),
            "group_count": len(set(groups)),
            "family_counts": dict(families),
            "digests": digests,
            "truth_opened": False,
        }
    overlaps = cross_role_overlap_counts(role_sets)
    if any(overlaps.values()):
        raise RuntimeError(f"cross-role metadata overlap: {overlaps}")
    audit["overlap_counts"] = overlaps
    audit["confirmation_exclusion_digest_verified"] = True
    return audit


def parameter_manifest(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "numel": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
    ]


def parameter_manifest_digest(model: torch.nn.Module) -> str:
    return canonical_sha256(parameter_manifest(model))


def initial_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_bytes({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def fresh_model_cpu(seed: int = SEED) -> tuple[ParentAnchoredBlockCorrector, str, str]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ParentAnchoredBlockCorrector(
        condition_channels=MODEL_CONFIG["condition_channels"],
        block_size=MODEL_CONFIG["block_size"],
        width=MODEL_CONFIG["width"],
        spectral_rank=MODEL_CONFIG["spectral_rank"],
        modes=MODEL_CONFIG["modes"],
        depth=MODEL_CONFIG["depth"],
        maximum_correction_ratio=MODEL_CONFIG["maximum_correction_ratio"],
        boundary_blend_frames=MODEL_CONFIG["boundary_blend_frames"],
        activation_checkpointing=True,
        hard_free_surface=MODEL_CONFIG["hard_free_surface"],
    ).cpu()
    if parameter_count(model) != MODEL_CONFIG["parameter_count"]:
        raise RuntimeError("fresh model parameter-count drift")
    return model, initial_state_digest(model), parameter_manifest_digest(model)


def assert_training_environment() -> dict[str, Any]:
    observed = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    expected = {
        "CUBLAS_WORKSPACE_CONFIG": None,
        "CUDA_VISIBLE_DEVICES": "0",
        "deterministic_algorithms": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
    }
    if observed != expected:
        raise RuntimeError(f"r3-default training environment drift: {observed}")
    return observed


def render_parent_context(
    coefficients: torch.Tensor, *, start: int, block_size: int = 32, time_count: int = 401
) -> tuple[torch.Tensor, int]:
    valid = min(int(block_size), int(time_count) - int(start))
    if start < 2 or valid <= 0:
        raise ValueError("invalid block start")
    indices = list(range(start - 2, start + valid))
    rendered = render_retained_rfft_frames(
        coefficients, time_count=time_count, frame_indices=indices
    )
    history, future = rendered[:, :2], rendered[:, 2:]
    if valid < block_size:
        future = torch.cat((future, future[:, -1:].expand(-1, block_size - valid, -1, -1)), 1)
    return torch.cat((history, future), 1), valid


def training_window(
    model: ParentAnchoredBlockCorrector,
    coefficients: torch.Tensor,
    condition: torch.Tensor,
    cache: GuardedParentCache,
    position: int,
    *,
    loss_start: int,
    purpose: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block = model.block_size
    total = cache.time_count
    if loss_start > 2:
        warm_start = loss_start - block
        warm_context, warm_valid = render_parent_context(
            coefficients, start=warm_start, block_size=block, time_count=total
        )
        with torch.no_grad():
            warm = model.forward_block(
                warm_context,
                warm_context[:, :2],
                condition,
                block_start=warm_start,
                total_frames=total,
            )
        corrected_history = warm[:, warm_valid - 2 : warm_valid].detach()
    else:
        initial_context, _ = render_parent_context(
            coefficients, start=2, block_size=block, time_count=total
        )
        corrected_history = initial_context[:, :2]
    predictions, targets, parents = [], [], []
    for block_index in range(2):
        start = int(loss_start) + block_index * block
        if start >= total:
            break
        context, valid = render_parent_context(
            coefficients, start=start, block_size=block, time_count=total
        )
        prediction = model.forward_block(
            context, corrected_history, condition, block_start=start, total_frames=total
        )
        predictions.append(prediction[:, :valid])
        targets.append(cache.truth(position, start, start + valid, prediction.device, purpose=purpose))
        parents.append(context[:, 2 : 2 + valid])
        corrected_history = prediction[:, valid - 2 : valid]
    return tuple(torch.cat(values, 1) for values in (predictions, targets, parents))


def window_loss(
    model: ParentAnchoredBlockCorrector,
    cache: GuardedParentCache,
    position: int,
    start: int,
    device: torch.device,
    *,
    purpose: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    coefficients, condition = cache.parent_and_condition(position, device)
    prediction, target, parent = training_window(
        model, coefficients, condition, cache, position, loss_start=start, purpose=purpose
    )
    frame, delta = unbiased_window_weights(
        loss_start=start,
        sample_length=int(prediction.shape[1]),
        time_count=cache.time_count,
        block_size=model.block_size,
        rollout_blocks=2,
        device=device,
        dtype=prediction.dtype,
    )
    energy, delta_energy = cache.full_future_energies(position, purpose=purpose)
    loss, terms = record_energy_squared_loss(
        prediction,
        target,
        parent,
        frame_weights=frame,
        delta_weights=delta,
        full_target_energy=energy,
        full_target_delta_energy=delta_energy,
        **LOSS_CONFIG,
    )
    return loss, terms


def flatten_current_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("unexpected missing accumulated gradient")
    value = torch.cat([parameter.grad.detach().cpu().double().reshape(-1) for parameter in parameters])
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite accumulated gradient")
    return value


def flatten_autograd(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=True)
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("unexpected missing individual gradient")
    value = torch.cat([gradient.detach().cpu().double().reshape(-1) for gradient in gradients])
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite individual gradient")
    return value


def vector_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = left.double().flatten(), right.double().flatten()
    left_norm, right_norm = float(left.norm()), float(right.norm())
    denominator = left_norm * right_norm
    if denominator < 1.0e-24:
        cosine = 1.0 if torch.equal(left, right) else 0.0
    else:
        cosine = float(torch.dot(left, right) / denominator)
    return {
        "cosine": cosine,
        "relative_norm_difference": float((left - right).norm()) / max(right_norm, 1.0e-12),
        "norm_ratio": left_norm / max(right_norm, 1.0e-12),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def k4_accumulate_and_step(
    *,
    loss_factory: Callable[[int], torch.Tensor],
    optimizer: Any,
    scheduler: Any,
    parameters: Sequence[torch.nn.Parameter],
    clipper: Callable[[Sequence[torch.nn.Parameter], float], Any],
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    scalar_losses = []
    for window_index in range(4):
        loss = loss_factory(window_index)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite K4 window loss")
        scalar_losses.append(float(loss.detach()))
        (loss / 4.0).backward()
        del loss
    gradient_norm = clipper(parameters, 1.0)
    if not math.isfinite(float(gradient_norm)):
        raise FloatingPointError("non-finite K4 gradient norm")
    optimizer.step()
    scheduler.step()
    return {"mean_loss": float(np.mean(scalar_losses)), "gradient_norm": float(gradient_norm)}


def expected_lr(update: int) -> float:
    return 3.0e-6 + 0.5 * (3.0e-4 - 3.0e-6) * (
        1.0 + math.cos(math.pi * int(update) / SCHEDULER_T_MAX)
    )


def nested_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, Mapping):
        return all(nested_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_tensors_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def checkpoint_purpose_allows_pilot_initialization(payload: Mapping[str, Any]) -> bool:
    return payload.get("purpose") != "smoke_only" and not bool(
        payload.get("pilot_init_forbidden", False)
    )


def assert_checkpoint_not_pilot_initialization(payload: Mapping[str, Any]) -> None:
    if not checkpoint_purpose_allows_pilot_initialization(payload):
        raise PermissionError("smoke-only checkpoint is forbidden as pilot initialization")


def relative_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return prediction.sub(target).double().flatten(1).norm(dim=1) / target.double().flatten(1).norm(dim=1).clamp_min(1.0e-16)


@torch.inference_mode()
def evaluate_calibration(
    model: ParentAnchoredBlockCorrector,
    cache: GuardedParentCache,
    positions: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for position in positions:
        coefficients, condition = cache.parent_and_condition(position, device)
        parent = render_retained_rfft_frames(
            coefficients,
            time_count=cache.time_count,
            frame_indices=torch.arange(cache.time_count, device=device),
        )
        parent = parent.clone()
        parent[..., 0, :] = 0.0
        prediction = model.rollout(parent, condition)
        target = cache.truth(position, 0, cache.time_count, device, purpose="calibration")
        candidate = float(relative_rows(prediction[:, 2:], target[:, 2:])[0])
        baseline = float(relative_rows(parent[:, 2:], target[:, 2:])[0])
        rows.append(
            {
                "sample_id": str(cache.sample_ids[position]),
                "family": str(cache.families[position]),
                "candidate": candidate,
                "parent": baseline,
            }
        )
    result = {
        "record_count": len(rows),
        "candidate_mean": float(np.mean([row["candidate"] for row in rows])),
        "parent_mean": float(np.mean([row["parent"] for row in rows])),
        "nonworse_count": sum(row["candidate"] <= row["parent"] for row in rows),
        "per_family_candidate": {},
        "rows": rows,
        "validation_opened": False,
        "test_id_opened": False,
    }
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        result["per_family_candidate"][family] = float(
            np.mean([row["candidate"] for row in family_rows])
        )
    return result


def pilot_acceptance(metrics: Mapping[str, Any], *, update: int, learning_rate: float) -> dict[str, Any]:
    gates = {
        "candidate_mean": float(metrics["candidate_mean"]) <= 0.47955565810136125,
        "nonworse_count": int(metrics["nonworse_count"]) >= 30,
        "uniform": float(metrics["per_family_candidate"]["uniform"]) <= 0.37022280539921215,
        "layered": float(metrics["per_family_candidate"]["layered"]) <= 0.3149375678241338,
        "anomaly": float(metrics["per_family_candidate"]["anomaly"]) <= 0.5241317602092531,
        "marmousi": float(metrics["per_family_candidate"]["marmousi"]) <= 0.6795251190565753,
        "update": int(update) == TOTAL_UPDATES,
        "learning_rate": math.isclose(
            float(learning_rate), EXPECTED_LR_UPDATE_204, rel_tol=1.0e-12, abs_tol=1.0e-15
        ),
        "finite": all(
            math.isfinite(float(value))
            for value in [metrics["candidate_mean"], learning_rate, *metrics["per_family_candidate"].values()]
        ),
    }
    return {"passed": all(gates.values()), "gates": gates}


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    identity: Mapping[str, Any],
    epoch: int,
    update: int,
    purpose: str,
    initial_digest: str,
) -> dict[str, Any]:
    return {
        "schema": "transfer_dg_parent_anchored_k4_checkpoint_v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "update": int(update),
        "next_record_cursor": 0,
        "offsets": list(OFFSETS),
        "scheduler_t_max": SCHEDULER_T_MAX,
        "identity": dict(identity),
        "purpose": purpose,
        "pilot_init_forbidden": purpose == "smoke_only",
        "initial_state_digest": initial_digest,
        "confirmation_opened": False,
        "validation_opened": False,
        "test_id_opened": False,
    }


def _resolved(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def validate_stage_prerequisite(prereg: Mapping[str, Any], name: str) -> None:
    item = prereg["prerequisites"][name]
    if item.get("status") != "passed":
        raise RuntimeError(f"K4 prerequisite is not passed: {name}")
    path = _resolved(item["path"])
    if not path.is_file() or sha256(path) != item.get("sha256"):
        raise RuntimeError(f"K4 prerequisite binding drift: {name}")


def validate_training_invocation(
    args: argparse.Namespace, prereg: Mapping[str, Any]
) -> tuple[str, Path]:
    if prereg.get("schema") != "transfer_dg_parent_anchored_k4_preregistration_v1":
        raise RuntimeError("K4 preregistration schema drift")
    if prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("K4 candidate drift")
    mode = "smoke" if bool(args.smoke) else "pilot"
    required_status = {
        "smoke": "mechanism_passed_smoke_pending_independent_audit",
        "pilot": "pilot_pending_independent_audit",
    }[mode]
    if prereg.get("status") != required_status:
        raise RuntimeError(f"K4 {mode} status contract rejected")
    paths = prereg["paths"]
    supplied = {
        "preregistration": args.preregistration.resolve(),
        "cache": args.cache.resolve(),
        "source_h5": args.source_h5.resolve(),
        f"{mode}_output_dir": args.output_dir.resolve(),
    }
    for key, supplied_path in supplied.items():
        if supplied_path != _resolved(paths[key]):
            raise RuntimeError(f"K4 {mode} path override rejected: {key}")
    validate_stage_prerequisite(prereg, "mechanism")
    if mode == "pilot":
        validate_stage_prerequisite(prereg, "smoke")
    return mode, _resolved(paths[f"{mode}_output_dir"])


def validate_training_bindings(
    args: argparse.Namespace, prereg: Mapping[str, Any]
) -> dict[str, str]:
    bindings = prereg["bindings"]
    paths = prereg["paths"]
    binding_paths = {
        "trainer_sha256": Path(__file__),
        "mechanism_audit_sha256": ROOT / "scripts/audit_transfer_dg_parent_anchored_k4.py",
        "launcher_sha256": ROOT / "scripts/launch_transfer_dg_parent_anchored_k4.py",
        "test_sha256": ROOT / "tests/saved_time_phase_operator_v4/test_parent_anchored_k4.py",
        "corrector_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_block.py",
        "relative_loss_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_relative_loss.py",
        "r3_trainer_sha256": _resolved(paths["r3_trainer"]),
        "exact_gradient_audit_sha256": ROOT / "scripts/audit_transfer_dg_r3_exact_gradient_sampling.py",
        "exact_gradient_report_sha256": _resolved(paths["exact_gradient_report"]),
        "exact_gradient_selection_sha256": _resolved(paths["exact_gradient_selection"]),
        "cache_sha256": args.cache,
        "source_h5_sha256": args.source_h5,
        "selection_136_manifest_sha256": _resolved(paths["selection_136_manifest"]),
        "full_2800_manifest_sha256": _resolved(paths["full_2800_manifest"]),
        "baseline_metrics_sha256": _resolved(paths["baseline_metrics"]),
        "r3_best_sha256": _resolved(paths["r3_best"]),
        "parent_checkpoint_sha256": _resolved(paths["parent_checkpoint"]),
    }
    hashes = {key: sha256(path) for key, path in binding_paths.items()}
    drift = [key for key, value in hashes.items() if bindings.get(key) != value]
    if drift:
        raise RuntimeError(f"K4 training binding drift: {drift}")
    baseline_lines = _resolved(paths["baseline_metrics"]).read_bytes().splitlines()
    line3_sha256 = hashlib.sha256(baseline_lines[2]).hexdigest() if len(baseline_lines) >= 3 else ""
    if line3_sha256 != bindings["baseline_line3_sha256"]:
        raise RuntimeError("baseline line-3 digest drift")
    hashes["baseline_line3_sha256"] = line3_sha256
    for name, item in prereg["prerequisites"].items():
        if item.get("status") == "passed":
            path = _resolved(item["path"])
            observed = sha256(path)
            if observed != item.get("sha256"):
                raise RuntimeError(f"K4 prerequisite changed during binding audit: {name}")
            hashes[f"prerequisite_{name}_sha256"] = observed
    return hashes


def training_failure_payload(
    *,
    error: Exception,
    started: float,
    output_dir: Path,
    input_hashes_before: Mapping[str, str],
    preregistration_sha256_before: str,
    environment: Mapping[str, Any],
    argv: Sequence[str],
    guard: TruthAccessGuard | None,
    peak_allocated_bytes: int,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": repr(error),
        "input_hashes_before": dict(input_hashes_before),
        "preregistration_sha256_observed_before": preregistration_sha256_before,
        "argv": list(argv),
        "environment": dict(environment),
        "truth_ledger": guard.summary() if guard is not None else None,
        "resources": {
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": int(peak_allocated_bytes),
            "new_disk_bytes": directory_size_bytes(output_dir),
        },
        "confirmation_opened": False,
        "validation_opened": False,
        "test_id_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    mode_name, fixed_output_dir = validate_training_invocation(args, prereg)
    if fixed_output_dir.exists():
        raise FileExistsError("refusing to reuse K4 output directory")
    fixed_output_dir.mkdir(parents=True)
    terminal_path = fixed_output_dir / "terminal.json"
    started = time.time()
    cache = None
    guard = None
    input_hashes_before: dict[str, str] = {}
    preregistration_sha256_before = sha256(args.preregistration)
    environment: dict[str, Any] = {}
    try:
        bindings = prereg["bindings"]
        paths = prereg["paths"]
        input_hashes_before = validate_training_bindings(args, prereg)
        environment = assert_training_environment()
        model, init_digest, manifest_digest = fresh_model_cpu(SEED)
        if init_digest != prereg["fresh_initialization"]["initial_state_digest"]:
            raise RuntimeError("fresh initial-state digest drift")
        if manifest_digest != prereg["fresh_initialization"]["parameter_manifest_digest"]:
            raise RuntimeError("parameter manifest digest drift")
        if "checkpoint" in vars(args) or "model_state" in vars(args):
            raise RuntimeError("checkpoint/model-state CLI input unexpectedly exists")
        torch.set_num_threads(16)
        evaluation_model = copy.deepcopy(model).cpu().eval() if args.pilot else None
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=SCHEDULER_T_MAX, eta_min=3.0e-6
        )
        purpose = "smoke" if args.smoke else "training"
        guard = TruthAccessGuard()
        cache = GuardedParentCache(args.cache, args.source_h5, purpose=purpose, guard=guard)
        fit = cache.positions("fit")
        calibration = cache.positions("calibration")
        confirmation = cache.positions("confirmation")
        if (len(fit), len(calibration), len(confirmation)) != (68, 34, 34):
            raise RuntimeError("role census drift")
        identity = {
            "schema": "transfer_dg_parent_anchored_k4_identity_v1",
            "candidate": CANDIDATE,
            "mode": mode_name,
            "bindings": bindings,
            "input_hashes_before": input_hashes_before,
            "preregistration": str(args.preregistration.resolve()),
            "preregistration_sha256_observed_before": preregistration_sha256_before,
            "output_dir": str(fixed_output_dir),
            "argv": list(sys.argv),
            "environment": environment,
            "role_audit": cache.role_audit,
            "initial_state_digest": init_digest,
            "parameter_manifest_digest": manifest_digest,
            "checkpoint_loaded": False,
            "offsets": list(OFFSETS),
            "scheduler_t_max": SCHEDULER_T_MAX,
            "confirmation_opened": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(identity, fixed_output_dir / "run_identity.json")
        print(
            json.dumps(
                {
                    "event": "k4_identity",
                    "candidate": CANDIDATE,
                    "mode": mode_name,
                    "preregistration_sha256": preregistration_sha256_before,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        parameters = tuple(model.parameters())
        if args.smoke:
            position = next(index for index in fit if str(cache.sample_ids[index]) == "train_uniform_00014")
            starts = k4_starts(str(cache.sample_ids[position]), 1)
            state_before = initial_state_digest(model)
            individual, scalar_losses = [], []
            for start in starts:
                loss, _ = window_loss(model, cache, position, start, device, purpose="smoke")
                scalar_losses.append(float(loss.detach()))
                individual.append(flatten_autograd(loss, parameters))
                del loss
            if initial_state_digest(model) != state_before:
                raise RuntimeError("individual-gradient smoke mutated fresh state")
            mean_gradient = torch.stack(individual).mean(0)
            produced_losses: list[float] = []

            def loss_factory(window_index: int) -> torch.Tensor:
                loss, _ = window_loss(
                    model, cache, position, starts[window_index], device, purpose="smoke"
                )
                produced_losses.append(float(loss.detach()))
                return loss

            optimizer.zero_grad(set_to_none=True)
            for window_index in range(4):
                loss = loss_factory(window_index)
                (loss / 4.0).backward()
                del loss
            accumulated = flatten_current_gradients(parameters)
            comparison = vector_comparison(accumulated, mean_gradient)
            scalar_absdiff = abs(float(np.mean(produced_losses)) - float(np.mean(scalar_losses)))
            if not (
                comparison["cosine"] >= 0.999999
                and comparison["relative_norm_difference"] <= 1.0e-5
                and scalar_absdiff <= 1.0e-7
            ):
                raise RuntimeError("smoke accumulation algebra gate failed")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError("non-finite smoke gradient")
            optimizer.step()
            scheduler.step()
            if scheduler.last_epoch != 1 or not math.isclose(
                scheduler.get_last_lr()[0], expected_lr(1), rel_tol=1.0e-12, abs_tol=1.0e-15
            ):
                raise RuntimeError("smoke scheduler gate failed")
            if not (
                nested_tensors_finite(model.state_dict())
                and nested_tensors_finite(optimizer.state_dict())
                and nested_tensors_finite(scheduler.state_dict())
            ):
                raise FloatingPointError("non-finite smoke model/optimizer/scheduler state")
            checkpoint = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                identity=identity,
                epoch=1,
                update=1,
                purpose="smoke_only",
                initial_digest=init_digest,
            )
            latest = fixed_output_dir / "latest.pt"
            atomic_checkpoint(checkpoint, latest, immutable=True)
            event = {
                "event": "smoke_update",
                "update": 1,
                "starts": list(starts),
                "comparison": comparison,
                "mean_scalar_loss_absolute_difference": scalar_absdiff,
                "gradient_norm": float(gradient_norm),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            atomic_json(event, fixed_output_dir / "metrics.json")
            input_hashes_after = validate_training_bindings(args, prereg)
            preregistration_sha256_after = sha256(args.preregistration)
            if input_hashes_after != input_hashes_before or preregistration_sha256_after != preregistration_sha256_before:
                raise RuntimeError("smoke input/preregistration mutated")
            resources = {
                "elapsed_seconds": time.time() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "new_disk_bytes": directory_size_bytes(fixed_output_dir),
            }
            if (
                resources["elapsed_seconds"] > 180
                or resources["peak_allocated_bytes"] >= 8 * 2**30
                or resources["new_disk_bytes"] >= 16 * 2**20
            ):
                raise RuntimeError("smoke resource budget exceeded")
            terminal = {
                "status": "smoke_complete",
                "update": 1,
                "latest_checkpoint": str(latest.resolve()),
                "latest_checkpoint_sha256": sha256(latest),
                "checkpoint_purpose": "smoke_only",
                "pilot_init_forbidden": True,
                "truth_ledger": guard.summary(),
                "input_hashes_before": input_hashes_before,
                "input_hashes_after": input_hashes_after,
                "preregistration_sha256_observed_before": preregistration_sha256_before,
                "preregistration_sha256_observed_after": preregistration_sha256_after,
                "preregistration_hash_unchanged": True,
                "argv": list(sys.argv),
                "environment": environment,
                "resources": resources,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
            }
            atomic_json(terminal, terminal_path)
            return 0
        update = 0
        best_mean = math.inf
        final_metrics = None
        metrics_path = fixed_output_dir / "metrics.jsonl"
        for epoch in range(1, EPOCHS + 1):
            order = np.asarray(fit, dtype=np.int64)
            np.random.default_rng(SEED + 1009 * epoch).shuffle(order)
            epoch_losses = []
            model.train()
            for position in order.tolist():
                starts = k4_starts(str(cache.sample_ids[position]), epoch)

                def loss_factory(window_index: int) -> torch.Tensor:
                    loss, _ = window_loss(
                        model, cache, position, starts[window_index], device, purpose="training"
                    )
                    return loss

                result = k4_accumulate_and_step(
                    loss_factory=loss_factory,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    parameters=parameters,
                    clipper=torch.nn.utils.clip_grad_norm_,
                )
                update += 1
                epoch_losses.append(result["mean_loss"])
            if update != epoch * UPDATES_PER_EPOCH:
                raise RuntimeError("epoch update-count drift")
            if not (
                nested_tensors_finite(model.state_dict())
                and nested_tensors_finite(optimizer.state_dict())
                and nested_tensors_finite(scheduler.state_dict())
            ):
                raise FloatingPointError("non-finite pilot model/optimizer/scheduler state")
            if evaluation_model is None:
                raise RuntimeError("pilot CPU evaluation model is absent")
            evaluation_model.load_state_dict(
                {name: value.detach().cpu() for name, value in model.state_dict().items()},
                strict=True,
            )
            cache.purpose = "calibration"
            final_metrics = evaluate_calibration(
                evaluation_model, cache, calibration, torch.device("cpu")
            )
            cache.purpose = "training"
            event = {
                "event": "epoch",
                "epoch": epoch,
                "update": update,
                "learning_rate": scheduler.get_last_lr()[0],
                "mean_train_loss": float(np.mean(epoch_losses)),
                "calibration": final_metrics,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                identity=identity,
                epoch=epoch,
                update=update,
                purpose="pilot",
                initial_digest=init_digest,
            )
            epoch_path = fixed_output_dir / f"epoch_{epoch:03d}.pt"
            atomic_checkpoint(payload, epoch_path, immutable=True)
            atomic_checkpoint(payload, fixed_output_dir / "latest.pt", immutable=False)
            if final_metrics["candidate_mean"] < best_mean:
                best_mean = final_metrics["candidate_mean"]
                atomic_checkpoint(payload, fixed_output_dir / "best.pt", immutable=False)
        if update != TOTAL_UPDATES or final_metrics is None:
            raise RuntimeError("pilot did not reach frozen epoch-3 gate")
        learning_rate = scheduler.get_last_lr()[0]
        if not math.isclose(learning_rate, EXPECTED_LR_UPDATE_204, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise RuntimeError("update-204 learning-rate drift")
        acceptance = pilot_acceptance(final_metrics, update=update, learning_rate=learning_rate)
        input_hashes_after = validate_training_bindings(args, prereg)
        preregistration_sha256_after = sha256(args.preregistration)
        if input_hashes_after != input_hashes_before or preregistration_sha256_after != preregistration_sha256_before:
            raise RuntimeError("pilot input/preregistration mutated")
        terminal = {
            "status": "accepted_for_full_k4_trainonly_preregistration" if acceptance["passed"] else "rejected",
            "epoch": 3,
            "update": update,
            "learning_rate": learning_rate,
            "epoch_3_calibration": final_metrics,
            "acceptance": acceptance,
            "checkpoint_hashes": {
                path.name: sha256(path)
                for path in sorted(fixed_output_dir.glob("*.pt"))
            },
            "truth_ledger": guard.summary(),
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "preregistration_sha256_observed_before": preregistration_sha256_before,
            "preregistration_sha256_observed_after": preregistration_sha256_after,
            "preregistration_hash_unchanged": True,
            "argv": list(sys.argv),
            "environment": environment,
            "resources": {
                "elapsed_seconds": time.time() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "new_disk_bytes": directory_size_bytes(fixed_output_dir),
            },
            "future_authorization": "Only a separate full train-only preregistration may be proposed after acceptance.",
            "confirmation_opened": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        if (
            terminal["resources"]["elapsed_seconds"] > 1800
            or terminal["resources"]["peak_allocated_bytes"] >= 8 * 2**30
            or terminal["resources"]["new_disk_bytes"] >= 64 * 2**20
        ):
            raise RuntimeError("pilot resource budget exceeded")
        atomic_json(terminal, terminal_path)
        return 0
    except Exception as error:
        failure = training_failure_payload(
            error=error,
            started=started,
            output_dir=fixed_output_dir,
            input_hashes_before=input_hashes_before,
            preregistration_sha256_before=preregistration_sha256_before,
            environment=environment,
            argv=sys.argv,
            guard=guard,
            peak_allocated_bytes=(
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
            ),
        )
        failure["traceback"] = traceback.format_exc()
        try:
            failure["input_hashes_after"] = validate_training_bindings(args, prereg)
            failure["preregistration_sha256_observed_after"] = sha256(args.preregistration)
            failure["preregistration_hash_unchanged"] = (
                failure["preregistration_sha256_observed_after"] == preregistration_sha256_before
            )
        except Exception as after_error:
            failure["after_audit_error"] = repr(after_error)
        atomic_json(
            failure,
            terminal_path,
        )
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
