#!/usr/bin/env python3
"""r50 ``wkb48_rank32_pbg`` -- 3-record overfit capacity pilot (r49 rollback).

Scientific contract (rollback of r49; capacity basis: r48b
``trainonly_spectral_capacity``; WKB amplitude-rank evidence from the
``saved_time_phase_operator_v4`` design notes):
  * Data is only ``artifacts/pbg_factorized_fno_marm700_eikonal_v3/dataset_v1.h5``
    together with ``split_full560_70_70_seed372.json``; only the fit-panel head
    (first three ``train_*`` sample ids) is opened.  Rows are resolved through
    the sample-id -> row mapping, and every opened row must carry HDF5 split
    metadata ``train`` and a ``train_`` id prefix.  calibration / confirmation /
    validation / test_id panels are never opened.
  * Model inputs are velocity, source_map/source_parameters, coordinates,
    eikonal travel time, and analytic source parameters only.  The P_bg
    wavefield column is used exclusively as train labels and never enters the
    model input.
  * SINGLE CONCEPTUAL VARIABLE vs r49: the Helmholtz head switches from r49's
    raw oscillatory rank-0 independent head (wkb_phase=false, rank=0, which
    FAILED at agg ~1.09) to the WKB / geometric-optics LOW-RANK head
    (wkb_phase=true, rank=32).  WKB factors the fast oscillation out
    analytically via the eikonal travel time, so the head learns 8 shared
    SMOOTH amplitude basis fields plus a small per-frequency, per-record
    complex mixing instead of 96 fully oscillatory complex fields.  Every
    other setting (backbone, data panel, optimizer, gates) is held identical
    to r49 so the comparison isolates the phase/representation factor.
  * The head emits the 48 cos + 48 sin bins at 64x64 through the rank-32
    factorization and the fixed inverse synthesis renders [B, 401, 64, 64] in
    retarded time.  The supervision target follows r48b: the loader's
    bilinear+antialias 64x64 spatial sampling over all 401 stored time points,
    encode_pressure, then a forward-norm rFFT over bins 0..47 (cos=2Re /
    sin=-2Im, DC special case) WITH the WKB arrival-phase rotation
    (arrival_time_s = the same eikonal travel time the synthesis consumes).
    Global pressure_scale_pa=1.894229157173348e-08 and amplitude=1; no
    per-record or per-frequency target normalization.
  * Training follows r4/r49: AdamW dense 1e-4 / backbone 5e-5, seed 372, 800
    updates, evaluation every 50, microbatch 1, equal-weight per-record
    coefficient-relative-L2, prefix gradient clipping
    local_field=20 / medium_encoder=2 / source_encoder=5.  Evaluation disables
    the causal gate, the free-surface factor, and hard causality so the output
    IS the 48-bin WKB inverse synthesis.
  * Decision gates live in the run configuration (never hard-coded here).  This
    is a train-overfit capacity check and must not be presented as the final
    candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts  # noqa: E402
from grouped_ufno_mionet_v3.training.checkpoint import save_checkpoint_atomic  # noqa: E402
from saved_time_phase_operator_v4.data import split_pilot_batch  # noqa: E402
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec  # noqa: E402
from saved_time_phase_operator_v4.probe import ProbeVariant  # noqa: E402
from saved_time_phase_operator_v4.streaming_metrics import (  # noqa: E402
    ExactWavefieldMetricAccumulator,
)
from scripts.diagnose_capacity_ladder_overfit import (  # noqa: E402
    build_base_config,
    build_capacity_optimizer,
    build_probe_config,
    coefficient_microbatch_backward_weights,
    direct_frequency_relative_l2_squared,
    direct_frequency_target_coefficients,
)
from scripts.diagnose_saved_time_temporal_three_record_overfit import _dataset  # noqa: E402
from scripts.diagnose_trainonly_spectral_capacity import LoaderWavefieldReader  # noqa: E402
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot  # noqa: E402
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _gradient_report,
    clip_trainable_gradients,
)
from scripts.train_saved_time_v4_probe import _atomic_hardlink, _atomic_json, _digest, _model  # noqa: E402

_SCHEMA = "wkb48_rank32_pbg_pilot_v1"
_TERMINAL_SCHEMA = "wkb48_rank32_pbg_pilot_terminal_v1"
_PRESSURE_SCALE_PA = 1.894229157173348e-08
_CAPACITY_ONLY = (
    "train-overfit capacity pilot only; accepted/strong capacity here never "
    "claims the final candidate"
)
_WAVEFIELD_USE = "fit_panel_train_labels_only"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _decode_column(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def resolve_fit_sample_rows(
    h5_path: str | Path,
    split_manifest_path: str | Path,
    sample_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Fit-only contract: map sample ids by the HDF5 sample_id column.

    Every requested id must come from the split manifest's fit panel
    (``sample_ids.train``), exist in the HDF5, carry split metadata ``train``,
    and start with ``train_``.  Returns one identity dict per fit record in the
    requested order.  Calibration/confirmation/validation/test_id panels are
    never resolved here.
    """
    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    fit_panel = [str(value) for value in manifest["sample_ids"]["train"]]
    if len(set(fit_panel)) != len(fit_panel):
        raise ValueError("split manifest fit panel contains duplicate sample ids")
    requested = tuple(str(value) for value in sample_ids)
    if not requested:
        raise ValueError("fit sample ids must be nonempty")
    missing_from_panel = [value for value in requested if value not in fit_panel]
    if missing_from_panel:
        raise ValueError(
            "requested fit sample ids are not in the manifest fit panel: "
            + ", ".join(missing_from_panel)
        )
    with h5py.File(str(h5_path), "r", swmr=True) as h5:
        id_column = _decode_column(h5["sample_id"][:])
        split_column = _decode_column(h5["split"][:])
        if len(id_column) != len(split_column):
            raise ValueError("sample_id and split columns have unequal length")
        id_to_row = {sample_id: row for row, sample_id in enumerate(id_column)}
        if len(id_to_row) != len(id_column):
            raise ValueError("HDF5 sample_id column contains duplicate values")
        sha_column = (
            _decode_column(h5["sample_sha256"][:]) if "sample_sha256" in h5 else None
        )
        if sha_column is not None and len(sha_column) != len(id_column):
            raise ValueError("sample_sha256 column length changed")
        resolved: list[dict[str, object]] = []
        for sample_id in requested:
            if sample_id not in id_to_row:
                raise ValueError(f"sample_id {sample_id!r} is missing from the HDF5")
            row = int(id_to_row[sample_id])
            if split_column[row] != "train":
                raise ValueError(
                    f"sample_id {sample_id!r} maps to VDS row {row} with split "
                    f"metadata {split_column[row]!r} instead of 'train'"
                )
            if not sample_id.startswith("train_"):
                raise ValueError(
                    f"sample_id {sample_id!r} does not start with 'train_'"
                )
            resolved.append(
                {
                    "sample_id": sample_id,
                    "row": row,
                    "sample_sha256": (
                        sha_column[row] if sha_column is not None else None
                    ),
                }
            )
    return resolved


def bilinear_antialias_resize(field: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """Loader-identical 2D resize: bilinear, align_corners=False, antialias=True.

    Matches ``PinoHDF5Dataset._resize_wavefield`` / ``_resize_2d`` so the pilot
    observes exactly the deterministic sampling r48b used.
    """
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError("bilinear_antialias_resize expects [batch,1,z,x]")
    return F.interpolate(
        field,
        size=(int(spatial_size), int(spatial_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def resample_coordinate_axis(axis: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """Loader-consistent coordinate axis for the resized 2D grid.

    The physical coordinate of an output pixel of the bilinear+antialias resize
    is the resampled value of the coordinate field itself (the resize is linear),
    so this applies the same 2D rule to a coordinate ramp and reads the axis back.
    """
    values = torch.as_tensor(axis, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("coordinate axis must be a 1-D sequence")
    field = values[None, None, None, :].expand(1, 1, values.numel(), values.numel())
    resized = bilinear_antialias_resize(field, int(spatial_size))[0, 0]
    return resized[0]


def resample_source_map(
    source_map: torch.Tensor, spatial_size: int
) -> torch.Tensor:
    """Resample the source map and restore the unit-mass contract."""
    resized = bilinear_antialias_resize(source_map, int(spatial_size))
    mass = resized.sum(dim=(-2, -1), keepdim=True)
    if torch.any(mass <= 0.0) or not torch.isfinite(mass).all():
        raise FloatingPointError("resampled source map has non-positive mass")
    return resized / mass


def resample_wavefield_frames(
    frames: torch.Tensor, spatial_size: int
) -> torch.Tensor:
    """[R,T,Z,X] -> [R,T,S,S] with the loader's per-frame rule."""
    if frames.ndim != 4:
        raise ValueError("wavefield frames must be [records,time,z,x]")
    records, times = frames.shape[:2]
    flat = frames.reshape(records * times, 1, frames.shape[2], frames.shape[3])
    resized = bilinear_antialias_resize(flat, int(spatial_size))
    return resized.reshape(records, times, int(spatial_size), int(spatial_size))


def validate_run_config(config: Mapping[str, object]) -> None:
    if str(config.get("schema")) != "wkb48_rank32_pbg_pilot_run_v1":
        raise ValueError("run config schema mismatch")
    sample_ids = [str(value) for value in config.get("sample_ids", ())]
    if len(sample_ids) != 3 or len(set(sample_ids)) != 3:
        raise ValueError("pilot requires exactly three distinct fit sample ids")
    if any(not value.startswith("train_") for value in sample_ids):
        raise ValueError("pilot sample ids must start with 'train_'")
    if int(config.get("spatial_size", 0)) < 2:
        raise ValueError("spatial_size must be at least 2")
    helmholtz = config.get("helmholtz")
    if not isinstance(helmholtz, Mapping):
        raise ValueError("run config requires a helmholtz section")
    if int(helmholtz.get("frequencies", 0)) <= 0:
        raise ValueError("helmholtz frequencies must be positive")
    if not bool(helmholtz.get("wkb_phase", False)):
        raise ValueError("wkb48_rank32 pilot requires wkb_phase=true")
    if int(helmholtz.get("rank", 0)) <= 0:
        raise ValueError("wkb48_rank32 pilot requires the low-rank head (rank>0)")
    probe = config.get("probe")
    if not isinstance(probe, Mapping):
        raise ValueError("run config requires a probe section")
    for key in ("dense_depth", "dense_spectral_rank", "dense_modes"):
        if int(probe.get(key, 0)) <= 0:
            raise ValueError(f"probe {key} must be positive")
    channels = tuple(int(value) for value in probe.get("local_field_channels", ()))
    if len(channels) != 4 or channels[0] != 1:
        raise ValueError(
            "local_field_channels must be the four-level multiplier list "
            "[1, 1, 2, 2] (leading multiplier 1)"
        )
    if any(int(value) <= 0 for value in channels):
        raise ValueError("local_field_channels must all be positive")
    train = config.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("run config requires a train section")
    for key in ("dense_learning_rate", "backbone_learning_rate", "seed"):
        if not math.isfinite(float(train.get(key, float("nan")))):
            raise ValueError(f"train {key} must be finite")
    if int(train.get("updates", 0)) <= 0 or int(train.get("evaluate_every", 0)) <= 0:
        raise ValueError("updates and evaluate_every must be positive")
    if int(train.get("microbatch_records", 0)) != 1:
        raise ValueError("microbatch_records must be 1 (equal-weight per-record loss)")
    clip = train.get("gradient_clip")
    if not isinstance(clip, Mapping) or str(clip.get("mode")) != "prefix_limits":
        raise ValueError("gradient clip must use prefix_limits mode")
    limits = clip.get("prefix_limits")
    if not isinstance(limits, Mapping):
        raise ValueError("gradient clip prefix_limits is required")
    for prefix in ("local_field", "medium_encoder", "source_encoder"):
        if float(limits.get(prefix, float("nan"))) <= 0.0:
            raise ValueError(f"gradient clip prefix {prefix} must be positive")
    eval_cfg = config.get("eval")
    if not isinstance(eval_cfg, Mapping):
        raise ValueError("run config requires an eval section")
    if bool(eval_cfg.get("causal_gate", True)):
        raise ValueError("pilot eval requires causal_gate=false")
    if bool(eval_cfg.get("free_surface_factor", True)):
        raise ValueError("pilot eval requires free_surface_factor=false")
    if bool(eval_cfg.get("hard_causality", True)):
        raise ValueError("pilot eval requires hard_causality=false")
    if int(eval_cfg.get("frames", 0)) != 401:
        raise ValueError("pilot eval requires the full 401 stored time points")
    decision = config.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("run config requires a decision section")
    accepted = float(decision.get("accepted_capacity_aggregate_maximum", float("nan")))
    record_gate = float(decision.get("accepted_capacity_record_maximum", float("nan")))
    strong = float(decision.get("strong_capacity_aggregate_maximum", float("nan")))
    rejected = float(decision.get("rejected_aggregate_minimum", float("nan")))
    if not 0.0 < strong <= accepted < rejected:
        raise ValueError(
            "decision gates must satisfy 0 < strong <= accepted < rejected"
        )
    if not math.isfinite(record_gate) or record_gate <= 0.0:
        raise ValueError("accepted_capacity_record_maximum must be positive and finite")


def decide_capacity(
    aggregate: float,
    record_maximum: float,
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Apply the config-held decision gates (never hard-coded here)."""
    accepted = float(decision["accepted_capacity_aggregate_maximum"])
    record_gate = float(decision["accepted_capacity_record_maximum"])
    strong = float(decision["strong_capacity_aggregate_maximum"])
    rejected = float(decision["rejected_aggregate_minimum"])
    value = float(aggregate)
    if not math.isfinite(value):
        raise FloatingPointError("aggregate capacity metric is non-finite")
    if value >= rejected:
        status = "rejected"
        interpretation = (
            "train-overfit aggregate >= rejected gate; capacity is insufficient"
        )
    elif value <= strong:
        status = "strong_capacity"
        interpretation = (
            "train-overfit aggregate <= strong gate; capacity is strong "
            "(overfit pilot only)"
        )
    elif value <= accepted and float(record_maximum) <= float(record_gate):
        status = "accepted_capacity"
        interpretation = (
            "train-overfit aggregate <= accepted gate and every record <= "
            "record gate (overfit pilot only)"
        )
    else:
        status = "inconclusive"
        interpretation = "train-overfit capacity sits between the config gates"
    return {
        "status": status,
        "interpretation": interpretation,
        "decision_gates": {
            "accepted_capacity_aggregate_maximum": accepted,
            "accepted_capacity_record_maximum": record_gate,
            "strong_capacity_aggregate_maximum": strong,
            "rejected_aggregate_minimum": rejected,
        },
        "all_saved_aggregate_relative_l2": value,
        "all_saved_record_maximum": float(record_maximum),
    }


# --------------------------------------------------------------------------- #
# Fit-only data plumbing
# --------------------------------------------------------------------------- #
class PbgTravelTimeSource:
    """Sample-id view of the eikonal travel-time column resampled to 64x64."""

    def __init__(
        self, h5_path: str | Path, rows_by_id: Mapping[str, int], spatial_size: int
    ) -> None:
        self.path = str(Path(h5_path).resolve())
        self.rows_by_id = {str(key): int(value) for key, value in rows_by_id.items()}
        self.spatial_size = int(spatial_size)
        self._h5: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def read(self, sample_ids: Sequence[str]) -> torch.Tensor:
        samples = tuple(str(value) for value in sample_ids)
        missing = [value for value in samples if value not in self.rows_by_id]
        if missing:
            raise KeyError(f"travel time source is missing sample ids: {missing[:3]}")
        handle = self._file()
        rows = [
            np.asarray(handle["travel_time_s"][self.rows_by_id[value]], dtype=np.float32)
            for value in samples
        ]
        stacked = torch.from_numpy(np.stack(rows, axis=0))
        return bilinear_antialias_resize(stacked[:, None], self.spatial_size)[:, 0]

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class FitOnlyFullTraceStore:
    """Complete 64x64 P_bg train labels read through the loader itself.

    ``LoaderWavefieldReader`` reuses ``PinoHDF5Dataset``'s deterministic
    bilinear+antialias sampling over every stored time point (stride 1), so the
    coefficient target is bit-aligned with the r48b capacity definition.  Only
    fit rows (resolved by sample id) are ever opened.
    """

    def __init__(
        self, h5_path: str | Path, rows_by_id: Mapping[str, int], spatial_size: int
    ) -> None:
        self.path = str(Path(h5_path).resolve())
        self.rows_by_id = {str(key): int(value) for key, value in rows_by_id.items()}
        self.reader = LoaderWavefieldReader(self.path, int(spatial_size))
        self._h5: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getitem__(self, sample_id: str) -> np.ndarray:
        key = str(sample_id)
        if key not in self.rows_by_id:
            raise KeyError(f"fit-only trace store is missing sample id {key!r}")
        row = int(self.rows_by_id[key])
        resized = self.reader.read(self._file(), row)  # [64,64,401]
        if resized.shape != (64, 64, 401):
            raise ValueError("resized fit trace shape changed")
        if not np.isfinite(resized).all():
            raise FloatingPointError("non-finite fit trace label")
        return np.transpose(resized, (2, 0, 1)).copy()  # [401,64,64]

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class Resample64PilotDataset(Dataset):
    """Wrap the exact stored-time dataset with the 64x64 pilot grid.

    Velocity, source map, travel time, coordinates, and dense targets are
    resampled with the loader-identical rule; the source map is renormalized to
    unit mass.  Wavefield-derived query fields stay labels and are unused by
    the coefficient update / evaluation.
    """

    def __init__(self, inner: Dataset, travel_source, spatial_size: int) -> None:
        self.inner = inner
        self.travel_source = travel_source
        self.spatial_size = int(spatial_size)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        import dataclasses

        batch = self.inner[index]
        velocity = bilinear_antialias_resize(
            batch.velocity_mps, self.spatial_size
        )
        source_map = resample_source_map(batch.source_map, self.spatial_size)
        targets = resample_wavefield_frames(
            batch.dense_target_physical, self.spatial_size
        )
        x_m = resample_coordinate_axis(batch.x_m, self.spatial_size)
        z_m = resample_coordinate_axis(batch.z_m, self.spatial_size)
        travel = self.travel_source.read(batch.sample_id)
        return dataclasses.replace(
            batch,
            velocity_mps=velocity,
            source_map=source_map,
            dense_target_physical=targets,
            x_m=x_m,
            z_m=z_m,
            dense_travel_time_s=travel,
        )


def build_direct48_schedule(
    record_indices: Sequence[int], *, updates: int
) -> tuple[FullSupportStepSpec, ...]:
    selected = tuple(int(value) for value in record_indices)
    count = int(updates)
    if len(selected) != 3 or count <= 0:
        raise ValueError("direct48 schedule requires three records and positive updates")
    return tuple(
        FullSupportStepSpec(
            step=950_000 + update,
            epoch=0,
            record_indices=selected,
            appearance_indices=(update,) * len(selected),
        )
        for update in range(count)
    )


def _capture_wkb_coefficients(
    model,
    prepared,
    tensors,
    dense_grid,
    *,
    frequency_count: int,
) -> torch.Tensor:
    """Capture the effective [R, 2*nf, H, W] WKB amplitude fields the head emits.

    The rank>0 Helmholtz head has no single ``head`` conv to hook (the r49
    rank-0 capture path does not exist); the effective cos/sin coefficients are
    assembled inside ``_HelmholtzSynthesisField.forward`` from
    ``basis_head(rendered)`` and the per-record-conditioned mixing.  This
    helper registers a forward hook on the synthesis module, drives the real
    local-field forward with a single query time, and reconstructs the
    [cos; sin] coefficient tensor from the captured ``(a, b)`` basis mixing
    before the analytic retarded-time sum.  The result is exactly the smooth
    amplitude field the inverse synthesis consumes, so supervision on it is
    supervision on the synthesis output.  The forward stays query-independent:
    the captured ``a``/``b`` do not depend on the probe ``time_s``.
    """
    local_field = model.local_field
    synthesis = local_field.helmholtz_synthesis
    if synthesis is None or int(synthesis.rank) <= 0:
        raise RuntimeError("WKB coefficient capture requires the low-rank head (rank>0)")
    if int(synthesis.late_rank) > 0:
        raise RuntimeError("r50 does not exercise the late-rank head")
    nf = int(synthesis.num_frequencies)
    if nf != int(frequency_count):
        raise RuntimeError("synthesis frequency count changed")
    captured: dict[str, torch.Tensor] = {}
    original_forward = synthesis.forward

    def capturing_forward(rendered, arrival, time_s, saved_time_values, *, domain_t_s, late_record_gate=None):
        records, _, height, w = rendered.shape
        basis = synthesis.basis_head(rendered).reshape(records, synthesis.rank, height * w)
        pooled = rendered.mean(dim=(-2, -1))
        delta = synthesis.mix_condition(pooled).reshape(records, 2, nf, synthesis.rank)
        cos_mix = synthesis.cos_mix[None] + delta[:, 0]
        sin_mix = synthesis.sin_mix[None] + delta[:, 1]
        a = torch.einsum("bjr,brs->bjs", cos_mix.to(rendered.dtype), basis)
        b = torch.einsum("bjr,brs->bjs", sin_mix.to(rendered.dtype), basis)
        captured["a"] = a
        captured["b"] = b
        captured["height"] = torch.tensor(height)
        captured["w"] = torch.tensor(w)
        return original_forward(
            rendered, arrival, time_s, saved_time_values,
            domain_t_s=domain_t_s, late_record_gate=late_record_gate,
        )

    synthesis.forward = capturing_forward
    try:
        model.dense_normalized(
            prepared,
            tensors["requested_time_s"][:, :1],
            dense_grid=dense_grid,
            time_block=1,
            apply_correction=False,
        )
    finally:
        synthesis.forward = original_forward
    if "a" not in captured or "b" not in captured:
        raise RuntimeError("WKB synthesis was not invoked exactly once")
    height = int(captured["height"].item())
    w = int(captured["w"].item())
    a = captured["a"].reshape(-1, nf, height, w)
    b = captured["b"].reshape(-1, nf, height, w)
    return torch.cat((a, b), dim=1)


def _wkb48_coefficient_update(
    model,
    optimizer,
    batch,
    normalizer,
    device: torch.device,
    *,
    full_targets,
    frequency_count: int,
    microbatch_records: int,
    saved_time_s: torch.Tensor,
    family_gradient_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Backpropagate WKB smooth-amplitude labels from the full 401-frame truth.

    Mirrors ``_direct_frequency_coefficient_update`` but for the rank>0 WKB
    head: the prediction is the effective cos/sin amplitude field the inverse
    synthesis assembles (captured via a forward pre-hook on the synthesis
    module), and the target applies the SAME WKB arrival-phase rotation to the
    r48b-style raw spectrum so the supervision lands on the smooth envelope
    the U-Net can actually represent.  No 401-frame graph is built.
    """
    synthesis = model.local_field.helmholtz_synthesis
    if synthesis is None or int(synthesis.rank) <= 0:
        raise RuntimeError("wkb48 update requires the low-rank WKB head (rank>0)")
    if int(synthesis.late_rank) > 0:
        raise RuntimeError("r50 does not exercise the late-rank head")
    optimizer.zero_grad(set_to_none=True)
    micros = tuple(split_pilot_batch(batch, microbatch_records=int(microbatch_records)))
    backward_weights = coefficient_microbatch_backward_weights(
        micros, family_gradient_weights
    )
    relative_values: list[float] = []
    for micro, backward_weight in zip(micros, backward_weights, strict=True):
        tensors = _to_device(micro, device)
        source = tensors["source_parameters"]
        prepared = model.prepare_sources(
            model.encode_medium(tensors["velocity_mps"], normalizer),
            source,
            tensors["source_map"],
            normalizer,
            record_to_medium=tensors["record_to_medium"],
        )
        dense_grid = model.prepare_dense_grid(
            prepared,
            x_m=tensors["x_m"],
            z_m=tensors["z_m"],
            travel_time_s=(
                None
                if micro.dense_travel_time_s is None
                else micro.dense_travel_time_s.to(
                    device, non_blocking=device.type == "cuda"
                )
            ),
        )
        prediction = _capture_wkb_coefficients(
            model, prepared, tensors, dense_grid, frequency_count=int(frequency_count)
        )
        physical = torch.stack(
            [
                torch.from_numpy(full_targets[sample_id])
                for sample_id in micro.sample_id
            ],
            dim=0,
        ).to(device=device, dtype=tensors["dense_target_physical"].dtype)
        normalized = normalizer.encode_pressure(physical, source[:, 4])
        target_coefficients = direct_frequency_target_coefficients(
            normalized,
            int(frequency_count),
            arrival_time_s=micro.dense_travel_time_s.to(device),
            saved_time_s=saved_time_s.to(device),
        )
        loss = direct_frequency_relative_l2_squared(prediction, target_coefficients)
        (loss * float(backward_weight)).backward()
        relative_values.append(math.sqrt(max(float(loss.detach()), 0.0)))
    return {
        "coefficient_relative_l2": float(sum(relative_values) / len(relative_values)),
        "total": float(sum(value * value for value in relative_values) / len(relative_values)),
        "full_trace_label_frames": 401.0,
    }


def select_direct48_indices(manifest, sample_ids: Sequence[str]) -> tuple[int, ...]:
    """Local manifest-record indices for the requested fit sample ids."""
    by_id = {record.sample_id: index for index, record in enumerate(manifest.records)}
    result = tuple(int(by_id[str(value)]) for value in sample_ids)
    selected_records = tuple(manifest.records[index] for index in result)
    if any(record.split != "train" for record in selected_records):
        raise ValueError("direct48 sample ids must resolve to train records")
    return result


def build_pilot_dataset(
    config,
    base,
    manifest,
    indices,
    *,
    split: str,
    schedule,
    time_policy: str,
    frames_per_record: int,
    travel_source,
    spatial_size: int,
):
    inner = _dataset(
        config,
        base,
        manifest,
        indices,
        split=split,
        schedule=schedule,
        time_policy=time_policy,
        frames_per_record=frames_per_record,
    )
    return Resample64PilotDataset(inner, travel_source, spatial_size)


# --------------------------------------------------------------------------- #
# Evaluation (eval flags fully off: output is exactly the 48-bin synthesis)
# --------------------------------------------------------------------------- #
def assert_pilot_eval_flags(model, config: Mapping[str, object]) -> None:
    """Fail loudly unless every eval-side flag is off.

    With causal_gate / free_surface_factor / hard_causality all disabled the
    model output IS the 48-bin inverse synthesis, which is the contract of the
    direct-frequency capacity pilot.
    """
    if getattr(model.local_field, "helmholtz_apply_causal_gate", True):
        raise RuntimeError("pilot evaluation requires the causal gate to be off")
    if bool(getattr(model, "dense_apply_free_surface_factor", True)):
        raise RuntimeError(
            "pilot evaluation requires the free-surface factor to be off"
        )
    if bool(config["loss"].get("hard_causality", True)):
        raise RuntimeError("pilot evaluation requires hard causality to be off")


@torch.inference_mode()
def _evaluate_pilot(
    model,
    base,
    manifest,
    normalizer,
    device: torch.device,
    config,
    indices,
    *,
    split: str,
    time_policy: str,
    frames_per_record: int,
    travel_source,
    spatial_size: int,
    evaluation_macro_records: int | None = None,
    evaluation_time_block: int = 1,
) -> dict[str, object]:
    assert_pilot_eval_flags(model, config)
    selected = tuple(int(value) for value in indices)
    macro = len(selected) if evaluation_macro_records is None else int(
        evaluation_macro_records
    )
    if macro <= 0:
        raise ValueError("evaluation_macro_records must be positive")
    schedule = tuple(
        FullSupportStepSpec(
            step=990_000 + step,
            epoch=0,
            record_indices=selected[start : start + macro],
            appearance_indices=(0,) * len(selected[start : start + macro]),
        )
        for step, start in enumerate(range(0, len(selected), macro))
    )
    dataset = build_pilot_dataset(
        config,
        base,
        manifest,
        indices,
        split=split,
        schedule=schedule,
        time_policy=time_policy,
        frames_per_record=frames_per_record,
        travel_source=travel_source,
        spatial_size=spatial_size,
    )
    predicted_metrics = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    reference_metrics = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    error_total = 0.0
    target_total = 0.0
    model.eval()
    for batch_index in range(len(dataset)):
        batch = dataset[batch_index]
        for micro in split_pilot_batch(batch, microbatch_records=1):
            tensors = _to_device(micro, device)
            source = tensors["source_parameters"]
            prepared = model.prepare_sources(
                model.encode_medium(tensors["velocity_mps"], normalizer),
                source,
                tensors["source_map"],
                normalizer,
                record_to_medium=tensors["record_to_medium"],
            )
            dense_grid = model.prepare_dense_grid(
                prepared,
                x_m=tensors["x_m"],
                z_m=tensors["z_m"],
                travel_time_s=(
                    None
                    if micro.dense_travel_time_s is None
                    else micro.dense_travel_time_s.to(
                        device, non_blocking=device.type == "cuda"
                    )
                ),
            )
            prediction = model.dense_normalized(
                prepared,
                tensors["requested_time_s"],
                dense_grid=dense_grid,
                time_block=int(evaluation_time_block),
                apply_correction=False,
            )
            if int(frames_per_record) == len(manifest.time_s) and (
                prediction.shape[1] != len(manifest.time_s)
            ):
                raise RuntimeError(
                    "pilot evaluation did not synthesize every stored time"
                )
            if prediction.shape[-2:] != (int(spatial_size), int(spatial_size)):
                raise RuntimeError("pilot evaluation spatial shape changed")
            reference = prediction
            target = normalizer.encode_pressure(
                tensors["dense_target_physical"], source[:, 4]
            )
            metric_onset = torch.clamp(
                source[:, 3] - source[:, 2].reciprocal(),
                min=float(manifest.time_s[0]),
            )
            onset_indices = torch.searchsorted(
                torch.as_tensor(manifest.time_s, device=device),
                metric_onset.contiguous(),
            ).cpu().tolist()
            common = {
                "families": micro.medium_type,
                "group_ids": micro.group_id,
                "sample_ids": micro.sample_id,
                "time_indices": micro.left_index,
                "source_onset_indices": onset_indices,
            }
            predicted_metrics.update(prediction, target, **common)
            reference_metrics.update(reference, target, **common)
            error_total += float((prediction.float() - target.float()).square().sum())
            target_total += float(target.float().square().sum())
    metrics = predicted_metrics.finalize()
    reference = reference_metrics.finalize()
    metrics["coarse_metrics"] = reference
    metrics["reference_kind"] = "coarse"
    metrics["correction_to_coarse_l2_ratio"] = 0.0
    metrics["relative_improvement_vs_coarse"] = 0.0
    metrics["aggregate_relative_l2_energy_weighted"] = math.sqrt(
        max(error_total, 0.0)
    ) / math.sqrt(max(target_total, 1.0e-16))
    return metrics


def save_pilot_checkpoint(
    root: Path,
    *,
    model,
    optimizer,
    update: int,
    manifest_digest: str,
    config_digest: str,
    aggregate_relative_l2: float,
) -> Path:
    checkpoint_path = root / "checkpoints" / f"update_{int(update):04d}.pt"
    save_checkpoint_atomic(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=int(update),
        manifest_digest=manifest_digest,
        config_digest=config_digest,
        metrics={
            "triplet_aggregate_relative_l2": float(aggregate_relative_l2),
        },
    )
    _atomic_hardlink(checkpoint_path, root / "latest.pt")
    return checkpoint_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--evaluate-every", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda; the real pilot is single-GPU)",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    base_config_path = Path(args.base_config).resolve()
    run_config_path = Path(args.run_config).resolve()
    run_config = json.loads(run_config_path.read_text()) if run_config_path.suffix == ".json" else None
    if run_config is None:
        import yaml

        run_config = yaml.safe_load(run_config_path.read_text()) or {}
    validate_run_config(run_config)
    base_config = build_base_config(128, base_config=str(base_config_path))
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0
    if (root / "run_identity.json").exists():
        raise FileExistsError(
            f"incomplete previous run in {root}: run_identity.json exists without "
            "terminal.json; move the directory before restarting"
        )

    device = torch.device(str(args.device) if args.device else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    seed = int(run_config["train"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    manifest = build_manifest(base_config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base_config.data.expected_train_records,
            "validation": base_config.data.expected_validation_records,
        },
    )
    if len(manifest.time_s) != 401:
        raise ValueError("pilot requires the 401-point stored time axis")
    normalizer = load_normalizer(base_config, manifest.digest)
    if not math.isclose(
        normalizer.metadata.pressure_scale_pa,
        _PRESSURE_SCALE_PA,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ):
        raise ValueError(
            "normalization pressure_scale_pa does not match the pilot contract"
        )
    if normalizer.metadata.record_count != int(
        base_config.data.expected_train_records
    ):
        raise ValueError(
            "normalization record count does not match the train census"
        )

    spatial_size = int(run_config["spatial_size"])
    sample_ids = tuple(str(value) for value in run_config["sample_ids"])
    fit_identities = resolve_fit_sample_rows(
        base_config.data.source_h5,
        base_config.data.manifest_json,
        sample_ids,
    )
    rows_by_id = {identity["sample_id"]: identity["row"] for identity in fit_identities}
    indices = select_direct48_indices(manifest, sample_ids)

    helmholtz = run_config["helmholtz"]
    probe = run_config["probe"]
    train_cfg = run_config["train"]
    eval_cfg = run_config["eval"]
    decision_cfg = run_config["decision"]
    updates = int(args.updates if args.updates is not None else train_cfg["updates"])
    evaluate_every = int(
        args.evaluate_every
        if args.evaluate_every is not None
        else train_cfg["evaluate_every"]
    )
    if updates <= 0 or evaluate_every <= 0:
        raise ValueError("updates and evaluate_every must be positive")
    if args.smoke:
        updates = min(updates, 2)
        evaluate_every = 1

    variant = ProbeVariant(
        depth=int(probe["dense_depth"]),
        use_local_phase=True,
        spectral_rank=int(probe["dense_spectral_rank"]),
        modes=int(probe["dense_modes"]),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_channel_multipliers=tuple(
            int(value) for value in probe["local_field_channels"]
        ),
        local_field_causal_width_s=float(probe["local_field_causal_width_s"]),
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(helmholtz["frequencies"]),
        local_field_helmholtz_synthesis_wkb_phase=True,
        local_field_helmholtz_synthesis_rank=int(helmholtz["rank"]),
    )
    model = _model(base_config, manifest, variant).to(device)
    model.local_field.helmholtz_apply_causal_gate = False
    model.dense_apply_free_surface_factor = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    synthesis = model.local_field.helmholtz_synthesis
    if int(synthesis.rank) != int(helmholtz["rank"]) or int(synthesis.rank) <= 0:
        raise RuntimeError("wkb48 pilot requires the low-rank WKB head (rank>0)")
    if not bool(synthesis.wkb_phase):
        raise RuntimeError("wkb48 pilot requires wkb_phase=true")
    if int(synthesis.num_frequencies) != int(helmholtz["frequencies"]):
        raise RuntimeError("helmholtz frequency count changed")
    if int(synthesis.basis_head.out_channels) != int(helmholtz["rank"]):
        raise RuntimeError("WKB basis head rank changed")

    config = build_probe_config(
        dense_lr=float(train_cfg["dense_learning_rate"]),
        backbone_lr=float(train_cfg["backbone_learning_rate"]),
        temporal_lr=1.0e-5,
        seed=seed,
        travel_time_h5=None,
        family_gradient_weights=None,
    )
    config["travel_time_h5"] = None
    config["loss"]["hard_causality"] = False
    clip_cfg = train_cfg["gradient_clip"]
    config["optimizer"]["gradient_clip"] = float(clip_cfg["maximum_norm"])
    config["optimizer"]["gradient_clip_mode"] = str(clip_cfg["mode"])
    config["optimizer"]["gradient_clip_prefix_limits"] = {
        str(key): float(value) for key, value in clip_cfg["prefix_limits"].items()
    }
    assert_pilot_eval_flags(model, config)
    optimizer = build_capacity_optimizer(
        model,
        dense_lr=float(train_cfg["dense_learning_rate"]),
        backbone_lr=float(train_cfg["backbone_learning_rate"]),
        local_field_lr=None,
    )

    travel_source = PbgTravelTimeSource(
        base_config.data.source_h5, rows_by_id, spatial_size
    )
    try:
        schedule = build_direct48_schedule(indices, updates=updates)
        train_data = build_pilot_dataset(
            config,
            base_config,
            manifest,
            indices,
            split="train",
            schedule=schedule,
            time_policy="appearance16",
            frames_per_record=4,
            travel_source=travel_source,
            spatial_size=spatial_size,
        )
        code_bindings = {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "base_config": str(base_config_path),
            "base_config_sha256": _sha256_file(base_config_path),
            "run_config": str(run_config_path),
            "run_config_sha256": _sha256_file(run_config_path),
            "source_h5": str(Path(base_config.data.source_h5).resolve()),
            "source_h5_sha256": _sha256_file(Path(base_config.data.source_h5).resolve()),
            "split_manifest": str(Path(base_config.data.manifest_json).resolve()),
            "split_manifest_sha256": _sha256_file(
                Path(base_config.data.manifest_json).resolve()
            ),
            "normalization_json": str(
                Path(base_config.data.normalization_json).resolve()
            ),
            "normalization_json_sha256": _sha256_file(
                Path(base_config.data.normalization_json).resolve()
            ),
            "travel_time_content_sha256": _travel_time_content_sha256(
                base_config.data.source_h5
            ),
            "vds_provenance": _vds_provenance(base_config.data.source_h5),
        }
        identity = {
            "schema": _SCHEMA,
            "base_config": str(base_config_path),
            "run_config": str(run_config_path),
            "manifest_digest": manifest.digest,
            "fit_sample_ids": sample_ids,
            "fit_rows": [int(item["row"]) for item in fit_identities],
            "fit_sample_sha256": {
                item["sample_id"]: item["sample_sha256"] for item in fit_identities
            },
            "spatial_size": spatial_size,
            "helmholtz": dict(helmholtz),
            "probe": dict(probe),
            "decision": dict(decision_cfg),
            "train": {
                "dense_learning_rate": float(train_cfg["dense_learning_rate"]),
                "backbone_learning_rate": float(train_cfg["backbone_learning_rate"]),
                "seed": seed,
                "updates": updates,
                "evaluate_every": evaluate_every,
                "microbatch_records": int(train_cfg["microbatch_records"]),
                "gradient_clip": dict(clip_cfg),
            },
            "eval": {
                "causal_gate": False,
                "free_surface_factor": False,
                "hard_causality": False,
                "frames": int(eval_cfg["frames"]),
            },
            "parameter_count": parameter_count,
            "access": {
                "fit_opened": True,
                "fit_record_count": len(fit_identities),
                "calibration_opened": False,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
                "hdf5_datasets_read": [
                    "sample_id",
                    "split",
                    "sample_sha256",
                    "time_s",
                    "x_m",
                    "z_m",
                    "velocity_mps (fit rows only)",
                    "source_map (fit rows only)",
                    "source_x_m/source_z_m/source_f0_hz/source_t0_s/source_amplitude (fit rows only)",
                    "travel_time_s (fit rows only)",
                    "wavefield (fit rows only, train labels)",
                ],
                "wavefield_use": _WAVEFIELD_USE,
            },
            "bindings": code_bindings,
            "capacity_only": _CAPACITY_ONLY,
        }
        identity["run_digest"] = _digest(identity)
        _atomic_json(identity, root / "run_identity.json")

        eval_frames = 32 if args.smoke else int(eval_cfg["frames"])

        def evaluate(frames: int) -> dict[str, object]:
            return _evaluate_pilot(
                model,
                base_config,
                manifest,
                normalizer,
                device,
                config,
                indices,
                split="train",
                time_policy="all_saved" if frames == len(manifest.time_s) else "validation_fixed",
                frames_per_record=frames,
                travel_source=travel_source,
                spatial_size=spatial_size,
                evaluation_macro_records=1 if frames == len(manifest.time_s) else 3,
                evaluation_time_block=frames,
            )

        baseline = evaluate(eval_frames)
        _append_jsonl(
            root / "metrics.jsonl",
            {"event": "baseline", "update": 0, "metrics": baseline},
        )
        baseline_checkpoint = save_pilot_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            update=0,
            manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            aggregate_relative_l2=float(baseline["aggregate_relative_l2"]),
        )
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        best_score = float(baseline["aggregate_relative_l2"])
        best_checkpoint = str(baseline_checkpoint)
        last_update = 0
        final_components: dict[str, float] = {}
        full_targets = FitOnlyFullTraceStore(
            base_config.data.source_h5, rows_by_id, spatial_size
        )
        try:
            for update, batch in enumerate(train_data, start=1):
                model.train()
                components = _wkb48_coefficient_update(
                    model,
                    optimizer,
                    batch,
                    normalizer,
                    device,
                    full_targets=full_targets,
                    frequency_count=int(helmholtz["frequencies"]),
                    microbatch_records=int(train_cfg["microbatch_records"]),
                    saved_time_s=torch.as_tensor(manifest.time_s),
                    family_gradient_weights=None,
                )
                final_components = dict(components)
                gradients = _gradient_report(
                    model, ("local_field", "medium_encoder", "source_encoder")
                )
                gradient_norm, clipping = clip_trainable_gradients(
                    model,
                    maximum_norm=float(config["optimizer"]["gradient_clip"]),
                    mode=str(config["optimizer"]["gradient_clip_mode"]),
                    prefix_limits=config["optimizer"]["gradient_clip_prefix_limits"],
                    return_report=True,
                )
                optimizer.step()
                last_update = update
                _append_jsonl(
                    root / "updates.jsonl",
                    {
                        "event": "optimizer_update",
                        "update": update,
                        "loss_components": components,
                        "gradient_norm_before_clip": gradient_norm,
                        "gradient_clipping": clipping,
                        "gradient_prefixes": gradients,
                        "gpu": _gpu_snapshot(),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                if update % evaluate_every and update != updates:
                    continue
                metrics = evaluate(eval_frames)
                checkpoint_path = save_pilot_checkpoint(
                    root,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    manifest_digest=manifest.digest,
                    config_digest=identity["run_digest"],
                    aggregate_relative_l2=float(metrics["aggregate_relative_l2"]),
                )
                score = float(metrics["aggregate_relative_l2"])
                if score <= best_score:
                    best_score = score
                    best_checkpoint = str(checkpoint_path)
                _append_jsonl(
                    root / "metrics.jsonl",
                    {
                        "event": "evaluation",
                        "update": update,
                        "metrics": metrics,
                        "peak_cuda_bytes": (
                            int(torch.cuda.max_memory_allocated())
                            if device.type == "cuda"
                            else None
                        ),
                        "elapsed_seconds": time.monotonic() - started,
                        "checkpoint": str(checkpoint_path),
                    },
                )
                print(
                    json.dumps(
                        {"update": update, "agg": score}, sort_keys=True
                    ),
                    flush=True,
                )
        finally:
            full_targets.close()

        if args.smoke:
            terminal = {
                "schema": _TERMINAL_SCHEMA,
                "status": "smoke_complete",
                "updates_completed": last_update,
                "best_fixed_aggregate_relative_l2": best_score,
                "parameter_count": parameter_count,
                "peak_cuda_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda"
                    else None
                ),
                "elapsed_seconds": time.monotonic() - started,
                "access": identity["access"],
                "capacity_only": _CAPACITY_ONLY,
                "bindings": {
                    **code_bindings,
                    "run_identity_sha256": _sha256_file(root / "run_identity.json"),
                },
            }
            _atomic_json(terminal, terminal_path)
            print(json.dumps(terminal, sort_keys=True), flush=True)
            return 0

        all_saved = evaluate(int(eval_cfg["frames"]))
        record_relative = all_saved.get("source_relative_l2")
        if not isinstance(record_relative, dict):
            raise RuntimeError("all-saved evaluation did not report per-record metrics")
        record_maximum = float(max(record_relative.values()))
        decision = decide_capacity(
            float(all_saved["aggregate_relative_l2"]),
            record_maximum,
            decision_cfg,
        )
        terminal = {
            "schema": _TERMINAL_SCHEMA,
            "status": decision["status"],
            "interpretation": decision["interpretation"],
            "capacity_only": _CAPACITY_ONLY,
            "updates_completed": last_update,
            "best_checkpoint": best_checkpoint,
            "best_fixed_aggregate_relative_l2": best_score,
            "initial_aggregate_relative_l2": float(
                baseline["aggregate_relative_l2"]
            ),
            "all_saved_metrics": all_saved,
            "coefficient_relative_l2": float(
                final_components.get("coefficient_relative_l2", float("nan"))
            ),
            "coefficient_loss_total": float(
                final_components.get("total", float("nan"))
            ),
            "decision": decision,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda"
                else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "access": identity["access"],
            "bindings": {
                **code_bindings,
                "run_identity_sha256": _sha256_file(root / "run_identity.json"),
                "terminal_checkpoint_sha256": (
                    _sha256_file(Path(best_checkpoint)) if best_checkpoint else None
                ),
            },
        }
        _atomic_json(terminal, terminal_path)
        print(json.dumps(terminal, sort_keys=True), flush=True)
        return 0
    finally:
        travel_source.close()


def _travel_time_content_sha256(h5_path: str | Path) -> str:
    with h5py.File(str(h5_path), "r", swmr=True) as h5:
        return str(h5.attrs.get("travel_time_content_sha256", ""))


def _vds_provenance(h5_path: str | Path) -> dict[str, object]:
    """Pin the VDS provenance attrs (shard manifest, generator config)."""
    with h5py.File(str(h5_path), "r", swmr=True) as h5:
        return {
            str(key): str(value)
            for key, value in h5.attrs.items()
            if key
            in {
                "manifest_sha256",
                "config_sha256",
                "background_source_manifest_sha256",
                "included_splits",
                "schema_version",
                "axis_order",
                "vds_sample_count",
            }
        }


if __name__ == "__main__":
    raise SystemExit(main())
