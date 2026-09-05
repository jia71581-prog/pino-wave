"""Offline training utilities for snapshot-only wave propagation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FAMILIES = ("uniform", "layered", "marmousi")


class SnapshotWindowDataset(Dataset):
    """Balanced, deterministic wavefront-context windows from stored trajectories.

    Source metadata is used offline only to avoid sampling a pre-source all-zero
    context. It is not returned and therefore cannot enter the model.
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        split: str,
        history_frames: int,
        rollout_steps: int,
        seed: int,
        records_per_family: int,
        samples_per_epoch: int | None = None,
        fixed_context_ends: tuple[int, ...] = (),
        minimum_context_end: int = 48,
        maximum_context_end: int = 240,
        maximum_context_fraction: float = 1.0 / 3.0,
        post_peak_cycles: float = 0.5,
        record_selection: str = "random",
    ) -> None:
        self.h5_path = str(Path(h5_path))
        self.split = str(split)
        self.history_frames = int(history_frames)
        self.rollout_steps = int(rollout_steps)
        self.seed = int(seed)
        self.epoch = 0
        self._h5 = None
        if self.history_frames < 4 or self.rollout_steps <= 0:
            raise ValueError("snapshot windows require history>=4 and rollout>0")
        with h5py.File(self.h5_path, "r") as h5:
            split_values = np.asarray(h5["split"][:])
            family_values = np.asarray(h5["medium_type"][:])
            self.time_count = int(h5["wavefield"].shape[1])
            self.dt_s = float(h5.attrs["dt_output_s"])
            source_t0 = np.asarray(h5["source_t0_s"][:], dtype=np.float64)
            source_f0 = np.asarray(h5["source_f0_hz"][:], dtype=np.float64)
        context_fraction = float(maximum_context_fraction)
        if not 0.0 < context_fraction <= 1.0:
            raise ValueError("maximum_context_fraction must lie in (0,1]")
        # Strict inequality keeps every observed index out of the middle-time bin.
        self.maximum_context_fraction = context_fraction
        self.early_context_limit = int(
            np.ceil(context_fraction * (self.time_count - 1)) - 1
        )
        rng = np.random.default_rng(self.seed)
        selection = str(record_selection)
        if selection not in {"random", "first"}:
            raise ValueError("record_selection must be 'random' or 'first'")
        by_family: dict[str, list[int]] = {}
        for family in FAMILIES:
            candidates = np.flatnonzero(
                (split_values == self.split.encode()) & (family_values == family.encode())
            )
            count = int(records_per_family)
            if count <= 0:
                count = int(candidates.size)
            if candidates.size < count:
                raise ValueError(
                    f"{self.split}/{family} has {candidates.size} records, needs {count}"
                )
            chosen = (
                candidates[:count]
                if selection == "first"
                else rng.choice(candidates, size=count, replace=False)
            )
            by_family[family] = sorted(int(value) for value in chosen)
        common_count = min(len(values) for values in by_family.values())
        self.records = [
            by_family[family][index]
            for index in range(common_count)
            for family in FAMILIES
        ]
        self.record_family = {
            record: family for family, records in by_family.items() for record in records
        }
        upper_limit = min(
            int(maximum_context_end),
            self.early_context_limit,
            self.time_count - self.rollout_steps - 1,
        )
        self.context_bounds: dict[int, tuple[int, int]] = {}
        for record in self.records:
            energetic_start = int(
                np.ceil(
                    (source_t0[record] + float(post_peak_cycles) / source_f0[record])
                    / self.dt_s
                )
            )
            lower = max(
                self.history_frames - 1,
                int(minimum_context_end),
                energetic_start + self.history_frames - 1,
            )
            if lower > upper_limit:
                raise ValueError(
                    f"record {record} has no admissible energetic snapshot window"
                )
            self.context_bounds[record] = (lower, upper_limit)

        self.fixed_context_ends = tuple(int(value) for value in fixed_context_ends)
        if self.fixed_context_ends:
            self.items = []
            for record in self.records:
                lower, upper = self.context_bounds[record]
                for end in self.fixed_context_ends:
                    if lower <= end <= upper:
                        self.items.append((record, end))
            if not self.items:
                raise ValueError("fixed validation snapshot windows are empty")
            self.samples_per_epoch = len(self.items)
        else:
            if samples_per_epoch is None or int(samples_per_epoch) <= 0:
                raise ValueError("training snapshot dataset needs samples_per_epoch")
            self.items = []
            self.samples_per_epoch = int(samples_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.items:
            record, context_end = self.items[int(index)]
        else:
            record = self.records[int(index) % len(self.records)]
            lower, upper = self.context_bounds[record]
            mixed = (
                self.seed
                + 1_000_003 * self.epoch
                + 7_919 * int(index)
                + 104_729 * int(record)
            )
            context_end = lower + int(mixed % (upper - lower + 1))
        start = int(context_end) - self.history_frames + 1
        stop = int(context_end) + 1
        h5 = self._file()
        history = np.asarray(
            h5["wavefield"][record, start:stop], dtype=np.float32
        )
        target = np.asarray(
            h5["wavefield"][record, stop : stop + self.rollout_steps],
            dtype=np.float32,
        )
        return {
            "wavefield_history": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "family": self.record_family[record],
            "record": int(record),
            "context_end": int(context_end),
            "context_end_fraction": float(context_end / max(self.time_count - 1, 1)),
        }


def snapshot_rollout_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    frame_weight: float = 0.5,
    temporal_difference_weight: float = 0.1,
    spatial_gradient_weight: float = 0.1,
    spectrum_weight: float = 0.1,
    frame_energy_floor_fraction: float = 0.05,
    multi_horizon_weight: float = 0.5,
    multi_horizon_fractions: tuple[float, ...] | list[float] = (0.25, 0.5, 1.0),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Energy-normalized multi-step loss for free-running snapshot rollout."""

    pred = torch.as_tensor(prediction).float()
    ref = torch.as_tensor(target, device=pred.device).float()
    if pred.shape != ref.shape or pred.ndim != 4:
        raise ValueError("prediction and target must match [batch,time,z,x]")
    error = pred - ref
    window = error.flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp_min(1e-12)
    window_loss = window.square().mean()
    prefix_ends = sorted(
        {
            max(1, min(pred.shape[1], int(round(float(fraction) * pred.shape[1]))))
            for fraction in multi_horizon_fractions
            if 0.0 < float(fraction) <= 1.0
        }
    )
    if not prefix_ends:
        raise ValueError("multi_horizon_fractions must contain a value in (0,1]")
    prefix_losses = []
    for end in prefix_ends:
        prefix_error = error[:, :end].flatten(1).norm(dim=1)
        prefix_norm = ref[:, :end].flatten(1).norm(dim=1).clamp_min(1e-12)
        prefix_losses.append((prefix_error / prefix_norm).square().mean())
    multi_horizon_loss = torch.stack(prefix_losses).mean()
    frame_error = error.flatten(2).norm(dim=2)
    frame_norm = ref.flatten(2).norm(dim=2)
    frame_floor = float(frame_energy_floor_fraction) * frame_norm.amax(dim=1, keepdim=True)
    frame_loss = (
        frame_error / torch.maximum(frame_norm, frame_floor).clamp_min(1e-12)
    ).square().mean()
    if pred.shape[1] > 1:
        pred_dt = pred[:, 1:] - pred[:, :-1]
        ref_dt = ref[:, 1:] - ref[:, :-1]
        temporal_loss = (
            (pred_dt - ref_dt).flatten(1).norm(dim=1)
            / ref_dt.flatten(1).norm(dim=1).clamp_min(1e-12)
        ).square().mean()
    else:
        temporal_loss = pred.new_zeros(())
    pred_dx, ref_dx = pred[..., 1:] - pred[..., :-1], ref[..., 1:] - ref[..., :-1]
    pred_dz, ref_dz = pred[..., 1:, :] - pred[..., :-1, :], ref[..., 1:, :] - ref[..., :-1, :]
    gradient_loss = 0.5 * (
        (pred_dx - ref_dx).square().mean() / ref_dx.square().mean().clamp_min(1e-12)
        + (pred_dz - ref_dz).square().mean() / ref_dz.square().mean().clamp_min(1e-12)
    )
    pred_spectrum = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    ref_spectrum = torch.fft.rfft2(ref, dim=(-2, -1), norm="ortho")
    spectrum_loss = (
        (pred_spectrum - ref_spectrum).abs().flatten(1).norm(dim=1)
        / ref_spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    ).square().mean()
    total = (
        window_loss
        + float(multi_horizon_weight) * multi_horizon_loss
        + float(frame_weight) * frame_loss
        + float(temporal_difference_weight) * temporal_loss
        + float(spatial_gradient_weight) * gradient_loss
        + float(spectrum_weight) * spectrum_loss
    )
    return total, {
        "window": window_loss,
        "multi_horizon": multi_horizon_loss,
        "frame": frame_loss,
        "temporal_difference": temporal_loss,
        "spatial_gradient": gradient_loss,
        "spectrum": spectrum_loss,
    }


def summarize_snapshot_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty snapshot rows")
    family_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        family_values[str(row["family"])].append(float(row["relative_l2"]))
    return {
        "window_count": len(rows),
        "aggregate_record_mean_relative_l2": float(
            np.mean([float(row["relative_l2"]) for row in rows])
        ),
        "family_record_mean_relative_l2": {
            family: float(np.mean(values))
            for family, values in sorted(family_values.items())
        },
    }


__all__ = [
    "FAMILIES",
    "SnapshotWindowDataset",
    "snapshot_rollout_loss",
    "summarize_snapshot_rows",
]
