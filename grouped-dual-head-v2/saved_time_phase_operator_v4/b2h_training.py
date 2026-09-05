"""Training utilities for the B2-H physical residual propagator.

The smoke protocol intentionally uses short, teacher-initialized trajectory
windows.  It verifies the learned closure, DDP/checkpoint plumbing and recurrent
stability; it is not the final leak-free full-operator validation.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FAMILIES = ("uniform", "layered", "marmousi")


class SequenceWindowDataset(Dataset):
    """Lazy HDF5 trajectory windows with deterministic epoch-dependent starts."""

    def __init__(
        self,
        h5_path: str,
        *,
        split: str,
        rollout_steps: int,
        seed: int,
        records_per_family: int,
        history_steps: int = 0,
        samples_per_epoch: int | None = None,
        fixed_starts: tuple[int, ...] = (),
    ) -> None:
        self.h5_path = str(Path(h5_path))
        self.split = str(split)
        self.rollout_steps = int(rollout_steps)
        self.history_steps = int(history_steps)
        self.seed = int(seed)
        self.epoch = 0
        self._h5 = None
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.history_steps not in (0, 1):
            raise ValueError("history_steps must be 0 or 1")

        with h5py.File(self.h5_path, "r") as h5:
            split_values = np.asarray(h5["split"][:])
            family_values = np.asarray(h5["medium_type"][:])
            self.time_count = int(h5["wavefield"].shape[1])
        rng = np.random.default_rng(self.seed)
        by_family: dict[str, list[int]] = {}
        for family in FAMILIES:
            mask = (split_values == self.split.encode()) & (
                family_values == family.encode()
            )
            candidates = np.flatnonzero(mask)
            if candidates.size < records_per_family:
                raise ValueError(
                    f"{split}/{family} has {candidates.size} records, "
                    f"needs {records_per_family}"
                )
            selected = rng.choice(
                candidates, size=int(records_per_family), replace=False
            )
            by_family[family] = sorted(int(value) for value in selected)
        # Interleave families so any prefix remains balanced.
        self.records = [
            by_family[family][index]
            for index in range(int(records_per_family))
            for family in FAMILIES
        ]
        self.record_family = {
            record: family for family, records in by_family.items() for record in records
        }

        self.fixed_starts = tuple(int(value) for value in fixed_starts)
        maximum_start = self.time_count - self.rollout_steps - 2
        if maximum_start < 0:
            raise ValueError("rollout exceeds the stored trajectory")
        if any(
            value < self.history_steps or value > maximum_start
            for value in self.fixed_starts
        ):
            raise ValueError(
                f"fixed_starts must lie in [{self.history_steps},{maximum_start}]"
            )
        if self.fixed_starts:
            self.items = [
                (record, start)
                for record in self.records
                for start in self.fixed_starts
            ]
            self.samples_per_epoch = len(self.items)
        else:
            if samples_per_epoch is None or int(samples_per_epoch) <= 0:
                raise ValueError("training dataset needs samples_per_epoch > 0")
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
            record, start = self.items[int(index)]
        else:
            record = self.records[int(index) % len(self.records)]
            maximum_start = self.time_count - self.rollout_steps - 2
            mixed = (
                self.seed
                + 1_000_003 * self.epoch
                + 7_919 * int(index)
                + 104_729 * int(record)
            )
            available = maximum_start - self.history_steps + 1
            start = self.history_steps + int(mixed % available)
        h5 = self._file()
        stop = start + self.rollout_steps + 2
        frames = np.asarray(
            h5["wavefield"][
                record, start - self.history_steps : stop
            ],
            dtype=np.float32,
        )
        offset = self.history_steps
        source = np.asarray(
            h5["source_wavelet"][
                record, start + 1 : start + 1 + self.rollout_steps
            ],
            dtype=np.float32,
        )
        source_parameters = np.asarray(
            [
                h5["source_f0_hz"][record],
                h5["source_t0_s"][record],
                h5["source_amplitude"][record],
            ],
            dtype=np.float32,
        )
        return {
            "history": torch.from_numpy(
                frames[0:1]
                if self.history_steps
                else np.empty((0, *frames.shape[-2:]), dtype=np.float32)
            ),
            "p0": torch.from_numpy(frames[offset : offset + 1]),
            "p1": torch.from_numpy(frames[offset + 1 : offset + 2]),
            "target": torch.from_numpy(frames[offset + 2 :, None]),
            "velocity": torch.from_numpy(
                np.asarray(h5["velocity_mps"][record], dtype=np.float32)[None]
            ),
            "source_map": torch.from_numpy(
                np.asarray(h5["source_map"][record], dtype=np.float32)[None]
            ),
            "source_series": torch.from_numpy(source),
            "source_parameters": torch.from_numpy(source_parameters),
            "initial_time_s": torch.tensor(
                float(h5["time_s"][start + 1]), dtype=torch.float32
            ),
            "family": self.record_family[record],
            "record": int(record),
            "start": int(start),
        }


def relative_window_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float,
) -> torch.Tensor:
    """Mean squared frame-relative error with a per-window energy floor."""

    error_norm = (prediction - target).float().flatten(start_dim=3).norm(dim=-1)
    target_norm = target.float().flatten(start_dim=3).norm(dim=-1)
    peak = target_norm.amax(dim=1, keepdim=True)
    denominator = torch.maximum(
        target_norm, float(energy_floor_fraction) * peak
    ).clamp_min(1.0e-12)
    return ((error_norm / denominator) ** 2).mean()


def metric_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    families: list[str],
    records: torch.Tensor,
    starts: torch.Tensor,
    time_count: int,
    energy_floor_fraction: float,
) -> list[dict[str, object]]:
    """Return JSON-safe per-window statistics for distributed aggregation."""

    pred = prediction.detach().float().cpu()
    ref = target.detach().float().cpu()
    rows: list[dict[str, object]] = []
    for index, family in enumerate(families):
        difference = pred[index] - ref[index]
        reference_norm = ref[index].flatten().norm().clamp_min(1.0e-12)
        relative = float(difference.flatten().norm() / reference_norm)
        frame_norm = ref[index].flatten(start_dim=2).norm(dim=-1)
        floor = float(energy_floor_fraction) * frame_norm.max().clamp_min(1.0e-12)
        informative = frame_norm > floor
        dot = (pred[index] * ref[index]).flatten(start_dim=2).sum(dim=-1)
        pred_norm = pred[index].flatten(start_dim=2).norm(dim=-1)
        phase = dot / (pred_norm * frame_norm).clamp_min(1.0e-12)
        phase_values = phase[informative]
        bins: dict[str, list[float]] = {}
        start = int(starts[index])
        for offset in range(ref.shape[1]):
            absolute_time = start + 2 + offset
            fraction = absolute_time / max(time_count - 1, 1)
            name = "early" if fraction < 1.0 / 3.0 else (
                "middle" if fraction < 2.0 / 3.0 else "late"
            )
            error_square = float(difference[offset].double().square().sum())
            target_square = float(ref[index, offset].double().square().sum())
            state = bins.setdefault(name, [0.0, 0.0])
            state[0] += error_square
            state[1] += target_square
        rows.append(
            {
                "record": int(records[index]),
                "start": start,
                "family": str(family),
                "relative_l2": relative,
                "phase_sum": float(phase_values.sum()) if phase_values.numel() else 0.0,
                "phase_count": int(phase_values.numel()),
                "bins": bins,
            }
        )
    return rows


def summarize_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty metric rows")
    family_values: defaultdict[str, list[float]] = defaultdict(list)
    bin_state: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    phase_sum = 0.0
    phase_count = 0
    for row in rows:
        family_values[str(row["family"])].append(float(row["relative_l2"]))
        phase_sum += float(row["phase_sum"])
        phase_count += int(row["phase_count"])
        for name, values in dict(row["bins"]).items():
            bin_state[name][0] += float(values[0])
            bin_state[name][1] += float(values[1])
    return {
        "scope": "teacher_initialized_fixed_windows_smoke",
        "window_count": len(rows),
        "aggregate_relative_l2": float(
            np.mean([float(row["relative_l2"]) for row in rows])
        ),
        "family_relative_l2": {
            name: float(np.mean(values))
            for name, values in sorted(family_values.items())
        },
        "time_bin_relative_l2": {
            name: float(np.sqrt(error / max(target, 1.0e-24)))
            for name, (error, target) in sorted(bin_state.items())
        },
        "phase_correlation": phase_sum / max(phase_count, 1),
    }


__all__ = [
    "FAMILIES",
    "SequenceWindowDataset",
    "metric_rows",
    "relative_window_loss",
    "summarize_metric_rows",
]
