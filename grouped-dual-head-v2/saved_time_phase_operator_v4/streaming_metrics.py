"""Bounded-memory metrics for arbitrary exact stored-time wavefield blocks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch


@dataclass
class _RecordState:
    family: str
    group_id: str
    error_square: float = 0.0
    target_square: float = 0.0
    frame_count: int = 0


class ExactWavefieldMetricAccumulator:
    """Aggregate exact-time metrics while retaining only per-record scalars."""

    def __init__(
        self,
        *,
        energy_floor_fraction: float = 0.01,
        require_unique: bool = False,
        stored_time_count: int = 401,
    ) -> None:
        if not 0.0 < float(energy_floor_fraction) <= 1.0:
            raise ValueError("energy floor fraction must lie in (0,1]")
        if int(stored_time_count) <= 1:
            raise ValueError("stored time count must exceed one")
        self.energy_floor_fraction = float(energy_floor_fraction)
        self.require_unique = bool(require_unique)
        self.stored_time_count = int(stored_time_count)
        self._records: dict[str, _RecordState] = {}
        self._pairs: set[tuple[str, int]] = set()
        self._unique_times: set[int] = set()
        self._frame_count = 0
        self._near_zero_frames = 0
        self._error_square = 0.0
        self._element_count = 0
        self._phase_sum = 0.0
        self._phase_count = 0
        # displacement / transport diagnostics (Codex co-primary gate for r3/r4):
        # phase_correlation rewards amplitude alignment and can stay flat while the
        # wavefront is mis-placed, so we also track how far the predicted energy is
        # spatially shifted from the reference -- the quantity a warp must reduce.
        self._centroid_shift_sum = 0.0   # sum over informative frames of |centroid_p - centroid_t| (cells)
        self._xcorr_shift_sum = 0.0      # sum over informative frames of the xcorr peak offset (cells)
        self._displacement_count = 0
        self._time_error: defaultdict[str, float] = defaultdict(float)
        self._time_target: defaultdict[str, float] = defaultdict(float)
        # per-(family, time_bin) cross error/target: resolves whether a family's
        # residual is late-concentrated (temporal lever) or time-flat (spatial/width
        # limit).  Additive; keyed by (family, bin).  See ARCH_CANDIDATES ceiling bound.
        self._family_time_error: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._family_time_target: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._spectrum_error: defaultdict[str, float] = defaultdict(float)
        self._spectrum_target: defaultdict[str, float] = defaultdict(float)

    def _time_bin(self, time_index: int, onset_index: int | None) -> str:
        if onset_index is not None:
            if time_index < onset_index:
                return "pre_onset"
            active = max(self.stored_time_count - onset_index, 1)
            phase = min(2, 3 * (time_index - onset_index) // active)
            return ("early", "middle", "late")[phase]
        phase = min(3, 4 * time_index // self.stored_time_count)
        return ("pre_onset", "early", "middle", "late")[phase]

    @staticmethod
    def _spectrum_masks(height: int, width: int, device: torch.device):
        z_frequency = torch.fft.fftfreq(height, device=device).abs()
        x_frequency = torch.fft.rfftfreq(width, device=device).abs()
        radius = torch.sqrt(z_frequency[:, None].square() + x_frequency[None, :].square())
        radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
        return {
            "low": radius <= 1.0 / 3.0,
            "middle": (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
            "high": radius > 2.0 / 3.0,
        }

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        families: Sequence[str],
        group_ids: Sequence[str],
        sample_ids: Sequence[str],
        time_indices: torch.Tensor,
        source_onset_indices: Sequence[int] | None = None,
    ) -> None:
        predicted = torch.as_tensor(prediction)
        reference = torch.as_tensor(target, device=predicted.device)
        if predicted.shape != reference.shape or predicted.ndim != 4:
            raise ValueError("prediction and target must match [record,time,z,x]")
        if torch.is_complex(predicted) or torch.is_complex(reference):
            raise ValueError(
                "wavefield metrics require real scalar [record,time,z,x] fields"
            )
        records, times, height, width = predicted.shape
        labels = tuple(str(value) for value in families)
        groups = tuple(str(value) for value in group_ids)
        samples = tuple(str(value) for value in sample_ids)
        indices = torch.as_tensor(time_indices, dtype=torch.long).cpu()
        if not (len(labels) == len(groups) == len(samples) == records):
            raise ValueError("metric metadata must contain one value per record")
        if len(set(samples)) != len(samples):
            raise ValueError("sample IDs must be unique within one metric update")
        if indices.shape != (records, times):
            raise ValueError("time indices must match the record/time dimensions")
        if bool((indices < 0).any()) or bool((indices >= self.stored_time_count).any()):
            raise ValueError("metric time index is outside the stored axis")
        onsets = (
            (None,) * records
            if source_onset_indices is None
            else tuple(int(value) for value in source_onset_indices)
        )
        if len(onsets) != records:
            raise ValueError("source onset indices must contain one value per record")
        new_pairs = {
            (samples[record], int(indices[record, time]))
            for record in range(records)
            for time in range(times)
        }
        if len(new_pairs) != records * times:
            raise ValueError("duplicate sample/time pair inside metric update")
        if self.require_unique and self._pairs.intersection(new_pairs):
            raise ValueError("duplicate sample/time pair across metric updates")

        difference = predicted.float() - reference.float()
        error_square_by_record = difference.double().square().flatten(1).sum(dim=1)
        target_square_by_record = reference.double().square().flatten(1).sum(dim=1)
        for record, sample_id in enumerate(samples):
            state = self._records.get(sample_id)
            if state is None:
                state = _RecordState(labels[record], groups[record])
                self._records[sample_id] = state
            elif state.family != labels[record] or state.group_id != groups[record]:
                raise ValueError(f"metric metadata changed for sample {sample_id}")
            state.error_square += float(error_square_by_record[record])
            state.target_square += float(target_square_by_record[record])
            state.frame_count += times

        error_frame = difference.double().square().flatten(2).sum(dim=2)
        target_frame = reference.double().square().flatten(2).sum(dim=2)
        predicted_frame = predicted.double().square().flatten(2).sum(dim=2)
        dot_frame = (predicted.double() * reference.double()).flatten(2).sum(dim=2)
        informative = target_frame > 1.0e-16
        phase_denominator = torch.sqrt(predicted_frame * target_frame).clamp_min(1.0e-16)
        self._phase_sum += float((dot_frame / phase_denominator)[informative].sum())
        self._phase_count += int(informative.sum())
        self._near_zero_frames += int((~informative).sum())

        # --- displacement / transport diagnostics (per informative frame) ---
        # Energy centroid shift: distance between the |field|^2 centres of mass of
        # prediction and reference, in cell units.  Computed from 1-D marginals so no
        # [R,T,H,W] weight tensor is materialised beyond the squared fields.
        weight_p = predicted.float().square()
        weight_t = reference.float().square()
        z_axis = torch.arange(height, device=predicted.device, dtype=torch.float64)
        x_axis = torch.arange(width, device=predicted.device, dtype=torch.float64)
        total_p = predicted_frame.clamp_min(1.0e-16)
        total_t = target_frame.clamp_min(1.0e-16)
        cp_z = (weight_p.sum(dim=-1).double() * z_axis).sum(dim=-1) / total_p
        cp_x = (weight_p.sum(dim=-2).double() * x_axis).sum(dim=-1) / total_p
        ct_z = (weight_t.sum(dim=-1).double() * z_axis).sum(dim=-1) / total_t
        ct_x = (weight_t.sum(dim=-2).double() * x_axis).sum(dim=-1) / total_t
        centroid_shift = torch.sqrt((cp_z - ct_z).square() + (cp_x - ct_x).square())
        self._centroid_shift_sum += float(centroid_shift[informative].sum())
        predicted_fft = torch.fft.rfft2(predicted.float(), norm="ortho")
        target_fft = torch.fft.rfft2(reference.float(), norm="ortho")
        cross = torch.fft.irfft2(predicted_fft * target_fft.conj(), s=(height, width), norm="ortho")
        peak = cross.flatten(2).argmax(dim=-1)
        peak_z = torch.div(peak, width, rounding_mode="floor").double()
        peak_x = (peak % width).double()
        peak_z = torch.where(peak_z > height / 2, peak_z - height, peak_z)
        peak_x = torch.where(peak_x > width / 2, peak_x - width, peak_x)
        xcorr_shift = torch.sqrt(peak_z.square() + peak_x.square())
        self._xcorr_shift_sum += float(xcorr_shift[informative].sum())
        self._displacement_count += int(informative.sum())

        for record in range(records):
            for time in range(times):
                name = self._time_bin(int(indices[record, time]), onsets[record])
                self._time_error[name] += float(error_frame[record, time])
                self._time_target[name] += float(target_frame[record, time])
                fam_key = (labels[record], name)
                self._family_time_error[fam_key] += float(error_frame[record, time])
                self._family_time_target[fam_key] += float(target_frame[record, time])

        masks = self._spectrum_masks(height, width, predicted.device)
        spectrum_difference = predicted_fft - target_fft
        for name, mask in masks.items():
            self._spectrum_error[name] += float(spectrum_difference[..., mask].abs().square().sum())
            self._spectrum_target[name] += float(target_fft[..., mask].abs().square().sum())

        self._pairs.update(new_pairs)
        self._unique_times.update(int(value) for value in indices.flatten().tolist())
        self._frame_count += records * times
        self._error_square += float(error_square_by_record.sum())
        self._element_count += int(predicted.numel())

    @staticmethod
    def _relative(error_square: float, target_square: float) -> float:
        return math.sqrt(max(error_square, 0.0)) / math.sqrt(max(target_square, 1.0e-16))

    def finalize(self) -> dict[str, object]:
        if not self._records:
            raise ValueError("cannot finalize empty streamed metrics")
        record_relative = {
            sample_id: self._relative(state.error_square, state.target_square)
            for sample_id, state in self._records.items()
        }
        by_family: defaultdict[str, list[float]] = defaultdict(list)
        by_medium: defaultdict[str, list[float]] = defaultdict(list)
        for sample_id, state in self._records.items():
            value = record_relative[sample_id]
            by_family[state.family].append(value)
            by_medium[state.group_id].append(value)
        return {
            "record_count": len(self._records),
            "frame_count": self._frame_count,
            "near_zero_frame_count": self._near_zero_frames,
            "unique_time_index_count": len(self._unique_times),
            "aggregate_relative_l2": float(np.mean(tuple(record_relative.values()))),
            "family_relative_l2": {
                name: float(np.mean(values)) for name, values in sorted(by_family.items())
            },
            "medium_relative_l2": {
                name: float(np.mean(values)) for name, values in sorted(by_medium.items())
            },
            "source_relative_l2": dict(sorted(record_relative.items())),
            "time_bin_relative_l2": {
                name: self._relative(self._time_error[name], self._time_target[name])
                for name in ("pre_onset", "early", "middle", "late")
                if name in self._time_error
            },
            "family_time_bin_relative_l2": {
                family: {
                    name: self._relative(
                        self._family_time_error[(family, name)],
                        self._family_time_target[(family, name)],
                    )
                    for name in ("pre_onset", "early", "middle", "late")
                    if (family, name) in self._family_time_error
                }
                for family in sorted({fam for fam, _ in self._family_time_error})
            },
            "spectrum_relative_l2": {
                name: self._relative(self._spectrum_error[name], self._spectrum_target[name])
                for name in ("low", "middle", "high")
            },
            "rmse": math.sqrt(self._error_square / max(self._element_count, 1)),
            "phase_correlation": self._phase_sum / max(self._phase_count, 1),
            "centroid_shift_cells": self._centroid_shift_sum / max(self._displacement_count, 1),
            "xcorr_peak_shift_cells": self._xcorr_shift_sum / max(self._displacement_count, 1),
            "energy_floor_fraction": self.energy_floor_fraction,
        }


__all__ = ["ExactWavefieldMetricAccumulator"]
