#!/usr/bin/env python3
"""R26 development pilot: tail-risk and spectral coarse-field correction.

This script deliberately reuses the audited R25 data loader, evaluator, and
training driver while replacing only the fit-frame sampling rule, residual
model, and training loss.  High-fidelity fields remain training-only targets.
The R25 holdout is a development set here; validation and test_id stay sealed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_PATH = Path(__file__).resolve()
R25_PATH = SCRIPT_PATH.with_name("train_r25_coarse_residual_operator.py")
SPEC = importlib.util.spec_from_file_location("r25_base_train", R25_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R25 training driver: {R25_PATH}")
r25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r25
SPEC.loader.exec_module(r25)


class TailAwareFitFrameDataset(r25.FitFrameDataset):
    """Repeat Marmousi and top-quartile coarse-error records during fitting."""

    def __init__(self, collection: r25.CacheCollection):
        super().__init__(collection)
        metadata: list[tuple[float, str]] = []
        for file_index, local_index in collection.records:
            handle = collection.handles[file_index]
            error_square = float(handle["baseline_error_square_norm"][local_index])
            target_square = float(handle["target_square_norm"][local_index])
            relative = math.sqrt(error_square / max(target_square, 1.0e-30))
            family = str(handle["family"].asstr()[local_index])
            metadata.append((relative, family))
        threshold = float(np.quantile([value[0] for value in metadata], 0.75))
        mapping: list[int] = []
        repetition_counts: list[int] = []
        for record_position, (relative, family) in enumerate(metadata):
            repeats = 1 + int(relative >= threshold) + int(family == "marmousi")
            repetition_counts.append(repeats)
            start = record_position * self.time_count
            for _ in range(repeats):
                mapping.extend(range(start, start + self.time_count))
        self.mapping = tuple(mapping)
        self.tail_threshold = threshold
        self.repetition_counts = tuple(repetition_counts)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int):
        return super().__getitem__(self.mapping[int(index)])


class LearnedComplexSpectralConv2d(nn.Module):
    """Learn global low-frequency complex Fourier contractions."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        shape = (self.width, self.width, self.modes, self.modes, 2)
        self.weight_top = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.weight_bottom = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        scale = 1.0 / math.sqrt(self.width * self.width)
        nn.init.uniform_(self.weight_top, -scale, scale)
        nn.init.uniform_(self.weight_bottom, -scale, scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        modes_y = min(self.modes, height // 2)
        modes_x = min(self.modes, width // 2 + 1)
        with torch.autocast(device_type=value.device.type, enabled=False):
            spectrum = torch.fft.rfft2(value.float(), norm="ortho")
            output = torch.zeros(
                value.shape[0],
                self.width,
                height,
                width // 2 + 1,
                dtype=spectrum.dtype,
                device=value.device,
            )
            top = torch.view_as_complex(self.weight_top.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            bottom = torch.view_as_complex(self.weight_bottom.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            output[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, :modes_y, :modes_x], top
            )
            output[:, :, -modes_y:, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, -modes_y:, :modes_x], bottom
            )
            return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class SpectralResidualBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = LearnedComplexSpectralConv2d(width, modes)
        self.local = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.mix = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.spectral(value) + self.mix(self.local(value))
        return value + F.gelu(self.norm(update))


class TailSpectralResidualUNet(r25.CoarseResidualUNet):
    """R25 local U-Net plus a zero-gated global spectral residual branch."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        spectral_width = 16
        self.spectral_stem = nn.Conv2d(r25.INPUT_CHANNELS, spectral_width, 1)
        self.spectral_blocks = nn.Sequential(
            SpectralResidualBlock(spectral_width, modes=24),
            SpectralResidualBlock(spectral_width, modes=24),
        )
        self.spectral_output = nn.Conv2d(spectral_width, 1, 1)
        nn.init.zeros_(self.spectral_output.weight)
        nn.init.zeros_(self.spectral_output.bias)
        self.spectral_cap = min(0.12, float(correction_cap))

    def forward(
        self, features: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        local = super().forward(features, active=active)
        spectral = self.spectral_blocks(self.spectral_stem(features))
        spectral = self.spectral_cap * torch.tanh(self.spectral_output(spectral)[:, 0])
        if active is not None:
            spectral = spectral * active[:, None, None]
        correction = torch.clamp(
            local + spectral, -self.correction_cap, self.correction_cap
        )
        correction = correction.clone()
        correction[:, 0, :] = 0.0
        return correction


def tail_risk_loss(
    correction: torch.Tensor,
    coarse: torch.Tensor,
    truth: torch.Tensor,
    mean_energy: torch.Tensor,
    *,
    hinge_weight: float,
    gradient_weight: float,
):
    """Optimize average accuracy and a batch CVaR proxy simultaneously."""

    prediction = coarse + correction
    denominator = mean_energy.clamp_min(1.0e-7)
    candidate_mse = (prediction - truth).square().mean(dim=(1, 2))
    parent_mse = (coarse - truth).square().mean(dim=(1, 2))
    candidate_relative_square = candidate_mse / denominator
    parent_relative_square = parent_mse / denominator

    tail_count = max(1, int(math.ceil(0.5 * candidate_relative_square.numel())))
    tail = torch.topk(candidate_relative_square, k=tail_count).values.mean()
    parent_scale = torch.sqrt(parent_relative_square.detach().clamp_min(1.0e-12))
    difficulty = (parent_scale / parent_scale.mean().clamp_min(1.0e-8)).clamp(0.5, 3.0)
    difficulty_weighted = (
        candidate_relative_square * difficulty
    ).sum() / difficulty.sum().clamp_min(1.0e-8)
    hinge = F.relu(
        torch.sqrt(candidate_relative_square.clamp_min(1.0e-12))
        - torch.sqrt(parent_relative_square.clamp_min(1.0e-12))
    ).square()
    gradient = r25.spatial_gradient_loss(prediction, truth, denominator)
    correction_energy = correction.square().mean()
    total = (
        0.25 * candidate_relative_square.mean()
        + 0.75 * difficulty_weighted
        + 1.00 * tail
        + float(hinge_weight) * hinge.mean()
        + float(gradient_weight) * gradient
        + 1.0e-5 * correction_energy
    )
    return total, {
        "relative_square": candidate_relative_square.mean().detach(),
        "parent_relative_square": parent_relative_square.mean().detach(),
        "hinge": hinge.mean().detach(),
        "gradient": gradient.detach(),
        "correction_energy": correction_energy.detach(),
    }


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required argument {name}") from error


def write_preregistration() -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "schema": "r26_tail_spectral_preregistration_v1",
        "status": "frozen_before_training",
        "role": "development_pilot_on_r25_train_holdout",
        "model": {
            "local_branch": "r25_coarse_residual_unet",
            "spectral_branch": "two_global_complex_fourier_blocks_width16_modes24",
            "spectral_cap": 0.12,
            "total_correction_cap_from_cli": argument_value("--correction-cap"),
            "zero_gated_identity_start": True,
        },
        "training": {
            "sampling": "repeat_all_marmousi_and_top_quartile_parent_error_records_once",
            "loss": "0.25_mean_plus_0.75_parent_difficulty_weighted_plus_1.0_batch_top50pct_CVaR_proxy",
            "truth_role": "training_only_target_and_training_weight_not_deployment_input",
        },
        "success_gate": {
            "development_record_rel_l2_mean_lte": 0.05,
            "development_record_rel_l2_max_lte": 0.05,
            "validation_and_test_remain_sealed": True,
        },
        "evidence_boundary": "Passing this pilot is not final validation evidence; it only authorizes a fresh group-disjoint R26 train holdout experiment.",
        "literature_basis": [
            "https://arxiv.org/abs/2204.06684",
            "https://openreview.net/forum?id=Qv6468llWS",
            "https://doi.org/10.21314/JOR.2000.038",
        ],
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r25_driver_sha256": r25.sha256_file(R25_PATH),
    }
    r25.atomic_json(preregistration, output_dir / "r26_preregistration.json")


def write_terminal_sidecar() -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    base_terminal = json.loads((output_dir / "terminal.json").read_text(encoding="utf-8"))
    sidecar = {
        "schema": "r26_tail_spectral_terminal_v1",
        "status": base_terminal["status"],
        "best_epoch": base_terminal["best_epoch"],
        "best_metrics": base_terminal["best_metrics"],
        "checkpoint": base_terminal["checkpoint"],
        "checkpoint_sha256": base_terminal["checkpoint_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r25_driver_sha256": r25.sha256_file(R25_PATH),
    }
    r25.atomic_json(sidecar, output_dir / "r26_terminal.json")


def main() -> None:
    write_preregistration()
    r25.FitFrameDataset = TailAwareFitFrameDataset
    r25.CoarseResidualUNet = TailSpectralResidualUNet
    r25.train_loss = tail_risk_loss
    r25.main()
    write_terminal_sidecar()


if __name__ == "__main__":
    main()
