#!/usr/bin/env python3
"""Measure train-only oracle capacity of a physical-head channel transform span."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)
from saved_time_phase_operator_v4.instance_adaptation.contracts import (  # noqa: E402
    future_indices,
)
from saved_time_phase_operator_v4.phase_carrier import (  # noqa: E402
    rotate_complex_pairs,
    travel_phase_carrier,
)
from scripts.run_transfer_dg_pyramid_instance_adaptation_cpu import (  # noqa: E402
    PHYSICAL_SCALE,
    PublicPyramidRecord,
    relative_l2,
    render_time,
    whiten_basis,
)
from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    model_prediction,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    atomic_json,
    sha256,
)


def weighted_flatten(value: torch.Tensor) -> torch.Tensor:
    weights = torch.full((value.shape[-4],), math.sqrt(2.0), dtype=value.dtype)
    weights[0] = 1.0
    return (value * weights.reshape((1,) * (value.ndim - 4) + (-1, 1, 1, 1))).reshape(
        *value.shape[:-4], -1
    )


class PhysicalHeadCapture:
    def __init__(self, model: PyramidMoECoupledWaveOperator) -> None:
        self.activation: torch.Tensor | None = None
        self.handle = model.physical_head[3].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self.activation = output.detach()

    def close(self) -> None:
        self.handle.remove()


def physical_head_modes(
    model: PyramidMoECoupledWaveOperator,
    capture: PhysicalHeadCapture,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if capture.activation is None:
        raise RuntimeError("physical-head activation hook did not run")
    frequency_count = int(batch["frequency_hz"].numel())
    activation = capture.activation.reshape(frequency_count, 192, 221, 241)
    final = model.physical_head[4]
    weight = final.weight[:, :, 0, 0]
    modes = torch.einsum("frzx,cr->rfczx", activation, weight)
    modes = modes[..., :201, 20:221].clone()
    modes[..., 0, :] = 0.0
    carrier = travel_phase_carrier(batch["travel_physical"], batch["frequency_hz"])
    rotated = rotate_complex_pairs(
        modes.reshape(192 * frequency_count, 2, 201, 201),
        carrier[None]
        .expand(192, -1, -1, -1, -1)
        .reshape(192 * frequency_count, 2, 201, 201),
    )
    return rotated.reshape(192, frequency_count, 2, 201, 201)


@torch.inference_mode()
def predict_and_build_head_basis(
    model: PyramidMoECoupledWaveOperator,
    public: PublicPyramidRecord,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = []
    mode_blocks = []
    capture = PhysicalHeadCapture(model)
    try:
        for start in range(0, 64, BLOCK):
            batch = public.block(start, torch.device("cpu"))
            capture.activation = None
            prediction, _ = model_prediction(model, batch)
            predictions.append(prediction.detach().cpu())
            mode_blocks.append(physical_head_modes(model, capture, batch).detach().cpu())
            print(
                json.dumps(
                    {
                        "event": "prediction_block_complete",
                        "block_start": start,
                        "block_stop": start + BLOCK,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del batch, prediction
            gc.collect()
    finally:
        capture.close()
    return torch.cat(predictions, dim=0).float(), torch.cat(mode_blocks, dim=1).float()


def candidate_metrics(
    parent_coefficients: torch.Tensor,
    correction: torch.Tensor,
    truth: torch.Tensor,
    future: torch.Tensor,
) -> dict[str, float]:
    parent_field = render_time(parent_coefficients, time_count=401)
    candidate_field = render_time(parent_coefficients + correction, time_count=401)
    parent_full = relative_l2(parent_field, truth)
    candidate_full = relative_l2(candidate_field, truth)
    parent_future = relative_l2(parent_field[future], truth[future])
    candidate_future = relative_l2(candidate_field[future], truth[future])
    return {
        "correction_ratio": float(
            correction.norm() / parent_coefficients.norm().clamp_min(1.0e-30)
        ),
        "parent_full_relative_l2": parent_full,
        "candidate_full_relative_l2": candidate_full,
        "full_relative_improvement": 1.0 - candidate_full / max(parent_full, 1.0e-300),
        "parent_future_relative_l2": parent_future,
        "candidate_future_relative_l2": candidate_future,
        "future_relative_improvement": 1.0
        - candidate_future / max(parent_future, 1.0e-300),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("CPU-only oracle requires CUDA_VISIBLE_DEVICES to be empty")
    if len(args.residual_cache) != 4 or len(args.travel) != 4:
        raise RuntimeError("oracle requires four residual and four travel shards")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    observed = {
        "diagnostic_sha256": sha256(Path(__file__)),
        "integration_runner_sha256": sha256(
            ROOT / "scripts/run_transfer_dg_pyramid_instance_adaptation_cpu.py"
        ),
        "model_sha256": sha256(ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py"),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        "source_h5_sha256": sha256(args.source_h5),
    }
    for key, digest in observed.items():
        if digest != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {digest}")
    if args.sample_id != prereg["data"]["sample_id"]:
        raise RuntimeError("sample differs from preregistration")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    identity = {
        "schema": "transfer_dg_pyramid_head_basis_oracle_cpu_identity_v1",
        "pid": os.getpid(),
        "device": "cpu",
        "threads": args.threads,
        "sample_id": args.sample_id,
        "bindings": observed,
        "oracle_future_truth": True,
        "deployment_candidate": False,
        "validation_opened": False,
        "test_id_opened": False,
        "started_unix_s": started,
    }
    atomic_json(identity, args.output_dir / "run_identity.json")
    print(json.dumps({"event": "identity", **identity}, sort_keys=True), flush=True)

    public = None
    try:
        public = PublicPyramidRecord(args.residual_cache, args.travel, args.sample_id)
        checkpoint = torch.load(
            args.candidate_checkpoint, map_location="cpu", weights_only=False
        )
        model = PyramidMoECoupledWaveOperator(use_mhc=True)
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_meta = {
            key: checkpoint.get(key)
            for key in ("schema", "epoch", "next_step", "update")
        }
        del checkpoint
        model.eval().requires_grad_(False)
        if parameter_count(model) != int(prereg["model"]["parameter_count"]):
            raise RuntimeError("model parameter count drift")

        parent, raw_basis = predict_and_build_head_basis(model, public)
        del model
        gc.collect()
        channel_norms = raw_basis.reshape(raw_basis.shape[0], -1).norm(dim=1)
        top_k = int(prereg["oracle"]["top_energy_channels"])
        if top_k <= 0 or top_k > raw_basis.shape[0]:
            raise RuntimeError("registered physical-head channel count is invalid")
        selected_norms, selected_indices = torch.topk(channel_norms, top_k)
        selection = {
            "source_channel_count": int(raw_basis.shape[0]),
            "selected_channel_count": top_k,
            "selected_channel_indices": [int(value) for value in selected_indices],
            "selected_norm_min": float(selected_norms.min()),
            "selected_norm_max": float(selected_norms.max()),
        }
        raw_basis = raw_basis[selected_indices]
        basis, basis_meta = whiten_basis(
            raw_basis,
            parent,
            relative_tolerance=float(prereg["oracle"]["basis_relative_tolerance"]),
        )
        del raw_basis
        gc.collect()

        with h5py.File(args.source_h5, "r", swmr=True) as handle:
            samples = handle["sample_id"].asstr()[:]
            matches = np.flatnonzero(samples == args.sample_id)
            if len(matches) != 1:
                raise RuntimeError("oracle source sample selection is not singular")
            source_index = int(matches[0])
            if str(handle["split"].asstr()[source_index]) != "train":
                raise RuntimeError("oracle truth is not train-only")
            truth = torch.from_numpy(
                np.asarray(handle["wavefield"][source_index], dtype=np.float32)
            )
            time_s = torch.from_numpy(np.asarray(handle["time_s"][:], dtype=np.float32))
            t0 = float(handle["source_t0_s"][source_index])
            f0 = float(handle["source_f0_hz"][source_index])
        onset_start = int(
            torch.searchsorted(time_s, torch.tensor(t0 - 1.0 / f0), right=False)
        )
        observed_indices = (onset_start, onset_start + 1)
        future = future_indices(401, observed_indices)
        target_spectrum = torch.fft.rfft(truth, dim=0, norm="ortho")[:64] / PHYSICAL_SCALE
        target_pairs = torch.stack((target_spectrum.real, target_spectrum.imag), dim=1)

        error_flat = weighted_flatten(parent - target_pairs).float()
        basis_flat = weighted_flatten(basis).float()
        gram = basis_flat @ basis_flat.T
        right = -(basis_flat @ error_flat)
        ridge = float(prereg["oracle"]["relative_ridge"]) * float(
            torch.diagonal(gram).mean()
        )
        normal = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        coefficients = torch.linalg.solve(normal.double(), right.double()).float()
        correction = torch.einsum("k,kfczx->fczx", coefficients, basis)
        unrestricted_ratio = float(correction.norm() / parent.norm().clamp_min(1.0e-30))
        cap = float(prereg["oracle"]["trust_ratio_cap"])
        capped = correction * min(1.0, cap / max(unrestricted_ratio, 1.0e-30))
        zero = torch.zeros_like(parent)
        report = {
            "schema": "transfer_dg_pyramid_head_basis_oracle_cpu_report_v1",
            "status": "complete",
            "claim_scope": "train-only oracle capacity diagnostic; oracle coefficients are forbidden for deployment",
            "sample_id": args.sample_id,
            "family": public.family,
            "checkpoint": checkpoint_meta,
            "observed_indices": observed_indices,
            "basis": basis_meta,
            "selection": selection,
            "relative_ridge": float(prereg["oracle"]["relative_ridge"]),
            "unrestricted": candidate_metrics(parent, correction, truth, future),
            "trust_capped": candidate_metrics(parent, capped, truth, future),
            "parent": candidate_metrics(parent, zero, truth, future),
            "unmodeled_relative_l2_floor": relative_l2(
                render_time(target_pairs, time_count=401), truth
            ),
            "oracle_future_truth": True,
            "deployment_candidate": False,
            "elapsed_s": time.time() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(report, args.output_dir / "report.json")
        atomic_json(
            {
                "schema": "transfer_dg_pyramid_head_basis_oracle_cpu_terminal_v1",
                "status": "complete",
                "report": str((args.output_dir / "report.json").resolve()),
                "elapsed_s": report["elapsed_s"],
                "validation_opened": False,
                "test_id_opened": False,
            },
            args.output_dir / "terminal.json",
        )
        print(json.dumps({"event": "complete", **report}, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        atomic_json(
            {
                "schema": "transfer_dg_pyramid_head_basis_oracle_cpu_terminal_v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
                "validation_opened": False,
                "test_id_opened": False,
            },
            args.output_dir / "terminal.json",
        )
        raise
    finally:
        if public is not None:
            public.close()


if __name__ == "__main__":
    raise SystemExit(main())
