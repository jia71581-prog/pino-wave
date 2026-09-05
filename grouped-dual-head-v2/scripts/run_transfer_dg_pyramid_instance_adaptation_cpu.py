#!/usr/bin/env python3
"""Run one leakage-safe Transfer-DG adaptation and post-seal CPU evaluation."""
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

from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402
from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)
from saved_time_phase_operator_v4.instance_adaptation.contracts import (  # noqa: E402
    future_indices,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import (  # noqa: E402
    GuardedOnsetDataset,
)
from saved_time_phase_operator_v4.instance_adaptation.transfer_dg_adapt import (  # noqa: E402
    TransferDGConfig,
    build_transfer_dg_linear_system,
    pairs_to_complex,
    solve_transfer_dg_system,
)
from saved_time_phase_operator_v4.phase_carrier import (  # noqa: E402
    rotate_complex_pairs,
    travel_phase_carrier,
)
from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    model_prediction,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    atomic_checkpoint,
    atomic_json,
    sha256,
)


PHYSICAL_SCALE = 1.0e-8


class PublicPyramidRecord:
    """Build model inputs while never reading residual or target datasets."""

    def __init__(
        self,
        residual_paths: list[Path],
        travel_paths: list[Path],
        sample_id: str,
    ) -> None:
        self.residual_handles: list[h5py.File] = []
        self.travel_handles: list[h5py.File] = []
        self.sample_id = str(sample_id)
        residual_location = None
        travel_location = None
        try:
            for file_index, path in enumerate(residual_paths):
                handle = h5py.File(path, "r", swmr=True)
                self.residual_handles.append(handle)
                if handle.attrs.get("status") != "complete":
                    raise RuntimeError(f"incomplete residual cache: {path}")
                if handle.attrs.get("validation_opened") or handle.attrs.get("test_id_opened"):
                    raise RuntimeError("residual cache split marker is open")
                samples = handle["sample_id"].asstr()[:]
                matches = np.flatnonzero(samples == self.sample_id)
                if len(matches):
                    if residual_location is not None or len(matches) != 1:
                        raise RuntimeError("duplicate public residual-cache sample")
                    residual_location = (file_index, int(matches[0]))
            for file_index, path in enumerate(travel_paths):
                handle = h5py.File(path, "r", swmr=True)
                self.travel_handles.append(handle)
                if handle.attrs.get("status") != "complete":
                    raise RuntimeError(f"incomplete travel cache: {path}")
                if handle.attrs.get("validation_opened") or handle.attrs.get("test_id_opened"):
                    raise RuntimeError("travel cache split marker is open")
                samples = handle["sample_id"].asstr()[:]
                matches = np.flatnonzero(samples == self.sample_id)
                if len(matches):
                    if travel_location is not None or len(matches) != 1:
                        raise RuntimeError("duplicate public travel-cache sample")
                    travel_location = (file_index, int(matches[0]))
            if residual_location is None or travel_location is None:
                raise KeyError(f"sample is absent from public caches: {self.sample_id}")

            residual_handle = self.residual_handles[residual_location[0]]
            residual_local = residual_location[1]
            travel_handle = self.travel_handles[travel_location[0]]
            travel_local = travel_location[1]
            if float(residual_handle.attrs.get("physical_scale", -1.0)) != PHYSICAL_SCALE:
                raise RuntimeError("physical normalization scale drift")
            self.family = str(residual_handle["family"].asstr()[residual_local])
            self.medium = np.asarray(
                residual_handle["medium"][residual_local], dtype=np.float32
            )
            self.source_map = np.asarray(
                residual_handle["source_map"][residual_local], dtype=np.float32
            )
            self.parameters = np.asarray(
                residual_handle["source_parameters"][residual_local], dtype=np.float32
            )
            self.source_wavelet = np.asarray(
                residual_handle["source_wavelet"][residual_local], dtype=np.float32
            )
            self.frequency_hz = np.asarray(
                residual_handle["frequency_hz"][:], dtype=np.float32
            )
            self.travel_physical = np.asarray(
                travel_handle["travel_physical_s"][travel_local], dtype=np.float32
            )
            self.travel_exterior = np.asarray(
                travel_handle["travel_exterior_s"][travel_local], dtype=np.float32
            )
            self.wavelet_fft = np.fft.rfft(self.source_wavelet, norm="ortho")
            self.wavelet_fft /= max(float(np.max(np.abs(self.wavelet_fft))), 1.0e-12)
            if self.frequency_hz.shape != (64,):
                raise RuntimeError("public frequency axis is not the registered 64 bins")
        except Exception:
            self.close()
            raise

    def block(self, start: int, device: torch.device) -> dict[str, torch.Tensor]:
        frequencies = range(int(start), int(start) + BLOCK)
        x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
        z = np.arange(221, dtype=np.float32) * 10.0
        xx, zz = np.meshgrid(x, z)
        extended = np.zeros((221, 241), dtype=np.float32)
        extended[:201, 20:221] = self.source_map
        sx, sz, f0, t0 = (float(value) for value in self.parameters)
        sources = []
        scalars = []
        for frequency in frequencies:
            wave = self.wavelet_fft[frequency]
            sources.append(
                np.stack(
                    (
                        extended,
                        extended * float(wave.real),
                        extended * float(wave.imag),
                        np.clip((xx - sx) / 2000.0, -1.2, 1.2),
                        np.clip((zz - sz) / 2000.0, -0.2, 1.2),
                    ),
                    axis=0,
                ).astype(np.float32)
            )
            scalars.append(
                np.asarray(
                    (
                        float(self.frequency_hz[frequency]) / 200.0,
                        f0 / 30.0,
                        t0 / 0.2,
                        float(wave.real),
                        float(wave.imag),
                    ),
                    dtype=np.float32,
                )
            )
        return {
            "sample_id": self.sample_id,
            "family": self.family,
            "medium": torch.from_numpy(self.medium)[None, None]
            .expand(1, BLOCK, -1, -1, -1)
            .to(device),
            "source": torch.from_numpy(np.stack(sources))[None].to(device),
            "scalars": torch.from_numpy(np.stack(scalars))[None].to(device),
            "travel_physical": torch.from_numpy(self.travel_physical)[None]
            .expand(BLOCK, -1, -1)
            .to(device),
            "travel_exterior": torch.from_numpy(self.travel_exterior)[None]
            .expand(BLOCK, -1, -1)
            .to(device),
            "frequency_hz": torch.from_numpy(
                self.frequency_hz[int(start) : int(start) + BLOCK]
            ).to(device),
        }

    def close(self) -> None:
        for handle in self.residual_handles + self.travel_handles:
            if handle.id.valid:
                handle.close()
        self.residual_handles = []
        self.travel_handles = []


class BranchTrunkCapture:
    def __init__(self, model: PyramidMoECoupledWaveOperator) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.handles = [
            model.trunk_basis.register_forward_hook(self._hook("trunk")),
            model.branch_context.register_forward_hook(self._hook("branch")),
            model.medium_mionet_branch.register_forward_hook(self._hook("medium")),
            model.source_mionet_branch.register_forward_hook(self._hook("source")),
        ]

    def _hook(self, name: str):
        def capture(_module, _inputs, output) -> None:
            self.values[name] = output.detach()

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def branch_trunk_modes(
    model: PyramidMoECoupledWaveOperator,
    capture: BranchTrunkCapture,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    required = {"trunk", "branch", "medium", "source"}
    if set(capture.values) != required:
        raise RuntimeError("branch-trunk hooks did not capture every registered tensor")
    frequency_count = int(batch["frequency_hz"].numel())
    rank = int(model.branch_trunk_rank)
    trunk = capture.values["trunk"].reshape(frequency_count, rank, 221, 241)
    branch = capture.values["branch"].reshape(frequency_count, 12, rank)
    medium = capture.values["medium"].reshape(frequency_count, 12, rank)
    source = capture.values["source"].reshape(frequency_count, 12, rank)
    amplitudes = (
        branch[:, :2] * model.branch_trunk_scale[:2][None, :, None]
        + medium[:, :2]
        * source[:, :2]
        / math.sqrt(rank)
        * model.mionet_scale[:2][None, :, None]
    )
    modes = torch.einsum("fcr,frzx->rfczx", amplitudes, trunk)
    modes = modes[..., :201, 20:221].clone()
    modes[..., 0, :] = 0.0
    carrier = travel_phase_carrier(batch["travel_physical"], batch["frequency_hz"])
    rotated = rotate_complex_pairs(
        modes.reshape(rank * frequency_count, 2, 201, 201),
        carrier[None]
        .expand(rank, -1, -1, -1, -1)
        .reshape(rank * frequency_count, 2, 201, 201),
    )
    return rotated.reshape(rank, frequency_count, 2, 201, 201)


def whiten_basis(
    basis: torch.Tensor,
    parent: torch.Tensor,
    *,
    relative_tolerance: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    flattened = basis.reshape(basis.shape[0], -1)
    gram = flattened @ flattened.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    maximum = float(eigenvalues.max())
    keep = eigenvalues > maximum * float(relative_tolerance)
    if not bool(keep.any()):
        raise RuntimeError("learned branch-trunk basis is numerically empty")
    transform = (
        eigenvectors[:, keep].T
        / eigenvalues[keep].sqrt().clamp_min(1.0e-12)[:, None]
    ).float()
    orthogonal = transform @ flattened
    scale = parent.norm() / math.sqrt(int(keep.sum()))
    orthogonal = orthogonal.reshape(int(keep.sum()), *basis.shape[1:]) * scale
    metadata = {
        "input_rank": int(basis.shape[0]),
        "effective_rank": int(keep.sum()),
        "gram_eigenvalue_min_kept": float(eigenvalues[keep].min()),
        "gram_eigenvalue_max": maximum,
        "relative_tolerance": float(relative_tolerance),
        "mode_norm": float(scale),
    }
    return orthogonal, metadata


@torch.inference_mode()
def predict_and_build_basis(
    model: PyramidMoECoupledWaveOperator,
    public: PublicPyramidRecord,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cpu")
    predictions = []
    mode_blocks = []
    capture = BranchTrunkCapture(model)
    try:
        for start in range(0, 64, BLOCK):
            batch = public.block(start, device)
            capture.values.clear()
            prediction, _ = model_prediction(model, batch)
            predictions.append(prediction.detach().cpu())
            mode_blocks.append(branch_trunk_modes(model, capture, batch).detach().cpu())
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
    return (
        torch.cat(predictions, dim=0).float(),
        torch.cat(mode_blocks, dim=1).float(),
        torch.from_numpy(public.frequency_hz).float(),
    )


def render_time(coefficients: torch.Tensor, *, time_count: int) -> torch.Tensor:
    retained = pairs_to_complex(coefficients.float()) * PHYSICAL_SCALE
    full = torch.zeros(
        (time_count // 2 + 1, *retained.shape[-2:]), dtype=retained.dtype
    )
    full[: retained.shape[0]] = retained
    return torch.fft.irfft(full, n=time_count, dim=0, norm="ortho")


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction.double() - target.double()).norm()
        / target.double().norm().clamp_min(1.0e-30)
    )


def evaluate_after_seal(
    adapted_path: Path,
    source_h5: Path,
    source_index: int,
    observed_indices: tuple[int, int],
) -> dict[str, object]:
    payload = torch.load(adapted_path, map_location="cpu", weights_only=False)
    parent_field = render_time(payload["parent_coefficients"], time_count=401)
    adapted_field = render_time(payload["adapted_coefficients"], time_count=401)
    with h5py.File(source_h5, "r", swmr=True) as handle:
        truth = torch.from_numpy(
            np.asarray(handle["wavefield"][int(source_index)], dtype=np.float32)
        )
    future = future_indices(401, observed_indices)
    thirds = torch.tensor_split(future, 3)
    parent_future = relative_l2(parent_field[future], truth[future])
    adapted_future = relative_l2(adapted_field[future], truth[future])
    parent_full = relative_l2(parent_field, truth)
    adapted_full = relative_l2(adapted_field, truth)
    onset = torch.tensor(observed_indices, dtype=torch.long)
    return {
        "time_count": 401,
        "future_start_index": int(future[0]),
        "future_frame_count": int(future.numel()),
        "parent_full_relative_l2": parent_full,
        "adapted_full_relative_l2": adapted_full,
        "full_relative_improvement": 1.0 - adapted_full / max(parent_full, 1.0e-300),
        "parent_future_relative_l2": parent_future,
        "adapted_future_relative_l2": adapted_future,
        "future_relative_improvement": 1.0 - adapted_future / max(parent_future, 1.0e-300),
        "parent_onset_relative_l2": relative_l2(parent_field[onset], truth[onset]),
        "adapted_onset_relative_l2": relative_l2(adapted_field[onset], truth[onset]),
        "future_time_bands": {
            name: {
                "start": int(indices[0]),
                "stop": int(indices[-1]),
                "parent_relative_l2": relative_l2(parent_field[indices], truth[indices]),
                "adapted_relative_l2": relative_l2(adapted_field[indices], truth[indices]),
            }
            for name, indices in zip(("early", "middle", "late"), thirds, strict=True)
        },
        "adapted_target_0p05_pass": adapted_future <= 0.05,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("CPU-only run requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if len(args.residual_cache) != 4 or len(args.travel) != 4:
        raise RuntimeError("the registered run requires four residual and four travel shards")
    observed_bindings = {
        "runner_sha256": sha256(Path(__file__)),
        "adapt_core_sha256": sha256(
            ROOT / "saved_time_phase_operator_v4/instance_adaptation/transfer_dg_adapt.py"
        ),
        "model_sha256": sha256(ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py"),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        "source_h5_sha256": sha256(args.source_h5),
    }
    for index, path in enumerate(args.residual_cache):
        observed_bindings[f"residual_summary_{index}_sha256"] = sha256(
            path.with_suffix(path.suffix + ".summary.json")
        )
    for index, path in enumerate(args.travel):
        observed_bindings[f"travel_summary_{index}_sha256"] = sha256(
            path.with_suffix(path.suffix + ".summary.json")
        )
    for key, observed in observed_bindings.items():
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    if args.sample_id != prereg["data"]["sample_id"]:
        raise RuntimeError("sample differs from preregistration")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    identity = {
        "schema": "transfer_dg_pyramid_instance_adaptation_cpu_identity_v1",
        "pid": os.getpid(),
        "device": "cpu",
        "threads": args.threads,
        "sample_id": args.sample_id,
        "bindings": observed_bindings,
        "validation_opened": False,
        "test_id_opened": False,
        "started_unix_s": started,
    }
    atomic_json(identity, args.output_dir / "run_identity.json")
    print(json.dumps({"event": "identity", **identity}, sort_keys=True), flush=True)

    public = None
    guarded_dataset = None
    try:
        manifest = build_manifest(args.source_h5)
        guarded_dataset = GuardedOnsetDataset(
            args.source_h5,
            manifest,
            split="train",
            sample_ids=(args.sample_id,),
        )
        if len(guarded_dataset) != 1:
            raise RuntimeError("guarded sample selection is not singular")
        guarded = guarded_dataset[0]
        if guarded.medium_type != prereg["data"]["family"]:
            raise RuntimeError("guarded sample family drift")
        public = PublicPyramidRecord(args.residual_cache, args.travel, args.sample_id)
        if public.family != guarded.medium_type:
            raise RuntimeError("public-cache family differs from guarded record")

        checkpoint = torch.load(
            args.candidate_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_meta = {
            key: checkpoint.get(key)
            for key in (
                "schema",
                "epoch",
                "next_step",
                "update",
                "validation_opened",
                "test_id_opened",
            )
        }
        if checkpoint_meta["validation_opened"] or checkpoint_meta["test_id_opened"]:
            raise RuntimeError("candidate checkpoint has opened a sealed split")
        model = PyramidMoECoupledWaveOperator(use_mhc=True)
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        model.eval().requires_grad_(False)
        if parameter_count(model) != int(prereg["model"]["parameter_count"]):
            raise RuntimeError("candidate parameter count drift")

        parent_coefficients, raw_basis, frequency_hz = predict_and_build_basis(model, public)
        del model
        gc.collect()
        basis, basis_meta = whiten_basis(
            raw_basis,
            parent_coefficients,
            relative_tolerance=float(prereg["adaptation"]["basis_relative_tolerance"]),
        )
        del raw_basis
        gc.collect()

        config = TransferDGConfig(**prereg["adaptation"]["transfer_dg_config"])
        observed_normalized = guarded.observed_wavefield.float() / PHYSICAL_SCALE
        velocity_saved = guarded.velocity_mps.float()
        if velocity_saved.shape == (1, 201, 201):
            velocity_saved = velocity_saved[0]
        if velocity_saved.shape != (201, 201):
            raise RuntimeError(
                f"guarded saved-grid velocity shape drift: {tuple(velocity_saved.shape)}"
            )
        system = build_transfer_dg_linear_system(
            parent_coefficients,
            basis,
            velocity_saved,
            frequency_hz,
            observed_normalized,
            guarded.observed_indices,
            time_count=401,
            source_pairs=None,
            config=config,
        )
        result = solve_transfer_dg_system(
            system,
            parent_coefficients,
            basis,
            config=config,
        )
        adapted_payload = {
            "schema": "transfer_dg_pyramid_instance_adapted_coefficients_v1",
            "sample_id": args.sample_id,
            "family": guarded.medium_type,
            "parent_checkpoint_sha256": observed_bindings["candidate_checkpoint_sha256"],
            "parent_coefficients": parent_coefficients,
            "adapted_coefficients": result.candidate.cpu(),
            "adaptation_coefficients": result.coefficients.cpu(),
            "observed_indices": guarded.observed_indices,
            "basis": basis_meta,
            "accepted_online_objective": result.accepted,
            "future_truth_used": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        adapted_path = args.output_dir / "adapted_coefficients.pt"
        atomic_checkpoint(adapted_payload, adapted_path)
        adaptation_terminal = {
            "schema": "transfer_dg_pyramid_instance_adaptation_terminal_v1",
            "status": "accepted" if result.accepted else "rolled_back_to_parent",
            "sample_id": args.sample_id,
            "family": guarded.medium_type,
            "checkpoint": checkpoint_meta,
            "observed_indices": guarded.observed_indices,
            "access_audit": guarded.audit.payload(),
            "basis": basis_meta,
            "parent_online_objective": result.parent_objective,
            "candidate_online_objective": result.candidate_objective,
            "online_objective_improvement": 1.0
            - result.candidate_objective / max(result.parent_objective, 1.0e-300),
            "correction_ratio": result.correction_ratio,
            "condition_number": result.condition_number,
            "adapted_coefficients": str(adapted_path.resolve()),
            "adapted_coefficients_sha256": sha256(adapted_path),
            "future_truth_used": False,
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(adaptation_terminal, args.output_dir / "adaptation_terminal.json")
        print(
            json.dumps({"event": "adaptation_sealed", **adaptation_terminal}, sort_keys=True),
            flush=True,
        )

        del basis, system, result, parent_coefficients
        gc.collect()
        evaluation = evaluate_after_seal(
            adapted_path,
            args.source_h5,
            guarded.source_index,
            guarded.observed_indices,
        )
        evaluation_payload = {
            "schema": "transfer_dg_pyramid_instance_adaptation_evaluation_v1",
            "status": "complete",
            "scope": "post-seal train-only single-record diagnostic",
            "sample_id": args.sample_id,
            "family": guarded.medium_type,
            "checkpoint": checkpoint_meta,
            "adaptation_status": adaptation_terminal["status"],
            "metrics": evaluation,
            "future_truth_opened_only_after_adaptation_serialized": True,
            "elapsed_s": time.time() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(evaluation_payload, args.output_dir / "evaluation.json")
        terminal = {
            "schema": "transfer_dg_pyramid_instance_adaptation_run_terminal_v1",
            "status": "complete",
            "adaptation": adaptation_terminal,
            "evaluation": evaluation_payload,
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(terminal, args.output_dir / "terminal.json")
        print(json.dumps({"event": "complete", **evaluation_payload}, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        failure = {
            "schema": "transfer_dg_pyramid_instance_adaptation_run_terminal_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(failure, args.output_dir / "terminal.json")
        raise
    finally:
        if public is not None:
            public.close()
        if guarded_dataset is not None:
            guarded_dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
