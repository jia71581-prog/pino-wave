#!/usr/bin/env python3
"""Train one paired Background-WFP/FNO E1 lane on train-only cache shards."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    ExteriorCPMLContract,
    extend_velocity_to_exterior,
)
from saved_time_phase_operator_v4.wfp import (  # noqa: E402
    BackgroundFrequencyOperator,
    parameter_count,
)


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
CALIBRATION_FREQUENCIES = (1, 4, 8, 12, 16, 24, 32, 48, 63)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class CacheCollection:
    def __init__(
        self, paths: list[Path], manifest: dict[str, object], *, expected_count: int = 256
    ) -> None:
        self.paths = [path.resolve() for path in paths]
        self.handles: list[h5py.File] = []
        self.records: list[tuple[int, int, str, str, str]] = []
        expected = str(manifest["selection_sha256"])
        try:
            for file_index, path in enumerate(self.paths):
                summary_path = path.with_suffix(path.suffix + ".summary.json")
                summary = json.loads(summary_path.read_text())
                if summary.get("status") != "complete" or summary.get("selection_sha256") != expected:
                    raise RuntimeError(f"invalid WFP cache summary: {summary_path}")
                if summary.get("validation_opened") or summary.get("test_id_opened"):
                    raise RuntimeError("WFP cache violates split boundary")
                handle = h5py.File(path, "r", swmr=True)
                self.handles.append(handle)
                if handle.attrs.get("status") != "complete" or handle.attrs.get("manifest_selection_sha256") != expected:
                    raise RuntimeError(f"invalid WFP cache: {path}")
                for local in range(len(handle["sample_id"])):
                    self.records.append((
                        file_index,
                        local,
                        str(handle["sample_id"].asstr()[local]),
                        str(handle["family"].asstr()[local]),
                        str(handle["role"].asstr()[local]),
                    ))
            if len(self.records) != int(expected_count):
                raise RuntimeError(f"E1 cache census mismatch: {len(self.records)}")
            self.frequency_count = int(self.handles[0].attrs["frequency_count"])
            self.physical_scale = float(self.handles[0].attrs["physical_scale"])
            self.auxiliary_scales = np.asarray(self.handles[0]["auxiliary_scales"], dtype=np.float32)
            self.profiles = {
                key: np.asarray(self.handles[0]["profiles"][key])
                for key in self.handles[0]["profiles"]
            }
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for handle in self.handles:
            handle.close()
        self.handles = []

    def positions(self, role: str) -> list[int]:
        return [index for index, row in enumerate(self.records) if row[4] == role]

    def split_positions(self) -> tuple[list[int], list[int], list[int]]:
        fit = self.positions("fit")
        calibration, confirmation = [], []
        for family in FAMILIES:
            values = sorted(
                [index for index, row in enumerate(self.records) if row[3] == family and row[4] == "holdout"],
                key=lambda index: self.records[index][2],
            )
            if len(values) != 16:
                raise RuntimeError(f"{family} holdout count changed")
            calibration.extend(values[::2])
            confirmation.extend(values[1::2])
        return fit, calibration, confirmation

    def raw(self, position: int, frequency: int) -> dict[str, object]:
        file_index, local, sample_id, family, role = self.records[position]
        handle = self.handles[file_index]
        return {
            "sample_id": sample_id,
            "family": family,
            "role": role,
            "velocity": np.asarray(handle["velocity_bg_saved_mps"][local], dtype=np.float32),
            "source_map": np.asarray(handle["source_map"][local], dtype=np.float32),
            "source_parameters": np.asarray(handle["source_parameters"][local], dtype=np.float32),
            "source_wavelet": np.asarray(handle["source_wavelet"][local], dtype=np.float32),
            "physical": np.asarray(handle["physical_coeff_norm"][local, frequency], dtype=np.float32),
            "auxiliary": np.asarray(handle["auxiliary_coeff_norm"][local, frequency], dtype=np.float32),
            "frequency_hz": float(handle["frequency_hz"][frequency]),
        }

    def targets(self, position: int, frequency: int) -> tuple[np.ndarray, np.ndarray, float]:
        file_index, local, _, _, _ = self.records[position]
        handle = self.handles[file_index]
        return (
            np.asarray(handle["physical_coeff_norm"][local, frequency], dtype=np.float32),
            np.asarray(handle["auxiliary_coeff_norm"][local, frequency], dtype=np.float32),
            float(handle["frequency_hz"][frequency]),
        )


class FeatureBuilder:
    def __init__(self, collection: CacheCollection) -> None:
        self.collection = collection
        self.contract = ExteriorCPMLContract()
        profiles = collection.profiles
        sigma_max = max(float(np.max(profiles["sigma_x"])), float(np.max(profiles["sigma_z"])), 1.0)
        alpha_max = max(float(np.max(profiles["alpha_x"])), float(np.max(profiles["alpha_z"])), 1.0)
        self.profile_features = np.stack((
            profiles["sigma_x"] / sigma_max,
            profiles["sigma_z"] / sigma_max,
            (profiles["kappa_x"] - 1.0) / 2.0,
            (profiles["kappa_z"] - 1.0) / 2.0,
            profiles["alpha_x"] / alpha_max,
            profiles["alpha_z"] / alpha_max,
            profiles["active_x"].astype(np.float32),
            profiles["active_z"].astype(np.float32),
        ), axis=0).astype(np.float32)
        self.active = np.logical_or(profiles["active_x"], profiles["active_z"])
        self.physical_mask = np.zeros((221, 241), dtype=np.float32)
        self.physical_mask[:201, 20:221] = 1.0
        x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
        z = np.arange(221, dtype=np.float32) * 10.0
        self.xx, self.zz = np.meshgrid(x, z)
        self.static_cache: dict[int, dict[str, object]] = {}

    def build(self, position: int, frequency: int, device: torch.device):
        if position not in self.static_cache:
            raw = self.collection.raw(position, frequency)
            velocity = np.asarray(
                extend_velocity_to_exterior(raw["velocity"], self.contract), dtype=np.float32
            )
            log_velocity = np.log(np.maximum(velocity, 1.0))
            grad_z, grad_x = np.gradient(log_velocity)
            medium = np.concatenate((
                np.stack(((velocity - 4500.0) / 2500.0, 20.0 * grad_x, 20.0 * grad_z), axis=0),
                self.profile_features,
                self.physical_mask[None],
            ), axis=0).astype(np.float32)
            source_map = np.zeros((221, 241), dtype=np.float32)
            source_map[:201, 20:221] = raw["source_map"]
            wavelet_fft = np.fft.rfft(raw["source_wavelet"], norm="ortho")
            wavelet_fft /= max(float(np.max(np.abs(wavelet_fft))), 1.0e-12)
            self.static_cache[position] = {
                "sample_id": raw["sample_id"], "family": raw["family"],
                "role": raw["role"], "medium": medium, "source_map": source_map,
                "source_parameters": raw["source_parameters"], "wavelet_fft": wavelet_fft,
            }
        row = self.static_cache[position]
        medium = row["medium"]
        source_map = row["source_map"]
        wavelet_fft = row["wavelet_fft"]
        wave = wavelet_fft[frequency]
        sx, sz, f0, t0 = row["source_parameters"]
        source = np.stack((
            source_map,
            source_map * float(wave.real),
            source_map * float(wave.imag),
            np.clip((self.xx - float(sx)) / 2000.0, -1.2, 1.2),
            np.clip((self.zz - float(sz)) / 2000.0, -0.2, 1.2),
        ), axis=0).astype(np.float32)
        physical, auxiliary, frequency_hz = self.collection.targets(position, frequency)
        scalars = np.asarray((
            frequency_hz / 200.0,
            float(f0) / 30.0,
            float(t0) / 0.2,
            float(wave.real),
            float(wave.imag),
        ), dtype=np.float32)
        metadata = {
            "sample_id": row["sample_id"], "family": row["family"],
            "role": row["role"], "frequency_hz": frequency_hz,
        }
        return (
            torch.from_numpy(medium)[None].to(device),
            torch.from_numpy(source)[None].to(device),
            torch.from_numpy(scalars)[None].to(device),
            torch.from_numpy(physical)[None].to(device),
            torch.from_numpy(auxiliary)[None].to(device),
            metadata,
        )


def objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    auxiliary_prediction: torch.Tensor,
    auxiliary_target: torch.Tensor,
    active: torch.Tensor,
    cpml_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    physical = F.smooth_l1_loss(prediction.float(), target.float(), beta=0.05)
    mask = active[None, None].expand_as(auxiliary_target)
    auxiliary = F.smooth_l1_loss(
        auxiliary_prediction.float()[mask], auxiliary_target.float()[mask], beta=0.05
    )
    loss = physical + float(cpml_weight) * auxiliary
    return loss, {"physical": float(physical.detach()), "cpml": float(auxiliary.detach())}


@torch.inference_mode()
def evaluate(
    model: BackgroundFrequencyOperator,
    builder: FeatureBuilder,
    positions: list[int],
    frequencies: tuple[int, ...] | range,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    rows = []
    top_max = 0.0
    outer_max = 0.0
    started = time.perf_counter()
    active = torch.from_numpy(builder.active).to(device)
    frequency_values = tuple(int(value) for value in frequencies)
    for position in positions:
        physical_num = physical_den = auxiliary_num = auxiliary_den = 0.0
        metadata = None
        for start in range(0, len(frequency_values), 4):
            items = [
                builder.build(position, frequency, device)
                for frequency in frequency_values[start : start + 4]
            ]
            medium, source, scalars, target, auxiliary_target = (
                torch.cat([item[index] for item in items], dim=0)
                for index in range(5)
            )
            metadata = items[-1][5]
            prediction, auxiliary = model(medium, source, scalars)
            difference = prediction.double() - target.double()
            physical_num += float(difference.square().sum())
            physical_den += float(target.double().square().sum())
            mask = active[None, None].expand_as(auxiliary_target)
            aux_diff = auxiliary.double()[mask] - auxiliary_target.double()[mask]
            auxiliary_num += float(aux_diff.square().sum())
            auxiliary_den += float(auxiliary_target.double()[mask].square().sum())
            top_max = max(top_max, float(prediction[..., 0, :].abs().max()), float(auxiliary[:, :2, 0].abs().max()))
            outer_max = max(
                outer_max,
                float(auxiliary[:, :2, -1].abs().max()),
                float(auxiliary[:, :2, :, 0].abs().max()),
                float(auxiliary[:, :2, :, -1].abs().max()),
            )
        assert metadata is not None
        rows.append({
            "sample_id": metadata["sample_id"],
            "family": metadata["family"],
            "physical_relative_l2": math.sqrt(physical_num / max(physical_den, 1.0e-30)),
            "cpml_normalized_relative_l2": math.sqrt(auxiliary_num / max(auxiliary_den, 1.0e-30)),
        })
    by_family = {
        family: float(np.mean([row["physical_relative_l2"] for row in rows if row["family"] == family]))
        for family in FAMILIES
    }
    return {
        "record_count": len(rows),
        "frequency_count": len(frequency_values),
        "physical_mean": float(np.mean([row["physical_relative_l2"] for row in rows])),
        "physical_max": float(np.max([row["physical_relative_l2"] for row in rows])),
        "cpml_normalized_mean": float(np.mean([row["cpml_normalized_relative_l2"] for row in rows])),
        "per_family": by_family,
        "top_pressure_max_abs": top_max,
        "outer_pressure_max_abs": outer_max,
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("wfp", "fno"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--cpml-weight", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E1 lane: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    if preregistration.get("candidate") != "transfer_dg_wfp_e1_20260902":
        raise RuntimeError("wrong E1 preregistration")
    collection = CacheCollection(args.cache, manifest)
    fit, calibration, confirmation = collection.split_positions()
    if len(fit) != 192 or len(calibration) != 32 or len(confirmation) != 32:
        raise RuntimeError("E1 fit/calibration/confirmation census mismatch")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = BackgroundFrequencyOperator(
        medium_channels=12, source_channels=5, width=32, rank=16, depth=4,
        arm=args.arm, radii=(1, 2, 3, 4), fno_modes=32,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates, eta_min=args.lr * 0.01
    )
    builder = FeatureBuilder(collection)
    active = torch.from_numpy(builder.active).to(device)
    by_family = defaultdict(list)
    for position in fit:
        by_family[collection.records[position][3]].append(position)
    rng = np.random.default_rng(args.seed + 17)
    identity = {
        "schema": "transfer_dg_wfp_e1_lane_identity_v1",
        "arm": args.arm,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "updates": args.updates,
        "lr": args.lr,
        "cpml_weight": args.cpml_weight,
        "manifest_sha256": sha256(args.manifest),
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "wfp_module_sha256": sha256(ROOT / "saved_time_phase_operator_v4/wfp.py"),
        "cache_sha256": {
            str(path): json.loads(
                path.with_suffix(path.suffix + ".summary.json").read_text()
            )["output_sha256"]
            for path in args.cache
        },
        "fit_count": len(fit), "calibration_count": len(calibration),
        "confirmation_count": len(confirmation),
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(identity, args.output_dir / "run_identity.json")
    best = None
    started = time.time()
    metrics_path = args.output_dir / "metrics.jsonl"
    for update in range(1, args.updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        components = []
        for family in FAMILIES:
            position = int(rng.choice(by_family[family]))
            frequency = int(rng.integers(0, collection.frequency_count))
            medium, source, scalars, target, auxiliary_target, _ = builder.build(
                position, frequency, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, auxiliary = model(medium, source, scalars)
                loss, component = objective(
                    prediction, target, auxiliary, auxiliary_target, active, args.cpml_weight
                )
                loss = loss / len(FAMILIES)
            loss.backward()
            components.append(component)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update == 1 or update % 20 == 0:
            event = {
                "event": "update", "update": update,
                "physical_loss": float(np.mean([row["physical"] for row in components])),
                "cpml_loss": float(np.mean([row["cpml"] for row in components])),
                "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            metrics = evaluate(
                model, builder, calibration, CALIBRATION_FREQUENCIES, device
            )
            event = {"event": "calibration", "update": update, "metrics": metrics}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            score = float(metrics["physical_mean"])
            checkpoint = {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "update": update, "identity": identity, "calibration": metrics,
            }
            atomic_checkpoint(checkpoint, args.output_dir / "latest.pt")
            if best is None or score < best["score"]:
                best = {"score": score, "update": update, "metrics": metrics}
                atomic_checkpoint(checkpoint, args.output_dir / "best.pt")
                atomic_json(best, args.output_dir / "best.json")
            print(json.dumps({"event": "calibration", "update": update, "score": score}), flush=True)
    if best is None:
        raise RuntimeError("E1 lane produced no checkpoint")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    calibration_full = evaluate(model, builder, calibration, range(collection.frequency_count), device)
    confirmation_full = evaluate(model, builder, confirmation, range(collection.frequency_count), device)
    terminal = {
        "schema": "transfer_dg_wfp_e1_lane_terminal_v1",
        "status": "complete",
        "arm": args.arm, "seed": args.seed,
        "best_update": best["update"],
        "best_calibration_subset": best["metrics"],
        "calibration_full": calibration_full,
        "confirmation_full": confirmation_full,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "best_checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_s": time.time() - started,
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(terminal, args.output_dir / "terminal.json")
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    collection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
