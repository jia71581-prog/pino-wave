#!/usr/bin/env python3
"""PA-CORA P1b: stable three-window continuation from a V9 spectral parent."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.b2_v5_components import per_record_relative_l2, temporal_band_relative_l2
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")
TEMPORAL_SLICES = {"early": (0, 19), "middle": (19, 38), "late": (38, 56)}
FFT_BANDS = ((0, 5), (5, 12), (12, 29))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _f0_bin(value: float) -> str:
    if value < 15.0:
        return "low"
    if value < 22.0:
        return "middle"
    return "high"


def curriculum_slot_weights(epoch: int) -> tuple[float, float, float]:
    """Onset-first curriculum frozen before launch."""
    if int(epoch) <= 2:
        return (1.0, 0.0, 0.0)
    if int(epoch) <= 5:
        return (0.7, 0.3, 0.0)
    return (0.6, 0.25, 0.15)


class ThreeWindowCache:
    def __init__(self, paths: list[Path], manifests: list[dict], *, key: str = "cond"):
        if len(paths) != 3 or len(manifests) != 3:
            raise ValueError("P1b requires exactly three windows")
        self.handles = []
        self.manifests = manifests
        self.key = key
        for path, manifest in zip(paths, manifests):
            handle = h5py.File(path, "r", swmr=True)
            self.handles.append(handle)
            if handle.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError(f"unexpected cache schema: {path}")
            if handle.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError(f"cache/manifest drift: {path}")
            if key not in handle:
                raise RuntimeError(f"conditioning dataset absent: {key}")
        counts = {int(handle["target"].shape[0]) for handle in self.handles}
        if len(counts) != 1:
            raise RuntimeError("window caches have different record counts")
        self.record_count = counts.pop()
        self.onset_target_norm = np.empty(self.record_count, dtype=np.float64)
        for index in range(self.record_count):
            target = self.handles[0]["target"][index, 8:].astype(np.float32)
            self.onset_target_norm[index] = np.linalg.norm(target.astype(np.float64))
        self.strata = defaultdict(list)
        for index, row in enumerate(manifests[0]["records"]):
            self.strata[(str(row["family"]), _f0_bin(float(row["source_f0_hz"])))].append(index)
        self.stratum_keys = sorted(self.strata)

    def batch(self, slot: int, indices: list[int]):
        handle = self.handles[int(slot)]
        base = np.stack([handle["base_seq"][index] for index in indices]).astype(np.float32)
        target = np.stack([handle["target"][index] for index in indices]).astype(np.float32)
        conditioning = np.stack([handle[self.key][index] for index in indices]).astype(np.float32)
        return (
            torch.from_numpy(base)[:, :, None],
            torch.from_numpy(target)[:, :, None],
            torch.from_numpy(conditioning),
            torch.from_numpy(self.onset_target_norm[np.asarray(indices)].astype(np.float32)),
        )

    def balanced_indices(
        self, rng: np.random.Generator, count: int, offset: int
    ) -> list[int]:
        values = []
        for position in range(int(count)):
            key = self.stratum_keys[(int(offset) + position) % len(self.stratum_keys)]
            pool = self.strata[key]
            values.append(int(pool[int(rng.integers(0, len(pool)))]))
        return values

    def close(self) -> None:
        for handle in self.handles:
            handle.close()
        self.handles = []


def robust_multiphase_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    onset_target_norm: torch.Tensor,
    *,
    denominator_floor_fraction: float = 0.25,
    spectral_weight: float = 0.1,
    spectral_energy_floor_fraction: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Relative loss with a same-source onset-energy denominator floor."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must match [B,T,1,Z,X]")
    batch = prediction.shape[0]
    onset = torch.as_tensor(
        onset_target_norm, dtype=prediction.dtype, device=prediction.device
    ).reshape(batch)
    error_norm = (prediction.float() - target.float()).flatten(1).norm(dim=1)
    target_norm = target.float().flatten(1).norm(dim=1)
    denominator = torch.maximum(
        target_norm, float(denominator_floor_fraction) * onset.float()
    ).clamp_min(1.0e-8)
    relative = (error_norm / denominator).mean()

    prediction_fft = torch.fft.rfft(prediction.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    target_total = target_fft.abs().square().sum((1, 2, 3, 4))
    # torch.fft.rfft is unnormalized, so match the time-domain onset energy to
    # the FFT-energy scale before using it as a robust denominator floor.
    onset_floor_energy = onset.float().square() * prediction.shape[1]
    spectral_rows = []
    for start, stop in FFT_BANDS:
        band_target = target_fft[:, start:stop]
        band_error = prediction_fft[:, start:stop] - band_target
        numerator = band_error.abs().square().sum((1, 2, 3, 4)).sqrt()
        band_energy = band_target.abs().square().sum((1, 2, 3, 4))
        floor = float(spectral_energy_floor_fraction) * torch.maximum(
            target_total, onset_floor_energy
        )
        spectral_rows.append(numerator / torch.maximum(band_energy, floor).sqrt().clamp_min(1.0e-8))
    spectral = torch.stack(spectral_rows, dim=1).mean()
    loss = relative + float(spectral_weight) * spectral
    return loss, {
        "relative_l2_robust": relative.detach(),
        "spectral_robust": spectral.detach(),
        "target_floor_active_fraction": (target_norm < float(denominator_floor_fraction) * onset).float().mean().detach(),
    }


def _load_holdout(paths: list[Path], manifests: list[dict], *, key: str = "cond") -> list[dict]:
    outputs = []
    for path, manifest in zip(paths, manifests):
        with h5py.File(path, "r", swmr=True) as cache:
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("holdout cache/manifest drift")
            outputs.append(
                {
                    "base": torch.from_numpy(cache["base_seq"][:].astype(np.float32))[:, :, None],
                    "target": torch.from_numpy(cache["target"][:].astype(np.float32))[:, :, None],
                    "cond": torch.from_numpy(cache[key][:].astype(np.float32)),
                    "families": cache["family"][:].astype(str),
                }
            )
    return outputs


@torch.inference_mode()
def evaluate_phase(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    micro_records: int,
) -> dict:
    model.eval()
    relative_rows, correction_rows, frequency_rows = [], [], []
    temporal_rows = {name: [] for name in TEMPORAL_SLICES}
    for lo in range(0, len(data["families"]), micro_records):
        hi = min(lo + micro_records, len(data["families"]))
        anchor = data["base"][lo:hi].to(device)
        target = data["target"][lo:hi].to(device)
        prediction = model.forward_anchored(
            anchor,
            data["cond"][lo:hi].to(device),
            initial_state=target[:, :8, 0],
        )
        future, truth, anchor_future = prediction[:, 8:], target[:, 8:], anchor[:, 8:]
        relative_rows.append(per_record_relative_l2(future.double(), truth.double()).cpu())
        correction_rows.append(per_record_relative_l2(future.double(), anchor_future.double()).cpu())
        frequency_rows.append(temporal_band_relative_l2(future, truth, energy_floor_fraction=0.005).cpu())
        for name, (start, stop) in TEMPORAL_SLICES.items():
            temporal_rows[name].append(
                per_record_relative_l2(future[:, start:stop].double(), truth[:, start:stop].double()).cpu()
            )
    relative = torch.cat(relative_rows)
    frequency = torch.cat(frequency_rows)
    return {
        "aggregate": float(relative.mean()),
        "maximum": float(relative.max()),
        "nonworse_vs_anchor": None,
        "correction_energy": float(torch.cat(correction_rows).mean()),
        "per_family": {
            family: float(relative[torch.from_numpy(data["families"] == family)].mean())
            for family in FAMILIES
        },
        "temporal": {name: float(torch.cat(values).mean()) for name, values in temporal_rows.items()},
        "frequency": {
            name: float(frequency[:, index].mean())
            for index, name in enumerate(("low", "middle", "high"))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, action="append", required=True)
    parser.add_argument("--fit-cache", type=Path, action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", required=True)
    parser.add_argument("--holdout-cache", type=Path, action="append", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--micro-records", type=int, default=4)
    args = parser.parse_args()
    if len(args.fit_manifest) != 3 or len(args.fit_cache) != 3:
        raise ValueError("P1b requires three fit windows")
    if len(args.holdout_manifest) != 3 or len(args.holdout_cache) != 3:
        raise ValueError("P1b requires three holdout windows")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    fit = None
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if _sha256(Path(__file__)) != bindings["trainer_sha256"]:
            raise RuntimeError("trainer binding drift")
        if _sha256(args.checkpoint) != bindings["checkpoint_sha256"][str(args.checkpoint)]:
            raise RuntimeError("checkpoint binding drift")
        fit_manifests = [json.loads(path.read_text()) for path in args.fit_manifest]
        holdout_manifests = [json.loads(path.read_text()) for path in args.holdout_manifest]
        for path in args.fit_manifest:
            if _sha256(path) != bindings["fit_manifest_sha256"][str(path)]:
                raise RuntimeError(f"fit manifest drift: {path}")
        for path in args.fit_cache:
            if _sha256(path) != bindings["fit_cache_sha256"][str(path)]:
                raise RuntimeError(f"fit cache drift: {path}")
        for path in args.holdout_manifest:
            if _sha256(path) != bindings["holdout_manifest_sha256"][str(path)]:
                raise RuntimeError(f"holdout manifest drift: {path}")
        for path in args.holdout_cache:
            if _sha256(path) != bindings["holdout_cache_sha256"][str(path)]:
                raise RuntimeError(f"holdout cache drift: {path}")
        fit = ThreeWindowCache(args.fit_cache, fit_manifests)
        holdouts = _load_holdout(args.holdout_cache, holdout_manifests)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity_parent = checkpoint["identity"]
        if identity_parent["arm"] != "spectral" or identity_parent["cond_channels"] != 7:
            raise RuntimeError("P1b checkpoint is not a V9 spectral parent")
        device = torch.device("cuda")
        model = FrameConditionedPropagator(
            state_channels=8,
            cond_channels=7,
            width=64,
            spectral_rank=32,
            modes=24,
            depth=4,
            gate_init=1.0,
            activation_checkpointing=True,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.gate.requires_grad_(False)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + 101)
        steps_per_epoch = 240
        total_steps = args.epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=1.0e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.01
        )
        identity = {
            "schema": "pa_cora_p1b_curriculum_lane_identity_v1",
            "method": "PA-CORA",
            "stage": "P1b_stable_multiphase_continuation",
            "seed": args.seed,
            "parent_checkpoint": str(args.checkpoint),
            "parent_checkpoint_sha256": _sha256(args.checkpoint),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "micro_records": args.micro_records,
            "denominator_floor_fraction": 0.25,
            "curriculum": {"epochs_1_2": [1.0, 0.0, 0.0], "epochs_3_5": [0.7, 0.3, 0.0], "epochs_6_10": [0.6, 0.25, 0.15]},
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        initial_phases = [evaluate_phase(model, data, device, args.micro_records) for data in holdouts]
        _atomic_json({"phases": initial_phases}, args.output_dir / "initial_holdout.json")
        best = {"epoch": 0, "phases": initial_phases}
        torch.save({"model_state": model.state_dict(), "epoch": 0, "identity": identity, "metrics": best}, args.output_dir / "best.pt")
        _atomic_json(best, args.output_dir / "best.json")
        started = time.time()
        stratum_offset = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            weights = curriculum_slot_weights(epoch)
            losses, components, slot_counts = [], [], [0, 0, 0]
            for _ in range(steps_per_epoch):
                slot = int(rng.choice(3, p=np.asarray(weights) / np.sum(weights)))
                indices = fit.balanced_indices(rng, args.micro_records, stratum_offset)
                stratum_offset += args.micro_records
                anchor, target, conditioning, onset_norm = fit.batch(slot, indices)
                anchor, target = anchor.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_anchored(
                    anchor,
                    conditioning.to(device),
                    initial_state=target[:, :8, 0],
                )
                loss, row = robust_multiphase_loss(
                    prediction[:, 8:], target[:, 8:], onset_norm.to(device)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite P1b loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                components.append({key: float(value) for key, value in row.items()})
                slot_counts[slot] += len(indices)
            phases = [evaluate_phase(model, data, device, args.micro_records) for data in holdouts]
            event = {
                "event": "epoch",
                "epoch": epoch,
                "seed": args.seed,
                "slot_weights": weights,
                "slot_sample_counts": slot_counts,
                "train_loss": float(np.mean(losses)),
                "train_components": {
                    key: float(np.mean([row[key] for row in components]))
                    for key in components[0]
                },
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                "phases": phases,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if phases[0]["aggregate"] < best["phases"][0]["aggregate"]:
                best = {"epoch": epoch, "phases": phases}
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "identity": identity, "metrics": best}, args.output_dir / "best.pt")
                _atomic_json(best, args.output_dir / "best.json")
        passed = best["phases"][0]["aggregate"] < initial_phases[0]["aggregate"]
        _atomic_json(
            {
                "status": "passed" if passed else "rejected",
                "best_epoch": best["epoch"],
                "initial_primary": initial_phases[0]["aggregate"],
                "best_primary": best["phases"][0]["aggregate"],
                "relative_gain": (initial_phases[0]["aggregate"] - best["phases"][0]["aggregate"]) / initial_phases[0]["aggregate"],
                "elapsed_s": round(time.time() - started, 1),
            },
            terminal,
        )
        fit.close()
        return 0
    except Exception as error:
        import traceback
        if fit is not None:
            fit.close()
        _atomic_json({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
