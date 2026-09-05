#!/usr/bin/env python3
"""Train the first train-only parent-anchored block-32 correction pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
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

from saved_time_phase_operator_v4.parent_anchored_block import (  # noqa: E402
    ParentAnchoredBlockCorrector,
    parameter_count,
    parent_anchored_block_loss,
    render_retained_rfft_frames,
)
from saved_time_phase_operator_v4.parent_anchored_relative_loss import (  # noqa: E402
    record_energy_squared_loss,
    unbiased_window_weights,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    atomic_checkpoint,
    atomic_json,
    sha256,
)


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PHYSICAL_SCALE = 1.0e-8


def stable_block_index(sample_id: str, epoch: int, block_count: int) -> int:
    digest = hashlib.sha256(f"pab32:{sample_id}:{int(epoch)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % int(block_count)


def assert_group_disjoint_roles(group_ids, roles) -> None:
    if len(group_ids) != len(roles):
        raise ValueError("group and role vectors must have identical lengths")
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for group_id, role in zip(group_ids, roles):
        roles_by_group[str(group_id)].add(str(role))
    if any(len(group_roles) != 1 for group_roles in roles_by_group.values()):
        raise RuntimeError("one parent-cache group appears in multiple roles")


class ParentBlockCache:
    def __init__(self, path: Path, source_h5: Path) -> None:
        self.path = path.resolve()
        self.source_path = source_h5.resolve()
        self.cache = h5py.File(self.path, "r", swmr=True)
        self.source = h5py.File(self.source_path, "r", swmr=True)
        try:
            if self.cache.attrs.get("schema", "") != "transfer_dg_parent_anchored_block_cache_v1":
                raise RuntimeError("unexpected parent block-cache schema")
            if self.cache.attrs.get("status", "") != "complete":
                raise RuntimeError("parent block cache is incomplete")
            if self.cache.attrs.get("split", "") != "train":
                raise RuntimeError("parent block cache is not train-only")
            if self.cache.attrs.get("validation_opened") or self.cache.attrs.get("test_id_opened"):
                raise RuntimeError("parent block cache has an opened sealed split")
            if float(self.cache.attrs.get("physical_scale", -1.0)) != PHYSICAL_SCALE:
                raise RuntimeError("parent block-cache physical scale drift")
            self.sample_ids = self.cache["sample_id"].asstr()[:]
            self.families = self.cache["family"].asstr()[:]
            self.roles = self.cache["role"].asstr()[:]
            self.group_ids = self.cache["group_id"].asstr()[:]
            self.source_indices = np.asarray(self.cache["source_index"], dtype=np.int64)
            self.time_count = int(self.cache.attrs["time_count"])
            self.block_size = 32
            self._future_energy_cache: dict[int, tuple[float, float]] = {}
            if self.time_count != 401:
                raise RuntimeError("registered saved-time count is not 401")
            if self.cache["parent_coefficients"].shape != (
                len(self.sample_ids), 64, 2, 201, 201
            ):
                raise RuntimeError("parent coefficient cache shape drift")
            if self.cache["condition"].shape != (len(self.sample_ids), 9, 201, 201):
                raise RuntimeError("public conditioning cache shape drift")
            assert_group_disjoint_roles(self.group_ids.tolist(), self.roles.tolist())
            for local, source_index in enumerate(self.source_indices):
                sample = self.source["sample_id"][int(source_index)]
                split = self.source["split"][int(source_index)]
                sample = sample.decode() if isinstance(sample, bytes) else str(sample)
                split = split.decode() if isinstance(split, bytes) else str(split)
                if sample != self.sample_ids[local] or split != "train":
                    raise RuntimeError("source sample/split differs from parent cache")
        except Exception:
            self.close()
            raise

    def positions(self, role: str) -> list[int]:
        return np.flatnonzero(self.roles == role).astype(np.int64).tolist()

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

    def truth(self, position: int, start: int, stop: int, device: torch.device) -> torch.Tensor:
        if not 0 <= int(start) < int(stop) <= self.time_count:
            raise ValueError("truth slice lies outside the registered time axis")
        values = np.asarray(
            self.source["wavefield"][int(self.source_indices[position]), int(start) : int(stop)],
            dtype=np.float32,
        )
        return torch.from_numpy(values / PHYSICAL_SCALE)[None].to(device)

    def full_future_energies(self, position: int) -> tuple[float, float]:
        """Exact normalized train-only field and first-difference energies."""

        key = int(position)
        if key not in self._future_energy_cache:
            source_index = int(self.source_indices[key])
            total_energy = 0.0
            delta_energy = 0.0
            previous = None
            for start in range(2, self.time_count, 64):
                stop = min(start + 64, self.time_count)
                values = np.asarray(
                    self.source["wavefield"][source_index, start:stop],
                    dtype=np.float32,
                ) / PHYSICAL_SCALE
                tensor = torch.from_numpy(values).double()
                total_energy += float(tensor.square().sum())
                if previous is not None:
                    delta_energy += float((tensor[0] - previous).square().sum())
                if tensor.shape[0] > 1:
                    delta_energy += float((tensor[1:] - tensor[:-1]).square().sum())
                previous = tensor[-1]
            if not np.isfinite(total_energy) or not np.isfinite(delta_energy):
                raise FloatingPointError("non-finite complete-record train energy")
            if total_energy <= 0.0 or delta_energy <= 0.0:
                raise RuntimeError("complete-record train energy must be positive")
            self._future_energy_cache[key] = (total_energy, delta_energy)
        return self._future_energy_cache[key]

    def close(self) -> None:
        for handle_name in ("cache", "source"):
            handle = getattr(self, handle_name, None)
            if handle is not None and handle.id.valid:
                handle.close()


def render_parent_context(
    coefficients: torch.Tensor,
    *,
    start: int,
    block_size: int,
    time_count: int,
) -> tuple[torch.Tensor, int]:
    valid = min(int(block_size), int(time_count) - int(start))
    if start < 2 or valid <= 0:
        raise ValueError("invalid parent block start")
    indices = list(range(start - 2, start + valid))
    rendered = render_retained_rfft_frames(
        coefficients, time_count=time_count, frame_indices=indices
    )
    history, future = rendered[:, :2], rendered[:, 2:]
    if valid < block_size:
        future = torch.cat(
            (future, future[:, -1:].expand(-1, block_size - valid, -1, -1)), dim=1
        )
    return torch.cat((history, future), dim=1), valid


def training_window(
    model: ParentAnchoredBlockCorrector,
    coefficients: torch.Tensor,
    condition: torch.Tensor,
    cache: ParentBlockCache,
    position: int,
    *,
    loss_start: int,
    rollout_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One detached pushforward block followed by supervised free-running blocks."""

    block = model.block_size
    total = cache.time_count
    if loss_start > 2:
        warm_start = loss_start - block
        warm_context, warm_valid = render_parent_context(
            coefficients, start=warm_start, block_size=block, time_count=total
        )
        warm_history = warm_context[:, :2]
        with torch.no_grad():
            warm = model.forward_block(
                warm_context,
                warm_history,
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

    predictions, parents, targets = [], [], []
    for block_index in range(int(rollout_blocks)):
        start = int(loss_start) + block_index * block
        if start >= total:
            break
        context, valid = render_parent_context(
            coefficients, start=start, block_size=block, time_count=total
        )
        prediction = model.forward_block(
            context,
            corrected_history,
            condition,
            block_start=start,
            total_frames=total,
        )
        predictions.append(prediction[:, :valid])
        parents.append(context[:, 2 : 2 + valid])
        targets.append(cache.truth(position, start, start + valid, prediction.device))
        corrected_history = prediction[:, valid - 2 : valid]
    if not predictions:
        raise RuntimeError("training window produced no supervised frame")
    return tuple(torch.cat(values, dim=1) for values in (predictions, targets, parents))


def relative_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction - target).double().flatten(1).norm(dim=1)
        / target.double().flatten(1).norm(dim=1).clamp_min(1.0e-16)
    )


@torch.inference_mode()
def evaluate(
    model: ParentAnchoredBlockCorrector,
    cache: ParentBlockCache,
    positions: list[int],
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    for position in positions:
        coefficients, condition = cache.parent_and_condition(position, device)
        indices = torch.arange(cache.time_count, device=device)
        parent = render_retained_rfft_frames(
            coefficients, time_count=cache.time_count, frame_indices=indices
        )
        parent[..., 0, :] = 0.0
        started = time.perf_counter()
        prediction = model.rollout(parent, condition)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        correction_seconds = time.perf_counter() - started
        target = cache.truth(position, 0, cache.time_count, device)
        future = slice(2, None)
        candidate = float(relative_rows(prediction[:, future], target[:, future])[0])
        baseline = float(relative_rows(parent[:, future], target[:, future])[0])
        correction = float(relative_rows(prediction[:, future], parent[:, future])[0])
        temporal = {}
        for name, start, stop in (
            ("early", 2, 135), ("middle", 135, 268), ("late", 268, 401)
        ):
            temporal[name] = float(
                relative_rows(prediction[:, start:stop], target[:, start:stop])[0]
            )
        prediction_spectrum = torch.fft.rfft(prediction[:, 2:].float(), dim=1, norm="ortho")
        target_spectrum = torch.fft.rfft(target[:, 2:].float(), dim=1, norm="ortho")
        spectral = {}
        for name, start, stop in (
            ("low", 0, 64), ("middle", 64, 128), ("high", 128, 200)
        ):
            numerator = (prediction_spectrum[:, start:stop] - target_spectrum[:, start:stop]).abs().square().sum().sqrt()
            denominator = target_spectrum[:, start:stop].abs().square().sum().clamp_min(1.0e-16).sqrt()
            spectral[name] = float(numerator / denominator)
        rows.append(
            {
                "sample_id": str(cache.sample_ids[position]),
                "family": str(cache.families[position]),
                "candidate": candidate,
                "parent": baseline,
                "improvement": 1.0 - candidate / max(baseline, 1.0e-16),
                "correction_relative_l2": correction,
                "temporal": temporal,
                "spectral": spectral,
                "corrector_seconds": correction_seconds,
            }
        )
        del coefficients, condition, parent, prediction, target
    result = {
        "record_count": len(rows),
        "candidate_mean": float(np.mean([row["candidate"] for row in rows])),
        "parent_mean": float(np.mean([row["parent"] for row in rows])),
        "nonworse_count": sum(row["candidate"] <= row["parent"] for row in rows),
        "correction_relative_l2_mean": float(np.mean([row["correction_relative_l2"] for row in rows])),
        "corrector_seconds_mean": float(np.mean([row["corrector_seconds"] for row in rows])),
        "corrector_seconds_p95": float(np.quantile([row["corrector_seconds"] for row in rows], 0.95)),
        "per_family_candidate": {},
        "per_family_parent": {},
        "temporal": {},
        "spectral": {},
        "rows": rows,
    }
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        if family_rows:
            result["per_family_candidate"][family] = float(np.mean([row["candidate"] for row in family_rows]))
            result["per_family_parent"][family] = float(np.mean([row["parent"] for row in family_rows]))
    for band in ("early", "middle", "late"):
        result["temporal"][band] = float(np.mean([row["temporal"][band] for row in rows]))
    for band in ("low", "middle", "high"):
        result["spectral"][band] = float(np.mean([row["spectral"][band] for row in rows]))
    result["improvement_fraction"] = 1.0 - result["candidate_mean"] / max(result["parent_mean"], 1.0e-16)
    result["validation_opened"] = False
    result["test_id_opened"] = False
    return result


def checkpoint_payload(model, optimizer, identity, *, epoch: int, update: int) -> dict:
    return {
        "schema": "transfer_dg_parent_anchored_block32_checkpoint_v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": identity["model_config"],
        "identity": identity,
        "epoch": int(epoch),
        "update": int(update),
        "validation_opened": False,
        "test_id_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse block-corrector output directory")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    data = None
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for key, path in {
            "trainer_sha256": Path(__file__),
            "corrector_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_block.py",
            "cache_sha256": args.cache,
            "source_h5_sha256": args.source_h5,
        }.items():
            observed = sha256(path)
            if observed != bindings[key]:
                raise RuntimeError(f"block-corrector binding drift: {key} {observed}")
        if prereg["training"].get("loss_variant") == "record_energy_squared_v1":
            loss_path = ROOT / "saved_time_phase_operator_v4/parent_anchored_relative_loss.py"
            observed = sha256(loss_path)
            if observed != bindings["relative_loss_sha256"]:
                raise RuntimeError(f"relative-loss binding drift: {observed}")
        if args.max_updates and args.max_updates != int(prereg["smoke"]["max_updates"]):
            raise RuntimeError("unregistered max-updates override")

        data = ParentBlockCache(args.cache, args.source_h5)
        fit = data.positions("fit")
        calibration = data.positions("calibration")
        confirmation = data.positions("confirmation")
        expected_roles = prereg["data"]["role_counts"]
        if {"fit": len(fit), "calibration": len(calibration), "confirmation": len(confirmation)} != expected_roles:
            raise RuntimeError("parent block-cache role census drift")
        role_groups = {
            role: {str(data.group_ids[index]) for index in positions}
            for role, positions in (("fit", fit), ("calibration", calibration), ("confirmation", confirmation))
        }
        if any(role_groups[a] & role_groups[b] for a, b in (("fit", "calibration"), ("fit", "confirmation"), ("calibration", "confirmation"))):
            raise RuntimeError("fit/calibration/confirmation group leakage")

        config = prereg["training"]
        model_config = prereg["model"]
        device = torch.device("cuda")
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        model = ParentAnchoredBlockCorrector(
            condition_channels=int(model_config["condition_channels"]),
            block_size=int(model_config["block_size"]),
            width=int(model_config["width"]),
            spectral_rank=int(model_config["spectral_rank"]),
            modes=int(model_config["modes"]),
            depth=int(model_config["depth"]),
            maximum_correction_ratio=float(model_config["maximum_correction_ratio"]),
            boundary_blend_frames=int(model_config["boundary_blend_frames"]),
            activation_checkpointing=True,
            hard_free_surface=True,
        ).to(device)
        if parameter_count(model) != int(model_config["parameter_count"]):
            raise RuntimeError("block-corrector parameter count drift")
        torch.set_num_threads(int(config["cpu_inference_threads"]))
        evaluation_model = ParentAnchoredBlockCorrector(
            condition_channels=int(model_config["condition_channels"]),
            block_size=int(model_config["block_size"]),
            width=int(model_config["width"]),
            spectral_rank=int(model_config["spectral_rank"]),
            modes=int(model_config["modes"]),
            depth=int(model_config["depth"]),
            maximum_correction_ratio=float(model_config["maximum_correction_ratio"]),
            boundary_blend_frames=int(model_config["boundary_blend_frames"]),
            activation_checkpointing=False,
            hard_free_surface=True,
        ).cpu().eval()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        total_updates = int(config["epochs"]) * len(fit)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_updates, 1),
            eta_min=float(config["learning_rate"]) * float(config["eta_ratio"]),
        )
        identity = {
            "schema": "transfer_dg_parent_anchored_block32_identity_v1",
            "candidate": prereg["candidate"],
            "model_config": model_config,
            "parameter_count": parameter_count(model),
            "cache": str(args.cache.resolve()),
            "cache_sha256": bindings["cache_sha256"],
            "source_h5": str(args.source_h5.resolve()),
            "source_h5_sha256": bindings["source_h5_sha256"],
            "parent_checkpoint_sha256": prereg["parent"]["checkpoint_sha256"],
            "training": config,
            "fit_count": len(fit),
            "calibration_count": len(calibration),
            "confirmation_count": len(confirmation),
            "deployment_inputs": ["parent trajectory", "public medium/source conditioning", "optional two onset observations"],
            "forbidden_inputs": ["future truth", "oracle coefficient", "validation labels", "test_id labels"],
            "evaluation_device": "cpu",
            "cpu_inference_threads": int(config["cpu_inference_threads"]),
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(identity, args.output_dir / "run_identity.json")
        metrics_path = args.output_dir / "metrics.jsonl"
        best_path = args.output_dir / "best.pt"
        latest_path = args.output_dir / "latest.pt"
        update = 0
        best = None
        block_starts = list(range(2, data.time_count, model.block_size))
        started = time.time()
        for epoch in range(1, int(config["epochs"]) + 1):
            order = np.asarray(fit, dtype=np.int64)
            np.random.default_rng(int(config["seed"]) + 1009 * epoch).shuffle(order)
            sums = defaultdict(float)
            count = 0
            model.train()
            for position in order.tolist():
                block_index = stable_block_index(
                    str(data.sample_ids[position]), epoch, len(block_starts)
                )
                loss_start = block_starts[block_index]
                coefficients, condition = data.parent_and_condition(position, device)
                optimizer.zero_grad(set_to_none=True)
                prediction, target, parent = training_window(
                    model,
                    coefficients,
                    condition,
                    data,
                    position,
                    loss_start=loss_start,
                    rollout_blocks=int(config["rollout_blocks"]),
                )
                if config.get("loss_variant") == "record_energy_squared_v1":
                    frame_weights, delta_weights = unbiased_window_weights(
                        loss_start=loss_start,
                        sample_length=int(prediction.shape[1]),
                        time_count=data.time_count,
                        block_size=model.block_size,
                        rollout_blocks=int(config["rollout_blocks"]),
                        device=prediction.device,
                        dtype=prediction.dtype,
                    )
                    full_energy, full_delta_energy = data.full_future_energies(position)
                    loss, terms = record_energy_squared_loss(
                        prediction,
                        target,
                        parent,
                        frame_weights=frame_weights,
                        delta_weights=delta_weights,
                        full_target_energy=full_energy,
                        full_target_delta_energy=full_delta_energy,
                        derivative_weight=float(config["derivative_weight"]),
                        spectral_weight=float(config["spectral_weight"]),
                        nonworse_weight=float(config["nonworse_weight"]),
                        correction_weight=float(config["correction_weight"]),
                        spectral_floor_fraction=float(config["spectral_floor_fraction"]),
                    )
                else:
                    loss, terms = parent_anchored_block_loss(
                        prediction,
                        target,
                        parent,
                        derivative_weight=float(config["derivative_weight"]),
                        spectral_weight=float(config["spectral_weight"]),
                        nonworse_weight=float(config["nonworse_weight"]),
                        correction_weight=float(config["correction_weight"]),
                        spectral_floor_fraction=float(config["spectral_floor_fraction"]),
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite block loss at update {update + 1}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(f"non-finite gradient at update {update + 1}")
                optimizer.step()
                scheduler.step()
                update += 1
                count += 1
                sums["loss"] += float(loss.detach())
                for key, value in terms.items():
                    sums[key] += float(value)
                if args.max_updates and update >= args.max_updates:
                    atomic_checkpoint(
                        checkpoint_payload(model, optimizer, identity, epoch=epoch, update=update),
                        latest_path,
                    )
                    event = {
                        "event": "smoke_update",
                        "epoch": epoch,
                        "update": update,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                        "validation_opened": False,
                        "test_id_opened": False,
                    }
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    atomic_json(
                        {"status": "smoke_complete", "epoch": epoch, "update": update, "checkpoint": str(latest_path.resolve())},
                        terminal,
                    )
                    print(json.dumps(event, sort_keys=True), flush=True)
                    return 0
                del coefficients, condition, prediction, target, parent, loss

            evaluation_model.load_state_dict(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                strict=True,
            )
            calibration_metrics = evaluate(
                evaluation_model, data, calibration, torch.device("cpu")
            )
            event = {
                "event": "epoch",
                "epoch": epoch,
                "update": update,
                "elapsed_s": time.time() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": {key: value / max(count, 1) for key, value in sums.items()},
                "calibration": calibration_metrics,
                "validation_opened": False,
                "test_id_opened": False,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            payload = checkpoint_payload(model, optimizer, identity, epoch=epoch, update=update)
            atomic_checkpoint(payload, latest_path)
            if best is None or calibration_metrics["candidate_mean"] < best["candidate_mean"]:
                best = {"epoch": epoch, "candidate_mean": calibration_metrics["candidate_mean"]}
                atomic_checkpoint(payload, best_path)

        if not best_path.exists():
            raise RuntimeError("training completed without a best checkpoint")
        selected = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state"], strict=True)
        evaluation_model.load_state_dict(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            strict=True,
        )
        confirmation_metrics = evaluate(
            evaluation_model, data, confirmation, torch.device("cpu")
        )
        acceptance = prereg["acceptance"]
        family_nonworse = all(
            confirmation_metrics["per_family_candidate"][family]
            <= float(acceptance["maximum_family_regression_ratio"])
            * confirmation_metrics["per_family_parent"][family]
            for family in FAMILIES
        )
        accepted = (
            confirmation_metrics["improvement_fraction"]
            >= float(acceptance["minimum_confirmation_improvement_fraction"])
            and family_nonworse
        )
        terminal_payload = {
            "status": "accepted" if accepted else "rejected",
            "claim_scope": "train-only architecture pilot; validation and test_id remain sealed",
            "selected_epoch": int(selected["epoch"]),
            "selected_checkpoint": str(best_path.resolve()),
            "selected_checkpoint_sha256": sha256(best_path),
            "confirmation": confirmation_metrics,
            "family_nonworse": family_nonworse,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(terminal_payload, terminal)
        print(json.dumps(terminal_payload, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        atomic_json(
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            terminal,
        )
        raise
    finally:
        if data is not None:
            data.close()


if __name__ == "__main__":
    raise SystemExit(main())
