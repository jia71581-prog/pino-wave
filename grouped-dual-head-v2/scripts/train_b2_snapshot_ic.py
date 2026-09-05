#!/usr/bin/env python3
"""Anchored-B2 snapshot-IC trainer (one lane = one arm x one seed x one GPU).

Preregistration: results/b2_snapshot_ic_preregistration_20260830.json
Manifest:        results/b2_dev_snapshot_manifest_90rec_20260830.json

Deployment contract: inference inputs are (a) static medium/source features,
(b) the first N_VISIBLE true snapshots of the record window, (c) the solve-free
warp_r1 render as the anchor.  Truth frames >= N_VISIBLE are train-only
supervision; loss/metric are computed only there.

Stage 1 (once, shared): render the warp_r1 base_seq for all 90 manifest records
over their frozen windows and cache to disk (fp16) together with the encoded
targets and static conditioning, so the four lanes never re-render.
Stage 2 (per lane): train CausalSemigroupPropagator width=64 for 60 epochs,
evaluating on all 90 records every epoch; emit best.pt / metrics.jsonl /
terminal.json in the r25-style layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.propagator import CausalSemigroupPropagator


class FrameConditionedPropagator(CausalSemigroupPropagator):
    """v2: per-frame correction sees the CURRENT anchor frame.

    The ep23 interruption diagnosed a structural throttle in the base class:
    ``p_k = base_k + gate*decode(h_k)`` decodes a latent that has never seen
    ``base_k`` (the anchor enters only the NEXT step's recurrence), and the
    decoder is two 1x1 convs with zero spatial receptive field -- so the
    correction cannot express even a small wavefront displacement of the
    current render.  All four lanes pinned train_loss == eval == baseline
    (underfit, not overfit).

    Two changes, mirroring the proven R25 CoarseResidualUNet recipe of
    computing the correction FROM the frame being corrected:
      * decode input = step_norm(h_k + cond_latent + base_latent_k), the same
        driven state the recurrence consumes, so the correction is conditioned
        on the current anchor frame;
      * decoder = 3x3 conv -> GELU -> 3x3 conv (spatial receptive field),
        final conv zero-init by the trainer for the exact no-op warm start.
    Free-running ``forward`` is NOT overridden (anchored-only line).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        width = self.width
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

    def forward_anchored(self, base_seq, cond, initial_state=None):
        if base_seq.ndim != 5 or base_seq.shape[2] != 1:
            raise ValueError("base_seq must be [B, K, 1, Z, X]")
        if cond.ndim != 4:
            raise ValueError("cond must be [B, cond_channels, Z, X]")
        b, k, _, z, x = base_seq.shape
        if initial_state is None:
            initial_state = base_seq[:, 0].expand(
                b, self.encoder[0].in_channels - cond.shape[1], z, x
            )
        hidden = self.encoder(torch.cat((initial_state, cond), dim=1))
        cond_latent = self.cond_projection(cond)
        interface_maps = self._interface_maps(cond)
        frames = []
        for j in range(int(k)):
            base_j = base_seq[:, j]
            base_latent = self.base_projection(base_j)
            driven = self.step_norm(hidden + cond_latent + base_latent)
            frames.append(
                self._apply_output_boundary(
                    base_j + self.gate * self.decoder(driven)
                )
            )
            if self.activation_checkpointing and self.training and hidden.requires_grad:
                from torch.utils.checkpoint import checkpoint
                hidden = checkpoint(
                    self._step_anchored,
                    hidden,
                    cond_latent,
                    base_latent,
                    *(() if interface_maps is None else (interface_maps,)),
                    use_reentrant=False,
                )
            else:
                hidden = self._step_anchored(
                    hidden, cond_latent, base_latent, interface_maps
                )
        return torch.stack(frames, dim=1)

PREREG = ROOT / "results/b2_snapshot_ic_preregistration_20260830.json"
MANIFEST = ROOT / "results/b2_dev_snapshot_manifest_90rec_20260830.json"
CACHE = ROOT / "results/b2_snapshot_ic_cache_20260830.h5"
WARP_CFG = ROOT / "configs/saved_time_v4/generated/local_field_w128_warp_r1.yaml"
WARP_RUN = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/local_field_w128_hicap/warp_r1/run"
)
PARENT_IDENTITY = (
    ROOT / "configs/saved_time_v4/generated/"
    "w2_parent_identity_legacy_norm_marmousi1_4m_v2.json"
)
ARM_STATE_CHANNELS = {"ic2": 2, "ic8": 8}
FAMILIES = ("uniform", "layered", "marmousi")


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_cond(velocity, x_m, z_m, source_x, source_z):
    """7-channel deployment-legal conditioning (r25 static_features recipe)."""
    value = np.asarray(velocity, dtype=np.float32)
    logv = np.log(np.maximum(value, 1.0))
    grad_z, grad_x = np.gradient(logv)
    grad_x = np.clip(20.0 * grad_x, -4.0, 4.0)
    grad_z = np.clip(20.0 * grad_z, -4.0, 4.0)
    xx, zz = np.meshgrid(x_m.astype(np.float32), z_m.astype(np.float32))
    distance = np.sqrt((xx - float(source_x)) ** 2 + (zz - float(source_z)) ** 2)
    source_map = np.exp(-0.5 * (distance / 20.0) ** 2)
    source_map /= max(float(source_map.max()), 1.0e-12)
    travel = distance / np.maximum(value, 1.0)
    x_norm = 2.0 * xx / max(float(x_m[-1]), 1.0) - 1.0
    z_norm = 2.0 * zz / max(float(z_m[-1]), 1.0) - 1.0
    return np.stack(
        [
            np.clip((value - 4500.0) / 2500.0, -2.0, 2.0),
            grad_x,
            grad_z,
            source_map.astype(np.float32),
            np.clip(travel, 0.0, 1.5).astype(np.float32),
            x_norm.astype(np.float32),
            z_norm.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def build_cache(
    manifest: dict,
    device: torch.device,
    *,
    cache_path: Path = CACHE,
    data_split: str = "validation",
) -> None:
    """Stage 1: render base_seq + encode targets for all manifest records."""
    import importlib.util

    saved_argv = sys.argv
    sys.argv = ["b2_cache"]
    spec = importlib.util.spec_from_file_location(
        "train_v4", str(ROOT / "scripts/train_saved_time_v4_full_support.py")
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    sys.argv = saved_argv
    from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
    from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
    from grouped_ufno_mionet_v3.data.pilot import PilotStepSpec

    config = yaml.safe_load(WARP_CFG.read_text())
    config["parent_checkpoint"] = str(WARP_RUN / "best.pt")
    config["parent_checkpoint_identity"] = str(WARP_RUN / "run_identity.json")
    config["parent_identity"] = str(PARENT_IDENTITY)
    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["allow_parent_manifest_mismatch"] = True
    config["checkpoint_transfer"] = transfer
    base, v3_manifest, parent_identity = train._load_context(config)
    model = train._load_parent_model(config, base, v3_manifest, parent_identity, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    normalizer = load_normalizer(base, v3_manifest.digest)

    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5, v3_manifest, split=data_split,
        schedule=[PilotStepSpec(step=0, record_indices=(0,))],
        query_points=1, seed=0,
    )
    index_of = {
        rec.source_index: pos for pos, rec in enumerate(dataset.records.records)
    }

    records = manifest["records"]
    k_frames = int(manifest["k_frames"])
    source_h5 = manifest["source_h5"]
    with h5py.File(source_h5, "r", swmr=True) as f:
        time_axis = np.asarray(f["time_s"][:], dtype=np.float64)
        x_m = np.asarray(f["x_m"][:]); z_m = np.asarray(f["z_m"][:])
        conds = np.stack([
            _static_cond(
                np.asarray(f["velocity_mps"][row["source_index"]]), x_m, z_m,
                float(f["source_x_m"][row["source_index"]]),
                float(f["source_z_m"][row["source_index"]]),
            ) for row in records
        ])

    n = len(records)
    height = width = conds.shape[-1]
    partial = cache_path.with_name(cache_path.name + f".partial.{os.getpid()}")
    started = time.time()
    with h5py.File(partial, "w") as out:
        out.attrs["schema"] = "b2_snapshot_ic_cache_v1"
        out.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
        out.attrs["data_split"] = data_split
        out.attrs["warp_checkpoint_sha256"] = _sha256_file(WARP_RUN / "best.pt")
        out.create_dataset("sample_id", data=np.array(
            [row["sample_id"] for row in records], dtype=object
        ), dtype=h5py.string_dtype())
        out.create_dataset("family", data=np.array(
            [row["family"] for row in records], dtype=object
        ), dtype=h5py.string_dtype())
        out.create_dataset("window_start", data=np.array(
            [row["window_start"] for row in records], dtype=np.int64
        ))
        out.create_dataset("cond", data=conds.astype(np.float16))
        base_ds = out.create_dataset(
            "base_seq", shape=(n, k_frames, height, width), dtype=np.float16
        )
        target_ds = out.create_dataset(
            "target", shape=(n, k_frames, height, width), dtype=np.float16
        )
        with torch.no_grad():
            for row_index, row in enumerate(records):
                position = index_of[row["source_index"]]
                times = time_axis[row["window_start"]:row["window_start"] + k_frames]
                batch = dataset.materialize_spec(
                    PilotStepSpec(step=0, record_indices=(position,))
                )
                tensors = _to_device(batch, device)
                source = tensors["source_parameters"]
                prepared = model.prepare_sources(
                    model.encode_medium(tensors["velocity_mps"], normalizer),
                    source, tensors["source_map"], normalizer,
                    record_to_medium=tensors["record_to_medium"],
                )
                chunks = []
                for lo in range(0, k_frames, 8):
                    requested = torch.as_tensor(
                        times[lo:lo + 8], dtype=torch.float32, device=device
                    )[None]
                    chunks.append(model.dense_normalized(
                        prepared, requested,
                        x_m=tensors["x_m"], z_m=tensors["z_m"], time_block=1,
                    ).float().cpu())
                base_ds[row_index] = torch.cat(chunks, dim=1)[0].numpy().astype(np.float16)
                truth = dataset.records.read_wavefield(position, times).values
                truth = normalizer.encode_pressure(truth[None].to(device), source[:, 4])
                target_ds[row_index] = truth[0].float().cpu().numpy().astype(np.float16)
                if row_index % 10 == 0:
                    print(json.dumps({
                        "event": "cache_progress", "record": row_index, "of": n,
                        "elapsed_s": round(time.time() - started, 1),
                    }), flush=True)
    os.replace(partial, cache_path)
    print(json.dumps({
        "event": "cache_done", "path": str(cache_path),
        "bytes": cache_path.stat().st_size,
        "elapsed_s": round(time.time() - started, 1),
    }), flush=True)


def train_lane(arm: str, seed: int, manifest: dict, device: torch.device,
               output_dir: Path, *, epochs: int, lr: float, micro_records: int,
               width: int = 64, spectral_rank: int = 32) -> None:
    n_visible = int(manifest["n_visible"])
    ic_frames = ARM_STATE_CHANNELS[arm]
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(CACHE, "r", swmr=True) as cache:
        if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
            raise RuntimeError("cache does not match the frozen manifest")
        base_seq = torch.from_numpy(cache["base_seq"][:].astype(np.float32))[:, :, None]
        target = torch.from_numpy(cache["target"][:].astype(np.float32))[:, :, None]
        cond = torch.from_numpy(cache["cond"][:].astype(np.float32))
        families = cache["family"][:].astype(str)

    n_records = base_seq.shape[0]
    eval_slice = slice(n_visible, None)
    initial_state_all = target[:, n_visible - ic_frames:n_visible, 0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FrameConditionedPropagator(
        state_channels=ic_frames, cond_channels=cond.shape[1],
        width=width, spectral_rank=spectral_rank, modes=24, depth=4,
        gate_init=1.0, activation_checkpointing=True,
    ).to(device)
    # Preregistered gate-collapse fix: frozen unit gate + zero-init output conv.
    torch.nn.init.zeros_(model.decoder[-1].weight)
    torch.nn.init.zeros_(model.decoder[-1].bias)
    model.gate.requires_grad_(False)

    steps_per_epoch = (n_records + micro_records - 1) // micro_records
    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )

    def _evaluate():
        rel = []
        with torch.no_grad():
            for lo in range(0, n_records, micro_records):
                sl = slice(lo, min(lo + micro_records, n_records))
                pred = model.forward_anchored(
                    base_seq[sl].to(device), cond[sl].to(device),
                    initial_state=initial_state_all[sl].to(device),
                )[:, eval_slice]
                t = target[sl, eval_slice].to(device)
                p = pred.reshape(pred.shape[0], -1).double()
                tt = t.reshape(t.shape[0], -1).double()
                rel.append(((p - tt).pow(2).sum(1).sqrt()
                            / tt.pow(2).sum(1).clamp_min(1e-16).sqrt()).cpu())
        rel = torch.cat(rel)
        per_family = {
            name: float(rel[torch.from_numpy(families == name)].mean())
            for name in FAMILIES
        }
        return {"aggregate": float(rel.mean()), "max": float(rel.max()),
                "per_family": per_family}

    with torch.no_grad():
        b = base_seq[:, eval_slice].reshape(n_records, -1).double()
        t = target[:, eval_slice].reshape(n_records, -1).double()
        rel = (b - t).pow(2).sum(1).sqrt() / t.pow(2).sum(1).clamp_min(1e-16).sqrt()
        baseline = {"aggregate": float(rel.mean()), "max": float(rel.max()),
                    "per_family": {
                        name: float(rel[torch.from_numpy(families == name)].mean())
                        for name in FAMILIES}}
    identity = {
        "schema": "b2_snapshot_ic_lane_v1", "arm": arm, "seed": seed,
        "ic_frames": ic_frames, "epochs": epochs, "lr": lr,
        "micro_records": micro_records,
        "width": width, "spectral_rank": spectral_rank,
        "manifest_selection_sha256": manifest["selection_sha256"],
        "preregistration": str(PREREG),
        "trainer_sha256": _sha256_file(Path(__file__)),
        "baseline": baseline,
    }
    _atomic_json(identity, output_dir / "run_identity.json")
    print(json.dumps({"event": "baseline", "arm": arm, "seed": seed, **baseline}),
          flush=True)

    order_rng = np.random.default_rng(seed + 101)
    best = None
    started = time.time()
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            order = order_rng.permutation(n_records)
            epoch_loss = []
            for lo in range(0, n_records, micro_records):
                idx = torch.from_numpy(order[lo:lo + micro_records].copy())
                optimizer.zero_grad(set_to_none=True)
                pred = model.forward_anchored(
                    base_seq[idx].to(device), cond[idx].to(device),
                    initial_state=initial_state_all[idx].to(device),
                )
                p = pred[:, eval_slice].reshape(pred.shape[0], -1)
                t = target[idx, eval_slice].to(device).reshape(len(idx), -1)
                loss = ((p - t).pow(2).sum(1).clamp_min(0).sqrt()
                        / t.pow(2).sum(1).clamp_min(1e-16).sqrt()).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss.append(float(loss.detach()))
            model.eval()
            metrics = _evaluate()
            row = {
                "event": "epoch", "arm": arm, "seed": seed, "epoch": epoch,
                "train_loss": float(np.mean(epoch_loss)),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1), **metrics,
            }
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if best is None or metrics["aggregate"] < best["aggregate"]:
                best = {"epoch": epoch, **metrics}
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "identity": identity, "metrics": metrics},
                           output_dir / "best.pt")
                _atomic_json(best, output_dir / "best.json")
        _atomic_json({"status": "complete", "arm": arm, "seed": seed,
                      "best": best, "baseline": baseline,
                      "elapsed_s": round(time.time() - started, 1)},
                     output_dir / "terminal.json")
    except Exception as error:
        import traceback
        _atomic_json({"status": "failed", "arm": arm, "seed": seed,
                      "error": repr(error), "traceback": traceback.format_exc()},
                     output_dir / "terminal.json")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "train"), required=True)
    parser.add_argument("--arm", choices=tuple(ARM_STATE_CHANNELS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--micro-records", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--spectral-rank", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    device = torch.device("cuda")
    if args.stage == "cache":
        if CACHE.exists():
            raise FileExistsError(f"cache already exists: {CACHE}")
        build_cache(manifest, device)
        return 0
    if not (args.arm and args.seed and args.output_dir):
        raise SystemExit("--stage train requires --arm --seed --output-dir")
    if not CACHE.exists():
        raise SystemExit(f"run --stage cache first: {CACHE}")
    train_lane(args.arm, args.seed, manifest, device, args.output_dir,
               epochs=args.epochs, lr=args.lr, micro_records=args.micro_records,
               width=args.width, spectral_rank=args.spectral_rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
