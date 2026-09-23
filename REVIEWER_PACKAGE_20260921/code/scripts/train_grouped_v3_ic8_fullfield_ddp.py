#!/usr/bin/env python
"""DDP (4-GPU) ic8-fullfield V3 trainer.

Data-parallel over the ic8_fullfield pipeline (IC snapshots + CPML features +
full-horizon dense supervision).  One model, global batch = batch_records (e.g.
48), each rank handles batch_records/world_size records per step with DDP
gradient all-reduction.  Reuses the ic8_fullfield helpers unchanged; only the
loop is DDP-wrapped.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, sys, time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR)); sys.path.insert(0, str(WORKDIR / "src"))
from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import build_pilot_schedule, PilotBatch
from grouped_ufno_mionet_v3.losses import V3LossResult, compute_v3_losses
from grouped_ufno_mionet_v3.training.checkpoint import _restore_rng, save_checkpoint_atomic
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer
from scripts.train_grouped_v3 import build_model
from scripts.train_grouped_v3_ic8_fullfield import (
    _atomic_hardlink, _dataset, _weights,
    evaluate_ic_batch, load_normalizer,
)
from scripts.train_grouped_v3_pilot import dense_at_pilot_queries, _to_device, _atomic_json as _pilot_json

_MAIN_KEYS = ("velocity_mps", "source_parameters", "source_map", "requested_time_s",
              "dense_target_physical", "query_coords", "query_target_physical",
              "query_probability", "ic_snapshots_physical")

# velocity_mps is indexed by unique medium, every other _MAIN_KEYS entry by
# record.  pack_v3_groups deduplicates media per group_id, so a balanced
# three-family batch carries 12 records over 6 media while a uniform-only batch
# carries 12 of each.  Sharding therefore has to slice the record dimension and
# then gather the media those records actually reference.
_RECORD_KEYS = tuple(name for name in _MAIN_KEYS if name != "velocity_mps") + ("record_to_medium",)

# Keys the forward path consumes.  Rank 0 reads the full batch off disk and
# broadcasts only these (x_m/z_m are the shared grid axes, not sliced).
_BROADCAST_KEYS = _MAIN_KEYS + ("record_to_medium", "x_m", "z_m")


def _broadcast_batch(dataset, step, rank, world, device):
    """Materialize one step's batch on rank 0 only, then broadcast the tensors
    each rank needs to every rank.  Shards per-record tensors by rank stride
    and gathers each rank's medium rows exactly as _rank_batch does (the
    media are deduplicated by pack_v3_groups, so media count != record count)."""
    if world <= 1:
        return _to_device(dataset[step], device)
    def slc(v): return v[rank::world]
    keys = list(_BROADCAST_KEYS)
    import torch.distributed as _dist
    meta_list = [None]
    if rank == 0:
        batch = dataset[step]
        meta_list[0] = {k: (tuple(getattr(batch, k).shape), getattr(batch, k).dtype) for k in keys}
    _dist.broadcast_object_list(meta_list, src=0)
    meta = meta_list[0]
    local = {}
    for k in keys:
        shape, dtype = meta[k]
        if rank == 0:
            local[k] = getattr(batch, k).to(device, non_blocking=True)
        else:
            local[k] = torch.empty(shape, dtype=dtype, device=device)
        _dist.broadcast(local[k], src=0)
    batch_records = local["record_to_medium"]
    records = torch.arange(batch_records.nelement(), device=device)[rank::world]
    out = {k: slc(local[k]) for k in _RECORD_KEYS}
    velocity, mapping = _shard_media(local["velocity_mps"], batch_records, records)
    out["velocity_mps"] = velocity
    out["record_to_medium"] = mapping
    out["x_m"] = local["x_m"]
    out["z_m"] = local["z_m"]
    return out


def _setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


class ICForwardWrapper(torch.nn.Module):
    def __init__(self, operator, normalizer):
        super().__init__()
        self.operator = operator
        self.normalizer = normalizer

    def forward(self, tensors):
        src = tensors["source_parameters"]
        prepared = self.operator.prepare_sources(
            self.operator.encode_medium(tensors["velocity_mps"], self.normalizer),
            src, tensors["source_map"], self.normalizer,
            record_to_medium=tensors["record_to_medium"],
            ic_snapshots_physical=tensors.get("ic_snapshots_physical"),
        )
        dense = self.operator.dense_normalized(
            prepared, tensors["requested_time_s"],
            x_m=tensors["x_m"], z_m=tensors["z_m"], time_block=1,
        )
        query = self.operator.query_normalized(
            prepared, tensors["query_coords"], chunk_size=64,
        )
        return dense, query


def _shard_media(velocity_mps, record_to_medium, records):
    """Return this rank's medium rows plus a record_to_medium remapped onto them.

    ``records`` is the record-dimension index tensor already applied to the
    per-record tensors.  The old code assumed one medium per record and rebuilt
    the mapping as arange(local), which silently mismatched whenever
    pack_v3_groups deduplicated a group into fewer media than records."""
    if records.device != record_to_medium.device:
        records = records.to(record_to_medium.device)
    wanted = record_to_medium[records]
    media, local = torch.unique(wanted, sorted=True, return_inverse=True)
    return velocity_mps[media], local


def _update_residual_maps(
    residual_map: torch.Tensor | None,
    residual_time_profile: torch.Tensor | None,
    prediction_dense: torch.Tensor,
    target_dense: torch.Tensor,
    record_indices: torch.Tensor,
    *,
    world: int,
    device: torch.device,
    gamma: float = 0.95,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Update the residual-adaptive sampling statistics for one optimization step.

    Two separate objects, because space and time have different supports:

    * ``residual_map`` [nz,nx] sums |pred-target| over the batch's frames and
      averages over its records; it is a static spatial field shared by every
      record.
    * ``residual_time_profile`` [n_train_records, time] keeps a per-record
      profile, since a frame's residual depends on where that record's source
      sits in time.

    Both are bounded EMAs, so neither can collapse onto a single point, and both
    are all-reduced across ranks because each rank only sees its own shard of
    the batch.  Returns detached CPU tensors.
    """
    resid = (prediction_dense.float() - target_dense.float()).abs()

    updated_map = residual_map
    if residual_map is not None:
        spatial = resid.sum(dim=1).mean(dim=0)
        if world > 1:
            dist.all_reduce(spatial, op=dist.ReduceOp.SUM)
            spatial = spatial / world
        spatial = spatial.clamp_min(1.0e-8)
        spatial = spatial / spatial.max().clamp_min(1.0e-8)
        updated_map = (
            float(gamma) * residual_map.to(device) + (1.0 - float(gamma)) * spatial
        ).detach().cpu()

    updated_profile = residual_time_profile
    if residual_time_profile is not None:
        per_record = resid.mean(dim=(2, 3))  # [records, frames]
        rows = residual_time_profile.to(device)
        index = record_indices.to(device).long()
        # Several records can appear in one batch (batch_records / world per
        # rank), and a record may repeat across ranks, so accumulate with
        # index_add and divide by the number of contributions rather than
        # assigning row by row.
        flat_index = index.reshape(-1)
        frames = per_record.shape[1]
        if rows.shape[1] != frames:
            raise ValueError(
                f"residual_time_profile has {rows.shape[1]} frames but the dense "
                f"target has {frames}"
            )
        accumulated = torch.zeros(
            (rows.shape[0], frames), dtype=torch.float32, device=device
        )
        accumulated.index_add_(0, flat_index, per_record.float())
        counts = torch.zeros(rows.shape[0], dtype=torch.float32, device=device)
        counts.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.float32))
        if world > 1:
            dist.all_reduce(accumulated, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        seen = counts > 0
        step_mean = torch.zeros_like(accumulated)
        step_mean[seen] = accumulated[seen] / counts[seen, None]
        # Normalize each record's row to [0,1] so gamma means the same thing
        # for every record regardless of its absolute amplitude.
        row_max = step_mean.max(dim=1, keepdim=True).values.clamp_min(1.0e-8)
        step_mean = (step_mean / row_max).clamp(0.0, 1.0)
        updated_profile = rows.clone()
        updated_profile[seen] = (
            float(gamma) * rows[seen] + (1.0 - float(gamma)) * step_mean[seen]
        )
        updated_profile = updated_profile.detach().cpu()

    return updated_map, updated_profile


def _apply_group_hyperparameters(optimizer, config):
    """Return the param_groups of ``optimizer`` aligned with ``config``.

    ``Optimizer.load_state_dict`` restores param_groups verbatim, hyperparameters
    included, so a checkpoint written under a different config silently replaces
    the lr/weight_decay the optimizer was just constructed with.  The schema is
    otherwise identical, so this only overwrites the two hyperparameters that are
    config-derived at construction time.

    Returns ``(before_learning_rates, after_learning_rates)`` per group for audit.
    """
    before = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(config.train.learning_rate)
        group["weight_decay"] = float(config.train.weight_decay)
    after = [float(group["lr"]) for group in optimizer.param_groups]
    expected = float(config.train.learning_rate)
    for index, value in enumerate(after):
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise RuntimeError(
                f"param_groups[{index}].lr={value!r} does not match "
                f"config.train.learning_rate={expected!r} after loading optimizer state"
            )
    return before, after


def _rank_tensors(batch, rank, world, device):
    if world <= 1:
        return _to_device(batch, device)
    records = torch.arange(batch.source_parameters.shape[0])[rank::world]
    subset = {name: getattr(batch, name)[records] for name in _RECORD_KEYS}
    velocity, mapping = _shard_media(batch.velocity_mps, batch.record_to_medium, records)
    subset["velocity_mps"] = velocity
    subset["record_to_medium"] = mapping
    out = {name: v.to(device, non_blocking=True) for name, v in subset.items()}
    out["x_m"] = batch.x_m.to(device)
    out["z_m"] = batch.z_m.to(device)
    return out


def _rank_batch(batch, rank, world):
    if world <= 1:
        return batch
    def slc(v): return v[rank::world]
    _records = torch.arange(batch.source_parameters.shape[0])[rank::world]
    _velocity, _mapping = _shard_media(batch.velocity_mps, batch.record_to_medium, _records)
    return PilotBatch(
        step=batch.step, velocity_mps=_velocity,
        record_to_medium=_mapping,
        source_parameters=slc(batch.source_parameters), source_map=slc(batch.source_map),
        requested_time_s=slc(batch.requested_time_s),
        dense_target_physical=slc(batch.dense_target_physical),
        target_exact=slc(batch.target_exact),
        interpolation_alpha=slc(batch.interpolation_alpha) if batch.interpolation_alpha is not None else None,
        left_index=slc(batch.left_index), right_index=slc(batch.right_index),
        query_coords=slc(batch.query_coords), query_target_physical=slc(batch.query_target_physical),
        query_probability=slc(batch.query_probability), x_m=batch.x_m, z_m=batch.z_m,
        sample_id=batch.sample_id[rank::world], group_id=batch.group_id[rank::world],
        medium_type=batch.medium_type[rank::world],
        ic_snapshots_physical=slc(batch.ic_snapshots_physical) if batch.ic_snapshots_physical is not None else None,
    )


_DENSE_SPECTRAL_PREFIX = "dense_decoder.blocks."
_DENSE_SPECTRAL_NAMES = ("spectral.weight", "spectral.weight_top", "spectral.weight_bottom")


def _is_dense_spectral_key(key):
    """True for the dense decoder's spectral kernel weights.

    These are exactly the tensors the radial arm reparameterizes, so the reset
    control must discard the same set to isolate reset-and-retrain from the
    change in parameterization.  Matched by suffix rather than by shape because
    the rectangular form stores two half-plane tensors and the radial form one.
    """
    if not key.startswith(_DENSE_SPECTRAL_PREFIX):
        return False
    return any(key.endswith(name) for name in _DENSE_SPECTRAL_NAMES)


def _atomic_json(payload, destination):
    """Write standards-compliant JSON; unmeasured metrics must be None."""
    destination = Path(destination)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with partial.open("x") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _json_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _code_digests():
    files = set()
    for directory in ("grouped_ufno_mionet_v3", "src"):
        files.update((WORKDIR / directory).rglob("*.py"))
    for name in ("train_grouped_v3.py", "train_grouped_v3_pilot.py",
                 "train_grouped_v3_ic8_fullfield.py", "train_grouped_v3_ic8_fullfield_ddp.py"):
        files.add(WORKDIR / "scripts" / name)
    return {str(path.relative_to(WORKDIR)): _file_sha256(path) for path in sorted(files)}


def _load_exclusions(path, manifest):
    if path is None:
        return (), None
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("exclusion manifest digest mismatch")
    groups = payload["excluded_group_ids"]
    if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
        raise ValueError("excluded_group_ids must be a list of strings")
    if len(groups) != len(set(groups)):
        raise ValueError("duplicate excluded group IDs")
    train_groups = {record.group_id for record in manifest.records if record.split == "train"}
    if set(groups) - train_groups:
        raise ValueError("exclusions must contain only existing train group IDs")
    return tuple(sorted(groups)), _file_sha256(path)


def _training_data(config, manifest, *, train_only, excluded_group_ids=()):
    """The production split boundary: train-only never constructs validation."""
    schedule = build_pilot_schedule(
        manifest, split="train", steps=int(config.train.max_steps), seed=config.train.seed,
        families=config.data.train_families, batch_records=config.train.batch_records,
        excluded_group_ids=excluded_group_ids,
    )
    train_dataset = _dataset(config, manifest, split="train", schedule=schedule, seed=config.train.seed)
    validation_batch = None
    if not train_only:
        val_schedule = build_pilot_schedule(
            manifest, split="validation", steps=1, seed=config.train.seed + 7919,
            families=config.data.train_families,
        )
        val_dataset = _dataset(config, manifest, split="validation", schedule=val_schedule,
                               seed=config.train.seed + 7919)
        validation_batch = val_dataset[0]
    return train_dataset, validation_batch, schedule


def _training_census(manifest, excluded_group_ids):
    excluded = set(excluded_group_ids)
    rows = [dict(local_index=index, **asdict(record))
            for index, record in enumerate(record for record in manifest.records if record.split == "train")
            if record.group_id not in excluded]
    return {"eligible_records": len(rows),
            "eligible_by_family": dict(Counter(row["medium_type"] for row in rows)),
            "eligible_training_list_sha256": _json_sha256(rows),
            "excluded_group_ids": sorted(excluded)}


def _require_updates(start_step, max_steps):
    if int(max_steps) <= int(start_step):
        raise ValueError(f"refusing no-op training: max_steps={max_steps} <= start_step={start_step}")


def _load_fork(model, optimizer, state, config, *, manifest_digest,
               parent_config_digest, allow_new_params, reset_dense_spectral=False):
    """Actual fork path, shared by training and CPU lineage regression tests."""
    if state["manifest_digest"] != manifest_digest:
        raise ValueError("fork parent manifest digest mismatch")
    if parent_config_digest is None or state["config_digest"] != parent_config_digest:
        raise ValueError("fork parent config digest mismatch")
    if reset_dense_spectral and not allow_new_params:
        raise ValueError("reset_dense_spectral requires allow_new_params")
    parent_state = state["model_state"]
    loaded, fresh, parent_only, mismatch = [], [], [], []
    if allow_new_params:
        if optimizer.state:
            raise ValueError("partial fork requires an empty cold optimizer")
        own = model.state_dict()
        for key, value in own.items():
            if reset_dense_spectral and _is_dense_spectral_key(key):
                fresh.append(key)
            elif key not in parent_state:
                fresh.append(key)
            elif parent_state[key].shape != value.shape:
                mismatch.append(key)
            else:
                own[key] = parent_state[key]
                loaded.append(key)
        parent_only = [key for key in parent_state if key not in own]
        if mismatch:
            raise ValueError(f"fork shape mismatches: {mismatch[:8]}")
        if any(not _is_dense_spectral_key(key) for key in fresh + parent_only):
            raise ValueError("partial fork may change only the dense spectral kernels")
        model.load_state_dict(own, strict=True)
    else:
        model.load_state_dict(parent_state, strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        loaded = list(parent_state)
    lr_before, lr_after = _apply_group_hyperparameters(optimizer, config)
    own = model.state_dict()
    inherited = {key: {"parent_sha256": _tensor_sha256(parent_state[key]),
                       "child_sha256": _tensor_sha256(own[key])} for key in loaded}
    if any(item["parent_sha256"] != item["child_sha256"] for item in inherited.values()):
        raise RuntimeError("fork failed bitwise inherited tensor verification")
    fresh_hashes = {key: _tensor_sha256(own[key]) for key in fresh}
    parent_only_hashes = {key: _tensor_sha256(parent_state[key]) for key in parent_only}
    _restore_rng(state["rng_state"])
    return {"event": "fork_partial" if allow_new_params else "fork",
            "global_step": int(state["global_step"]), "epoch": int(state["epoch"]),
            "parent_config_digest": parent_config_digest, "child_config_digest": config.digest(),
            "optimizer": "fresh" if allow_new_params else "restored",
            "optimizer_state_entries_at_start": len(optimizer.state),
            "fresh_initialization_seed": int(config.train.seed), "subsequent_rng": "parent_checkpoint",
            "reset_dense_spectral": bool(reset_dense_spectral), "lr_before": lr_before, "lr_after": lr_after,
            "inherited_tensors": inherited, "fresh_tensors": fresh_hashes,
            "parent_only_tensors": parent_only_hashes,
            "inherited_keys_sha256": _json_sha256(sorted(loaded)),
            "fresh_keys_sha256": _json_sha256(sorted(fresh)),
            "parent_only_keys_sha256": _json_sha256(sorted(parent_only))}


def _collective_finite(value, *, world, device, label):
    finite = torch.as_tensor(value, device=device).isfinite().all().to(torch.int32)
    if world > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise RuntimeError(f"nonfinite DDP {label}; optimizer update refused on every rank")


def _load_resume(model, optimizer, state, config, *, manifest_digest):
    if state["manifest_digest"] != manifest_digest or state["config_digest"] != config.digest():
        raise ValueError("resume checkpoint manifest/config digest mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    lr_before, lr_after = _apply_group_hyperparameters(optimizer, config)
    _restore_rng(state["rng_state"])
    return int(state["global_step"]), lr_before, lr_after


def _spectral_summary(model, *, gradients=False, before=None):
    result = {}
    for key, parameter in model.named_parameters():
        if not _is_dense_spectral_key(key):
            continue
        value = parameter.grad if gradients else parameter.detach() - before[key]
        if value is None:
            result[key] = {"present": False, "finite": False, "real_l2": None, "imag_l2": None}
            continue
        value = value.detach()
        result[key] = {"present": True, "finite": bool(torch.isfinite(value).all().item()),
                       "real_l2": float(value[..., 0].norm().item()),
                       "imag_l2": float(value[..., 1].norm().item()),
                       "l2": float(value.norm().item()), "max_abs": float(value.abs().max().item())}
    return result


def _checked_optimizer_step(model, optimizer, *, gradient_clip, world, device, sample_update):
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    _collective_finite(0.0 if grads else float("nan"), world=world, device=device,
                       label="missing-gradient guard")
    flags = torch.stack([torch.isfinite(grad).all() for grad in grads])
    # Preserve a nonfinite sentinel for the common collective guard.
    _collective_finite(torch.where(flags.all(), torch.tensor(0., device=device),
                                   torch.tensor(float("nan"), device=device)),
                       world=world, device=device, label="gradient")
    spectral = _spectral_summary(model, gradients=True)
    before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()
              if sample_update and _is_dense_spectral_key(key)}
    # A finite collection of gradients can still have an overflowing norm.
    # Never raise locally before all ranks reach the collective refusal gate.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=False)
    _collective_finite(norm, world=world, device=device, label="preclip gradient norm")
    optimizer.step()
    return {"gradient_finite": True, "preclip_gradient_norm": float(norm),
            "dense_spectral_gradients_preclip": spectral,
            "dense_spectral_update": _spectral_summary(model, before=before) if sample_update else None}


class _CorrectionProbe:
    """Read-only hook, removed before backward, on one training micro per rank."""
    def __init__(self, model):
        self.decoder = model.dense_decoder
        self.raw_sum_squares = 0.0
        self.elements = 0
        self.invocations = 0

    def __enter__(self):
        def observe(module, inputs, output):
            self.raw_sum_squares += float(output.detach().double().square().sum().item())
            self.elements += output.numel()
            self.invocations += 1
        self.handle = self.decoder.output.register_forward_hook(observe)
        return self

    def __exit__(self, *unused):
        self.handle.remove()

    def report(self):
        scale = float(self.decoder.correction_scale.detach())
        raw = self.raw_sum_squares / self.elements if self.elements else None
        return {"scope": "first training microbatch on this rank; sampled normalized dense output; forward only",
                "raw_mean_square": raw, "scaled_mean_square": None if raw is None else raw * scale * scale,
                "scaling_definition": "raw_mean_square * correction_scale**2",
                "correction_scale": scale, "elements": self.elements, "output_invocations": self.invocations}


def _save_step_checkpoint(trainer, artifact_dir, *, steps_per_epoch, loss):
    """Save an absolute-step endpoint without replacing an epoch-named file."""
    destination = artifact_dir / "checkpoints" / f"checkpoint_step_{trainer.global_step:08d}.pt"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    checkpoint = save_checkpoint_atomic(
        destination, model=trainer.model, optimizer=trainer.optimizer,
        epoch=trainer.global_step // steps_per_epoch, global_step=trainer.global_step,
        manifest_digest=trainer.manifest_digest, config_digest=trainer.config_digest,
        metrics={"loss": float(loss)},
    )
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (int(reloaded["global_step"]) != trainer.global_step
            or reloaded["manifest_digest"] != trainer.manifest_digest
            or reloaded["config_digest"] != trainer.config_digest):
        raise RuntimeError("endpoint checkpoint identity failed reload verification")
    nonfinite = [key for key, value in reloaded["model_state"].items()
                 if not bool(torch.isfinite(value).all())]
    if nonfinite:
        raise RuntimeError(f"nonfinite endpoint model tensors: {nonfinite[:8]}")
    live = trainer.model.state_dict()
    if set(reloaded["model_state"]) != set(live) or any(
        _tensor_sha256(value) != _tensor_sha256(live[key])
        for key, value in reloaded["model_state"].items()
    ):
        raise RuntimeError("endpoint checkpoint model failed bitwise reload verification")
    _atomic_hardlink(checkpoint, artifact_dir / "latest.pt")
    return {"path": str(checkpoint.resolve()), "sha256": _file_sha256(checkpoint),
            "global_step": trainer.global_step, "reload_verified": True}


def _completed_terminal(*, global_step, start_step, train_only, best_score, checkpoint, identity_digest):
    return {"status": "complete", "global_step": int(global_step), "start_step": int(start_step),
            "updates_completed": int(global_step - start_step), "train_only": bool(train_only),
            "best_validation_score": float(best_score) if math.isfinite(best_score) else None,
            "validation": None if train_only else "legacy sampled validation",
            "checkpoint": checkpoint, "run_identity_sha256": identity_digest, "parallelism": "ddp"}


def _avg_validation(all_vals):
    ref = all_vals[0]
    out = {}
    for k in ("aggregate_dense_relative_l2", "aggregate_query_relative_l2",
              "aggregate_late_relative_l2"):
        out[k] = float(sum(float(v[k]) for v in all_vals) / len(all_vals))
    return out


def run_ddp(config, *, artifact_dir, device_name, resume_path=None,
        fork_path=None, parent_config_digest=None, fork_allow_new_params=False,
        reset_dense_spectral=False, train_only=False, exclude_groups_json=None,
        run_identity_name=None, config_path=None):
    rank, world, device = _setup()
    main_rank = rank == 0
    if config.data.ic_frames <= 0:
        raise ValueError("ic8 DDP requires data.ic_frames > 0")
    if config.train.batch_records % world != 0:
        raise ValueError(f"batch_records={config.train.batch_records} must be divisible by world={world}")
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(manifest, {"train": config.data.expected_train_records,
                                         "validation": config.data.expected_validation_records})
    normalizer = load_normalizer(config, manifest.digest)
    max_steps = int(config.train.max_steps)
    state_path = fork_path if fork_path is not None else resume_path
    # Parent Adam tensors are not needed on GPU, including in a cold fork.
    state = torch.load(state_path, map_location="cpu", weights_only=False) if state_path is not None else None
    _require_updates(int(state["global_step"]) if state is not None else 0, max_steps)
    excluded_groups, exclusions_sha = _load_exclusions(exclude_groups_json, manifest)
    if excluded_groups and not train_only:
        raise ValueError("group exclusions require explicit --train-only")
    train_dataset, validation_batch, schedule = _training_data(
        config, manifest, train_only=train_only, excluded_group_ids=excluded_groups,
    )

    # Residual-adaptive sampling state.  Both are None unless the arm enables
    # them, which leaves the legacy energy-weighted, fixed-phase path untouched.
    # The maps live on the training dataset and are mutated in place, so a
    # materialization later in the same process sees the freshest statistics.
    train_record_count = len(train_dataset.records)
    residual_map = None
    residual_time_profile = None
    if config.data.residual_adaptive_space:
        nz, nx = len(manifest.z_m), len(manifest.x_m)
        residual_map = torch.full((nz, nx), 1.0 / (nz * nx), dtype=torch.float32)
        train_dataset.residual_map = residual_map.clone()
    if config.data.residual_adaptive_time:
        residual_time_profile = torch.full(
            (train_record_count, config.data.dense_time_steps),
            1.0 / config.data.dense_time_steps,
            dtype=torch.float32,
        )
        train_dataset.residual_time_profile = residual_time_profile.clone()

    torch.manual_seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    # Activation checkpointing on the dense blocks is DDP-safe through the
    # non-reentrant implementation (see dense.py); the wide decoder needs it to
    # fit a 51-frame supervision window.
    model = build_model(
        config,
        dense_checkpoint=bool(config.data.dense_checkpoint),
        dense_outer_checkpoint=bool(config.data.dense_outer_checkpoint),
    ).to(device)
    wrapper = ICForwardWrapper(model, normalizer)
    ddp_model = DistributedDataParallel(wrapper, device_ids=[rank]) if world > 1 else wrapper
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                                  weight_decay=config.train.weight_decay)
    trainer = GuardedV3Trainer(model, optimizer, checkpoint_dir=artifact_dir / "checkpoints",
                               manifest_digest=manifest.digest, config_digest=config.digest(),
                               gradient_clip=config.train.gradient_clip)
    weights = _weights(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    best_path = artifact_dir / "best.pt"; best_validation_path = artifact_dir / "best_validation.json"
    best_score = math.inf
    if main_rank and not train_only and best_validation_path.exists():
        with best_validation_path.open() as fh:
            best_score = float(json.load(fh)["validation_score"])
    start_step = 0
    fork_audit = None
    if fork_path is not None:
        fork_audit = _load_fork(
            model, optimizer, state, config, manifest_digest=manifest.digest,
            parent_config_digest=parent_config_digest, allow_new_params=fork_allow_new_params,
            reset_dense_spectral=reset_dense_spectral,
        )
        trainer.global_step = start_step = fork_audit["global_step"]
        if main_rank:
            _atomic_json(fork_audit, artifact_dir / "fork_audit.json")
            print(json.dumps({"event": fork_audit["event"], "global_step": start_step,
                              "optimizer": fork_audit["optimizer"],
                              "fresh_keys": list(fork_audit["fresh_tensors"])}, allow_nan=False), flush=True)
    if resume_path is not None:
        start_step, lr_before, lr_after = _load_resume(
            model, optimizer, state, config, manifest_digest=manifest.digest,
        )
        trainer.global_step = start_step
        if main_rank:
            print(json.dumps({"event": "resume_lr_assert", "checkpoint": str(resume_path),
                              "configured_lr": float(config.train.learning_rate),
                              "lr_before": lr_before, "lr_after": lr_after,
                              "configured_weight_decay": float(config.train.weight_decay),
                              "groups": len(optimizer.param_groups)}), flush=True)
            print(json.dumps({"event": "resume", "checkpoint": str(resume_path),
                              "global_step": start_step, "epoch": int(state["epoch"])}), flush=True)
    state = None
    code_digests = _code_digests()
    identity = {
        "name": run_identity_name, "parallelism": "ddp", "world_size": world,
        "train_only": bool(train_only), "validation_labels_accessed": not train_only,
        "test_id_labels_accessed": False, "batch_records": config.train.batch_records,
        "records_per_rank": config.train.batch_records // world,
        "micro_batch_records": config.train.micro_batch_records,
        "gradient_accumulation": "legacy sum of micro losses; DDP average across ranks; unchanged",
        "ic_frames": config.data.ic_frames, "dense_time_steps": config.data.dense_time_steps,
        "manifest_digest": manifest.digest, "normalizer_sha256": _file_sha256(config.data.normalization_json),
        "config_digest": config.digest(), "effective_config": asdict(config),
        "config_file": str(Path(config_path).resolve()) if config_path else None,
        "config_file_sha256": _file_sha256(config_path) if config_path else None,
        "code_files": code_digests, "code_digest": _json_sha256(code_digests),
        "parent_checkpoint": str(Path(state_path).resolve()) if state_path else None,
        "parent_checkpoint_sha256": _file_sha256(state_path) if state_path else None,
        "parent_config_digest": parent_config_digest,
        "fork_allow_new_params": bool(fork_allow_new_params), "reset_dense_spectral": bool(reset_dense_spectral),
        "fork_audit_sha256": _json_sha256(fork_audit) if fork_audit else None,
        "exclusions_path": str(Path(exclude_groups_json).resolve()) if exclude_groups_json else None,
        "exclusions_sha256": exclusions_sha, **_training_census(manifest, excluded_groups),
        "schedule_sha256": _json_sha256([asdict(spec) for spec in schedule]),
        "executed_schedule_sha256": _json_sha256([asdict(spec) for spec in schedule[start_step:]]),
        "start_step": start_step, "max_steps": max_steps, "additional_updates": max_steps - start_step,
        "artifact_dir": str(artifact_dir.resolve()),
        "endpoint_checkpoint": str((artifact_dir / "checkpoints" / f"checkpoint_step_{max_steps:08d}.pt").resolve()),
    }
    identity_digest = _json_sha256(identity)
    if main_rank:
        _atomic_json(identity, artifact_dir / "run_identity.json")
    started = time.monotonic()
    last_checkpoint = None
    loss_mean = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Micro-batch the rank's records with DDP gradient accumulation: each micro
    # does its own forward+backward, but only the LAST micro triggers the DDP
    # all-reduce (no_sync on the earlier ones).  This keeps peak memory to a
    # single micro-batch while still producing a single global-batch gradient.
    micro = int(config.train.micro_batch_records)
    from scripts.train_grouped_v3_ic8_fullfield import _to_device as _ic8_to_device
    for step in range(start_step, max_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.monotonic()
        tensors = _broadcast_batch(train_dataset, step, rank, world, device)
        n_records = tensors["source_parameters"].shape[0]
        samples = list(range(0, n_records, micro))
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        components = {"weighted": Counter(), "unweighted": Counter()}
        correction_probe = None
        res_final = None
        # Latest micro-batch's normalized prediction/target, kept for the
        # residual-map update (detached: the maps are statistics, not a path
        # for gradient to flow back into the loss).
        last_dense_prediction = None
        last_dense_target = None
        for mi, start in enumerate(samples):
            idx = slice(start, start + micro)
            # Record-dimension tensors slice by idx; x_m/z_m are the grid axes
            # (201,) and must NOT be sliced.  record_to_medium is remapped to
            # 0..micro-1 so it indexes the micro's own medium rows, and
            # velocity_mps is sliced once (stacked per row in the uniform/IC8
            # batch).  ic_snapshots_physical is also a record-dim tensor.
            micro_idx = torch.arange(start, start + micro)
            micro_t = {
                name: tensors[name][micro_idx]
                for name in _RECORD_KEYS
                if name != "record_to_medium" and name in tensors
            }
            _micro_velocity, _micro_mapping = _shard_media(
                tensors["velocity_mps"], tensors["record_to_medium"], micro_idx)
            micro_t["velocity_mps"] = _micro_velocity
            micro_t["record_to_medium"] = _micro_mapping
            micro_t["x_m"] = tensors["x_m"]
            micro_t["z_m"] = tensors["z_m"]
            src = micro_t["source_parameters"]
            is_last = mi == len(samples) - 1
            if world > 1 and not is_last:
                ctx = ddp_model.no_sync()
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                if step == start_step and mi == 0:
                    with _CorrectionProbe(model) as probe:
                        pd, pq = ddp_model(micro_t)
                    correction_probe = probe.report()
                else:
                    pd, pq = ddp_model(micro_t)
                td = normalizer.encode_pressure(micro_t["dense_target_physical"], src[:, 4])
                tq = normalizer.encode_pressure(micro_t["query_target_physical"], src[:, 4])
                daq = dense_at_pilot_queries(pd, micro_t["query_coords"], micro_t["requested_time_s"],
                                              x_m=micro_t["x_m"], z_m=micro_t["z_m"])
                res = compute_v3_losses(prediction_query=pq, target_query=tq,
                                         query_probability=micro_t["query_probability"],
                                         prediction_dense=pd, target_dense=td, dense_at_query=daq,
                                         weights=weights,
                                         phase_energy_fraction=config.loss.phase_energy_fraction,
                                         relative_energy_floor_fraction=config.loss.relative_energy_floor_fraction)
                loss = res.total
                _collective_finite(loss.detach(), world=world, device=device, label="micro-batch loss")
                loss.backward()
            total_loss += float(loss.detach())
            for kind in components:
                for key, value in getattr(res, kind).items():
                    components[kind][key] += float(value.detach())
            res_final = res
            last_dense_prediction = pd.detach()
            last_dense_target = td.detach()
        gradient_report = _checked_optimizer_step(
            model, optimizer, gradient_clip=config.train.gradient_clip, world=world, device=device,
            sample_update=step in (start_step, max_steps - 1),
        )
        # Update the residual statistics from this step's prediction, after the
        # step so a materialization can never see a map derived from the very
        # batch it is about to build.  Only the last micro-batch's prediction is
        # retained, which is enough: the maps are bounded EMAs and the earlier
        # micros differ only by which records they cover.
        if (residual_map is not None or residual_time_profile is not None) and last_dense_prediction is not None:
            # This rank's records for this step are schedule step_indices[rank::world];
            # the retained prediction is the LAST micro-batch, i.e. the tail of
            # that shard, so index the same way rather than assuming it covers all.
            step_records = train_dataset.schedule[step].record_indices
            step_shard = torch.tensor(
                [step_records[position] for position in range(rank, len(step_records), world)],
                dtype=torch.long,
                device=device,
            )
            tail = step_shard[len(step_shard) - last_dense_prediction.shape[0] :]
            new_map, new_profile = _update_residual_maps(
                residual_map,
                residual_time_profile,
                last_dense_prediction,
                last_dense_target,
                tail,
                world=world,
                device=device,
                gamma=float(config.data.residual_ema_gamma),
            )
            if new_map is not None:
                residual_map = new_map
                train_dataset.residual_map.copy_(residual_map)
            if new_profile is not None:
                residual_time_profile = new_profile
                train_dataset.residual_time_profile.copy_(residual_time_profile)
        loss_mean = total_loss / len(samples)
        trainer.global_step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_report = {
            "rank": rank, "loss": loss_mean, "loss_finite": math.isfinite(loss_mean),
            "records_this_rank": n_records, "microbatches": len(samples),
            "loss_components_mean_over_micros": {
                kind: {key: value / len(samples) for key, value in values.items()}
                for kind, values in components.items()
            },
            "correction_scale_after_update": float(model.dense_decoder.correction_scale.detach()),
            "first_micro_correction_probe": correction_probe, **gradient_report,
            "step_seconds_synchronized": time.monotonic() - step_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
        }
        rank_reports = [local_report]
        if world > 1:
            rank_reports = [None] * world
            dist.all_gather_object(rank_reports, local_report)
        report = {"global_step": trainer.global_step,
                  "loss": sum(item["loss"] for item in rank_reports) / world,
                  "elapsed": time.monotonic() - started, "ranks": rank_reports,
                  "step_seconds_max_rank": max(item["step_seconds_synchronized"] for item in rank_reports),
                  "peak_gib": max(item["peak_allocated_gib"] for item in rank_reports) if device.type == "cuda" else None}
        if main_rank:
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
                fh.flush()
            print(json.dumps({"event": "train_step", "global_step": trainer.global_step,
                              "loss": report["loss"], "peak_gib": report["peak_gib"],
                              "step_seconds_max_rank": report["step_seconds_max_rank"]}, allow_nan=False), flush=True)
        if trainer.global_step % config.train.steps_per_epoch:
            continue
        if train_only:
            if main_rank:
                last_checkpoint = _save_step_checkpoint(
                    trainer, artifact_dir, steps_per_epoch=config.train.steps_per_epoch, loss=loss_mean,
                )
                print(json.dumps({"event": "train_only_epoch", "checkpoint": last_checkpoint}, allow_nan=False), flush=True)
            if world > 1:
                dist.barrier()
            continue
        ddp_model.eval()
        if world > 1:
            local = evaluate_ic_batch(model, _rank_batch(validation_batch, rank, world), normalizer,
                                      device=device, micro_batch=config.train.micro_batch_records)
            all_objs = [None] * world
            dist.all_gather_object(all_objs, local)
            validation = _avg_validation(all_objs)
        else:
            validation = evaluate_ic_batch(model, validation_batch, normalizer, device=device,
                                           micro_batch=config.train.micro_batch_records)
        score = float(validation["aggregate_dense_relative_l2"]) + float(validation["aggregate_query_relative_l2"])
        if main_rank:
            ckpt = trainer.save_epoch(trainer.global_step // config.train.steps_per_epoch,
                                      metrics={"loss": float(loss), "validation_score": score})
            _atomic_hardlink(ckpt, artifact_dir / "latest.pt")
            rep = {"epoch": trainer.global_step // config.train.steps_per_epoch,
                   "global_step": trainer.global_step, "checkpoint": str(ckpt.resolve()),
                   "validation_score": score, "validation": validation}
            _atomic_json(rep, artifact_dir / "validation_latest.json")
            if score < best_score:
                best_score = score
                _atomic_hardlink(ckpt, best_path)
                _atomic_json(rep, best_validation_path)
            print(json.dumps({"event": "epoch", **rep}, sort_keys=True), flush=True)
    if main_rank:
        if last_checkpoint is None or last_checkpoint["global_step"] != trainer.global_step:
            last_checkpoint = _save_step_checkpoint(
                trainer, artifact_dir, steps_per_epoch=config.train.steps_per_epoch, loss=loss_mean,
            )
    if world > 1:
        dist.barrier()
    if main_rank:
        _atomic_json(_completed_terminal(
            global_step=trainer.global_step, start_step=start_step, train_only=train_only,
            best_score=best_score, checkpoint=last_checkpoint, identity_digest=identity_digest,
        ), artifact_dir / "terminal_report.json")
    if world > 1:
        dist.destroy_process_group()
    return None if train_only else best_score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--train-only", action="store_true",
                    help="never construct validation data or select best; save the fixed-budget endpoint")
    ap.add_argument("--exclude-groups-json", default=None)
    ap.add_argument("--run-identity", default=None)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from (model+optimizer+rng, digest-checked)")
    ap.add_argument("--fork-from", default=None,
                    help="parent checkpoint to fork from under a NEW config digest")
    ap.add_argument("--parent-config-digest", default=None,
                    help="required with --fork-from: expected parent checkpoint config digest")
    ap.add_argument("--reset-dense-spectral", action="store_true",
                    help="with --fork-from --fork-allow-new-params: re-initialize the dense "
                         "decoder spectral kernels instead of loading them from the parent. "
                         "Reset control for the radial reparameterization arm.")
    ap.add_argument("--fork-allow-new-params", action="store_true",
                    help="with --fork-from: load the key intersection, keep fresh init for "
                         "new parameters, and start the optimizer from scratch (reparameterized arms)")
    args = ap.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    if args.max_steps is not None:
        from dataclasses import replace
        config = replace(config, train=replace(config.train, max_steps=int(args.max_steps)))
    if args.fork_from and args.resume:
        raise SystemExit("--fork-from and --resume are mutually exclusive")
    if args.fork_allow_new_params and not args.fork_from:
        raise SystemExit("--fork-allow-new-params requires --fork-from")
    if args.reset_dense_spectral and not (args.fork_from and args.fork_allow_new_params):
        raise SystemExit("--reset-dense-spectral requires --fork-from with --fork-allow-new-params")
    if args.train_only and (Path(args.artifact_dir) / "run_identity.json").exists():
        raise SystemExit("train-only requires a fresh artifact directory")
    try:
        run_ddp(config, artifact_dir=Path(args.artifact_dir), device_name="cuda",
                resume_path=Path(args.resume) if args.resume else None,
                fork_path=Path(args.fork_from) if args.fork_from else None,
                parent_config_digest=args.parent_config_digest,
                fork_allow_new_params=bool(args.fork_allow_new_params),
                reset_dense_spectral=bool(args.reset_dense_spectral),
                train_only=bool(args.train_only), exclude_groups_json=args.exclude_groups_json,
                run_identity_name=args.run_identity, config_path=args.config)
    except Exception as e:
        if os.environ.get("RANK", "0") == "0":
            _atomic_json({"status": "failed", "error": str(e), "traceback": __import__("traceback").format_exc()},
                         Path(args.artifact_dir) / "terminal_report.json")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
