#!/usr/bin/env python3
"""Atomically freeze one corrected, non-diagnostic AIS-MQFNO recipe."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

try:
    from scripts.generate_ais_configs import (
        B0_MODEL, STAGES, canonical_loss_profiles, configure_legacy_b0,
        label_sites_per_update,
    )
except ModuleNotFoundError:  # direct execution outside the repository cwd
    from generate_ais_configs import (
        B0_MODEL, STAGES, canonical_loss_profiles, configure_legacy_b0,
        label_sites_per_update,
    )


class InjectedFreezeFailure(RuntimeError):
    """Raised only by the explicit test failure hook."""


@dataclass(frozen=True)
class FrozenRecipeOutputs:
    root: Path
    manifest_path: Path
    config_64: Path
    config_128: Path
    config_400: Path
    config_b0: Path
    checkpoint_64: Path
    manifest: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _outputs(root: Path, manifest: dict[str, Any]) -> FrozenRecipeOutputs:
    return FrozenRecipeOutputs(
        root, root / "recipe.json", root / "config_64.yaml", root / "config_128.yaml",
        root / "config_400.yaml", root / "config_b0.yaml",
        root / "seed20260714_64_best.pt", manifest,
    )


def _strict_mapping(payload: object, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def materialize_stage_configs(
    selected: dict[str, Any], *, frozen_split_path: Path | None = None,
    frozen_normalization_path: Path | None = None,
    frozen_train_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    per_aux = label_sites_per_update(selected["loss"])
    schedule = [[2000, 2048], [10000, per_aux]]
    total = sum(updates * sites for updates, sites in schedule)
    outputs: dict[str, dict[str, Any]] = {}
    for stage in (64, 128, 400):
        cfg = deepcopy(selected)
        cfg["loss_profiles"] = canonical_loss_profiles(selected["loss"])
        if frozen_split_path is not None:
            cfg["data"]["split_manifest"] = str(frozen_split_path.absolute())
        if frozen_normalization_path is not None:
            cfg["normalization"]["stats_path"] = str(frozen_normalization_path.absolute())
        values = STAGES[stage]
        cfg["sampling"].update({key: values[key] for key in
                                ("target_height", "target_width", "global_size")})
        cfg["train"]["query_chunk_size"] = values["query_chunk_size"]
        cfg["train"]["learning_rate"] = values["learning_rate"]
        if stage == 64:
            cfg["train"]["phases"] = [
                {"name": "field_pretrain", "optimizer_updates": 2000,
                 "loss_profile": "field", "reset_optimizer": True,
                 "reset_scheduler": True, "init_from": "random"},
                {"name": "selected_auxiliary", "optimizer_updates": 1000,
                 "loss_profile": "selected_frozen_loss", "reset_optimizer": True,
                 "reset_scheduler": True, "init_from": "phase_best",
                 "init_phase": "field_pretrain"},
            ]
        else:
            parent = 64 if stage == 128 else 128
            cfg["train"]["phases"] = [
                {"name": f"selected_{stage}", "optimizer_updates": values["optimizer_updates"],
                 "loss_profile": "selected_frozen_loss", "reset_optimizer": True,
                 "reset_scheduler": True, "init_from": "external_checkpoint",
                 "external_checkpoint_stage": parent},
            ]
        cfg["experiment"].update({
            "optimizer_updates": 12000, "physical_scene_draws": 12000,
            "label_site_schedule": deepcopy(schedule), "full160_label_sites": total,
            "diagnostic_only": False,
        })
        outputs[f"config_{stage}.yaml"] = cfg
    b0 = deepcopy(outputs["config_400.yaml"])
    b0.pop("loss_profiles", None)
    b0["model"] = deepcopy(B0_MODEL)
    b0["train"]["phases"] = [{
        "name": "matched_budget_b0", "optimizer_updates": 12000,
        "loss_profile": "field", "reset_optimizer": True,
        "reset_scheduler": True, "init_from": "random",
    }]
    configure_legacy_b0(
        b0, smoke=False,
        run_name="ais_mqfno_full160_native400_20260714/baselines/b0",
    )
    if frozen_train_count is not None:
        batch_size = int(b0["train"]["batch_size"])
        updates_per_epoch = math.ceil(int(frozen_train_count) / batch_size)
        if updates_per_epoch <= 0 or 12000 % updates_per_epoch != 0:
            raise ValueError(
                "frozen B0 updates per epoch must divide 12000 exactly "
                f"(train_count={frozen_train_count}, batch_size={batch_size})"
            )
        b0["train"]["epochs"] = 12000 // updates_per_epoch
    b0["experiment"].update({
        "label_site_schedule": deepcopy(schedule), "full160_label_sites": total,
        "optimizer_updates": 12000, "physical_scene_draws": 12000,
    })
    outputs["config_b0.yaml"] = b0
    return outputs


def _verify_existing(
    root: Path, selection: dict[str, Any], expected_manifest: dict[str, Any]
) -> FrozenRecipeOutputs:
    recipe_path = root / "recipe.json"
    try:
        manifest = _strict_mapping(json.loads(recipe_path.read_bytes().decode("utf-8")), "recipe")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing frozen recipe is unreadable") from exc
    expected_pair = (selection["selected_config_sha256"], selection["selected_checkpoint_sha256"])
    actual_pair = (manifest.get("selected_config_sha256"), manifest.get("selected_checkpoint_sha256"))
    if actual_pair != expected_pair:
        raise FileExistsError("refusing to overwrite a different frozen recipe")
    if manifest != expected_manifest:
        raise ValueError("existing frozen recipe is not identical to the canonical manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("existing recipe file hash inventory is invalid")
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"existing frozen file hash mismatch: {name}")
    return _outputs(root, manifest)


def _canonical_manifest(
    selection_bytes: bytes,
    config_bytes: bytes,
    checkpoint_bytes: bytes,
    split_manifest_bytes: bytes,
    normalization_bytes: bytes,
    generated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inventory = {
        name: sha256_bytes(yaml.safe_dump(payload, sort_keys=True).encode("utf-8"))
        for name, payload in generated.items()
    }
    inventory["seed20260714_64_best.pt"] = sha256_bytes(checkpoint_bytes)
    inventory["split_manifest.json"] = sha256_bytes(split_manifest_bytes)
    inventory["normalization_stats.json"] = sha256_bytes(normalization_bytes)
    return {
        "schema_version": 1,
        "selection_manifest_sha256": sha256_bytes(selection_bytes),
        "selected_config_sha256": sha256_bytes(config_bytes),
        "selected_checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        "split_manifest": {
            "file": "split_manifest.json",
            "sha256": sha256_bytes(split_manifest_bytes),
        },
        "normalization_stats": {
            "file": "normalization_stats.json",
            "sha256": sha256_bytes(normalization_bytes),
        },
        "confirmation_seeds": [20260714, 20260715, 20260716],
        "seed_aggregation": "per_sample_median_then_paired_bootstrap",
        "required_individual_seed_passes": 2,
        "threshold": 0.30,
        "sealed_test_authorized": False,
        "files": inventory,
    }


def freeze_recipe(
    selection_manifest: Path, output_dir: Path, *, fail_before_publish: bool = False
) -> FrozenRecipeOutputs:
    selection_path = Path(selection_manifest)
    output_dir = Path(output_dir)
    selection_bytes = selection_path.read_bytes()
    try:
        selection = _strict_mapping(json.loads(selection_bytes.decode("utf-8")), "selection manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selection manifest is not valid UTF-8 JSON") from exc
    for field in ("selected_config", "selected_checkpoint", "selected_config_sha256",
                  "selected_checkpoint_sha256", "diagnostic_only"):
        if field not in selection:
            raise ValueError(f"selection manifest missing {field}")
    selected_config = Path(selection["selected_config"])
    selected_checkpoint = Path(selection["selected_checkpoint"])
    config_bytes = selected_config.read_bytes()
    checkpoint_bytes = selected_checkpoint.read_bytes()
    if sha256_bytes(config_bytes) != selection["selected_config_sha256"] or \
       sha256_bytes(checkpoint_bytes) != selection["selected_checkpoint_sha256"]:
        raise ValueError("selection source hash mismatch")
    try:
        selected = _strict_mapping(yaml.safe_load(config_bytes.decode("utf-8")), "selected config")
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("selected config is not valid UTF-8 YAML") from exc
    if selection["diagnostic_only"] is not False or selected.get("experiment", {}).get("diagnostic_only") is True:
        raise ValueError("only non-diagnostic recipes may be frozen")
    if selected.get("loss", {}).get("hh_reweight") is not True:
        raise ValueError("only corrected Hansen-Hurwitz recipes may be frozen")
    split_value = selected.get("data", {}).get("split_manifest")
    if not isinstance(split_value, str) or not split_value:
        raise ValueError("selected config requires a split manifest path")
    split_manifest_bytes = Path(split_value).read_bytes()
    try:
        split_payload = json.loads(split_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("split manifest is not valid UTF-8 JSON") from exc
    if not isinstance(split_payload, dict) or any(
        not isinstance(split_payload.get(name), list) for name in ("train", "val", "test")
    ):
        raise ValueError("split manifest requires train, val, and test lists")
    normalization_value = selected.get("normalization", {}).get("stats_path")
    if not isinstance(normalization_value, str) or not normalization_value:
        raise ValueError("selected config requires normalization stats")
    normalization_bytes = Path(normalization_value).read_bytes()
    try:
        normalization_payload = json.loads(normalization_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("normalization stats are not valid UTF-8 JSON") from exc
    if not isinstance(normalization_payload, dict):
        raise ValueError("normalization stats must be a mapping")
    generated = materialize_stage_configs(
        selected,
        frozen_split_path=output_dir / "split_manifest.json",
        frozen_normalization_path=output_dir / "normalization_stats.json",
        frozen_train_count=len(split_payload["train"]),
    )
    canonical_manifest = _canonical_manifest(
        selection_bytes, config_bytes, checkpoint_bytes, split_manifest_bytes,
        normalization_bytes, generated,
    )
    if output_dir.exists():
        return _verify_existing(output_dir, selection, canonical_manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.{next(tempfile._get_candidate_names())}.tmp"
    try:
        staging.mkdir()
        inventory: dict[str, str] = {}
        for name, payload in generated.items():
            data = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
            path = staging / name
            _write_fsynced(path, data)
            reread = path.read_bytes()
            if yaml.safe_load(reread.decode("utf-8")) != payload:
                raise ValueError(f"generated config round-trip mismatch: {name}")
            inventory[name] = sha256_bytes(reread)
        checkpoint_name = "seed20260714_64_best.pt"
        _write_fsynced(staging / checkpoint_name, checkpoint_bytes)
        inventory[checkpoint_name] = sha256_bytes(checkpoint_bytes)
        _write_fsynced(staging / "split_manifest.json", split_manifest_bytes)
        inventory["split_manifest.json"] = sha256_bytes(split_manifest_bytes)
        _write_fsynced(staging / "normalization_stats.json", normalization_bytes)
        inventory["normalization_stats.json"] = sha256_bytes(normalization_bytes)
        manifest = canonical_manifest
        if inventory != manifest["files"]:
            raise ValueError("staged frozen file hashes differ from the canonical inventory")
        manifest_data = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        _write_fsynced(staging / "recipe.json", manifest_data)
        _fsync_directory(staging)
        if fail_before_publish:
            raise InjectedFreezeFailure("injected failure before recipe publication")
        os.replace(staging, output_dir)
        _fsync_directory(output_dir.parent)
        return _outputs(output_dir, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = freeze_recipe(args.selection_manifest, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
