#!/usr/bin/env python3
"""Render an atomic B2-v5 cache with base and source-aware conditioning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.b2_v5_components import append_source_conditioning
from saved_time_phase_operator_v4.instance_adaptation.b2_v6_encoding import (
    enriched_physical_conditioning,
)
from scripts.train_b2_snapshot_ic import build_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cache.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to reuse cache or output directory")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    staging = args.cache.with_name(f"{args.cache.name}.v5stage.{os.getpid()}")
    try:
        manifest = json.loads(args.manifest.read_text())
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if manifest.get("schema") != "b2_v5_causal_stratified_manifest_v1":
            raise RuntimeError("manifest is not a B2-v5 causal manifest")
        if manifest.get("future_truth_opened_for_window_selection"):
            raise RuntimeError("manifest window selection used future truth")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        if _sha256(Path(__file__)) != bindings["cache_builder_sha256"]:
            raise RuntimeError("cache builder binding drift")
        if _sha256(args.manifest) != bindings["manifest_sha256"][str(args.manifest)]:
            raise RuntimeError("manifest binding drift")
        build_cache(manifest, torch.device("cuda"), cache_path=staging, data_split="train")
        source_path = Path(manifest["source_h5"])
        with h5py.File(source_path, "r", swmr=True) as source, h5py.File(
            staging, "r+"
        ) as cache:
            indices = np.asarray(
                [row["source_index"] for row in manifest["records"]], dtype=np.int64
            )
            f0 = np.asarray(source["source_f0_hz"][indices], dtype=np.float64)
            t0 = np.asarray(source["source_t0_s"][indices], dtype=np.float64)
            for row, value_f0, value_t0 in zip(manifest["records"], f0, t0):
                if not np.isclose(row["source_f0_hz"], value_f0):
                    raise RuntimeError("manifest/source f0 drift")
                if not np.isclose(row["source_t0_s"], value_t0):
                    raise RuntimeError("manifest/source t0 drift")
            base = cache["cond"]
            source_cond = cache.create_dataset(
                "cond_source",
                shape=(base.shape[0], base.shape[1] + 2, base.shape[2], base.shape[3]),
                dtype=np.float16,
            )
            for lo in range(0, len(indices), 16):
                hi = min(lo + 16, len(indices))
                source_cond[lo:hi] = append_source_conditioning(
                    base[lo:hi].astype(np.float32), f0[lo:hi], t0[lo:hi]
                ).astype(np.float16)
            physics_cond = cache.create_dataset(
                "cond_physics",
                shape=(base.shape[0], 20, base.shape[2], base.shape[3]),
                dtype=np.float16,
            )
            velocity = np.asarray(source["velocity_mps"][indices], dtype=np.float32)
            for lo in range(0, len(indices), 8):
                hi = min(lo + 8, len(indices))
                physics_cond[lo:hi] = enriched_physical_conditioning(
                    torch.from_numpy(base[lo:hi].astype(np.float32)),
                    torch.from_numpy(velocity[lo:hi, None]),
                    torch.from_numpy(f0[lo:hi].astype(np.float32)),
                    torch.from_numpy(t0[lo:hi].astype(np.float32)),
                ).numpy().astype(np.float16)
            cache.attrs["schema"] = "b2_v5_causal_cache_v1"
            cache.attrs["window_rule"] = manifest["window_rule"]
            cache.attrs["lead_cycles"] = manifest["lead_cycles"]
            cache.attrs["source_conditioning"] = "f0_norm=(f0-20)/10;t0_norm=(t0-0.10)/0.05"
            cache.flush()
        os.replace(staging, args.cache)
        identity = {
            "schema": "b2_v5_cache_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "cache": str(args.cache),
            "cache_sha256": _sha256(args.cache),
            "cache_builder_sha256": _sha256(Path(__file__)),
            "record_count": len(manifest["records"]),
            "conditioning_channels": {"control": 7, "source_cond": 9, "physics_cond": 20},
            "future_truth_opened_for_window_selection": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json({
            "status": "complete",
            "cache": str(args.cache),
            "cache_sha256": identity["cache_sha256"],
            "run_identity": str(args.output_dir / "run_identity.json"),
        }, terminal)
        print(json.dumps(json.loads(terminal.read_text()), indent=2))
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "staging_cache": str(staging),
        }, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
