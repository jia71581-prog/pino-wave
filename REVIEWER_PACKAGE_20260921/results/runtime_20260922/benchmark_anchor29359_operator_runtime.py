#!/usr/bin/env python3
"""Single-shot inference wall-clock for the anchor-29359 operator.

Timed protocol, matched to the traditional-solver reference
(``benchmark_lwc84_traditional_runtime.py``, schema
``lwc84_traditional_runtime_reference_v1``) so the two numbers divide:

  one record in, the full 401-frame [201, 201] pressure field materialised on
  the GPU in physical units, CUDA-synchronised timing, no disk I/O inside the
  timed region.

The timed region covers the whole deployment path: velocity encoding + source
preparation (prepare), then the dense decode of all 401 frames at
``time_block=1`` -- the same iter_dense_normalized loop the paper's panel
measurement uses, plus the pressure de-normalisation.  Reading the 8 stored IC
frames from HDF5 is timed separately and reported, but sits outside the
operator time because it is data loading, not computation; the traditional
solver needs no IC read at all (it starts from the source wavelet).

Same anchor discipline as the figure script: the anchor's frozen code snapshot
tree, sha-asserted checkpoint, and the three family-median records the paper's
figures already use, so every artifact talks about the same records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

REL = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/release_ic8_40m_20260910")
GATE_DIR = REL / "research/l1_gate_measure_20260913"
SNAPSHOT = REL / "research/l2_lwc84_ddp4_20260915_v1/snapshot"
sys.path.insert(0, str(SNAPSHOT))

from grouped_ufno_mionet_v3.config import V3Config  # noqa: E402
from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset  # noqa: E402
from scripts.train_grouped_v3_pilot import load_normalizer  # noqa: E402

for _module, _name in ((V3Config, "config"), (V3WavefieldDataset, "records")):
    _origin = Path(sys.modules[_module.__module__].__file__).resolve()
    if SNAPSHOT.resolve() not in _origin.parents:
        raise SystemExit(f"{_name} was imported from {_origin}, not the anchor snapshot")

sys.path.insert(0, str(GATE_DIR))
import measure_gates as mg  # noqa: E402

CONFIG = REL / "research/l2_lwc84_ddp4_20260915_v1/train.yaml"
CHECKPOINT = Path(
    "/root/autodl-tmp/staging/l2_lwc84_ddp4_20260915_v1/formal/checkpoints/checkpoint_step_00029359.pt")
CHECKPOINT_SHA = "6a45c074bf0fcac335b6ac69c0bc1527d7824a25106e34be041d81b7f7c03f24"
# The records the paper's Figures 1-2 already show: family median by
# full-future relative L2 over each family's frozen list.
RECORDS = ("train_uniform_00413", "train_layered_00299", "train_marmousi_00010")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    import math
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats-per-record", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--time-block", type=int, default=1,
                        help="frames decoded per forward pass; 1 is the panel evaluation "
                             "path, larger blocks are the deployment batching")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats_per_record <= 0 or args.warmup_runs < 0 or args.time_block <= 0:
        raise SystemExit("repeats and time block must be positive, warmups nonnegative")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the runtime comparison must run on the deployment GPU")
    mg.require_vram(device, 8.0, 0.0)

    cfg = V3Config.from_yaml(str(CONFIG))
    manifest = build_manifest(cfg.data.source_h5)
    dataset = V3WavefieldDataset(cfg.data.source_h5, manifest, split="train")
    normalizer = load_normalizer(cfg, manifest.digest)
    model, provenance = mg.load_checkpoint(CHECKPOINT, cfg, device, CHECKPOINT_SHA, False)
    if provenance["backfill"]["measures_untrained_parameters"]:
        raise SystemExit("checkpoint would measure untrained parameters")

    positions = mg.resolve_positions(dataset, list(RECORDS))
    prepared_records = []
    for sample_id, position in zip(RECORDS, positions):
        record = dataset[position]
        onset = mg.source_onset(record)
        prepared_records.append((sample_id, position, record, onset))

    def run_one(entry):
        """One full deployment pass; returns per-stage CUDA-synchronised seconds."""
        sample_id, position, record, onset = entry
        # Stage 0 (excluded from operator time): read the 8 stored IC frames.
        t0 = time.perf_counter()
        ic = dataset.read_wavefield(position, record.time_s[onset:onset + mg.IC_FRAMES])
        if not bool(ic.exact.all()):
            raise SystemExit("IC frames must be exact stored snapshots")
        ic_read_s = time.perf_counter() - t0

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        prepared = mg.prepare_record(model, normalizer, record, device, ic.values,
                                     float(record.time_s[onset]))
        torch.cuda.synchronize(device)
        prepare_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        frames = []
        for _, value in model.iter_dense_normalized(
                prepared, record.time_s.float().to(device),
                x_m=record.x_m.to(device), z_m=record.z_m.to(device),
                time_block=args.time_block):
            frames.append(value)
        prediction = torch.cat(frames, dim=1)
        pressure = normalizer.decode_pressure(
            prediction.float(), record.source_parameters[4:5].to(device))
        torch.cuda.synchronize(device)
        decode_s = time.perf_counter() - t2
        if tuple(pressure.shape) != (1, 401, 201, 201):
            raise SystemExit(f"operator returned shape {tuple(pressure.shape)}")
        peak = float(pressure.abs().max())
        del frames, prediction, pressure
        return ic_read_s, prepare_s, decode_s, peak

    for index in range(args.warmup_runs):
        run_one(prepared_records[index % len(prepared_records)])

    measurements = []
    for entry in prepared_records:
        for repeat in range(args.repeats_per_record):
            ic_read_s, prepare_s, decode_s, peak = run_one(entry)
            measurements.append({
                "sample_id": entry[0],
                "family": entry[2].medium_type,
                "repeat": repeat,
                "ic_read_s": ic_read_s,
                "prepare_s": prepare_s,
                "dense_decode_401f_s": decode_s,
                "operator_runtime_s": prepare_s + decode_s,
                "output_abs_peak_pa": peak,
            })

    runtimes = [m["operator_runtime_s"] for m in measurements]
    report = {
        "status": "complete",
        "schema": "anchor29359_operator_runtime_reference_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "task": "one record -> all 401 stored frames [201,201] in physical units",
            "time_block": args.time_block,
            "single_instance": True,
            "cuda_synchronized_timing": True,
            "includes_output_materialization": True,
            "excludes_disk_io": True,
            "ic_read_reported_but_excluded": True,
            "matches": "lwc84_traditional_runtime_reference_v1",
        },
        "config_file": str(CONFIG),
        "config_digest": cfg.digest(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "global_step": provenance["global_step"],
        "code_tree": str(SNAPSHOT),
        "script_sha256": file_sha256(Path(__file__)),
        "records": list(RECORDS),
        "record_selection": "the paper's figure records: family median by full-future relL2",
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "tf32": False,
        "warmup_runs": args.warmup_runs,
        "repeats_per_record": args.repeats_per_record,
        "measurements": measurements,
        "minimum_runtime_s": min(runtimes),
        "mean_runtime_s": sum(runtimes) / len(runtimes),
        "p50_runtime_s": nearest_rank(runtimes, 0.50),
        "p95_runtime_s": nearest_rank(runtimes, 0.95),
        "ic_read_s_mean": float(np.mean([m["ic_read_s"] for m in measurements])),
        "prepare_s_mean": float(np.mean([m["prepare_s"] for m in measurements])),
        "dense_decode_401f_s_mean": float(
            np.mean([m["dense_decode_401f_s"] for m in measurements])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "minimum_runtime_s", "mean_runtime_s", "p50_runtime_s", "p95_runtime_s",
        "prepare_s_mean", "dense_decode_401f_s_mean")}, indent=2))
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
