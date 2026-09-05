#!/usr/bin/env python3
"""Build and train the four-seed train-only phase/scatter-64 pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "transfer_dg_phase_scatter64_pilot_manifest_20260903.json"
PREREG = RESULTS / "transfer_dg_phase_scatter64_pilot_preregistration_20260903.json"
PARENT = RESULTS / "transfer_dg_wfp_full2800_pretraining_20260902/high_s372/best.pt"
BASE_CACHE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902"
TRAVEL = RESULTS / "transfer_dg_wfp_full2800_travel_20260902"
CACHE_BUILDER = ROOT / "scripts/build_transfer_dg_phase_scatter64_pilot_cache.py"
TRAINER = ROOT / "scripts/train_transfer_dg_phase_scatter64_pilot.py"
OUT = RESULTS / "transfer_dg_phase_scatter64_pilot_20260903"
SEEDS = (372, 733, 1009, 2027)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def launch(commands: list[list[str]], logs: list[Path]) -> tuple[list[int], list[int]]:
    processes = []
    handles = []
    for gpu, (command, log_path) in enumerate(zip(commands, logs)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["OMP_NUM_THREADS"] = "4"
        environment["PYTHONPATH"] = f'{ROOT}:{ROOT / "src"}'
        handle = log_path.open("w")
        handles.append(handle)
        processes.append(subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        ))
    codes = [process.wait() for process in processes]
    pids = [process.pid for process in processes]
    for handle in handles:
        handle.close()
    return codes, pids


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to reuse pilot output: {OUT}")
    prereg = json.loads(PREREG.read_text())
    bindings = prereg["bindings"]
    paths = {
        "pilot_manifest_sha256": MANIFEST,
        "parent_checkpoint_sha256": PARENT,
        "cache_builder_sha256": CACHE_BUILDER,
        "trainer_sha256": TRAINER,
        "phase_scatter_module_sha256": ROOT / "saved_time_phase_operator_v4/phase_scatter.py",
        "wfp_module_sha256": ROOT / "saved_time_phase_operator_v4/wfp.py",
        "launcher_sha256": Path(__file__),
    }
    for key, path in paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    if tuple(prereg["training"]["seeds"]) != SEEDS:
        raise RuntimeError("seed plan drift")
    OUT.mkdir(parents=True)
    cache_dir = OUT / "cache"
    lane_dir = OUT / "lanes"
    cache_dir.mkdir()
    lane_dir.mkdir()
    started = time.time()
    base_args = sum(
        (["--base-cache", str(BASE_CACHE / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    travel_args = sum(
        (["--travel", str(TRAVEL / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    cache_commands = []
    for worker in range(4):
        cache_commands.append([
            sys.executable, str(CACHE_BUILDER),
            "--manifest", str(MANIFEST),
            "--preregistration", str(PREREG),
            "--parent-checkpoint", str(PARENT),
            *base_args, *travel_args,
            "--output", str(cache_dir / f"shard_{worker}.h5"),
            "--worker-index", str(worker), "--worker-count", "4",
        ])
    atomic_json({
        "schema": "transfer_dg_phase_scatter64_pilot_identity_v1",
        "pid": os.getpid(),
        "state": "building_cache",
        "cache_commands": cache_commands,
        "gpu_ids": [0, 1, 2, 3],
        "parent_checkpoint_sha256": bindings["parent_checkpoint_sha256"],
        "pilot_manifest_sha256": bindings["pilot_manifest_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
        "started_unix_s": started,
    }, OUT / "run_identity.json")
    cache_codes, cache_pids = launch(
        cache_commands, [OUT / f"cache_worker_{worker}.log" for worker in range(4)]
    )
    if any(code != 0 for code in cache_codes):
        atomic_json({
            "schema": "transfer_dg_phase_scatter64_pilot_terminal_v1",
            "status": "cache_failed", "cache_return_codes": cache_codes,
            "cache_pids": cache_pids, "validation_opened": False,
            "test_id_opened": False, "elapsed_s": time.time() - started,
        }, OUT / "terminal.json")
        return 2
    cache_args = sum(
        (["--cache", str(cache_dir / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    train_commands = []
    for seed in SEEDS:
        train_commands.append([
            sys.executable, str(TRAINER), *cache_args,
            "--manifest", str(MANIFEST),
            "--preregistration", str(PREREG),
            "--output-dir", str(lane_dir / f"seed_{seed}"),
            "--seed", str(seed),
        ])
    identity = json.loads((OUT / "run_identity.json").read_text())
    identity.update({
        "state": "training", "cache_return_codes": cache_codes,
        "cache_pids": cache_pids, "train_commands": train_commands,
    })
    atomic_json(identity, OUT / "run_identity.json")
    train_codes, train_pids = launch(
        train_commands, [OUT / f"train_seed_{seed}.log" for seed in SEEDS]
    )
    terminals = []
    for seed in SEEDS:
        path = lane_dir / f"seed_{seed}/terminal.json"
        if path.exists():
            terminals.append(json.loads(path.read_text()))
    if len(terminals) != 4 or any(code != 0 for code in train_codes):
        summary = {
            "schema": "transfer_dg_phase_scatter64_pilot_summary_v1",
            "status": "failed", "train_return_codes": train_codes,
            "terminal_count": len(terminals), "validation_opened": False,
            "test_id_opened": False,
        }
    else:
        parent_mean = sum(row["confirmation"]["parent_mean"] for row in terminals) / 4.0
        candidate_mean = sum(row["confirmation"]["candidate_mean"] for row in terminals) / 4.0
        accepted = candidate_mean < parent_mean and all(
            row["confirmation"]["top_pressure_max_abs"] == 0.0 for row in terminals
        )
        summary = {
            "schema": "transfer_dg_phase_scatter64_pilot_summary_v1",
            "status": "accepted" if accepted else "rejected",
            "decision": "scale_to_full_train_pool" if accepted else "stop_phase_scatter_route",
            "parent_confirmation_mean": parent_mean,
            "candidate_confirmation_mean": candidate_mean,
            "relative_gain": 1.0 - candidate_mean / max(parent_mean, 1.0e-300),
            "per_family_parent": {
                family: sum(row["confirmation"]["per_family_parent"][family] for row in terminals) / 4.0
                for family in ("uniform", "layered", "anomaly", "marmousi")
            },
            "per_family_candidate": {
                family: sum(row["confirmation"]["per_family_candidate"][family] for row in terminals) / 4.0
                for family in ("uniform", "layered", "anomaly", "marmousi")
            },
            "lanes": {f"seed_{row['seed']}": row for row in terminals},
            "minimum_improvement_fraction": 0.0,
            "validation_opened": False,
            "test_id_opened": False,
        }
    atomic_json(summary, OUT / "summary.json")
    atomic_json({
        "schema": "transfer_dg_phase_scatter64_pilot_terminal_v1",
        "status": "complete" if len(terminals) == 4 and all(code == 0 for code in train_codes) else "failed",
        "cache_return_codes": cache_codes,
        "train_return_codes": train_codes,
        "train_pids": train_pids,
        "summary": str((OUT / "summary.json").resolve()),
        "elapsed_s": time.time() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }, OUT / "terminal.json")
    return 0 if len(terminals) == 4 and all(code == 0 for code in train_codes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
