#!/usr/bin/env python3
"""Supervise four-GPU prediction followed by one-shot validation scoring."""
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
PREREG = RESULTS / "transfer_dg_wfp_validation30_preregistration_20260903.json"
MANIFEST = RESULTS / "transfer_dg_wfp_validation30_manifest_20260903.json"
CHECKPOINT = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903/latest.pt"
TRAINING_TERMINAL = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903/terminal.json"
TRAINING_CACHE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902/shard_0.h5"
PREDICTOR = ROOT / "scripts/predict_transfer_dg_wfp_validation401.py"
SCORER = ROOT / "scripts/score_transfer_dg_wfp_validation401.py"
OUT = RESULTS / "transfer_dg_wfp_validation30_20260903"


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


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to reuse validation output: {OUT}")
    prereg = json.loads(PREREG.read_text())
    bindings = prereg["bindings"]
    bound_paths = {
        "checkpoint_sha256": CHECKPOINT,
        "training_terminal_sha256": TRAINING_TERMINAL,
        "validation_manifest_sha256": MANIFEST,
        "predictor_sha256": PREDICTOR,
        "scorer_sha256": SCORER,
        "launcher_sha256": Path(__file__),
        "wfp_module_sha256": ROOT / "saved_time_phase_operator_v4/wfp.py",
        "phase_module_sha256": ROOT / "saved_time_phase_operator_v4/phase_carrier.py",
        "eikonal_module_sha256": ROOT / "saved_time_phase_operator_v4/eikonal.py",
    }
    for key, path in bound_paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    training_terminal = json.loads(TRAINING_TERMINAL.read_text())
    if training_terminal.get("status") != "complete":
        raise RuntimeError("final pretraining terminal is not complete")
    if training_terminal.get("validation_opened") or training_terminal.get("test_id_opened"):
        raise RuntimeError("training lineage reports an opened frozen split")

    OUT.mkdir(parents=True)
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir()
    started = time.time()
    common = [
        "--manifest", str(MANIFEST),
        "--preregistration", str(PREREG),
        "--checkpoint", str(CHECKPOINT),
        "--training-cache", str(TRAINING_CACHE),
        "--worker-count", "4",
        "--frequency-batch", str(prereg["prediction"]["frequency_batch"]),
    ]
    processes = []
    logs = []
    commands = []
    for worker in range(4):
        output = prediction_dir / f"shard_{worker}.h5"
        command = [
            sys.executable,
            str(PREDICTOR),
            *common,
            "--output", str(output),
            "--worker-index", str(worker),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(worker)
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["OMP_NUM_THREADS"] = "4"
        environment["PYTHONPATH"] = f'{ROOT}:{ROOT / "src"}'
        log = (OUT / f"worker_{worker}.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        logs.append(log)
        commands.append(command)
    atomic_json(
        {
            "schema": "transfer_dg_wfp_validation401_supervisor_identity_v1",
            "pid": os.getpid(),
            "child_pids": [process.pid for process in processes],
            "gpu_ids": [0, 1, 2, 3],
            "commands": commands,
            "checkpoint_sha256": bindings["checkpoint_sha256"],
            "validation_manifest_sha256": bindings["validation_manifest_sha256"],
            "prediction_before_truth_scoring": True,
            "model_input_wavefield_frames": 0,
            "validation_future_truth_opened": False,
            "test_id_opened": False,
            "started_unix_s": started,
        },
        OUT / "run_identity.json",
    )
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(code != 0 for code in codes):
        atomic_json(
            {
                "schema": "transfer_dg_wfp_validation401_supervisor_terminal_v1",
                "status": "prediction_failed",
                "prediction_return_codes": codes,
                "validation_future_truth_opened": False,
                "test_id_opened": False,
                "elapsed_s": time.time() - started,
            },
            OUT / "terminal.json",
        )
        return 2

    predictions = [prediction_dir / f"shard_{worker}.h5" for worker in range(4)]
    score_command = [
        sys.executable,
        str(SCORER),
        "--manifest", str(MANIFEST),
        "--preregistration", str(PREREG),
    ]
    for prediction in predictions:
        score_command.extend(["--prediction", str(prediction)])
    score_command.extend(["--output-dir", str(OUT / "scored")])
    with (OUT / "scorer.log").open("w") as scorer_log:
        score_process = subprocess.Popen(
            score_command,
            cwd=ROOT,
            env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONPATH": f'{ROOT}:{ROOT / "src"}'},
            stdout=scorer_log,
            stderr=subprocess.STDOUT,
        )
        score_code = score_process.wait()
    score_terminal_path = OUT / "scored/terminal.json"
    score_terminal = (
        json.loads(score_terminal_path.read_text()) if score_terminal_path.exists() else None
    )
    complete = score_code == 0 and score_terminal and score_terminal.get("status") == "complete"
    atomic_json(
        {
            "schema": "transfer_dg_wfp_validation401_supervisor_terminal_v1",
            "status": "complete" if complete else "scoring_failed",
            "prediction_return_codes": codes,
            "scorer_return_code": score_code,
            "scorer_terminal": score_terminal,
            "elapsed_s": time.time() - started,
            "validation_future_truth_opened": bool(score_terminal),
            "test_id_opened": False,
        },
        OUT / "terminal.json",
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
