#!/usr/bin/env python3
"""Wait for E1 caches, audit them, run four paired lanes, and decide E1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results/transfer_dg_wfp_e1_cache_20260902"
OUTPUT = ROOT / "results/transfer_dg_wfp_e1_training_20260902"
MANIFEST = ROOT / "results/transfer_dg_wfp_e1_manifest_256_20260902.json"
PREREG = ROOT / "results/transfer_dg_wfp_e1_preregistration_20260902.json"
LANES = (("fno", 372), ("fno", 733), ("wfp", 372), ("wfp", 733))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wait_for_cache() -> dict[str, object]:
    terminal_path = CACHE_DIR / "terminal.json"
    while not terminal_path.is_file():
        time.sleep(30)
    terminal = json.loads(terminal_path.read_text())
    if terminal.get("status") != "complete" or int(terminal.get("record_count", -1)) != 256:
        raise RuntimeError(f"E1 cache supervisor failed: {terminal}")
    return terminal


def summarize(terminals: list[dict[str, object]]) -> dict[str, object]:
    grouped = {arm: [row for row in terminals if row["arm"] == arm] for arm in ("fno", "wfp")}
    arm_summary = {}
    for arm, rows in grouped.items():
        arm_summary[arm] = {
            "seed_confirmation": {
                str(row["seed"]): row["confirmation_full"] for row in rows
            },
            "mean_confirmation_physical": sum(
                float(row["confirmation_full"]["physical_mean"]) for row in rows
            ) / len(rows),
            "worst_seed_confirmation_physical": max(
                float(row["confirmation_full"]["physical_mean"]) for row in rows
            ),
            "mean_confirmation_cpml": sum(
                float(row["confirmation_full"]["cpml_normalized_mean"]) for row in rows
            ) / len(rows),
            "parameter_counts": [
                json.loads(
                    (Path(row["best_checkpoint"]).parent / "run_identity.json").read_text()
                )["parameter_count"]
                for row in rows
            ],
        }
    gain = 1.0 - arm_summary["wfp"]["mean_confirmation_physical"] / max(
        arm_summary["fno"]["mean_confirmation_physical"], 1.0e-30
    )
    boundary_ok = all(
        float(row["confirmation_full"]["top_pressure_max_abs"]) == 0.0
        and float(row["confirmation_full"]["outer_pressure_max_abs"]) == 0.0
        for row in terminals
    )
    finite = all(
        all(
            value is not None and float(value) == float(value)
            for value in (
                row["confirmation_full"]["physical_mean"],
                row["confirmation_full"]["cpml_normalized_mean"],
            )
        )
        for row in terminals
    )
    accepted = gain > 0.0 and boundary_ok and finite
    return {
        "schema": "transfer_dg_wfp_e1_summary_v1",
        "status": "accepted" if accepted else "rejected",
        "decision": "accepted_e1_expand_to_2800" if accepted else "rejected_e1_stop_expansion",
        "primary_metric": "two-seed mean confirmation physical 64-bin relative L2",
        "arms": arm_summary,
        "relative_gain_wfp_vs_fno": gain,
        "boundary_gate_passed": boundary_ok,
        "finite_gate_passed": finite,
        "all_four_families_present": all(
            set(row["confirmation_full"]["per_family"]) == {"uniform", "layered", "anomaly", "marmousi"}
            for row in terminals
        ),
        "conditional_next_step": "expand_cache_and_training_to_all_2800_train_records" if accepted else "diagnose_E1_without_new_data_generation",
        "validation_opened": False,
        "test_id_opened": False,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    queue_identity = {
        "schema": "transfer_dg_wfp_e1_training_queue_v1",
        "supervisor_pid": os.getpid(),
        "state": "waiting_for_cache",
        "cache_terminal": str(CACHE_DIR / "terminal.json"),
        "started_unix_s": time.time(),
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(queue_identity, OUTPUT / "queue_identity.json")
    cache_terminal = wait_for_cache()
    cache_paths = [CACHE_DIR / f"shard_{gpu}.h5" for gpu in range(4)]
    summaries = [
        json.loads(path.with_suffix(path.suffix + ".summary.json").read_text())
        for path in cache_paths
    ]
    if sum(int(row["record_count"]) for row in summaries) != 256:
        raise RuntimeError("E1 cache summaries do not total 256 records")
    trainer = ROOT / "scripts/train_transfer_dg_wfp_e1.py"
    module = ROOT / "saved_time_phase_operator_v4/wfp.py"
    audit = {
        "schema": "transfer_dg_wfp_e1_training_preregistration_audit_v1",
        "cleared_for_four_gpu_training": True,
        "preregistration_sha256": sha256(PREREG),
        "manifest_sha256": sha256(MANIFEST),
        "trainer_sha256": sha256(trainer),
        "wfp_module_sha256": sha256(module),
        "cache_supervisor_terminal_sha256": sha256(CACHE_DIR / "terminal.json"),
        "cache_output_sha256": {
            str(path): summary["output_sha256"]
            for path, summary in zip(cache_paths, summaries, strict=True)
        },
        "record_count": 256,
        "validation_opened": False,
        "test_id_opened": False,
        "veto_reason": None,
    }
    atomic_json(audit, OUTPUT / "preregistration_audit.json")
    processes, logs = [], []
    commands = []
    training_started = time.time()
    for gpu, (arm, seed) in enumerate(LANES):
        lane = OUTPUT / f"{arm}_s{seed}"
        command = [
            sys.executable, str(trainer),
            *sum((["--cache", str(path)] for path in cache_paths), []),
            "--manifest", str(MANIFEST),
            "--preregistration", str(PREREG),
            "--output-dir", str(lane),
            "--arm", arm, "--seed", str(seed),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
        log = (OUTPUT / f"{arm}_s{seed}.log").open("w")
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        commands.append(command)
        processes.append(process)
        logs.append(log)
    atomic_json({
        "schema": "transfer_dg_wfp_e1_training_supervisor_identity_v1",
        "supervisor_pid": os.getpid(),
        "child_pids": [process.pid for process in processes],
        "commands": commands,
        "training_started_unix_s": training_started,
        "validation_opened": False,
        "test_id_opened": False,
    }, OUTPUT / "run_identity.json")
    return_codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    terminals = []
    if all(code == 0 for code in return_codes):
        terminals = [
            json.loads((OUTPUT / f"{arm}_s{seed}/terminal.json").read_text())
            for arm, seed in LANES
        ]
        summary = summarize(terminals)
        atomic_json(summary, OUTPUT / "summary.json")
    else:
        summary = {
            "schema": "transfer_dg_wfp_e1_summary_v1",
            "status": "failed",
            "decision": "failed_e1_stop_expansion",
            "return_codes": return_codes,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(summary, OUTPUT / "summary.json")
    terminal = {
        "schema": "transfer_dg_wfp_e1_training_supervisor_terminal_v1",
        "status": "complete" if all(code == 0 for code in return_codes) else "failed",
        "return_codes": return_codes,
        "summary": str(OUTPUT / "summary.json"),
        "summary_sha256": sha256(OUTPUT / "summary.json"),
        "elapsed_s": time.time() - training_started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(terminal, OUTPUT / "terminal.json")
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if terminal["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
