from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "saved_time_phase_operator_v4.postprocess_supervisor",
            "--gate",
            str(tmp_path / "gate.json"),
            "--run-terminal",
            str(tmp_path / "run_terminal.json"),
            "--sealed-report",
            str(tmp_path / "sealed_report.json"),
            "--figures-report",
            str(tmp_path / "figures_report.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_gate_waits_without_claiming_completion(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "wait_gate",
        "reason": "pilot gate is not available",
    }


def test_rejected_pilot_never_starts_long_run_evaluation(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": False}))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "pilot_rejected",
        "reason": "same-panel pilot gate did not pass",
    }


def test_passing_gate_waits_for_long_training_terminal(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "wait_run",
        "reason": "V66 long training is not complete",
    }


def test_failed_long_run_is_not_evaluated(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(
        json.dumps({"status": "failed", "reason": "cuda error"})
    )

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "run_failed",
        "reason": "V66 long training terminal is not complete",
    }


def test_complete_long_run_requests_sealed_evaluation_first(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(json.dumps({"status": "complete"}))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "evaluate_sealed",
        "reason": "full held-out evaluation is missing",
    }


def test_valid_sealed_evidence_requests_three_family_figures(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "sealed_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stored_times_only": True,
                "interpolated_targets": 0,
                "passes_accuracy_gate": False,
                "metrics": {
                    "record_count": 480,
                    "unique_time_index_count": 401,
                    "family_relative_l2": {
                        "uniform": 0.2,
                        "layered": 0.3,
                        "marmousi": 0.4,
                    },
                },
            }
        )
    )

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "render_figures",
        "reason": "three-family visual diagnostics are missing",
    }


def test_finished_figures_do_not_claim_success_when_accuracy_fails(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "sealed_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stored_times_only": True,
                "interpolated_targets": 0,
                "passes_accuracy_gate": False,
                "metrics": {
                    "record_count": 480,
                    "unique_time_index_count": 401,
                    "aggregate_relative_l2": 0.2,
                    "family_relative_l2": {
                        "uniform": 0.2,
                        "layered": 0.2,
                        "marmousi": 0.2,
                    },
                },
            }
        )
    )
    (tmp_path / "figures_report.json").write_text(
        json.dumps({"status": "complete", "families": {name: {} for name in ("uniform", "layered", "marmousi")}})
    )

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "continue_experiments",
        "reason": "sealed neural-operator accuracy target is not met",
    }


def test_complete_is_returned_only_for_verified_neural_operator_target(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "sealed_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stored_times_only": True,
                "interpolated_targets": 0,
                "passes_accuracy_gate": True,
                "metrics": {
                    "record_count": 480,
                    "unique_time_index_count": 401,
                    "aggregate_relative_l2": 0.099,
                    "family_relative_l2": {
                        "uniform": 0.11,
                        "layered": 0.10,
                        "marmousi": 0.119,
                    },
                },
            }
        )
    )
    (tmp_path / "figures_report.json").write_text(
        json.dumps({"status": "complete", "families": {name: {} for name in ("uniform", "layered", "marmousi")}})
    )

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": "complete",
        "reason": "sealed neural-operator accuracy and visual evidence are complete",
    }


def test_accuracy_boolean_cannot_override_failed_numeric_thresholds(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"passes": True}))
    (tmp_path / "run_terminal.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "sealed_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stored_times_only": True,
                "interpolated_targets": 0,
                "passes_accuracy_gate": True,
                "metrics": {
                    "record_count": 480,
                    "unique_time_index_count": 401,
                    "aggregate_relative_l2": 0.2,
                    "family_relative_l2": {
                        "uniform": 0.11,
                        "layered": 0.10,
                        "marmousi": 0.119,
                    },
                },
            }
        )
    )
    (tmp_path / "figures_report.json").write_text(
        json.dumps({"status": "complete", "families": {name: {} for name in ("uniform", "layered", "marmousi")}})
    )

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "continue_experiments"


def test_remote_launcher_uses_guard_before_any_gpu_evaluation() -> None:
    launcher = Path("scripts/run_remote_v66_postprocess.sh")

    assert launcher.is_file()
    text = launcher.read_text()
    guard = text.index("saved_time_phase_operator_v4.postprocess_supervisor")
    sealed = text.index("scripts/evaluate_saved_time_v4_full_support.py")
    figures = text.index("scripts/evaluate_saved_time_v62_three_family_figures.py")
    assert guard < sealed < figures
    assert '"wait_gate"|"wait_run"' in text
    assert '"continue_experiments"' in text
