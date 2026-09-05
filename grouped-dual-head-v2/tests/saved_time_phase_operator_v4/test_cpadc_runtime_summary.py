from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_cpadc_vs_lwc84_runtime import build_report


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _traditional() -> dict[str, object]:
    return {
        "status": "complete",
        "schema": "lwc84_traditional_runtime_reference_v1",
        "mean_runtime_s": 12.0,
        "minimum_runtime_s": 10.0,
        "p50_runtime_s": 11.0,
        "p95_runtime_s": 14.0,
        "measurements": [{"runtime_s": 10.0}, {"runtime_s": 14.0}],
        "protocol": {
            "same_gpu_as_deployment": True,
            "cuda_synchronized_timing": True,
            "includes_output_materialization": True,
            "excludes_disk_io": True,
            "saved_frames": 401,
            "saved_grid": [201, 201],
            "solver_grid": [401, 401],
        },
        "device": {"name": "test GPU"},
    }


def _cpadc() -> dict[str, object]:
    return {
        "records": [
            {
                "adaptation": {
                    "adaptation_elapsed_s": 0.5,
                    "total_inference_elapsed_s": 4.0,
                    "future_truth_used": False,
                    "cuda_synchronized_timing": True,
                    "runtime_protocol": (
                        "input_ready_to_401_frame_output_materialized_no_disk_io_v1"
                    ),
                }
            },
            {
                "adaptation": {
                    "adaptation_elapsed_s": 1.0,
                    "total_inference_elapsed_s": 6.0,
                    "future_truth_used": False,
                    "cuda_synchronized_timing": True,
                    "runtime_protocol": (
                        "input_ready_to_401_frame_output_materialized_no_disk_io_v1"
                    ),
                }
            },
        ]
    }


def test_build_report_includes_adaptation_in_total(tmp_path: Path) -> None:
    validation = _write(tmp_path / "validation.json", _cpadc())
    test_id = _write(tmp_path / "test_id.json", _cpadc())
    traditional = _write(tmp_path / "traditional.json", _traditional())

    report = build_report(
        validation_path=validation,
        test_id_path=test_id,
        traditional_path=traditional,
    )

    split = report["splits"][0]
    assert split["instance_fine_tuning"]["mean_s"] == pytest.approx(0.75)
    assert split["end_to_end_fine_tuning_plus_inference"]["mean_s"] == pytest.approx(5.0)
    assert split["inference_and_output_materialization"]["mean_s"] == pytest.approx(4.25)
    assert report["method"]["name"].startswith("Ours:")
    assert split["speedup"]["ratio_of_mean_runtimes"] == pytest.approx(2.4)
    assert split["speedup"]["conservative_p95_using_fastest_traditional"] == pytest.approx(10.0 / 6.0)
    assert split["speedup"]["mean_speedup_if_adaptation_were_zero"] == pytest.approx(
        12.0 / 4.25
    )
    assert split["speedup"]["tenfold_latency_budget_using_fastest_traditional_s"] == 1.0
    assert report["claim_gate"]["cpadc_archived_timing_metadata_complete"] is True
    assert report["claim_gate"]["speed_superiority_claim_allowed"] is False


def test_build_report_rejects_traditional_timing_without_materialization(
    tmp_path: Path,
) -> None:
    validation = _write(tmp_path / "validation.json", _cpadc())
    test_id = _write(tmp_path / "test_id.json", _cpadc())
    traditional_payload = _traditional()
    traditional_payload["protocol"]["includes_output_materialization"] = False
    traditional = _write(tmp_path / "traditional.json", traditional_payload)

    with pytest.raises(ValueError, match="includes_output_materialization"):
        build_report(
            validation_path=validation,
            test_id_path=test_id,
            traditional_path=traditional,
        )
