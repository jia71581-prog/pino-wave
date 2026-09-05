import csv
import json
from pathlib import Path

import pytest

from scripts.build_tgrs_experiment_evidence_bundle import build


BUNDLE = Path(
    "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813"
)


def test_bundle_manifest_has_no_broken_targets():
    rows = list(csv.DictReader((BUNDLE / "FILE_MANIFEST.csv").open(encoding="utf8")))
    assert len(rows) >= 78
    assert all((BUNDLE / row["bundle_path"]).exists() for row in rows)


def test_r5b_train_gate_census_is_complete_and_labelled():
    path = BUNDLE / "05_relative_error/r5b_fixed_train_gate_all_48_records.csv"
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    assert len(rows) == 48
    assert {row["evidence_split"] for row in rows} == {"train"}
    assert {row["evidence_scope"] for row in rows} == {
        "fixed_48_record_train_gate"
    }


def test_cpadc_validation_and_test_id_censuses_are_complete_and_sealed():
    path = BUNDLE / "05_relative_error/cpadc_r7_all_960_records.csv"
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    assert len(rows) == 960
    assert sum(row["split"] == "validation" for row in rows) == 480
    assert sum(row["split"] == "test_id" for row in rows) == 480
    assert all(row["all_saved_time_indices"] == "401" for row in rows)
    assert all(row["prediction_sealed_before_truth"] == "True" for row in rows)


def test_position_protocol_is_fixed_frequency_and_position_only():
    protocol = json.loads(
        (BUNDLE / "07_protocols_and_figure_plan/fixed19hz_position_protocol.json")
        .read_text(encoding="utf8")
    )
    text = json.dumps(protocol, sort_keys=True)
    assert "19.0" in text
    assert "position" in text
    assert (
        protocol["generalization_scope"][
            "source_frequency_generalization_in_scope"
        ]
        is False
    )
    assert protocol["generalization_scope"]["frequency_sweep_permitted"] is False


def test_historical_builder_refuses_to_overwrite_promoted_bundle(tmp_path: Path):
    (tmp_path / "LATEST_MODEL_PROMOTION.json").write_text("{}", encoding="utf8")

    with pytest.raises(FileExistsError, match="promoted latest-model"):
        build(tmp_path)
