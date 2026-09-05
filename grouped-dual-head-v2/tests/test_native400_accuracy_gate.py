from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from fno_acoustic.native400_gate import (
    CATEGORIES,
    METRICS,
    NativeGateCell,
    NativeGateResult,
    aggregate_candidate_rows_by_sample,
    aggregate_seed_gates,
    evaluate_native400_gate,
    native_stability_guards,
)
from fno_acoustic.query_census import SAMPLE_COLUMNS


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_native400_accuracy.py"


def _row(sample_id: int, category: str, scale: float = 1.0, *, seed: int = 1) -> dict:
    row = {name: 0.0 for name in SAMPLE_COLUMNS}
    row.update(
        sample_id=sample_id,
        category=category,
        split="val",
        seed=seed,
        predictor_family="candidate",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        normalization_sha256="d" * 64,
        receiver_geometry_sha256="e" * 64,
        relative_l2=1.0 * scale,
        relative_l2_q1=1.0 * scale,
        relative_l2_q2=1.0 * scale,
        relative_l2_q3=1.0 * scale,
        relative_l2_q4=1.0 * scale,
        receiver_relative_l2=1.2 * scale,
        receiver_relative_l2_q1=1.2 * scale,
        receiver_relative_l2_q2=1.2 * scale,
        receiver_relative_l2_q3=1.2 * scale,
        receiver_relative_l2_q4=1.2 * scale,
        field_q4_q1_ratio=1.0,
        receiver_q4_q1_ratio=1.0,
        active_time_coverage=1.0,
        active_time_error_slope_per_s=0.1,
        active_time_error_max=1.0,
        arrival_mae_s=0.01,
        arrival_miss_rate=0.1,
        arrival_target_coverage=1.0,
        receiver_lag_abs_s=0.01,
        receiver_xcorr_peak=0.9,
        receiver_phase_error=0.1,
        receiver_phase_coherence=0.9,
        energy_log_ratio=0.1,
        komega_relative_l2=1.0,
        komega_relative_l2_q4=1.0,
        komega_high=1.0,
        komega_high_q4=1.0,
        zero_relative_l2=2.0,
        zero_relative_l2_q4=2.0,
        zero_receiver_relative_l2=2.0,
        zero_receiver_relative_l2_q4=2.0,
        prediction_target_norm_ratio=1.0,
        prediction_target_pearson=0.9,
        sampler_ess=160000.0,
        sampler_duplicate_fraction=0.0,
        sampler_coverage=1.0,
        sampler_max_median_inverse_weight=1.0,
        prediction_finite=1.0,
        prediction_nonzero=1.0,
        output_height=400.0,
        output_width=400.0,
        output_time_steps=160.0,
    )
    return row


def paired_rows(improvement: float = 0.35, count: int = 12):
    baseline, candidate = [], []
    for group, category in enumerate(CATEGORIES):
        for i in range(count):
            sample_id = group * 1000 + i
            baseline.append(_row(sample_id, category, 1.0 + i / 100.0))
            candidate.append(_row(sample_id, category, (1.0 + i / 100.0) * (1.0 - improvement)))
    return baseline, candidate


def _gate(passed: bool) -> NativeGateResult:
    cell = NativeGateCell("uniform", "relative_l2", 1.0, 0.6, 0.4, 0.35, 0.45, passed)
    return NativeGateResult((cell,), {"stable": passed}, passed, "confirmed" if passed else "failed")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _seed_run(rows: list[dict], seed: int) -> list[dict]:
    return [
        dict(
            row,
            seed=seed,
            config_sha256=f"{seed + 1:x}" * 64,
            checkpoint_sha256=f"{seed + 4:x}" * 64,
        )
        for row in rows
    ]


def test_gate_requires_all_twelve_cells_and_ci_lower_bound_at_thirty_percent():
    baseline, candidate = paired_rows(0.35, 20)
    result = evaluate_native400_gate(baseline, candidate, 0.30, 400, 31)
    assert len(result.cells) == 12
    assert result.passed and result.classification == "confirmed"
    assert min(cell.ci_lower for cell in result.cells) >= 0.30


def test_point_improvement_without_ci_support_is_exploratory():
    baseline, candidate = paired_rows(0.35, 20)
    candidate[0]["field_q4_q1_ratio"] = 1.6
    result = evaluate_native400_gate(baseline, candidate, 0.30, 300, 37)
    assert all(cell.improvement >= 0.30 for cell in result.cells)
    assert not result.passed and result.classification == "exploratory"


def test_three_seed_gate_requires_exactly_two_seed_passes_and_aggregate_pass():
    result = aggregate_seed_gates([_gate(True), _gate(True), _gate(False)], _gate(True))
    assert result.passed and result.passed_seed_count == 2
    assert not aggregate_seed_gates([_gate(True), _gate(True), _gate(False)], _gate(False)).passed
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_seed_gates([_gate(True), _gate(True)], _gate(True))


def test_seed_aggregation_preserves_physical_sample_as_statistical_unit():
    _, base = paired_rows(0.1, 2)
    seeds = []
    for seed, factor in enumerate((1.0, 0.7, 0.4), 1):
        rows = []
        for source in _seed_run(base, seed):
            row = dict(source)
            for metric in METRICS:
                row[metric] *= factor
            rows.append(row)
        seeds.append(rows)
    aggregated = aggregate_candidate_rows_by_sample(seeds)
    assert len(aggregated) == len(seeds[0])
    assert len({(row["category"], row["sample_id"]) for row in aggregated}) == len(aggregated)
    assert aggregated[0]["relative_l2"] == pytest.approx(
        statistics.median(rows[0]["relative_l2"] for rows in seeds)
    )


def test_seed_aggregation_accepts_only_the_three_signed_task9_numeric_columns():
    _, rows = paired_rows(0.35, 2)
    seeds = []
    for seed, signed in enumerate((-0.8, -0.4, -0.2), 1):
        seeds.append([
            dict(
                row,
                active_time_error_slope_per_s=signed,
                receiver_xcorr_peak=signed,
                receiver_phase_coherence=signed,
            )
            for row in _seed_run(rows, seed)
        ])
    aggregated = aggregate_candidate_rows_by_sample(seeds)
    assert aggregated[0]["active_time_error_slope_per_s"] == pytest.approx(-0.4)
    assert aggregated[0]["receiver_xcorr_peak"] == pytest.approx(-0.4)
    assert aggregated[0]["receiver_phase_coherence"] == pytest.approx(-0.4)
    seeds[0][0]["energy_log_ratio"] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        aggregate_candidate_rows_by_sample(seeds)


def test_order_does_not_change_reproducible_paired_bootstrap():
    baseline, candidate = paired_rows(0.35, 8)
    expected = evaluate_native400_gate(baseline, candidate, 0.30, 250, 17)
    actual = evaluate_native400_gate(list(reversed(baseline)), candidate[::2] + candidate[1::2], 0.30, 250, 17)
    assert actual == expected


def test_join_is_paired_and_rejects_unpaired_duplicate_and_extra_categories():
    baseline, candidate = paired_rows()
    with pytest.raises(ValueError, match="identical"):
        evaluate_native400_gate(baseline, candidate[:-1], 0.30, 10, 1)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_native400_gate(baseline + [dict(baseline[0])], candidate, 0.30, 10, 1)
    extra_base = baseline + [_row(9999, "extra")]
    extra_cand = candidate + [_row(9999, "extra", 0.6)]
    with pytest.raises(ValueError, match="categories"):
        evaluate_native400_gate(extra_base, extra_cand, 0.30, 10, 1)


def test_each_category_requires_two_unique_physical_samples():
    baseline, candidate = paired_rows(count=1)
    with pytest.raises(ValueError, match="at least two"):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


def test_every_baseline_primary_metric_must_be_strictly_positive():
    baseline, candidate = paired_rows()
    baseline[0]["relative_l2"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


def test_finite_primary_values_that_overflow_the_mean_are_rejected():
    baseline, candidate = paired_rows()
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    for row in baseline:
        row["relative_l2"] = maximum
    with pytest.raises(ValueError, match="means must be finite"):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


@pytest.mark.parametrize(
    "field",
    ["seed", "predictor_family", "config_sha256", "checkpoint_sha256", "split",
     "split_manifest_sha256", "normalization_sha256", "receiver_geometry_sha256"],
)
def test_each_csv_rejects_mixed_run_metadata(field):
    baseline, candidate = paired_rows()
    candidate[0][field] = "different"
    with pytest.raises(ValueError, match="internally consistent"):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


@pytest.mark.parametrize(
    "field",
    ["split", "split_manifest_sha256", "normalization_sha256", "receiver_geometry_sha256"],
)
def test_baseline_candidate_reject_shared_metadata_mismatch(field):
    baseline, candidate = paired_rows()
    for row in candidate:
        row[field] = "b" * 64 if field != "split" else "test"
    with pytest.raises(ValueError, match="must match"):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), -1.0])
def test_primary_baseline_and_metric_validation(value):
    baseline, candidate = paired_rows()
    if value == 0.0:
        for row in baseline:
            if row["category"] == "uniform":
                row["relative_l2"] = value
    else:
        baseline[0]["relative_l2"] = value
    with pytest.raises(ValueError):
        evaluate_native400_gate(baseline, candidate, 0.30, 10, 1)


@pytest.mark.parametrize(
    "threshold,reps,seed",
    [(-0.1, 10, 1), (1.1, 10, 1), (True, 10, 1), (0.3, 0, 1), (0.3, True, 1), (0.3, 10, True)],
)
def test_gate_configuration_is_strict(threshold, reps, seed):
    baseline, candidate = paired_rows()
    with pytest.raises(ValueError):
        evaluate_native400_gate(baseline, candidate, threshold, reps, seed)


@pytest.mark.parametrize(
    "column,bad_value,guard_fragment",
    [
        ("field_q4_q1_ratio", 1.6, "field_q4_q1"),
        ("receiver_q4_q1_ratio", 1.6, "receiver_q4_q1"),
        ("zero_relative_l2_q4", 0.1, "better_than_zero_q4"),
        ("arrival_miss_rate", 0.2, "arrival_miss"),
        ("receiver_lag_abs_s", 0.02, "lag"),
        ("receiver_phase_error", 0.2, "phase"),
        ("komega_high_q4", 1.2, "high_k"),
        ("prediction_finite", 0.0, "finite_nonzero"),
        ("prediction_nonzero", 0.0, "finite_nonzero"),
        ("output_height", 399.0, "native_shape"),
    ],
)
def test_each_stability_guard_can_fail(column, bad_value, guard_fragment):
    baseline, candidate = paired_rows()
    for row in candidate:
        if row["category"] == "uniform":
            row[column] = bad_value
    guards = native_stability_guards(baseline, candidate)
    assert guards[f"uniform:{guard_fragment}"] is False


def test_missing_stability_column_and_non_numeric_boolean_are_rejected():
    baseline, candidate = paired_rows()
    del candidate[0]["receiver_phase_error"]
    with pytest.raises(ValueError, match="incomplete"):
        native_stability_guards(baseline, candidate)
    baseline, candidate = paired_rows()
    candidate[0]["prediction_finite"] = "0"
    with pytest.raises(ValueError, match="boolean"):
        native_stability_guards(baseline, candidate)


def test_ci_boundary_at_exact_threshold_passes():
    baseline, candidate = paired_rows(0.30, 8)
    result = evaluate_native400_gate(baseline, candidate, 0.30, 100, 3)
    assert all(cell.passed for cell in result.cells)


def test_aggregate_rejects_different_keys_duplicates_and_wrong_reducer():
    _, rows = paired_rows()
    with pytest.raises(ValueError, match="median"):
        aggregate_candidate_rows_by_sample([rows, rows, rows], reducer="mean")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_candidate_rows_by_sample([_seed_run(rows, 1) + [dict(_seed_run(rows, 1)[0])],
                                            _seed_run(rows, 2), _seed_run(rows, 3)])
    with pytest.raises(ValueError, match="identical"):
        aggregate_candidate_rows_by_sample([_seed_run(rows, 1), _seed_run(rows, 2), _seed_run(rows, 3)[:-1]])


def test_aggregate_rejects_repeated_seed_checkpoint_and_candidate_family():
    _, rows = paired_rows()
    seeds = [_seed_run(rows, seed) for seed in (1, 2, 3)]
    repeated_seed = [[dict(row, seed=1) for row in seed_rows] for seed_rows in seeds]
    with pytest.raises(ValueError, match="unique seeds"):
        aggregate_candidate_rows_by_sample(repeated_seed)
    repeated_checkpoint = [
        [dict(row, checkpoint_sha256="d" * 64) for row in seed_rows] for seed_rows in seeds
    ]
    with pytest.raises(ValueError, match="unique checkpoints"):
        aggregate_candidate_rows_by_sample(repeated_checkpoint)
    different_family = [*seeds[:2], [dict(row, predictor_family="other") for row in seeds[2]]]
    with pytest.raises(ValueError, match="predictor_family"):
        aggregate_candidate_rows_by_sample(different_family)


def test_candidate_seed_tokens_cannot_fake_three_independent_seeds():
    _, rows = paired_rows()
    seeds = [_seed_run(rows, seed) for seed in (1, 2, 3)]
    disguised = [
        [dict(row, seed=token) for row in seed_rows]
        for token, seed_rows in zip(("1", "01", "+1"), seeds, strict=True)
    ]
    with pytest.raises(ValueError, match="canonical nonnegative decimal"):
        aggregate_candidate_rows_by_sample(disguised)


def test_candidate_accepts_canonical_large_decimal_seed_and_baseline_label():
    baseline, candidate = paired_rows()
    baseline = [dict(row, seed="b0") for row in baseline]
    candidate = [dict(row, seed="20260714") for row in candidate]
    assert evaluate_native400_gate(baseline, candidate, 0.30, 10, 1).passed


def test_aggregate_sets_canonical_provenance_instead_of_copying_first_seed():
    _, rows = paired_rows(count=2)
    seeds = [_seed_run(rows, seed) for seed in (1, 2, 3)]
    aggregated = aggregate_candidate_rows_by_sample(seeds)
    expected_checkpoint = hashlib.sha256(
        "\n".join(sorted(seed[0]["checkpoint_sha256"] for seed in seeds)).encode()
    ).hexdigest()
    assert aggregated[0]["seed"] == "median_of_three"
    assert aggregated[0]["checkpoint_sha256"] == expected_checkpoint
    assert aggregated[0]["checkpoint_sha256"] != seeds[0][0]["checkpoint_sha256"]


def test_cli_single_seed_pass_and_fail_exit_and_payload(tmp_path: Path):
    baseline, passing = paired_rows(0.35, 8)
    _, failing = paired_rows(0.20, 8)
    baseline_csv, pass_csv, fail_csv = tmp_path / "b.csv", tmp_path / "p.csv", tmp_path / "f.csv"
    _write_csv(baseline_csv, baseline)
    _write_csv(pass_csv, passing)
    _write_csv(fail_csv, failing)
    for candidate_csv, code in ((pass_csv, 0), (fail_csv, 2)):
        output = tmp_path / f"result-{code}.json"
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
             "--candidate-csv", str(candidate_csv), "--output-json", str(output),
             "--bootstrap-replicates", "50", "--seed", "11"],
            cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == code, completed.stderr
        payload = json.loads(output.read_text())
        assert len(payload["cells"]) == 12
        assert payload["threshold"] == 0.30 and payload["seed"] == 11
        assert payload["bootstrap_replicates"] == 50
        assert set(payload["input_sha256"]) == {"baseline", "candidates"}


def test_cli_three_seed_aggregate_and_invalid_combinations(tmp_path: Path):
    baseline, candidate = paired_rows(0.35, 6)
    baseline_csv = tmp_path / "b.csv"
    _write_csv(baseline_csv, baseline)
    candidates = []
    for seed in range(3):
        path = tmp_path / f"c{seed}.csv"
        _write_csv(path, _seed_run(candidate, seed))
        candidates.append(path)
    output = tmp_path / "aggregate.json"
    command = [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv)]
    for path in candidates:
        command += ["--candidate-csv", str(path)]
    completed = subprocess.run(command + ["--aggregate-seeds", "--output-json", str(output),
                                "--bootstrap-replicates", "30"],
                               cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["passed_seed_count"] == 3 and payload["aggregate"]["passed"] is True
    invalid_output = tmp_path / "invalid.json"
    invalid = subprocess.run(command[:8] + ["--output-json", str(invalid_output)],
                             cwd=tmp_path, capture_output=True, text=True, check=False)
    assert invalid.returncode != 0 and not invalid_output.exists()


def test_cli_formal_output_dir_accepts_signed_columns_and_publishes_fixed_gate_json(tmp_path: Path):
    baseline, candidate = paired_rows(0.35, 6)
    for rows in (baseline, candidate):
        for row in rows:
            row["active_time_error_slope_per_s"] = -0.2
            row["receiver_xcorr_peak"] = -0.3
            row["receiver_phase_coherence"] = -0.4
    baseline_csv, candidate_csv = tmp_path / "b.csv", tmp_path / "c.csv"
    _write_csv(baseline_csv, baseline)
    _write_csv(candidate_csv, candidate)
    output_dir = tmp_path / "nested" / "formal-gate"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
         "--candidate-csv", str(candidate_csv), "--output-dir", str(output_dir),
         "--bootstrap-replicates", "30"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "gate.json").is_file()
    assert not list(output_dir.parent.glob(f".{output_dir.name}.*.tmp"))


def test_cli_formal_output_dir_publishes_failed_gate_and_rejects_existing_or_both_outputs(tmp_path: Path):
    baseline, candidate = paired_rows(0.20, 4)
    baseline_csv, candidate_csv = tmp_path / "b.csv", tmp_path / "c.csv"
    _write_csv(baseline_csv, baseline)
    _write_csv(candidate_csv, candidate)
    output_dir = tmp_path / "gate"
    command = [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
               "--candidate-csv", str(candidate_csv), "--output-dir", str(output_dir),
               "--bootstrap-replicates", "20"]
    failed_gate = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert failed_gate.returncode == 2
    assert json.loads((output_dir / "gate.json").read_text())["classification"] == "failed"
    existing = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert existing.returncode != 0
    both = subprocess.run(command + ["--output-json", str(tmp_path / "also.json")], cwd=tmp_path,
                          capture_output=True, text=True, check=False)
    assert both.returncode != 0 and not (tmp_path / "also.json").exists()


def test_cli_formal_three_seed_output_dir_and_invalid_input_leave_no_directory(tmp_path: Path):
    baseline, candidate = paired_rows(0.35, 4)
    baseline_csv = tmp_path / "b.csv"
    _write_csv(baseline_csv, baseline)
    candidates = []
    for seed in range(3):
        path = tmp_path / f"c{seed}.csv"
        _write_csv(path, _seed_run(candidate, seed))
        candidates.append(path)
    output_dir = tmp_path / "aggregate"
    command = [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv)]
    for path in candidates:
        command += ["--candidate-csv", str(path)]
    completed = subprocess.run(command + ["--aggregate-seeds", "--output-dir", str(output_dir),
                                "--bootstrap-replicates", "20"], cwd=tmp_path,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((output_dir / "gate.json").read_text())["passed_seed_count"] == 3
    bad_output = tmp_path / "bad"
    candidates[0].write_text("bad\n", encoding="utf-8")
    invalid = subprocess.run(command + ["--aggregate-seeds", "--output-dir", str(bad_output)],
                             cwd=tmp_path, capture_output=True, text=True, check=False)
    assert invalid.returncode != 0 and not bad_output.exists()
    assert not list(tmp_path.glob(".bad.*.tmp"))


def test_cli_rejects_same_candidate_file_repeated_three_times(tmp_path: Path):
    baseline, candidate = paired_rows(0.35, 4)
    baseline_csv, candidate_csv = tmp_path / "b.csv", tmp_path / "c.csv"
    _write_csv(baseline_csv, baseline)
    _write_csv(candidate_csv, _seed_run(candidate, 1))
    output_dir = tmp_path / "gate"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
         "--candidate-csv", str(candidate_csv), "--candidate-csv", str(candidate_csv),
         "--candidate-csv", str(candidate_csv), "--aggregate-seeds", "--output-dir", str(output_dir)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0 and not output_dir.exists()
    assert "unique" in completed.stderr


def test_csv_snapshot_is_read_once_and_hashes_the_exact_parsed_bytes(tmp_path: Path, monkeypatch):
    _, rows = paired_rows(0.35, 2)
    csv_path = tmp_path / "candidate.csv"
    _write_csv(csv_path, rows)
    original = csv_path.read_bytes()
    changed = original.replace(b"0.65", b"0.66")
    calls = 0
    real_read_bytes = Path.read_bytes

    def changing_read_bytes(path):
        nonlocal calls
        if path == csv_path:
            calls += 1
            return original if calls == 1 else changed
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    spec = importlib.util.spec_from_file_location("compare_native400_accuracy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot_rows, snapshot_hash = module.read_native400_snapshot(csv_path)
    assert calls == 1
    assert snapshot_hash == hashlib.sha256(original).hexdigest()
    assert snapshot_rows[0]["relative_l2"] == rows[0]["relative_l2"]


def test_cli_numeric_zero_boolean_fails_gate_without_becoming_truthy(tmp_path: Path):
    baseline, candidate = paired_rows()
    baseline_csv, candidate_csv = tmp_path / "b.csv", tmp_path / "c.csv"
    _write_csv(baseline_csv, baseline)
    candidate[0]["prediction_finite"] = "0"
    _write_csv(candidate_csv, candidate)
    output = tmp_path / "nested" / "gate.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
         "--candidate-csv", str(candidate_csv), "--output-json", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2 and output.exists()
    payload = json.loads(output.read_text())
    assert payload["stability"]["uniform:finite_nonzero"] is False
    assert not list(tmp_path.rglob("*.tmp"))


def test_cli_invalid_csv_does_not_publish_or_leave_temp(tmp_path: Path):
    baseline, candidate = paired_rows()
    baseline_csv, candidate_csv = tmp_path / "b.csv", tmp_path / "c.csv"
    _write_csv(baseline_csv, baseline)
    candidate[0]["relative_l2"] = "nan"
    _write_csv(candidate_csv, candidate)
    output = tmp_path / "nested" / "gate.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline-csv", str(baseline_csv),
         "--candidate-csv", str(candidate_csv), "--output-json", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0 and not output.exists()
    assert not list(tmp_path.rglob("*.tmp"))
