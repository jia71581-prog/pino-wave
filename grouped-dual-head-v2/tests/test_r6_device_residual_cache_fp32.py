"""CPU tests for the FP32 single-variable R6 residual-cache control.

Covers the fp32 storage contract (only coarse_norm/truth_norm change dtype),
the fp32 quantization diagnostic, prior-f16 parity checks, invocation and
static binding verification, and real R25 consumers (CacheCollection,
FitFrameDataset, train_loss, evaluate) reading fp32 caches that hold
non-float16-representable data, compared against independent native float32
in-memory baselines with preregistered tolerances. No GPU, no source truth.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

from scripts.build_r6_device_residual_cache import (
    CACHE_SCHEMA, FAMILIES, dataset_contract, role_time_indices, verify_stage_inputs,
)
from scripts.reattest_frozen_fine_grid_r6_train import dependency_closure, sha256
from scripts.smoke_r6_device_residual_cache_fp32 import (
    CANDIDATE, PRIOR_TERMINAL_SHA256, SOURCE_INDICES, STAGE, create_datasets,
    fp32_dataset_contract, prior_record_parity, quantization_diagnostic_fp32,
    selected_records, validate_cache_schema, validate_invocation,
)
from scripts.train_r25_coarse_residual_operator import (
    CacheCollection, CoarseResidualUNet, FitFrameDataset, evaluate, train_loss,
)


ROOT = Path(__file__).resolve().parents[1]
PREREG = STAGE / "preregistration.json"


def _prereg() -> dict:
    return json.loads(PREREG.read_text())


def _tolerances() -> dict:
    return _prereg()["consumer_tests"]


# ---------------------------------------------------------------- contract


def test_fp32_contract_changes_only_two_wavefield_dtypes() -> None:
    for role, count in (("fit", 3), ("development", 3)):
        base = dataset_contract(role, count)
        fp32 = fp32_dataset_contract(role, count)
        assert set(base) == set(fp32)
        for name in ("coarse_norm", "truth_norm"):
            assert base[name]["dtype"] == "float16" and fp32[name]["dtype"] == "float32"
            for key in ("shape", "chunks", "compression", "shuffle"):
                assert fp32[name][key] == base[name][key]
        assert fp32["static_features"]["dtype"] == "float16"
        for name in set(base) - {"coarse_norm", "truth_norm"}:
            assert fp32[name] == base[name]


def test_frozen_six_record_panel_constants() -> None:
    assert SOURCE_INDICES == {"fit": [0, 420, 2100], "development": [357, 968, 2375]}
    assert CANDIDATE == "r6_device_residual_cache_fp32_v1_20260905"


# ---------------------------------------------------- synthetic fp32 caches


def _build_fp32_cache(path: Path, role: str, count: int, seed: int) -> dict:
    """Create a schema-exact fp32 cache via the real create_datasets writer."""
    indices = role_time_indices(role)
    frames = len(indices)
    rng = np.random.default_rng(seed)
    time_s = np.linspace(0.0, 0.5, 401, dtype=np.float64)
    records = [{"sample_id": f"synthetic_{role}_{index:05d}", "group_id": f"synthetic:{role}:{index:05d}",
                "family": FAMILIES[index % 3], "sample_sha256": f"{index:064x}",
                "source_index": index, "split": "train"} for index in range(count)]
    coarse = rng.standard_normal((count, frames, 201, 201)).astype(np.float32)
    residual = np.float32(1e-3) * rng.standard_normal((count, frames, 201, 201)).astype(np.float32)
    truth = (coarse + residual).astype(np.float32)
    truth[:, 0] = 0.0
    static = rng.standard_normal((count, 7, 201, 201)).astype(np.float32)
    t0 = 0.05
    with h5py.File(path, "x") as cache:
        cache.attrs.update({"schema": CACHE_SCHEMA, "status": "complete", "subset": role, "role": role,
                            "parent_kind": "R6_device_resident_dt625us_restricted2x", "shard_index": 0, "shard_count": 4,
                            "parent_internal_dt_s": .000625, "parent_output_restriction_factor": 2,
                            "truth_policy": "train_only_supervision_not_deployment_input",
                            "selection_sha256": "5005f614d1ec7cbd77ddad440ac6132d0b79b5953aea70c566a8713d1433b937"})
        datasets = create_datasets(cache, records, role, time_s)
        validate_cache_schema(cache, role, count)
        for position in range(count):
            datasets["parent_full_sha256"][position] = "b" * 64
            datasets["field_scale"][position] = 1.0
            datasets["source_f0_hz"][position] = 12.0
            datasets["source_t0_s"][position] = t0
            datasets["coarse_norm"][position] = coarse[position]
            datasets["truth_norm"][position] = truth[position]
            datasets["static_features"][position] = static[position].astype(np.float16)
            energies = np.square(truth[position].astype(np.float64)).mean(axis=(1, 2))
            datasets["truth_frame_energy_max_norm"][position] = energies.max()
            datasets["truth_frame_energy_mean_norm"][position] = energies.mean()
            difference = coarse[position].astype(np.float64) - truth[position].astype(np.float64)
            datasets["baseline_error_square_norm"][position] = np.square(difference).sum()
            datasets["target_square_norm"][position] = np.square(truth[position].astype(np.float64)).sum()
    return {"path": path, "records": records, "coarse": coarse, "truth": truth,
            "static": static, "time_s": time_s[list(indices)], "t0": t0}


@pytest.fixture(scope="module")
def fit_cache(tmp_path_factory) -> dict:
    return _build_fp32_cache(tmp_path_factory.mktemp("fp32_fit") / "fit.h5", "fit", 2, 20260905)


@pytest.fixture(scope="module")
def development_cache(tmp_path_factory) -> dict:
    return _build_fp32_cache(tmp_path_factory.mktemp("fp32_dev") / "development.h5", "development", 1, 20260906)


def test_synthetic_fields_are_not_float16_representable(fit_cache) -> None:
    coarse = fit_cache["coarse"]; truth = fit_cache["truth"]
    assert (coarse.astype(np.float16).astype(np.float32) != coarse).any()
    assert (truth.astype(np.float16).astype(np.float32) != truth).any()
    native = truth[:, 1:].astype(np.float64) - coarse[:, 1:].astype(np.float64)
    f16 = (truth[:, 1:].astype(np.float16).astype(np.float64)
           - coarse[:, 1:].astype(np.float16).astype(np.float64))
    destroyed = np.linalg.norm(f16 - native) / np.linalg.norm(native)
    assert destroyed > 0.25  # f16 storage destroys this residual; fp32 must not


# ------------------------------------------------------------ R25 consumers


def test_cache_collection_reads_fp32_fit_cache_bit_exact(fit_cache) -> None:
    collection = CacheCollection([fit_cache["path"]], expected_subset="fit")
    try:
        assert collection.time_count == 64 and len(collection.records) == 2
        handle = collection.handles[0]
        assert str(handle["coarse_norm"].dtype) == "float32"
        for position in range(2):
            assert np.array_equal(handle["coarse_norm"][position], fit_cache["coarse"][position])
            assert np.array_equal(handle["truth_norm"][position], fit_cache["truth"][position])
    finally:
        collection.close()


def test_fit_frame_dataset_matches_native_float32_memory_baseline(fit_cache) -> None:
    collection = CacheCollection([fit_cache["path"]], expected_subset="fit")
    try:
        dataset = FitFrameDataset(collection)
        assert len(dataset) == 2 * 64
        for index in (0, 5, 63, 64, 100, 127):
            record, frame = divmod(index, 64)
            features, coarse, truth, mean_energy, active = dataset[index]
            native_coarse = torch.from_numpy(fit_cache["coarse"][record, frame])
            native_truth = torch.from_numpy(fit_cache["truth"][record, frame])
            assert torch.equal(coarse, native_coarse) and torch.equal(truth, native_truth)
            f16_coarse = torch.from_numpy(
                fit_cache["coarse"][record, frame].astype(np.float16).astype(np.float32))
            assert not torch.equal(coarse, f16_coarse)
            assert features.shape == (13, 201, 201) and torch.equal(features[0], coarse)
            static_f16 = torch.from_numpy(
                fit_cache["static"][record].astype(np.float16).astype(np.float32))
            assert torch.equal(features[1:8], static_f16)
            time_value = float(fit_cache["time_s"][frame])
            assert float(active) == float(time_value >= fit_cache["t0"])
            expected_energy = float(np.square(
                fit_cache["truth"][record].astype(np.float64)).mean(axis=(1, 2)).mean())
            assert float(mean_energy) == pytest.approx(expected_energy, rel=1e-6)
    finally:
        collection.close()


def test_development_subset_requires_explicit_wiring(development_cache) -> None:
    collection = CacheCollection([development_cache["path"]], expected_subset="development")
    try:
        assert collection.time_count == 401
        with pytest.raises(ValueError):
            FitFrameDataset(collection)
    finally:
        collection.close()
    with pytest.raises(RuntimeError, match="cache subset mismatch"):
        CacheCollection([development_cache["path"]], expected_subset="holdout")


def test_train_loss_nonzero_correction_gradient_and_independent_value(fit_cache) -> None:
    tolerances = _tolerances()
    collection = CacheCollection([fit_cache["path"]], expected_subset="fit")
    try:
        dataset = FitFrameDataset(collection)
        # post-onset frames only, so the active mask keeps the correction nonzero
        features, coarse, truth, mean_energy, active = default_collate(
            [dataset[index] for index in (10, 20, 40, 72)])
    finally:
        collection.close()
    assert float(active.min()) == 1.0
    alpha = nn.Parameter(torch.tensor(float(tolerances["constant_correction_value"])))
    correction = alpha * torch.ones_like(coarse) * active[:, None, None]
    loss, components = train_loss(correction, coarse, truth, mean_energy,
                                  hinge_weight=0.5, gradient_weight=0.03)
    assert torch.isfinite(loss) and float(components["correction_energy"]) > 0
    loss.backward()
    assert alpha.grad is not None and abs(float(alpha.grad)) > float(tolerances["gradient_abs_min"])

    value = np.float64(np.float32(tolerances["constant_correction_value"]))
    corr = value * active.numpy().astype(np.float64)[:, None, None] * np.ones((1, 201, 201))
    c64 = coarse.numpy().astype(np.float64); t64 = truth.numpy().astype(np.float64)
    prediction = c64 + corr
    denominator = np.maximum(mean_energy.numpy().astype(np.float64), 1e-7)
    candidate = ((prediction - t64) ** 2).mean(axis=(1, 2)) / denominator
    parent = ((c64 - t64) ** 2).mean(axis=(1, 2)) / denominator
    hinge = np.maximum(np.sqrt(np.maximum(candidate, 1e-12)) - np.sqrt(np.maximum(parent, 1e-12)), 0.0) ** 2
    gx = ((np.diff(prediction, axis=2) - np.diff(t64, axis=2)) ** 2).mean(axis=(1, 2))
    gz = ((np.diff(prediction, axis=1) - np.diff(t64, axis=1)) ** 2).mean(axis=(1, 2))
    gradient = ((gx + gz) / denominator).mean()
    expected = (candidate.mean() + 0.5 * hinge.mean() + 0.03 * gradient + 1e-5 * (corr ** 2).mean())
    assert float(loss) == pytest.approx(float(expected), rel=float(tolerances["train_loss_relative_tolerance"]))


class _ConstantCorrectionModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, features: torch.Tensor, *, active: torch.Tensor | None = None) -> torch.Tensor:
        correction = self.value.expand(features.shape[0], features.shape[2], features.shape[3])
        if active is not None:
            correction = correction * active[:, None, None]
        return correction


def test_evaluate_metrics_match_independent_native_baseline(development_cache) -> None:
    tolerances = _tolerances()
    value = float(tolerances["constant_correction_value"])
    relative = float(tolerances["evaluate_relative_tolerance"])
    collection = CacheCollection([development_cache["path"]], expected_subset="development")
    try:
        model = _ConstantCorrectionModel(value)
        report = evaluate(model, collection, device=torch.device("cpu"), batch_size=64, amp=False)
    finally:
        collection.close()
    assert len(report["records"]) == 1
    row = report["records"][0]

    coarse = development_cache["coarse"][0]; truth = development_cache["truth"][0]
    active = (development_cache["time_s"] >= development_cache["t0"]).astype(np.float32)
    corr = (np.float32(value) * active)[:, None, None] * np.ones((1, 201, 201), np.float32)
    prediction = coarse + corr.astype(np.float32)
    candidate_error = np.square(prediction.astype(np.float64) - truth.astype(np.float64)).sum()
    parent_error = np.square(coarse.astype(np.float64) - truth.astype(np.float64)).sum()
    target_square = np.square(truth.astype(np.float64)).sum()
    correction_square = np.square(corr.astype(np.float64)).sum()
    assert row["candidate_rel_l2"] == pytest.approx(math.sqrt(candidate_error / target_square), rel=relative)
    assert row["parent_rel_l2"] == pytest.approx(math.sqrt(parent_error / target_square), rel=relative)
    assert row["candidate_error_square"] == pytest.approx(candidate_error, rel=relative)
    assert row["parent_error_square"] == pytest.approx(parent_error, rel=relative)
    assert row["target_square"] == pytest.approx(target_square, rel=relative)
    assert row["correction_square"] == pytest.approx(correction_square, rel=relative)
    assert row["correction_square"] > 0
    aggregate = report["aggregate"]
    assert aggregate["parent_mean"] == pytest.approx(math.sqrt(parent_error / target_square), rel=relative)


def test_real_unet_consumes_fp32_features_with_nonzero_gradients(fit_cache) -> None:
    tolerances = _tolerances()
    probe = tolerances["unet_probe"]
    collection = CacheCollection([fit_cache["path"]], expected_subset="fit")
    try:
        dataset = FitFrameDataset(collection)
        # post-onset frames only, so the active mask keeps the correction nonzero
        features, coarse, truth, mean_energy, active = default_collate(
            [dataset[index] for index in (12, 76)])
    finally:
        collection.close()
    assert float(active.min()) == 1.0
    torch.manual_seed(int(probe["seed"]))
    model = CoarseResidualUNet(base_width=int(probe["base_width"]), correction_cap=0.5)
    nn.init.normal_(model.output.weight, std=float(probe["output_reinit_std"]))
    nn.init.constant_(model.output.bias, float(probe["output_reinit_bias"]))
    correction = model(features, active=active)
    assert float(correction.abs().max()) > 0  # explicitly not a zero-head probe
    loss, _ = train_loss(correction.float(), coarse, truth, mean_energy,
                         hinge_weight=0.5, gradient_weight=0.03)
    assert torch.isfinite(loss)
    loss.backward()
    total = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
    assert total > float(tolerances["gradient_abs_min"])
    assert float(model.stem.conv.weight.grad.abs().max()) > 0
    assert float(model.output.weight.grad.abs().max()) > 0


# ------------------------------------------------- fp32 diagnostic and gate


def test_fp32_diagnostic_exact_storage_gives_zero_eq_and_passes() -> None:
    rng = np.random.default_rng(7)
    parent = rng.standard_normal((3, 4, 4)).astype(np.float32)
    truth = (parent + np.float32(1e-3) * rng.standard_normal((3, 4, 4)).astype(np.float32)).astype(np.float32)
    static = rng.standard_normal((7, 4, 4)).astype(np.float32)
    result = quantization_diagnostic_fp32(parent, truth, parent.copy(), truth.copy(),
                                          static, static.astype(np.float16), [0, 200, 400])
    record = result["record"]
    assert record["E_q"] == 0.0 and record["E_q_over_R_native"] == 0.0
    assert record["residual_cosine"] == pytest.approx(1.0, rel=1e-12)
    assert result["gate"]["passed"] and result["field_storage_dtype"] == "float32"
    # E_q == 0 only proves lossless storage, not any learning benefit.


def test_fp32_diagnostic_rejects_float16_stored_arrays() -> None:
    parent = np.zeros((2, 3, 3), np.float32); truth = np.ones_like(parent)
    static = np.zeros((7, 3, 3), np.float32)
    with pytest.raises(RuntimeError, match="stored field contract"):
        quantization_diagnostic_fp32(parent, truth, parent.astype(np.float16), truth.astype(np.float16),
                                     static, static.astype(np.float16), [0, 200])


def test_fp32_diagnostic_still_rejects_f16_equivalent_content() -> None:
    parent = np.ones((3, 2, 2), np.float32)
    truth = parent + np.float32(.0001)
    static = np.zeros((7, 2, 2), np.float32)
    qp = parent.astype(np.float16).astype(np.float32)
    qt = truth.astype(np.float16).astype(np.float32)
    result = quantization_diagnostic_fp32(parent, truth, qp, qt, static, static.astype(np.float16), [0, 200, 400])
    assert result["record"]["E_q_over_R_native"] == 1
    assert not result["gate"]["passed"]
    assert result["gate"]["failure_classification"] == "cache_representation_rejected"


def test_fp32_diagnostic_zero_truth_and_static_bytes_are_explicit() -> None:
    zero = np.zeros((3, 2, 2), np.float32)
    static = np.zeros((7, 2, 2), np.float32)
    result = quantization_diagnostic_fp32(zero, zero, zero.copy(), zero.copy(),
                                          static, static.astype(np.float16), [0, 200, 400])
    assert result["record"]["truth_norm_zero"] and result["gate"]["passed"]
    corrupted = zero.copy(); corrupted[0, 0, 0] = 1e-6
    result = quantization_diagnostic_fp32(zero, zero, zero.copy(), corrupted,
                                          static, static.astype(np.float16), [0, 200, 400])
    assert not result["record"]["zero_truth_values_preserved"] and not result["gate"]["passed"]
    drifted = np.ones((7, 2, 2), np.float16)
    result = quantization_diagnostic_fp32(zero, np.ones_like(zero), zero.copy(), np.ones_like(zero),
                                          static, drifted, [0, 200, 400])
    assert not result["static_features_stored_float16_bytes_exact"] and not result["gate"]["passed"]


# ------------------------------------------------------- prior f16 parity


def test_prior_record_parity_synthetic(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    parent_norm = rng.standard_normal((2, 3, 3)).astype(np.float32)
    truth_norm = rng.standard_normal((2, 3, 3)).astype(np.float32)
    static = rng.standard_normal((7, 3, 3)).astype(np.float32)
    values = {"parent_norm": parent_norm, "truth_norm": truth_norm,
              "baseline_error_square_norm": 1.5, "target_square_norm": 2.5, "scale": 0.125}
    row = {"sample_id": "s", "group_id": "g", "family": "uniform",
           "sample_sha256": "a" * 64, "source_index": 7}
    text = h5py.string_dtype("utf-8")
    with h5py.File(tmp_path / "prior.h5", "x") as prior:
        for name in ("sample_id", "group_id", "family", "sample_sha256"):
            prior.create_dataset(name, data=np.asarray([row[name]], object), dtype=text)
        prior.create_dataset("source_index", data=np.asarray([7], np.int64))
        prior.create_dataset("parent_full_sha256", data=np.asarray(["c" * 64], object), dtype=text)
        prior.create_dataset("coarse_norm", data=parent_norm[None].astype(np.float16))
        prior.create_dataset("truth_norm", data=truth_norm[None].astype(np.float16))
        prior.create_dataset("static_features", data=static[None].astype(np.float16))
        prior.create_dataset("baseline_error_square_norm", data=np.asarray([1.5], np.float64))
        prior.create_dataset("target_square_norm", data=np.asarray([2.5], np.float64))
        prior.create_dataset("field_scale", data=np.asarray([0.125], np.float32))
    with h5py.File(tmp_path / "prior.h5", "r") as prior:
        parity = prior_record_parity(prior, 0, row, "c" * 64, values, static)
        assert all(parity.values())
        drifted = dict(values); drifted["truth_norm"] = truth_norm + np.float32(1.0)
        with pytest.raises(RuntimeError, match="do not reproduce prior f16 bytes"):
            prior_record_parity(prior, 0, row, "c" * 64, drifted, static)
        with pytest.raises(RuntimeError, match="parent full hash drift"):
            prior_record_parity(prior, 0, row, "d" * 64, values, static)


# ----------------------------------------- preregistration and static audit


def test_fp32_preregistration_freeze_contract() -> None:
    prereg = _prereg()
    assert prereg["schema"] == "r6_cache_fp32_smoke_preregistration_v1"
    assert prereg["candidate"] == CANDIDATE
    assert prereg["status"] == "fp32_smoke_prepared_pending_audit"
    assert not prereg["full_build_authorized"] and not prereg["training_authorized"]
    assert not prereg["promotion_authorized"]
    assert not prereg["confirmation_opened"] and not prereg["validation_opened"] and not prereg["test_id_opened"]
    assert prereg["budgets"]["smoke_wall_seconds"] == 180
    assert prereg["budgets"]["peak_bytes"] == 8 * 2 ** 30
    assert prereg["prerequisites"]["prior_f16_smoke"]["sha256"] == PRIOR_TERMINAL_SHA256
    assert prereg["prerequisites"]["prior_f16_smoke"]["status"] == "cache_representation_rejected"
    assert prereg["selection"]["source_indices"] == SOURCE_INDICES
    change = prereg["single_variable_change"]
    assert change["changed"] == ["coarse_norm dtype float16->float32", "truth_norm dtype float16->float32"]
    assert prereg["gate"]["E_q_zero_interpretation"] == "storage_lossless_only_not_learning_benefit"
    assert set(prereg["binding_paths"]) == set(prereg["bindings"])


def test_selected_records_matches_frozen_panel_and_prereg() -> None:
    panel = selected_records(_prereg())
    for role, indices in SOURCE_INDICES.items():
        assert [row["source_index"] for row in panel[role]] == indices
        assert [row["family"] for row in panel[role]] == list(FAMILIES)
        assert all(row["split"] == "train" for row in panel[role])


def test_validate_invocation_accepts_frozen_and_rejects_drift() -> None:
    prereg = _prereg()
    args = argparse.Namespace(smoke=True, preregistration=PREREG, device="cuda:0")
    validate_invocation(args, prereg)
    with pytest.raises(RuntimeError, match="only the fixed GPU0 smoke"):
        validate_invocation(argparse.Namespace(smoke=True, preregistration=PREREG, device="cuda:1"), prereg)
    drifted = dict(prereg); drifted["status"] = "cache_build_authorized"
    with pytest.raises(RuntimeError, match="status drift"):
        validate_invocation(args, drifted)
    escalated = dict(prereg); escalated["full_build_authorized"] = True
    with pytest.raises(RuntimeError, match="scope drift"):
        validate_invocation(args, escalated)


def test_verify_stage_inputs_static_without_hdf5_or_cuda(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("static binding verification must not open HDF5 or CUDA")
    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    prereg = _prereg()
    verified = verify_stage_inputs(prereg)
    assert verified["explicit_bindings"] == prereg["bindings"]
    assert verified["dependency_closure"]
    broken = dict(prereg); broken["binding_paths"] = dict(prereg["binding_paths"])
    del broken["binding_paths"]["R25"]
    with pytest.raises(RuntimeError, match="cache stage input binding drift"):
        verify_stage_inputs(broken)


def test_fp32_dependency_manifest_matches_local_closure() -> None:
    manifest = json.loads((STAGE / "dependency_manifest.json").read_text())
    assert manifest["roots"] == ["scripts/smoke_r6_device_residual_cache_fp32.py"]
    observed = [path.relative_to(ROOT).as_posix()
                for path in dependency_closure([ROOT / path for path in manifest["roots"]])]
    assert observed == [row["path"] for row in manifest["files"]]
    for row in manifest["explicit_bindings"]:
        path = Path(row["path"]); path = path if path.is_absolute() else ROOT / path
        assert sha256(path) == row["sha256"]


def test_smoke_script_has_no_direct_wavefield_access_and_keeps_seals() -> None:
    source = (ROOT / "scripts/smoke_r6_device_residual_cache_fp32.py").read_text()
    tree = ast.parse(source)
    count = sum(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and node.slice.value == "wavefield" for node in ast.walk(tree))
    assert count == 0 and "guard.read_truth" in source
    assert "additional_source_truth_reads" in source
    for name in ("confirmation_opened", "validation_opened", "test_id_opened"):
        assert name in source
    assert "full_build_authorized" in source and "training_authorized" in source
