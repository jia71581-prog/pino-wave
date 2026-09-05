"""r49 direct48_pbg pilot tests.

Covers the scientific contract of the minimal project-train overfit pilot:
sample-id/fit-only resolution, P_bg never entering the model input, the
48-bin direct-frequency target vs an independent NumPy reference, the
[64x64, 401] pilot shapes, all eval flags forced off, config-held decision
gates, and loader-consistent 64x64 resampling against the real fit rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402
from grouped_ufno_mionet_v3.data.pilot import PilotBatch  # noqa: E402
from grouped_ufno_mionet_v3.normalization import (  # noqa: E402
    PhysicalNormalizer,
    ScaleMetadata,
)
from saved_time_phase_operator_v4.operator import SavedTimePhaseOperatorV4  # noqa: E402
from scripts.diagnose_capacity_ladder_overfit import (  # noqa: E402
    _direct_frequency_coefficient_update,
    direct_frequency_target_coefficients,
)
from scripts.train_direct48_pbg_pilot import (  # noqa: E402
    Resample64PilotDataset,
    assert_pilot_eval_flags,
    bilinear_antialias_resize,
    build_direct48_schedule,
    decide_capacity,
    resample_coordinate_axis,
    resample_source_map,
    resample_wavefield_frames,
    resolve_fit_sample_rows,
    validate_run_config,
)

REAL_H5 = ROOT / "artifacts/pbg_factorized_fno_marm700_eikonal_v3/dataset_v1.h5"
REAL_SPLIT = ROOT / "artifacts/pbg_factorized_fno_marm700_eikonal_v3/split_full560_70_70_seed372.json"
NORMALIZATION_JSON = (
    ROOT / "artifacts/pbg_factorized_fno_marm700_eikonal_v3"
    / "normalization_v3_direct48_pbg.json"
)
RUN_CONFIG = ROOT / "configs/saved_time_v4/direct48_pbg_pilot_run.yaml"
BASE_CONFIG = ROOT / "configs/saved_time_v4/direct48_pbg_pilot.yaml"

REAL_FIT_IDS = ("train_marmousi_00000", "train_marmousi_00002", "train_marmousi_00003")
PRESSURE_SCALE_PA = 1.894229157173348e-08


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# synthetic source / helpers
# --------------------------------------------------------------------------- #
def _synthetic_source(tmp_path: Path):
    path = tmp_path / "source.h5"
    fit_ids = [f"train_marmousi_{i:05d}" for i in range(6)]
    ids = fit_ids + ["cal_marmousi_00000"]
    splits = ["train"] * 6 + ["calibration"]
    with h5py.File(path, "w") as h5:
        h5.create_dataset("sample_id", data=np.array(ids, dtype="S64"))
        h5.create_dataset("split", data=np.array(splits, dtype="S32"))
        h5.create_dataset(
            "sample_sha256", data=np.array([f"sha{i}" for i in range(len(ids))], dtype="S64")
        )
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_ids": {
                    "train": fit_ids,
                    "calibration": ["cal_marmousi_00000"],
                    "confirmation": [],
                    "validation": [],
                    "test_id": [],
                }
            }
        )
    )
    return path, manifest_path


def _normalizer() -> PhysicalNormalizer:
    return PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=2000.0,
            velocity_scale_mps=500.0,
            pressure_scale_pa=2.0e-8,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="test-train",
            allowed_medium_types=("uniform", "layered", "marmousi"),
            record_count=1,
            algorithm="test",
        )
    )


def _tiny_model(frequencies: int = 48):
    model = SavedTimePhaseOperatorV4(
        saved_time_s=torch.linspace(0.0, 1.0, 401, dtype=torch.float64),
        width=8,
        rank=6,
        spectral_rank=4,
        modes=(3, 2),
        dense_spectral_rank=6,
        dense_modes=4,
        dense_depth=2,
        dense_time_block=2,
        dense_local_field=True,
        dense_local_field_channel_multipliers=(1, 1, 2, 2),
        dense_local_field_causal_width_s=0.005,
        dense_local_field_residual=False,
        dense_local_field_helmholtz_synthesis=True,
        dense_local_field_helmholtz_synthesis_frequencies=frequencies,
        dense_local_field_helmholtz_synthesis_wkb_phase=False,
        dense_local_field_helmholtz_synthesis_rank=0,
        use_local_phase=True,
        heads=2,
        token_grid=2,
        position_bands=2,
        fourier_bands=2,
        gabor_scales_s=(0.03, 0.08),
        ray_samples=4,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
    )
    model.local_field.helmholtz_apply_causal_gate = False
    model.dense_apply_free_surface_factor = False
    return model


def _synthetic_batch(
    *,
    height: int = 9,
    width: int = 11,
    frames: int = 401,
    sample_id: str = "train_synthetic_0",
):
    return PilotBatch(
        step=0,
        velocity_mps=torch.full((1, 1, height, width), 2000.0),
        record_to_medium=torch.zeros(1, dtype=torch.long),
        source_parameters=torch.tensor([[400.0, 300.0, 10.0, 0.1, 1.0]]),
        source_map=(
            torch.zeros(1, 1, height, width).scatter_(
                2, torch.tensor([[[[2]]]]), 1.0
            )
        ),
        requested_time_s=torch.linspace(0.0, 1.0, frames)[None].expand(1, frames),
        dense_target_physical=torch.zeros(1, frames, height, width),
        target_exact=torch.zeros(1, frames, height, width),
        interpolation_alpha=torch.zeros(1, frames),
        left_index=torch.zeros(1, frames, dtype=torch.long),
        right_index=torch.zeros(1, frames, dtype=torch.long),
        query_coords=torch.zeros(1, 1, 2),
        query_target_physical=torch.zeros(1, 1),
        query_probability=torch.ones(1, 1),
        x_m=torch.linspace(0.0, 2000.0, width),
        z_m=torch.linspace(0.0, 2000.0, height),
        sample_id=(sample_id,),
        group_id=("marmousi",),
        medium_type=("marmousi",),
        dense_travel_time_s=None,
    )


# --------------------------------------------------------------------------- #
# 1. sample-id / fit-only contract
# --------------------------------------------------------------------------- #
def test_resolve_fit_sample_rows_maps_by_id_and_enforces_fit_only(tmp_path):
    source, manifest = _synthetic_source(tmp_path)
    resolved = resolve_fit_sample_rows(
        source, manifest, ["train_marmousi_00000", "train_marmousi_00002"]
    )
    assert [item["sample_id"] for item in resolved] == [
        "train_marmousi_00000",
        "train_marmousi_00002",
    ]
    assert [item["row"] for item in resolved] == [0, 2]
    assert [item["sample_sha256"] for item in resolved] == ["sha0", "sha2"]
    # fit rows only; no calibration/confirmation/validation/test_id touched
    for item in resolved:
        assert item["sample_id"].startswith("train_")


@pytest.mark.parametrize(
    "requested, match",
    [
        (
            ["train_marmousi_00000", "not_in_panel"],
            "not in the manifest fit panel",
        ),
        (["cal_marmousi_00000"], "not in the manifest fit panel"),
        (["train_marmousi_00006"], "missing from the HDF5"),
    ],
)
def test_resolve_fit_sample_rows_rejects_contract_violations(
    tmp_path, requested, match
):
    source, manifest = _synthetic_source(tmp_path)
    # "train_marmousi_00006" is in the manifest fit panel but absent from the
    # HDF5, so it must fail with the missing-row error (after the panel check).
    if requested == ["train_marmousi_00006"]:
        panel = json.loads(manifest.read_text())["sample_ids"]
        panel["train"] = panel["train"] + ["train_marmousi_00006"]
        manifest.write_text(json.dumps({"sample_ids": panel}))
    with pytest.raises(ValueError, match=match):
        resolve_fit_sample_rows(source, manifest, requested)


def test_resolve_fit_sample_rows_rejects_non_train_split_and_prefix(tmp_path):
    source, manifest = _synthetic_source(tmp_path)
    manifest_path = tmp_path / "split2.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_ids": {
                    "train": ["train_marmousi_00000", "train_marmousi_00001", "marmousi_bad_00002"],
                    "calibration": [],
                    "confirmation": [],
                    "validation": [],
                    "test_id": [],
                }
            }
        )
    )
    with h5py.File(source, "r+") as h5:
        splits = np.array(["train", "calibration", "train"], dtype="S32")
        h5["split"][0:3] = splits
    with pytest.raises(ValueError, match="instead of 'train'"):
        resolve_fit_sample_rows(source, manifest_path, ["train_marmousi_00001"])
    with h5py.File(source, "r+") as h5:
        # rename row 2's stored id so the prefix violation is reachable
        h5["sample_id"][2] = b"marmousi_bad_00002"
    with pytest.raises(ValueError, match="does not start with 'train_'"):
        resolve_fit_sample_rows(source, manifest_path, ["marmousi_bad_00002"])


def test_resolve_fit_sample_rows_rejects_empty_and_duplicate_columns(tmp_path):
    source, manifest = _synthetic_source(tmp_path)
    with h5py.File(source, "r+") as h5:
        ids = np.array(["train_marmousi_00000", "train_marmousi_00000"] + [b"x"] * 5, dtype="S64")
        h5["sample_id"][:] = ids
    with pytest.raises(ValueError, match="duplicate values"):
        resolve_fit_sample_rows(source, manifest, ["train_marmousi_00000"])


# --------------------------------------------------------------------------- #
# 2. P_bg wavefield never enters the model input
# --------------------------------------------------------------------------- #
def test_wavefield_labels_never_influence_model_forward():
    def run(seed: int, label_value: float):
        torch.manual_seed(seed)
        model = _tiny_model(frequencies=48)
        model.eval()
        batch = _synthetic_batch(frames=401)
        head = model.local_field.helmholtz_synthesis.head
        assert head.out_channels == 96
        captured: list[torch.Tensor] = []
        hook = head.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            components = _direct_frequency_coefficient_update(
                model,
                torch.optim.AdamW(model.parameters(), lr=1.0e-4),
                batch,
                _normalizer(),
                torch.device("cpu"),
                full_targets={
                    "train_synthetic_0": np.full(
                        (401, 9, 11), label_value, dtype=np.float32
                    )
                },
                frequency_count=48,
                microbatch_records=1,
                saved_time_s=torch.linspace(0.0, 1.0, 401),
                frequency_energy_floor_fraction=0.0,
            )
        finally:
            hook.remove()
        assert len(captured) == 1
        return captured[0].detach().clone(), components

    first, first_loss = run(0, 1.0e-8)
    second, second_loss = run(0, 2.0e-8)
    # identical initialization, different P_bg labels: the forward must be
    # bit-identical, proving the wavefield never enters the model input
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.shape == (1, 96, 9, 11)
    # the labels DO enter the loss: the coefficient loss must change
    assert first_loss["coefficient_relative_l2"] != pytest.approx(
        second_loss["coefficient_relative_l2"]
    )


# --------------------------------------------------------------------------- #
# 3. 48-bin coefficients vs an independent NumPy reference
# --------------------------------------------------------------------------- #
def test_48bin_coefficients_match_independent_numpy_reference():
    torch.manual_seed(11)
    records, frames, height, width = 2, 401, 16, 16
    target = torch.randn(records, frames, height, width)
    count = 48
    coefficients = direct_frequency_target_coefficients(target, count)
    assert coefficients.shape == (records, 2 * count, height, width)
    spectrum = np.fft.rfft(target.numpy(), axis=1, norm="forward")
    expected_cos = 2.0 * spectrum.real[:, :count]
    expected_sin = -2.0 * spectrum.imag[:, :count]
    # DC special case: cosine keeps the mean, sine is exactly zero
    expected_cos[:, 0] = spectrum.real[:, 0]
    expected_sin[:, 0] = 0.0
    expected = np.concatenate([expected_cos, expected_sin], axis=1)
    np.testing.assert_allclose(coefficients.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)
    # DC sanity: a constant trace yields only the DC cosine coefficient
    # (non-DC bins are zero up to float roundoff, not bit-exact zero)
    constant = torch.full((1, 401, 4, 4), 3.0)
    single = direct_frequency_target_coefficients(constant, 48)
    assert torch.isfinite(single).all()
    assert (single[0, 0] - 3.0).abs().max().item() < 1.0e-6
    assert single[0, 1:48].abs().max().item() < 1.0e-6
    assert single[0, 48:].abs().max().item() < 1.0e-6


# --------------------------------------------------------------------------- #
# 4. pilot shapes: head [B,96,64,64], fixed synthesis [B,401,64,64]
# --------------------------------------------------------------------------- #
def test_pilot_forward_renders_64x64_over_401_stored_points():
    model = _tiny_model(frequencies=48)
    normalizer = _normalizer()
    height = width = 64
    velocity = torch.full((1, 1, height, width), 2000.0)
    sources = torch.tensor([[1000.0, 1000.0, 10.0, 0.1, 1.0]])
    source_maps = torch.zeros(1, 1, height, width).scatter_(
        2, torch.tensor([[[[32]]]]), 1.0
    )
    prepared = model.prepare_sources(
        model.encode_medium(velocity, normalizer),
        sources,
        source_maps,
        normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long),
    )
    dense_grid = model.prepare_dense_grid(
        prepared,
        x_m=torch.linspace(0.0, 2000.0, width),
        z_m=torch.linspace(0.0, 2000.0, height),
        travel_time_s=None,
    )
    head = model.local_field.helmholtz_synthesis.head
    captured: list[torch.Tensor] = []
    hook = head.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    with torch.inference_mode():
        try:
            output = model.dense_normalized(
                prepared,
                torch.linspace(0.0, 1.0, 401)[None].expand(1, 401),
                dense_grid=dense_grid,
                time_block=401,
                apply_correction=False,
            )
        finally:
            hook.remove()
    assert output.shape == (1, 401, 64, 64)
    assert torch.isfinite(output).all()
    assert len(captured) == 1
    assert captured[0].shape == (1, 96, 64, 64)
    assert torch.isfinite(captured[0]).all()


# --------------------------------------------------------------------------- #
# 5. eval flags all off
# --------------------------------------------------------------------------- #
def test_eval_flag_guard_requires_all_flags_off():
    model = _tiny_model()
    config = {"loss": {"hard_causality": False}}
    assert_pilot_eval_flags(model, config)  # all off: passes
    model.local_field.helmholtz_apply_causal_gate = True
    with pytest.raises(RuntimeError, match="causal gate"):
        assert_pilot_eval_flags(model, config)
    model.local_field.helmholtz_apply_causal_gate = False
    model.dense_apply_free_surface_factor = True
    with pytest.raises(RuntimeError, match="free-surface"):
        assert_pilot_eval_flags(model, config)
    model.dense_apply_free_surface_factor = False
    with pytest.raises(RuntimeError, match="hard causality"):
        assert_pilot_eval_flags(model, {"loss": {"hard_causality": True}})


# --------------------------------------------------------------------------- #
# 6. run-config validation (gates live in config, not code)
# --------------------------------------------------------------------------- #
def _real_run_config() -> dict:
    import yaml

    return yaml.safe_load(RUN_CONFIG.read_text())


def test_validate_run_config_accepts_the_real_pilot_config():
    validate_run_config(_real_run_config())


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda cfg: cfg.update({"schema": "other"}), "schema mismatch"),
        (lambda cfg: cfg.update({"sample_ids": ["train_marmousi_00000"]}), "exactly three"),
        (
            lambda cfg: cfg.update(
                {"sample_ids": ["cal_marmousi_00000", "train_marmousi_00002", "train_marmousi_00003"]}
            ),
            "start with 'train_'",
        ),
        (
            lambda cfg: cfg["helmholtz"].update({"wkb_phase": True}),
            "wkb_phase=false",
        ),
        (lambda cfg: cfg["helmholtz"].update({"rank": 1}), "rank=0"),
        (
            lambda cfg: cfg["probe"].update({"local_field_channels": [1, 1]}),
            "four-level",
        ),
        (
            lambda cfg: cfg["train"].update({"microbatch_records": 2}),
            "microbatch_records must be 1",
        ),
        (
            lambda cfg: cfg["train"]["gradient_clip"].update({"mode": "global"}),
            "prefix_limits",
        ),
        (
            lambda cfg: cfg["train"]["gradient_clip"]["prefix_limits"].update({"local_field": 0.0}),
            "must be positive",
        ),
        (lambda cfg: cfg["eval"].update({"causal_gate": True}), "causal_gate=false"),
        (
            lambda cfg: cfg["eval"].update({"free_surface_factor": True}),
            "free_surface_factor=false",
        ),
        (
            lambda cfg: cfg["eval"].update({"hard_causality": True}),
            "hard_causality=false",
        ),
        (lambda cfg: cfg["eval"].update({"frames": 400}), "401"),
        (
            lambda cfg: cfg["decision"].update({"strong_capacity_aggregate_maximum": 0.20}),
            "0 < strong",
        ),
        (
            lambda cfg: cfg["decision"].update({"rejected_aggregate_minimum": 0.05}),
            "0 < strong",
        ),
    ],
)
def test_validate_run_config_rejects_contract_violations(mutate, match):
    cfg = _real_run_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_run_config(cfg)


# --------------------------------------------------------------------------- #
# 7. decision gates
# --------------------------------------------------------------------------- #
def test_decision_gate_boundaries():
    decision = {
        "accepted_capacity_aggregate_maximum": 0.10,
        "accepted_capacity_record_maximum": 0.12,
        "strong_capacity_aggregate_maximum": 0.06,
        "rejected_aggregate_minimum": 0.30,
    }
    assert decide_capacity(0.05, 0.10, decision)["status"] == "strong_capacity"
    assert decide_capacity(0.06, 0.12, decision)["status"] == "strong_capacity"
    assert decide_capacity(0.06 + 1.0e-9, 0.12, decision)["status"] == "accepted_capacity"
    assert decide_capacity(0.10, 0.12, decision)["status"] == "accepted_capacity"
    assert decide_capacity(0.10, 0.1200001, decision)["status"] == "inconclusive"
    assert decide_capacity(0.10 + 1.0e-9, 0.12, decision)["status"] == "inconclusive"
    assert decide_capacity(0.15, 0.05, decision)["status"] == "inconclusive"
    assert decide_capacity(0.30, 0.01, decision)["status"] == "rejected"
    assert decide_capacity(0.3000001, 0.01, decision)["status"] == "rejected"
    result = decide_capacity(0.07, 0.11, decision)
    assert result["status"] == "accepted_capacity"
    assert result["all_saved_aggregate_relative_l2"] == pytest.approx(0.07)
    assert "overfit" in result["interpretation"]


def test_decision_gate_config_order_is_enforced():
    cfg = _real_run_config()
    cfg["decision"]["strong_capacity_aggregate_maximum"] = 0.11  # > accepted 0.10
    with pytest.raises(ValueError, match="0 < strong"):
        validate_run_config(cfg)


# --------------------------------------------------------------------------- #
# 8. training schedule
# --------------------------------------------------------------------------- #
def test_build_direct48_schedule():
    schedule = build_direct48_schedule((0, 2, 3), updates=800)
    assert len(schedule) == 800
    assert all(entry.record_indices == (0, 2, 3) for entry in schedule)
    assert schedule[0].appearance_indices == (0, 0, 0)
    assert schedule[5].appearance_indices == (5, 5, 5)
    assert schedule[799].step == 950_000 + 799
    with pytest.raises(ValueError, match="three records"):
        build_direct48_schedule((0, 2), updates=10)
    with pytest.raises(ValueError, match="positive updates"):
        build_direct48_schedule((0, 2, 3), updates=0)


# --------------------------------------------------------------------------- #
# 9. 64x64 resampling pipeline
# --------------------------------------------------------------------------- #
class _StubTravel:
    def read(self, sample_ids):
        assert tuple(sample_ids) == ("train_synthetic_0",)
        return torch.zeros(1, 64, 64)


class _StubInner:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        batch = _synthetic_batch(height=201, width=201, frames=401)
        return batch


def test_resample64_wrapper_shapes_and_unit_mass():
    wrapper = Resample64PilotDataset(_StubInner(), _StubTravel(), 64)
    batch = wrapper[0]
    assert batch.velocity_mps.shape == (1, 1, 64, 64)
    assert batch.dense_target_physical.shape == (1, 401, 64, 64)
    assert batch.x_m.shape == (64,) and batch.z_m.shape == (64,)
    assert batch.dense_travel_time_s.shape == (1, 64, 64)
    assert abs(float(batch.source_map.sum()) - 1.0) < 1.0e-5
    assert batch.source_map.shape == (1, 1, 64, 64)


def test_resample_source_map_renormalizes_and_rejects_zero_mass():
    field = torch.ones(1, 1, 201, 201)
    resampled = resample_source_map(field, 64)
    assert resampled.shape == (1, 1, 64, 64)
    assert abs(float(resampled.sum()) - 1.0) < 1.0e-5
    with pytest.raises(FloatingPointError, match="non-positive mass"):
        resample_source_map(torch.zeros(1, 1, 201, 201), 64)


def test_resample_coordinate_axis_matches_the_2d_field_rule():
    axis = torch.linspace(0.0, 2000.0, 201)
    # values along the x axis (last dim); the field is constant along z
    from_field = bilinear_antialias_resize(
        axis[None, None, None, :].expand(1, 1, 201, 201), 64
    )[0, 0, 0]
    assert torch.equal(resample_coordinate_axis(axis, 64), from_field)
    resampled = resample_coordinate_axis(axis, 64)
    assert float(resampled[0]) == pytest.approx(13.64, abs=0.05)
    assert float(resampled[-1]) == pytest.approx(1986.36, abs=0.05)
    assert torch.all(torch.diff(resampled) > 0.0)


# --------------------------------------------------------------------------- #
# 10. loader-consistency on the real fit rows (labels are train-only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_H5.exists(), reason="real pbg dataset not present")
def test_resampler_matches_real_loader_on_fit_rows():
    from scripts.diagnose_trainonly_spectral_capacity import LoaderWavefieldReader

    reader = LoaderWavefieldReader(REAL_H5, 64)
    with h5py.File(REAL_H5, "r", swmr=True) as h5:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["sample_id"][:4]]
        assert ids[:4] == [
            "train_marmousi_00000",
            "train_marmousi_00001",
            "train_marmousi_00002",
            "train_marmousi_00003",
        ]
        splits = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["split"][:4]]
        assert splits == ["train"] * 4
        for row in (0, 2, 3):
            raw = np.asarray(h5["wavefield"][row], dtype=np.float32)  # [T,Z,X] NTZX
            assert raw.shape == (401, 201, 201)
            mine = resample_wavefield_frames(
                torch.from_numpy(raw[None]), 64
            )[0]
            loader = np.transpose(reader.read(h5, row), (2, 0, 1))  # -> [T,64,64]
            assert loader.shape == (401, 64, 64)
            torch.testing.assert_close(mine, torch.from_numpy(loader), rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not REAL_H5.exists(), reason="real pbg dataset not present")
def test_real_fit_rows_resolve_to_expected_rows_and_sha():
    resolved = resolve_fit_sample_rows(REAL_H5, REAL_SPLIT, REAL_FIT_IDS)
    assert [item["row"] for item in resolved] == [0, 2, 3]
    assert all(item["sample_id"].startswith("train_") for item in resolved)
    assert all(item["sample_sha256"] for item in resolved)


@pytest.mark.skipif(not REAL_H5.exists(), reason="real pbg dataset not present")
def test_real_coordinate_axes_match_the_loader_resampling_rule():
    with h5py.File(REAL_H5, "r", swmr=True) as h5:
        x_m = np.asarray(h5["x_m"][:], dtype=np.float32)
        z_m = np.asarray(h5["z_m"][:], dtype=np.float32)
    for axis in (x_m, z_m):
        resampled = resample_coordinate_axis(torch.from_numpy(axis), 64)
        assert float(resampled[0]) == pytest.approx(13.64, abs=0.05)
        assert float(resampled[-1]) == pytest.approx(1986.36, abs=0.05)


# --------------------------------------------------------------------------- #
# 11. normalization artifact / base config contract
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not NORMALIZATION_JSON.exists(), reason="pilot normalization not present")
def test_normalization_artifact_pins_contract_values():
    payload = json.loads(NORMALIZATION_JSON.read_text())
    assert payload["pressure_scale_pa"] == PRESSURE_SCALE_PA
    assert payload["record_count"] == 700
    assert payload["source_scales"] == [2000.0, 2000.0, 50.0, 1.2, 1.0]
    assert payload["algorithm"] == "direct48_pbg_pilot_v1"
    manifest = build_manifest(REAL_H5)
    assert payload["train_manifest_sha256"] == manifest.digest
    normalizer = PhysicalNormalizer.from_dict(payload, expected_manifest=manifest.digest)
    assert normalizer.metadata.pressure_scale_pa == PRESSURE_SCALE_PA
    assert normalizer.metadata.record_count == 700


@pytest.mark.skipif(not BASE_CONFIG.exists(), reason="pilot base config not present")
def test_real_base_config_loads_with_pilot_contract():
    from grouped_ufno_mionet_v3.config import V3Config

    base = V3Config.from_yaml(str(BASE_CONFIG))
    assert base.model.width == 128
    assert base.data.expected_train_records == 700
    assert base.data.expected_validation_records == 0
    assert Path(base.data.source_h5).resolve() == REAL_H5
    assert Path(base.data.manifest_json).resolve() == REAL_SPLIT
    assert Path(base.data.normalization_json).resolve() == NORMALIZATION_JSON


# --------------------------------------------------------------------------- #
# 12. end-to-end smoke of main() on a temporary synthetic dataset
# --------------------------------------------------------------------------- #
def _write_synthetic_pbg_dataset(path: Path, *, records: int = 3, grid: int = 17):
    frames = 401
    ids = ["train_marmousi_00000", "train_marmousi_00002", "train_marmousi_00003"]
    with h5py.File(path, "w") as h5:
        h5.create_dataset("medium_type", data=np.array(["marmousi"] * records, dtype="S16"))
        h5.create_dataset("split", data=np.array(["train"] * records, dtype="S16"))
        h5.create_dataset("sample_id", data=np.array(ids, dtype="S64"))
        h5.create_dataset(
            "group_id", data=np.array([f"marmousi_{i}" for i in range(records)], dtype="S32")
        )
        h5.create_dataset(
            "sample_sha256", data=np.array([f"sha-{i}" for i in range(records)], dtype="S64")
        )
        h5.create_dataset("split_id", data=np.arange(records, dtype=np.int64))
        h5.create_dataset("time_s", data=np.linspace(0.0, 1.0, frames, dtype=np.float64))
        h5.create_dataset("x_m", data=np.linspace(0.0, 2000.0, grid, dtype=np.float64))
        h5.create_dataset("z_m", data=np.linspace(0.0, 2000.0, grid, dtype=np.float64))
        z_grid = np.arange(grid)[:, None]
        x_grid = np.arange(grid)[None, :]
        velocity = (2500.0 + 2.0 * z_grid + 1.0 * x_grid).astype(np.float32)
        velocity = np.repeat(velocity[None], records, axis=0)  # [records, z, x]
        h5.create_dataset("velocity_mps", data=velocity)
        source_map = np.zeros((records, grid, grid), dtype=np.float32)
        source_map[:, 2, 3] = 1.0
        h5.create_dataset("source_map", data=source_map)
        h5.create_dataset("source_x_m", data=np.full((records,), 600.0, dtype=np.float32))
        h5.create_dataset("source_z_m", data=np.full((records,), 300.0, dtype=np.float32))
        h5.create_dataset("source_f0_hz", data=np.full((records,), 10.0, dtype=np.float32))
        h5.create_dataset("source_t0_s", data=np.full((records,), 0.1, dtype=np.float32))
        h5.create_dataset("source_amplitude", data=np.ones((records,), dtype=np.float32))
        h5.create_dataset("travel_time_s", data=np.zeros((records, grid, grid), dtype=np.float32))
        t = np.linspace(0.0, 1.0, frames)[:, None, None]
        r = np.sqrt(
            (np.arange(grid)[None, :, None] - 3.0) ** 2
            + (np.arange(grid)[None, None, :] - 2.0) ** 2
        )
        pulse = np.exp(-((t - 0.3) ** 2) / 0.01) * np.exp(-(r**2) / 8.0)
        wavefield = np.repeat(pulse[None], records, axis=0).astype(np.float32)
        h5.create_dataset("wavefield", data=wavefield)  # [records, time, z, x]
        h5.attrs["manifest_sha256"] = "synthetic-manifest"
        h5.attrs["config_sha256"] = "synthetic-config"
        h5.attrs["axis_order"] = "NTZX"
        h5.attrs["schema_version"] = "acoustic_lwc84_401_to_201_v1"
        h5.attrs["included_splits"] = '["train"]'
        h5.attrs["vds_sample_count"] = records
    return ids


def _write_synthetic_pilot_files(tmp_path: Path):
    from grouped_ufno_mionet_v3.data.index import build_manifest

    source = tmp_path / "synthetic.h5"
    ids = _write_synthetic_pbg_dataset(source)
    manifest = build_manifest(source)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "synthetic_pilot_v1",
                "project_data_scope": "synthetic",
                "strategy": "synthetic",
                "seed": 7,
                "train": 3,
                "val": 0,
                "test": 0,
                "vds": str(source),
                "test_id_opened": False,
                "validation_opened": False,
                "sample_counts": {"train": {"marmousi": 3}},
                "sample_ids": {
                    "train": ids,
                    "calibration": [],
                    "confirmation": [],
                    "validation": [],
                    "test_id": [],
                },
            }
        )
    )
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(
        json.dumps(
            {
                "velocity_center_mps": 2500.0,
                "velocity_scale_mps": 100.0,
                "pressure_scale_pa": PRESSURE_SCALE_PA,
                "source_scales": [2000.0, 2000.0, 50.0, 1.2, 1.0],
                "train_manifest_sha256": manifest.digest,
                "allowed_medium_types": ["uniform", "layered", "marmousi"],
                "record_count": 3,
                "algorithm": "direct48_pbg_pilot_v1",
            }
        )
    )
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "\n".join(
            [
                "model:",
                "  width: 16",
                "  spectral_rank: 8",
                "  modes: [6, 5, 4, 3]",
                "  dense_modes: [6, 5]",
                "  mionet_rank: 16",
                "  token_count: 64",
                "  heads: 4",
                "  position_bands: 4",
                "  fourier_bands: 6",
                "  gabor_scales_s: [0.01, 0.025, 0.05, 0.1]",
                "  ray_samples: 12",
                "  dense_time_block: 1",
                "  domain_x_m: 2000.0",
                "  domain_z_m: 2000.0",
                "  domain_t_s: 1.0",
                "data:",
                f"  source_h5: {source}",
                f"  manifest_json: {split_path}",
                f"  normalization_json: {normalization_path}",
                "  expected_train_records: 3",
                "  expected_validation_records: 0",
                "  continuous_fraction: 0.25",
                "loss: {}",
                "train:",
                "  optimizer: adamw",
                "  seed: 7",
                "  device: cpu",
            ]
        )
    )
    run_path = tmp_path / "run.yaml"
    run_path.write_text(
        "\n".join(
            [
                "schema: direct48_pbg_pilot_run_v1",
                f"base_config: {base_path}",
                "sample_ids: [train_marmousi_00000, train_marmousi_00002, train_marmousi_00003]",
                "spatial_size: 64",
                "helmholtz: {frequencies: 48, wkb_phase: false, rank: 0}",
                "probe:",
                "  dense_depth: 2",
                "  dense_spectral_rank: 16",
                "  dense_modes: 8",
                "  local_field_channels: [1, 1, 2, 2]",
                "  local_field_causal_width_s: 0.005",
                "  local_field_residual: false",
                "train:",
                "  dense_learning_rate: 0.0001",
                "  backbone_learning_rate: 5.0e-05",
                "  seed: 372",
                "  updates: 800",
                "  evaluate_every: 50",
                "  microbatch_records: 1",
                "  gradient_clip:",
                "    maximum_norm: 1.0",
                "    mode: prefix_limits",
                "    prefix_limits:",
                "      coordinate_encoder: 5.0",
                "      default: 1.0",
                "      dense_decoder: 20.0",
                "      fusion: 5.0",
                "      local_field: 20.0",
                "      medium_encoder: 2.0",
                "      source_encoder: 5.0",
                "      travel_branch: 5.0",
                "eval: {causal_gate: false, free_surface_factor: false, hard_causality: false, frames: 401}",
                "decision:",
                "  metric: all_saved_aggregate_relative_l2",
                "  accepted_capacity_aggregate_maximum: 0.10",
                "  accepted_capacity_record_maximum: 0.12",
                "  strong_capacity_aggregate_maximum: 0.06",
                "  rejected_aggregate_minimum: 0.30",
            ]
        )
    )
    return base_path, run_path, manifest, ids


def test_main_smoke_end_to_end_on_synthetic_dataset(tmp_path):
    from scripts.train_direct48_pbg_pilot import main

    base_path, run_path, manifest, ids = _write_synthetic_pilot_files(tmp_path)
    output = tmp_path / "out"
    code = main(
        [
            "--base-config",
            str(base_path),
            "--run-config",
            str(run_path),
            "--output-dir",
            str(output),
            "--updates",
            "2",
            "--evaluate-every",
            "1",
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert code == 0
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["schema"] == "direct48_pbg_pilot_terminal_v1"
    assert terminal["status"] == "smoke_complete"
    assert terminal["updates_completed"] == 2
    assert terminal["access"]["fit_opened"] is True
    assert terminal["access"]["calibration_opened"] is False
    assert terminal["access"]["confirmation_opened"] is False
    assert terminal["access"]["validation_opened"] is False
    assert terminal["access"]["test_id_opened"] is False
    assert terminal["access"]["wavefield_use"] == "fit_panel_train_labels_only"
    assert terminal["parameter_count"] > 0

    identity = json.loads((output / "run_identity.json").read_text())
    assert identity["schema"] == "direct48_pbg_pilot_v1"
    assert identity["fit_sample_ids"] == list(ids)
    assert identity["fit_rows"] == [0, 1, 2]  # synthetic file row order
    assert identity["manifest_digest"] == manifest.digest
    assert identity["helmholtz"] == {"frequencies": 48, "wkb_phase": False, "rank": 0}
    assert identity["eval"] == {
        "causal_gate": False,
        "free_surface_factor": False,
        "hard_causality": False,
        "frames": 401,
    }
    assert identity["run_digest"]
    # terminal binds the sha256 of the written file (indented JSON), while the
    # checkpoint config_digest binds the compact canonical content digest
    assert terminal["bindings"]["run_identity_sha256"] == _file_sha256(
        output / "run_identity.json"
    )

    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert [entry["update"] for entry in updates] == [1, 2]
    assert all("coefficient_relative_l2" in entry["loss_components"] for entry in updates)
    assert all(
        entry["gradient_prefixes"].keys() == {"local_field", "medium_encoder", "source_encoder"}
        for entry in updates
    )
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [entry["event"] for entry in metrics] == ["baseline", "evaluation", "evaluation"]
    assert all("aggregate_relative_l2" in entry["metrics"] for entry in metrics)

    checkpoints = sorted((output / "checkpoints").glob("update_*.pt"))
    assert [path.name for path in checkpoints] == [
        "update_0000.pt",
        "update_0001.pt",
        "update_0002.pt",
    ]
    assert (output / "latest.pt").exists()
    latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["manifest_digest"] == manifest.digest
    assert latest["config_digest"] == identity["run_digest"]
    assert latest["global_step"] == 2
    assert latest["optimizer_state"] is not None
    assert latest["rng_state"] is not None


def test_main_refuses_incomplete_or_finished_runs(tmp_path):
    from scripts.train_direct48_pbg_pilot import main

    base_path, run_path, _manifest, _ids = _write_synthetic_pilot_files(tmp_path)
    output = tmp_path / "out2"
    (output / "checkpoints").mkdir(parents=True)
    (output / "run_identity.json").write_text(json.dumps({"schema": "partial"}))
    with pytest.raises(FileExistsError, match="incomplete previous run"):
        main(
            [
                "--base-config",
                str(base_path),
                "--run-config",
                str(run_path),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
                "--smoke",
            ]
        )
    (output / "terminal.json").write_text(json.dumps({"schema": "done"}))
    code = main(
        [
            "--base-config",
            str(base_path),
            "--run-config",
            str(run_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert code == 0  # finished run -> terminal echoed, nothing touched
