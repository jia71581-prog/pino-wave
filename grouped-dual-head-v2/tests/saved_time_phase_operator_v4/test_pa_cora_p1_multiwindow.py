from __future__ import annotations

import h5py
import numpy as np

from scripts.build_pa_cora_p1_multiwindow_manifests import (
    derive_multiwindow_manifests,
)
from scripts.train_pa_cora_p1_multiwindow import MultiWindowCache
from scripts.analyze_pa_cora_p1_window_statistics import analyze_slot


def _base_manifest():
    records = []
    for index, family in enumerate(("uniform", "layered", "marmousi")):
        records.append(
            {
                "source_index": index,
                "sample_id": f"train_{family}_{index:05d}",
                "group_id": f"train:{family}:{index:05d}",
                "sample_sha256": str(index),
                "family": family,
                "source_f0_hz": 10.0 + index,
                "source_t0_s": 0.15,
                "window_start": 10 + 10 * index,
            }
        )
    return {
        "schema": "b2_v5_causal_stratified_manifest_v1",
        "selection_sha256": "base-selection",
        "split": "train",
        "source_h5": "synthetic.h5",
        "families": ["uniform", "layered", "marmousi"],
        "groups_per_family": 1,
        "n_visible": 8,
        "k_frames": 64,
        "records": records,
        "future_truth_opened_for_window_selection": False,
        "validation_opened": False,
        "test_id_opened": False,
    }


def test_multiwindow_slots_are_monotone_and_truth_independent():
    base = _base_manifest()
    time_s = np.arange(401, dtype=np.float64) * 0.0025
    manifests = derive_multiwindow_manifests(base, time_s, terminal_start_s=0.65)
    assert len(manifests) == 4
    for record_index, base_row in enumerate(base["records"]):
        starts = [m["records"][record_index]["window_start"] for m in manifests]
        assert starts[0] == base_row["window_start"]
        assert starts == sorted(starts)
        assert starts[-1] == 260
    assert all(not m["multiwindow"]["future_truth_used"] for m in manifests)
    sample_ids = [
        row["sample_id"] for manifest in manifests for row in manifest["records"]
    ]
    assert len(sample_ids) == len(set(sample_ids))


def test_p1_update_budget_matches_v9_record_exposure():
    p1_steps = (4 * 240 // 4) * 23
    p1_exposures = p1_steps * 4
    v9_steps = (240 // 4) * 90
    v9_exposures = v9_steps * 4
    assert p1_steps == 5520
    assert abs(p1_exposures - v9_exposures) / v9_exposures < 0.025


def test_streaming_multicache_batch_contract(tmp_path):
    manifests = derive_multiwindow_manifests(
        _base_manifest(), np.arange(401, dtype=np.float64) * 0.0025
    )
    paths = []
    for slot, manifest in enumerate(manifests):
        path = tmp_path / f"slot{slot}.h5"
        paths.append(path)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = "b2_v5_causal_cache_v1"
            handle.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
            handle.create_dataset("base_seq", data=np.zeros((3, 64, 5, 4), np.float16))
            handle.create_dataset("target", data=np.ones((3, 64, 5, 4), np.float16))
            handle.create_dataset("cond", data=np.zeros((3, 7, 5, 4), np.float16))
            handle.create_dataset(
                "family",
                data=np.asarray(["uniform", "layered", "marmousi"], dtype=object),
                dtype=h5py.string_dtype(),
            )
    cache = MultiWindowCache(paths, manifests, key="cond")
    try:
        base, target, conditioning = cache.batch([0, 4, 8, 11])
        assert base.shape == (4, 64, 1, 5, 4)
        assert target.shape == base.shape
        assert conditioning.shape == (4, 7, 5, 4)
        assert float(target.mean()) == 1.0
    finally:
        cache.close()


def test_window_statistics_reports_exact_anchor_relative_error(tmp_path):
    manifest = _base_manifest()
    manifest["records"] = manifest["records"][:1]
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "b2_v5_causal_cache_v1"
        handle.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
        handle.create_dataset("base_seq", data=np.zeros((1, 64, 41, 41), np.float16))
        handle.create_dataset("target", data=np.ones((1, 64, 41, 41), np.float16))
    rows = analyze_slot(path, manifest, slot=0)
    assert len(rows) == 1
    assert np.isclose(rows[0]["anchor_relative_l2"], 1.0)
    assert rows[0]["target_norm"] > 0.0
    assert len(rows[0]["frequency_target_energy_share_sampled"]) == 3
