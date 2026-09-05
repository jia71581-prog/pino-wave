from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reports.no_asvgd_group_meeting_20260710 import make_elastic_vti_section as slide_assets
from reports.no_asvgd_group_meeting_20260710.make_elastic_vti_section import (
    select_nearest_frames,
    summarize_evaluation_metrics,
    validate_manifest,
)
from scripts.benchmark_elastic_vti_pino import summarize_seconds


def test_velocity_display_uses_fixed_physical_limits() -> None:
    assert getattr(slide_assets, "VELOCITY_DISPLAY_LIMITS", None) == {
        "vp": (1400.0, 7000.0),
        "vs": (800.0, 4600.0),
    }


def test_coordinate_edges_convert_cell_centers_to_kilometres() -> None:
    coordinate_edges_km = getattr(slide_assets, "coordinate_edges_km", None)
    assert callable(coordinate_edges_km)
    assert coordinate_edges_km([5.0, 15.0, 25.0]) == (0.0, 0.03)


def test_receiver_gather_places_time_on_horizontal_axis() -> None:
    receiver_gather_time_x = getattr(slide_assets, "receiver_gather_time_x", None)
    assert callable(receiver_gather_time_x)
    field = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    gather = receiver_gather_time_x(field, receiver_depth_index=1)
    assert gather.shape == (3, 4)
    np.testing.assert_array_equal(gather, field[1, :, :])


def test_receiver_depth_is_selected_along_z_axis() -> None:
    target = np.zeros((4, 3, 2, 2), dtype=np.float64)
    target[1, :, :, :] = 5.0
    assert slide_assets._receiver_depth(target) == 1


def test_relative_l2_per_component_uses_physical_wavefields() -> None:
    relative_l2_per_component = getattr(slide_assets, "relative_l2_per_component", None)
    assert callable(relative_l2_per_component)
    target = np.ones((2, 2, 3, 2), dtype=np.float64)
    prediction = target.copy()
    prediction[..., 0] = 0.0
    prediction[..., 1] = 1.5
    assert relative_l2_per_component(prediction, target) == {"ux": 1.0, "uz": 0.5}


def test_select_validation_indices_uses_saved_split_order(tmp_path: Path) -> None:
    select_validation_indices = getattr(slide_assets, "select_validation_indices", None)
    assert callable(select_validation_indices)
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"val": list(range(40, 80))}), encoding="utf-8")
    assert select_validation_indices(split_path, 30) == list(range(40, 70))


def test_validate_distribution_metrics_requires_30_instances_per_model() -> None:
    validate_distribution_metrics = getattr(slide_assets, "validate_distribution_metrics", None)
    assert callable(validate_distribution_metrics)
    payload = {
        "protocol": {"instances_per_model": 30},
        "models": {
            name: {
                "indices": list(range(30)),
                "ux_relative_l2": [0.1] * 30,
                "uz_relative_l2": [0.2] * 30,
            }
            for name in ("uniform", "layered", "marmousi")
        },
    }
    validate_distribution_metrics(payload, expected_count=30)


def test_plot_distribution_boxplots_writes_png_and_pdf(tmp_path: Path) -> None:
    plot_distribution_boxplots = getattr(slide_assets, "plot_distribution_boxplots", None)
    assert callable(plot_distribution_boxplots)
    payload = {
        "protocol": {"instances_per_model": 30},
        "models": {
            name: {
                "indices": list(range(30)),
                "ux_relative_l2": np.linspace(0.05, 0.15, 30).tolist(),
                "uz_relative_l2": np.linspace(0.04, 0.12, 30).tolist(),
            }
            for name in ("uniform", "layered", "marmousi")
        },
    }
    output = tmp_path / "boxplots.png"
    plot_distribution_boxplots(payload, output)
    assert output.is_file() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").is_file() and output.with_suffix(".pdf").stat().st_size > 0


def test_select_nearest_frames_uses_requested_times() -> None:
    times = [0.0, 0.165, 0.335, 0.5]
    assert select_nearest_frames(times, [0.165, 0.335]) == [1, 2]


def test_marmousi_presentation_case_uses_requested_global_index() -> None:
    assert slide_assets.MODEL_SPECS["marmousi"]["sample_index"] == 300


def test_summarize_evaluation_metrics_reads_both_components(tmp_path: Path) -> None:
    metrics = {
        "per_component_mean": {
            "component_0": {"relative_l2": 0.1, "receiver_line_relative_l2": 0.2},
            "component_1": {"relative_l2": 0.3, "receiver_line_relative_l2": 0.4},
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert summarize_evaluation_metrics(path) == {
        "ux_relative_l2": 0.1,
        "uz_relative_l2": 0.3,
        "ux_receiver_relative_l2": 0.2,
        "uz_receiver_relative_l2": 0.4,
    }


def test_summarize_seconds_reports_median_iqr_and_range() -> None:
    result = summarize_seconds([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "median_s": 2.5,
        "q25_s": 1.75,
        "q75_s": 3.25,
        "min_s": 1.0,
        "max_s": 4.0,
    }


def test_validate_manifest_requires_all_models_and_figures() -> None:
    manifest = {
        "models": {
            name: {
                "figures": {kind: f"{name}_{kind}.png" for kind in ("ux", "uz", "receivers")},
                "metrics": {"relative_l2": 0.1},
            }
            for name in ("uniform", "layered", "marmousi")
        }
    }
    validate_manifest(manifest)
