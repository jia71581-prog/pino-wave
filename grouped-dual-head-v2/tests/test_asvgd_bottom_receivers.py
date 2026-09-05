from __future__ import annotations

import dis
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_asvgd_bottom_receivers import (
    StageConditionedAdam,
    apply_posterior_l2_guard,
    append_bottom_receivers,
    bottom_receiver_mse,
    build_l2_stage_conditioner,
    load_saved_runner,
    make_l2_optimizer_factory,
    patch_l2_optimizer,
    patch_legacy_map_log_format,
)


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_help_loads_preserved_runner_from_project_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_asvgd_bottom_receivers.py"),
            "--bottom-receivers",
            "2",
            "--help",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--model-source" in completed.stdout


def test_wrapper_help_lists_stage_conditioning_options() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_asvgd_bottom_receivers.py"),
            "--bottom-receivers",
            "2",
            "--help",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--l2-gradient-sigma-cells" in completed.stdout
    assert "--posterior-l2-guard-relative-nmse" in completed.stdout


def test_l2_optimizer_factory_conditions_only_the_first_lbfgs_construction() -> None:
    class FakeLBFGS:
        def __init__(self, params, lr, **kwargs) -> None:
            self.params = list(params)
            self.lr = lr
            self.kwargs = kwargs

    factory = make_l2_optimizer_factory(
        original_lbfgs=FakeLBFGS,
        adam_class=torch.optim.Adam,
        conditioner=None,
    )
    first = factory([torch.nn.Parameter(torch.zeros(()))], lr=0.1, max_iter=20)
    second = factory([torch.nn.Parameter(torch.zeros(()))], lr=0.2, max_iter=15)

    assert isinstance(first, StageConditionedAdam)
    assert isinstance(second, FakeLBFGS)
    assert second.lr == pytest.approx(0.2)
    assert second.kwargs["max_iter"] == 15


def test_stage_conditioner_defers_none_velocity_bounds_to_the_preserved_runner() -> None:
    runner = SimpleNamespace(
        parse_float_schedule=lambda _: [5.0, 8.0, 10.0, 15.0],
        bounded_velocity=lambda parameter, minimum, maximum: parameter,
    )
    args = SimpleNamespace(
        misfit_lowpass_hz="5,8,10,15",
        l2_gradient_sigma_cells="4,3,2,1",
        l2_huber_tv_weights="0,0,1e-4,5e-5",
        l2_huber_tv_delta=1.0e-3,
        l2_illumination_preconditioner="none",
        l2_illumination_max_gain=3.0,
        l2_iterations=4,
        min_vel=None,
        max_vel=None,
    )

    conditioner = build_l2_stage_conditioner(runner, args=args, geometry_context={})

    assert conditioner is not None
    assert conditioner.huber_tv_weights == [0.0, 0.0, 1.0e-4, 5.0e-5]


def test_posterior_guard_writes_l2_fallback_for_worse_asvgd(tmp_path: Path) -> None:
    l2_velocity = np.full((3, 3), 3000.0, dtype=np.float32)
    asvgd_velocity = np.full((3, 3), 3100.0, dtype=np.float32)
    l2_prediction = np.full((1, 2, 4), 1.0, dtype=np.float32)
    asvgd_prediction = np.full((1, 2, 4), 2.0, dtype=np.float32)
    np.savez(
        tmp_path / "comparison_result.npz",
        l2_velocity=l2_velocity,
        svdg_bayes_best_particle_velocity=asvgd_velocity,
        l2_prediction=l2_prediction,
        svdg_bayes_prediction=asvgd_prediction,
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "receiver_nmse_l2_fwi": 0.01,
                    "receiver_nmse_svdg_bayes_best_particle": 0.02,
                    "velocity_rel_l2_l2_fwi": 0.1,
                    "velocity_rel_l2_svdg_bayes_best_particle": 0.2,
                    "svdg_bayes_estimator": "best_particle",
                },
                "outputs": {},
            }
        )
    )

    report = apply_posterior_l2_guard(tmp_path, tolerance=0.02, runner=None)

    assert report["selected"] == "l2"
    assert (tmp_path / "posterior_l2_guard.json").is_file()
    with np.load(tmp_path / "comparison_result_l2_guarded.npz") as guarded:
        np.testing.assert_allclose(guarded["selected_velocity"], l2_velocity)
        np.testing.assert_allclose(guarded["raw_asvgd_selected_velocity"], asvgd_velocity)


def test_bottom_receiver_mse_uses_only_the_appended_receiver_rows() -> None:
    pred = torch.tensor([[[1.0], [1.0], [10.0], [10.0]]])
    target = torch.zeros_like(pred)

    assert bottom_receiver_mse(pred, target, bottom_receivers=2).item() == pytest.approx(100.0)


def test_patch_legacy_map_log_format_removes_float_format_from_mutable_loss() -> None:
    runner = load_saved_runner()

    assert patch_legacy_map_log_format(runner) is True
    instructions = list(dis.get_instructions(runner.run))
    loss_format = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.opname == "NOP"
        and instructions[index - 1].opname == "FORMAT_SIMPLE"
        and instructions[index - 2].argval == "iteration_loss"
    )
    assert instructions[loss_format - 1].positions.lineno == 1349


def test_patch_l2_optimizer_uses_adam_and_ignores_legacy_lbfgs_keywords() -> None:
    fake_torch = SimpleNamespace(optim=SimpleNamespace(LBFGS=object(), Adam=torch.optim.Adam))
    runner = SimpleNamespace(torch=fake_torch)
    parameter = torch.nn.Parameter(torch.ones(1))

    patch_l2_optimizer(runner, l2_optimizer="adam")
    optimizer = fake_torch.optim.LBFGS(
        [parameter],
        lr=0.03,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_change=1.0e-16,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.03)


def test_append_bottom_receivers_places_requested_receivers_above_bottom_boundary() -> None:
    receiver_locs = np.asarray([[2, 1], [2, 4]], dtype=np.int64)
    receiver_x_m = np.asarray([10.0, 40.0], dtype=np.float32)
    receiver_z_m = np.asarray([20.0, 20.0], dtype=np.float32)

    locations, x_m, z_m = append_bottom_receivers(
        receiver_locs,
        receiver_x_m,
        receiver_z_m,
        nx=6,
        nz=8,
        dx_m=10.0,
        pad_cells=3,
        bottom_receivers=3,
        receiver_margin_m=10.0,
        receiver_depth_m=20.0,
    )

    assert locations.shape == (5, 2)
    np.testing.assert_allclose(x_m[-3:], [10.0, 25.0, 40.0])
    np.testing.assert_allclose(z_m[-3:], [50.0, 50.0, 50.0])
    np.testing.assert_array_equal(locations[-3:, 0], [8, 8, 8])
    np.testing.assert_array_equal(locations[-3:, 1], [4, 5, 7])
