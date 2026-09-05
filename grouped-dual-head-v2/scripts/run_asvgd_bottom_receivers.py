#!/usr/bin/env python3
"""Launch the preserved ASVGD FWI runner with bottom receiver acquisition.

The editable historical runner is currently truncated, while its Python 3.13
bytecode cache remains executable.  This wrapper keeps that validated runner
read-only and changes only the acquisition geometry: former bottom source
positions become bottom receiver positions.
"""

from __future__ import annotations

import argparse
import dis
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as torch_f


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fwi_conditioning import (
    gaussian_smooth_2d,
    huber_tv_slowness_squared,
    illumination_gain,
    parse_stage_schedule,
    ray_illumination,
    select_l2_guard,
)


SAVED_RUNNER = ROOT / "scripts/__pycache__/run_marmousi_44shot_svdg_bayes_comparison.cpython-313.pyc"


@dataclass
class L2StageConditioner:
    """Apply optional, frequency-stage-specific conditioning to an L2 gradient."""

    total_iterations: int
    gradient_sigmas: list[float]
    huber_tv_weights: list[float]
    huber_tv_delta: float
    illumination_mode: str
    illumination_max_gain: float
    geometry_context: dict[str, Any]
    bounded_velocity: Callable[[torch.Tensor], torch.Tensor]
    reference_velocity_mps: Callable[[], float]
    iteration: int = 0
    _illumination_gain: torch.Tensor | None = None

    def _stage(self) -> int:
        stage_count = max(len(self.gradient_sigmas), len(self.huber_tv_weights), 1)
        return min(stage_count - 1, int(self.iteration * stage_count / max(int(self.total_iterations), 1)))

    def _gain(self, parameter: torch.Tensor) -> torch.Tensor | None:
        if self.illumination_mode == "none":
            return None
        if self._illumination_gain is None:
            required = {"nz", "nx", "pad_cells", "source_locs", "receiver_locs"}
            missing = required.difference(self.geometry_context)
            if missing:
                raise RuntimeError(f"L2 illumination needs geometry fields: {sorted(missing)}")
            sources = np.asarray(self.geometry_context["source_locs"], dtype=np.int64) - int(
                self.geometry_context["pad_cells"]
            )
            receivers = np.asarray(self.geometry_context["receiver_locs"], dtype=np.int64) - int(
                self.geometry_context["pad_cells"]
            )
            illumination = ray_illumination(
                int(self.geometry_context["nz"]),
                int(self.geometry_context["nx"]),
                source_ij=sources,
                receiver_ij=receivers,
            )
            self._illumination_gain = illumination_gain(
                illumination,
                max_gain=float(self.illumination_max_gain),
            )
        return self._illumination_gain.to(device=parameter.device, dtype=parameter.dtype)

    def apply(self, parameters: Iterable[torch.Tensor]) -> None:
        """Precondition gradients after a closure has evaluated the data misfit."""
        stage = self._stage()
        sigma = self.gradient_sigmas[min(stage, len(self.gradient_sigmas) - 1)] if self.gradient_sigmas else 0.0
        tv_weight = self.huber_tv_weights[min(stage, len(self.huber_tv_weights) - 1)] if self.huber_tv_weights else 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            if parameter.grad.ndim != 2:
                raise ValueError("stage-conditioned L2 expects a two-dimensional velocity parameter")
            conditioned = gaussian_smooth_2d(parameter.grad, sigma_cells=float(sigma))
            gain = self._gain(parameter)
            if gain is not None:
                conditioned = conditioned * gain
            parameter.grad.copy_(conditioned)
            if float(tv_weight) > 0.0:
                with torch.enable_grad():
                    velocity = self.bounded_velocity(parameter)
                    regularizer = huber_tv_slowness_squared(
                        velocity,
                        reference_velocity_mps=float(self.reference_velocity_mps()),
                        delta=float(self.huber_tv_delta),
                    )
                    regularizer_gradient = torch.autograd.grad(regularizer, parameter)[0]
                parameter.grad.add_(regularizer_gradient, alpha=float(tv_weight))
        self.iteration += 1


class StageConditionedAdam(torch.optim.Adam):
    """Adam whose closure gradient can be altered before each parameter update."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        conditioner: L2StageConditioner | None,
    ) -> None:
        super().__init__(params, lr=float(lr))
        self.conditioner = conditioner

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.conditioner is not None:
            parameters = [parameter for group in self.param_groups for parameter in group["params"]]
            self.conditioner.apply(parameters)
        super().step()
        return loss


def make_l2_optimizer_factory(
    *,
    original_lbfgs: Callable[..., Any],
    adam_class: type[torch.optim.Adam],
    conditioner: L2StageConditioner | None,
) -> Callable[..., Any]:
    """Return a factory that substitutes only the runner's first LBFGS instance."""
    factory_calls = 0

    def first_call_is_l2(params: Any, lr: float, **kwargs: Any) -> Any:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            # StageConditionedAdam subclasses torch.optim.Adam; the runner's
            # Adam class is checked for compatibility before this factory is installed.
            if adam_class is not torch.optim.Adam:
                raise TypeError("the preserved runner must expose torch.optim.Adam")
            return StageConditionedAdam(params, lr=float(lr), conditioner=conditioner)
        return original_lbfgs(params, lr=float(lr), **kwargs)

    return first_call_is_l2


def append_bottom_receivers(
    receiver_locs: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_z_m: np.ndarray,
    *,
    nx: int,
    nz: int,
    dx_m: float,
    pad_cells: int,
    bottom_receivers: int,
    receiver_margin_m: float,
    receiver_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append a horizontal receiver line above the physical bottom boundary."""
    count = int(bottom_receivers)
    if count < 0:
        raise ValueError("bottom_receivers must be non-negative")
    if count == 0:
        return receiver_locs, receiver_x_m, receiver_z_m
    width_m = (int(nx) - 1) * float(dx_m)
    height_m = (int(nz) - 1) * float(dx_m)
    if not 0.0 <= float(receiver_margin_m) < 0.5 * width_m:
        raise ValueError("receiver_margin_m must lie within the model width")
    if not 0.0 < float(receiver_depth_m) < height_m:
        raise ValueError("receiver_depth_m must lie within the model height")
    bottom_x_m = np.linspace(
        float(receiver_margin_m),
        width_m - float(receiver_margin_m),
        count,
        dtype=np.float32,
    )
    bottom_z_m = np.full(count, height_m - float(receiver_depth_m), dtype=np.float32)
    bottom_ix = np.clip(np.rint(bottom_x_m / float(dx_m)).astype(np.int64), 0, int(nx) - 1)
    bottom_iz = np.clip(np.rint(bottom_z_m / float(dx_m)).astype(np.int64), 0, int(nz) - 1)
    bottom_locs = np.column_stack([bottom_iz + int(pad_cells), bottom_ix + int(pad_cells)]).astype(np.int64)
    return (
        np.concatenate([np.asarray(receiver_locs, dtype=np.int64), bottom_locs], axis=0),
        np.concatenate([np.asarray(receiver_x_m, dtype=np.float32), bottom_x_m]),
        np.concatenate([np.asarray(receiver_z_m, dtype=np.float32), bottom_z_m]),
    )


def bottom_receiver_mse(pred: torch.Tensor, target: torch.Tensor, *, bottom_receivers: int) -> torch.Tensor:
    """Return MSE over the bottom receivers appended to the acquisition array."""
    count = int(bottom_receivers)
    if count <= 0:
        raise ValueError("bottom_receivers must be positive")
    if count > int(pred.shape[1]) or count > int(target.shape[1]):
        raise ValueError("bottom_receivers exceeds the receiver dimension")
    return torch_f.mse_loss(pred[:, -count:, :], target[:, -count:, :])


def load_saved_runner() -> Any:
    if not SAVED_RUNNER.is_file():
        raise FileNotFoundError(f"saved ASVGD runner bytecode is missing: {SAVED_RUNNER}")
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("saved_asvgd_runner", SAVED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load saved ASVGD runner: {SAVED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_legacy_map_log_format(runner: Any) -> bool:
    """Prevent the frozen runner's MAP logger from formatting a mutable loss list as a float."""
    code = runner.run.__code__
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if index < 2 or instruction.opname != "FORMAT_WITH_SPEC":
            continue
        value_instruction = instructions[index - 2]
        spec_instruction = instructions[index - 1]
        if (
            value_instruction.opname != "LOAD_DEREF"
            or value_instruction.argval != "iteration_loss"
            or spec_instruction.opname != "LOAD_CONST"
            or spec_instruction.argval != ".6e"
        ):
            continue
        patched_code = bytearray(code.co_code)
        patched_code[spec_instruction.offset] = dis.opmap["FORMAT_SIMPLE"]
        patched_code[instruction.offset] = dis.opmap["NOP"]
        runner.run.__code__ = code.replace(co_code=bytes(patched_code))
        return True
    return False


def build_l2_stage_conditioner(
    runner: Any,
    *,
    args: argparse.Namespace,
    geometry_context: dict[str, Any],
) -> L2StageConditioner | None:
    """Create an opt-in L2 gradient conditioner from the parsed experiment args."""
    stage_count = max(len(runner.parse_float_schedule(str(args.misfit_lowpass_hz))), 1)
    gradient_sigmas = parse_stage_schedule(
        str(args.l2_gradient_sigma_cells),
        stages=stage_count,
        option="--l2-gradient-sigma-cells",
    )
    huber_tv_weights = parse_stage_schedule(
        str(args.l2_huber_tv_weights),
        stages=stage_count,
        option="--l2-huber-tv-weights",
    )
    if any(value < 0.0 for value in gradient_sigmas):
        raise ValueError("--l2-gradient-sigma-cells values must be non-negative")
    if any(value < 0.0 for value in huber_tv_weights):
        raise ValueError("--l2-huber-tv-weights values must be non-negative")
    if float(args.l2_huber_tv_delta) <= 0.0:
        raise ValueError("--l2-huber-tv-delta must be positive")
    if not gradient_sigmas and not huber_tv_weights and str(args.l2_illumination_preconditioner) == "none":
        return None

    def bounds() -> tuple[float, float]:
        minimum = args.min_vel if args.min_vel is not None else geometry_context.get("min_vel")
        maximum = args.max_vel if args.max_vel is not None else geometry_context.get("max_vel")
        if minimum is None or maximum is None:
            raise RuntimeError("the preserved runner did not establish L2 velocity bounds before conditioning")
        return float(minimum), float(maximum)

    def bounded(parameter: torch.Tensor) -> torch.Tensor:
        minimum, maximum = bounds()
        return runner.bounded_velocity(parameter, minimum, maximum)

    def reference_velocity() -> float:
        minimum, maximum = bounds()
        return 0.5 * (minimum + maximum)

    return L2StageConditioner(
        total_iterations=int(args.l2_iterations),
        gradient_sigmas=gradient_sigmas,
        huber_tv_weights=huber_tv_weights,
        huber_tv_delta=float(args.l2_huber_tv_delta),
        illumination_mode=str(args.l2_illumination_preconditioner),
        illumination_max_gain=float(args.l2_illumination_max_gain),
        geometry_context=geometry_context,
        bounded_velocity=bounded,
        reference_velocity_mps=reference_velocity,
    )


def patch_l2_optimizer(
    runner: Any,
    *,
    l2_optimizer: str,
    conditioner: L2StageConditioner | None = None,
) -> None:
    """Select L2 Adam without silently replacing the later MAP LBFGS instance."""
    if l2_optimizer == "lbfgs":
        if conditioner is not None:
            raise ValueError("stage conditioning requires --l2-optimizer adam")
        return
    if l2_optimizer != "adam":
        raise ValueError(f"unsupported L2 optimizer: {l2_optimizer!r}")
    runner.torch.optim.LBFGS = make_l2_optimizer_factory(
        original_lbfgs=runner.torch.optim.LBFGS,
        adam_class=runner.torch.optim.Adam,
        conditioner=conditioner,
    )


def patch_velocity_bounds_capture(runner: Any, *, geometry_context: dict[str, Any]) -> None:
    """Record the concrete bounds chosen internally by the preserved runner."""
    original_bounded_velocity = runner.bounded_velocity

    def bounded_velocity_with_capture(parameter: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        geometry_context["min_vel"] = float(minimum)
        geometry_context["max_vel"] = float(maximum)
        return original_bounded_velocity(parameter, minimum, maximum)

    runner.bounded_velocity = bounded_velocity_with_capture


def patch_bottom_receiver_geometry(
    runner: Any,
    *,
    bottom_receivers: int,
    geometry_context: dict[str, Any] | None = None,
) -> None:
    """Replace the runner geometry factory while retaining all inversion code."""
    original_geometry = runner.make_marmousi_geometry

    def geometry_with_bottom_receivers(**kwargs: Any):
        requested_layout = str(kwargs["receiver_layout"])
        if not requested_layout.endswith("_plus_bottom"):
            return original_geometry(**kwargs)
        base_layout = requested_layout.removesuffix("_plus_bottom")
        if base_layout not in {"surface", "surface_plus_sides"}:
            raise ValueError(f"unsupported bottom-receiver base layout: {base_layout!r}")
        base_kwargs = dict(kwargs)
        base_kwargs["receiver_layout"] = base_layout
        source_locs, receiver_locs, source_x_m, source_z_m, receiver_x_m, receiver_z_m = original_geometry(**base_kwargs)
        receiver_locs, receiver_x_m, receiver_z_m = append_bottom_receivers(
            receiver_locs,
            receiver_x_m,
            receiver_z_m,
            nx=int(kwargs["nx"]),
            nz=int(kwargs["nz"]),
            dx_m=float(kwargs["dx_m"]),
            pad_cells=int(kwargs["pad_cells"]),
            bottom_receivers=int(bottom_receivers),
            receiver_margin_m=float(kwargs["receiver_margin_m"]),
            receiver_depth_m=float(kwargs["top_receiver_depth_m"]),
        )
        if geometry_context is not None:
            geometry_context.update(
                {
                    "nz": int(kwargs["nz"]),
                    "nx": int(kwargs["nx"]),
                    "pad_cells": int(kwargs["pad_cells"]),
                    "source_locs": np.asarray(source_locs, dtype=np.int64),
                    "receiver_locs": np.asarray(receiver_locs, dtype=np.int64),
                }
            )
        return source_locs, receiver_locs, source_x_m, source_z_m, receiver_x_m, receiver_z_m

    runner.make_marmousi_geometry = geometry_with_bottom_receivers


def patch_bottom_receiver_misfit(runner: Any, *, bottom_receivers: int, bottom_receiver_weight: float) -> None:
    """Add a balanced loss term for the receiver line appended at the bottom."""
    if float(bottom_receiver_weight) == 0.0:
        return
    original_receiver_misfit = runner.receiver_misfit

    def receiver_misfit_with_bottom_weight(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        obs_scale: torch.Tensor,
        dt_s: float,
        lowpass_hz: float | None,
        envelope_weight: float,
        ncc_weight: float,
        time_weight_power: float,
        late_time_start_s: float,
        late_time_weight: float,
    ) -> torch.Tensor:
        loss = original_receiver_misfit(
            pred,
            target,
            obs_scale=obs_scale,
            dt_s=dt_s,
            lowpass_hz=lowpass_hz,
            envelope_weight=envelope_weight,
            ncc_weight=ncc_weight,
            time_weight_power=time_weight_power,
            late_time_start_s=late_time_start_s,
            late_time_weight=late_time_weight,
        )
        pred_scaled = runner.lowpass_time(pred / obs_scale, cutoff_hz=lowpass_hz, dt_s=dt_s)
        target_scaled = runner.lowpass_time(target / obs_scale, cutoff_hz=lowpass_hz, dt_s=dt_s)
        time_weight = runner.make_time_weight(
            nt=int(pred_scaled.shape[-1]),
            dt_s=dt_s,
            device=pred_scaled.device,
            dtype=pred_scaled.dtype,
            time_weight_power=time_weight_power,
            late_time_start_s=late_time_start_s,
            late_time_weight=late_time_weight,
        )
        if time_weight is not None:
            pred_scaled = pred_scaled * time_weight
            target_scaled = target_scaled * time_weight
        return loss + float(bottom_receiver_weight) * bottom_receiver_mse(
            pred_scaled,
            target_scaled,
            bottom_receivers=bottom_receivers,
        )

    runner.receiver_misfit = receiver_misfit_with_bottom_weight


def _posterior_velocity_key(data: Any, estimator: str) -> str:
    names = (
        f"svdg_bayes_{estimator}_velocity",
        f"svgd_bayes_{estimator}_velocity",
    )
    for name in names:
        if name in data.files:
            return name
    raise KeyError(f"no posterior velocity array found for estimator {estimator!r}")


def apply_posterior_l2_guard(
    output_dir: Path,
    *,
    tolerance: float,
    runner: Any | None,
) -> dict[str, Any]:
    """Write a separate, loss-guarded L2/ASVGD recommendation after a run."""
    if float(tolerance) < 0.0:
        raise ValueError("posterior L2 guard tolerance must be non-negative")
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    result_path = output_dir / "comparison_result.npz"
    summary = json.loads(summary_path.read_text())
    metrics = summary["metrics"]
    estimator = str(metrics.get("svdg_bayes_estimator", "best_particle"))
    l2_nmse = float(metrics["receiver_nmse_l2_fwi"])
    candidate_nmse_key = f"receiver_nmse_svdg_bayes_{estimator}"
    candidate_nmse = float(metrics[candidate_nmse_key])
    decision = select_l2_guard(
        l2_nmse=l2_nmse,
        candidate_nmse=candidate_nmse,
        tolerance=float(tolerance),
    )

    with np.load(result_path) as data:
        candidate_velocity_key = _posterior_velocity_key(data, estimator)
        l2_velocity = np.asarray(data["l2_velocity"])
        candidate_velocity = np.asarray(data[candidate_velocity_key])
        l2_prediction = np.asarray(data["l2_prediction"])
        candidate_prediction = np.asarray(data["svdg_bayes_prediction"])
        selected_velocity = l2_velocity if decision.selected == "l2" else candidate_velocity
        selected_prediction = l2_prediction if decision.selected == "l2" else candidate_prediction
        guarded_result_path = output_dir / "comparison_result_l2_guarded.npz"
        np.savez_compressed(
            guarded_result_path,
            selected_velocity=selected_velocity,
            selected_prediction=selected_prediction,
            raw_asvgd_selected_velocity=candidate_velocity,
            raw_asvgd_selected_prediction=candidate_prediction,
            selection=np.asarray([decision.selected]),
        )

        guarded_figure_paths: dict[str, str] = {}
        required_figure_arrays = {
            "true_velocity",
            "initial_velocity",
            "svdg_map_velocity",
            "svdg_bayes_std_velocity",
            "source_x_m",
            "source_z_m",
            "receiver_x_m",
            "receiver_z_m",
        }
        if runner is not None and required_figure_arrays.issubset(set(data.files)):
            label = "L2 guard fallback" if decision.selected == "l2" else f"ASVGD {estimator} accepted"
            comparison_path = output_dir / "l2_guarded_imaging_comparison.png"
            runner.save_comparison_figure(
                comparison_path,
                true_v=np.asarray(data["true_velocity"]),
                init_v=np.asarray(data["initial_velocity"]),
                l2_v=l2_velocity,
                map_v=np.asarray(data["svdg_map_velocity"]),
                bayes_mean_v=selected_velocity,
                bayes_std_v=np.asarray(data["svdg_bayes_std_velocity"]),
                dx_m=float(summary["dx_m"]),
                metrics=metrics,
                model_label=str(summary["model_label"]),
                source_x_m=np.asarray(data["source_x_m"]),
                source_z_m=np.asarray(data["source_z_m"]),
                receiver_x_m=np.asarray(data["receiver_x_m"]),
                receiver_z_m=np.asarray(data["receiver_z_m"]),
                bayes_label=label,
            )
            gather_path = output_dir / "l2_guarded_receiver_gather.png"
            runner.save_gather_comparison(
                gather_path,
                observed=np.asarray(data["observed"]),
                l2_pred=l2_prediction,
                bayes_pred=selected_prediction,
                source_x_m=np.asarray(data["source_x_m"]),
                dt_s=float(summary["dt_s"]),
                bayes_label=label,
            )
            guarded_figure_paths = {
                "guarded_imaging_comparison": str(comparison_path.resolve()),
                "guarded_receiver_gather": str(gather_path.resolve()),
            }

    selected_rel_l2_key = "velocity_rel_l2_l2_fwi" if decision.selected == "l2" else f"velocity_rel_l2_svdg_bayes_{estimator}"
    report = {
        "selected": decision.selected,
        "candidate_estimator": estimator,
        "l2_receiver_nmse": decision.l2_nmse,
        "candidate_receiver_nmse": decision.candidate_nmse,
        "allowed_candidate_receiver_nmse": decision.allowed_nmse,
        "relative_nmse_tolerance": float(tolerance),
        "selected_velocity_rel_l2": float(metrics[selected_rel_l2_key]),
        "guarded_result": str(guarded_result_path.resolve()),
        **guarded_figure_paths,
    }
    guard_path = output_dir / "posterior_l2_guard.json"
    guard_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary["posterior_l2_guard"] = report
    summary.setdefault("outputs", {}).update(
        {
            "posterior_l2_guard": str(guard_path.resolve()),
            "comparison_result_l2_guarded": str(guarded_result_path.resolve()),
            **guarded_figure_paths,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--bottom-receivers", type=int, required=True)
    wrapper_parser.add_argument("--bottom-receiver-weight", type=float, default=0.0)
    wrapper_parser.add_argument("--l2-optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    wrapper_args, runner_argv = wrapper_parser.parse_known_args(argv)
    if int(wrapper_args.bottom_receivers) <= 0:
        raise ValueError("--bottom-receivers must be positive")
    if float(wrapper_args.bottom_receiver_weight) < 0.0:
        raise ValueError("--bottom-receiver-weight must be non-negative")
    runner = load_saved_runner()
    if not patch_legacy_map_log_format(runner):
        raise RuntimeError("unable to patch the saved runner's known MAP log-format defect")
    parser = runner.build_arg_parser()
    parser.add_argument(
        "--l2-gradient-sigma-cells",
        default="",
        help="Optional one/four stage-wise Gaussian sigmas applied to the L2 gradient, in grid cells.",
    )
    parser.add_argument(
        "--l2-huber-tv-weights",
        default="",
        help="Optional one/four stage-wise Huber-TV weights on squared slowness for conditioned L2 Adam.",
    )
    parser.add_argument(
        "--l2-huber-tv-delta",
        type=float,
        default=1.0e-3,
        help="Huber transition for the squared-slowness TV penalty.",
    )
    parser.add_argument(
        "--l2-illumination-preconditioner",
        choices=("none", "ray"),
        default="none",
        help="Optional static ray-density illumination approximation for the L2 gradient.",
    )
    parser.add_argument(
        "--l2-illumination-max-gain",
        type=float,
        default=3.0,
        help="Maximum inverse-square-root gain for the ray-density preconditioner.",
    )
    parser.add_argument(
        "--posterior-l2-guard-relative-nmse",
        type=float,
        default=0.02,
        help="Allow ASVGD to replace L2 only within this relative receiver-NMSE tolerance.",
    )
    args = parser.parse_args(runner_argv)
    if str(args.source_layout) != "surface":
        raise ValueError("bottom-receiver acquisition requires top/surface sources only")
    if str(args.receiver_layout) not in {"surface", "surface_plus_sides"}:
        raise ValueError("receiver layout must be surface or surface_plus_sides")
    args.receiver_layout = f"{args.receiver_layout}_plus_bottom"
    args.bottom_receivers = int(wrapper_args.bottom_receivers)
    args.bottom_receiver_weight = float(wrapper_args.bottom_receiver_weight)
    args.l2_optimizer = str(wrapper_args.l2_optimizer)
    geometry_context: dict[str, Any] = {}
    conditioner = build_l2_stage_conditioner(runner, args=args, geometry_context=geometry_context)
    patch_velocity_bounds_capture(runner, geometry_context=geometry_context)
    patch_l2_optimizer(
        runner,
        l2_optimizer=str(wrapper_args.l2_optimizer),
        conditioner=conditioner,
    )
    patch_bottom_receiver_geometry(
        runner,
        bottom_receivers=int(wrapper_args.bottom_receivers),
        geometry_context=geometry_context,
    )
    patch_bottom_receiver_misfit(
        runner,
        bottom_receivers=int(wrapper_args.bottom_receivers),
        bottom_receiver_weight=float(wrapper_args.bottom_receiver_weight),
    )
    if conditioner is not None:
        print(
            "[INFO] Stage-conditioned Adam enabled: "
            f"sigma={conditioner.gradient_sigmas or [0.0]}, "
            f"huber_tv={conditioner.huber_tv_weights or [0.0]}, "
            f"illumination={conditioner.illumination_mode}"
        )
    runner.run(args)
    guard_report = apply_posterior_l2_guard(
        Path(args.output_dir),
        tolerance=float(args.posterior_l2_guard_relative_nmse),
        runner=runner,
    )
    print(
        "[INFO] Posterior L2 guard: "
        f"selected={guard_report['selected']}, "
        f"candidate_nmse={guard_report['candidate_receiver_nmse']:.6e}, "
        f"l2_nmse={guard_report['l2_receiver_nmse']:.6e}"
    )


if __name__ == "__main__":
    main()
