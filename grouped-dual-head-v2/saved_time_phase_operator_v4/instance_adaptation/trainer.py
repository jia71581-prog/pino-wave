"""Deterministic two-stage optimizer and no-future acceptance gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import time

import torch
from torch.func import jacfwd, vmap
from torch.nn import functional as F

from .adapters import OnsetAdaptedV5
from .bridge import BridgeResult, make_onset_bridge
from .chonknoris import (
    ReducedChonknorisConfig,
    ReducedChonknorisResult,
    reduced_chonknoris_solve,
)
from .contracts import future_indices
from .data_guard import GuardedOnsetRecord
from .losses import (
    LossWeights,
    PhysicsPointSet,
    PhysicsSampling,
    build_fixed_physics_points,
    build_rad_physics_points,
    build_r3_rams_physics_points,
    instance_loss_terms,
    lwc84_residual,
    sample_fixed_physics_residual,
)


@dataclass
class AdaptationResult:
    accepted: bool
    rollback_reason: str | None
    steps: int
    lbfgs_closure_calls: int
    elapsed_s: float
    trainable_parameter_count: int
    total_parameter_count: int
    accessed_true_indices: tuple[int, ...]
    future_truth_used: bool
    baseline_physics: float
    candidate_physics: float
    energy_ratio: float
    adapter_state: dict[str, torch.Tensor]
    optimizer_name: str = "adam_lbfgs"
    residual_history: tuple[float, ...] = ()
    relaxation_history: tuple[float, ...] = ()
    step_size_history: tuple[float, ...] = ()
    contraction_history: tuple[float, ...] = ()
    condition_history: tuple[float, ...] = ()
    learned_factor_steps: int = 0
    exact_factor_fallbacks: int = 0
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["adapter_state"] = {
            key: value.detach().cpu() for key, value in self.adapter_state.items()
        }
        return payload


def _batched_record(record: GuardedOnsetRecord, device: torch.device):
    velocity = record.velocity_mps.to(device)
    if velocity.ndim == 3:
        velocity = velocity.unsqueeze(0)
    elif velocity.ndim != 4:
        raise ValueError("record velocity must be [1,z,x] or [record,1,z,x]")
    return (
        velocity,
        record.source_parameters.to(device).unsqueeze(0),
        record.observed_wavefield.to(device).unsqueeze(0),
        record.time_s.to(device),
    )


def _adapter_state(model: OnsetAdaptedV5) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("conditioner.")
        or name.startswith("residual.")
        or name == "latent_delta"
        or name == "residual_gate"
    }


def _restore_adapter_state(model: OnsetAdaptedV5, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update({name: value.to(current[name].device) for name, value in state.items()})
    model.load_state_dict(current, strict=True)


def _zero_residual_state(model: OnsetAdaptedV5) -> dict[str, torch.Tensor]:
    """Return a safe zero-correction state while retaining the conditioner."""
    return {
        name: torch.zeros_like(value.detach())
        for name, value in model.residual.state_dict().items()
    }


def _restore_residual_state(model: OnsetAdaptedV5, state: dict[str, torch.Tensor]) -> None:
    model.residual.load_state_dict(
        {name: value.to(next(model.parameters()).device) for name, value in state.items()},
        strict=True,
    )


def build_chonknoris_residual_function(
    model: OnsetAdaptedV5,
    parent: torch.Tensor,
    velocity: torch.Tensor,
    source: torch.Tensor,
    observed: torch.Tensor,
    time_s: torch.Tensor,
    points: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    pde_weight: float,
    pool_size: int,
    pde_normalization_scale: torch.Tensor | None = None,
) -> callable:
    """Build the online residual map using no future ground-truth frames.

    The observation block is spatially pooled before linearization so the
    residual Jacobian remains small.  The LWC-84 block is already sampled at a
    fixed, auditable point set and self-normalized, so it is valid in the
    normalizer's O(1) deployment space.
    """

    spatial_size = max(1, min(int(pool_size), *parent.shape[-2:]))
    selected = list(observed_indices)
    observed_reference = observed.detach()
    parent_observed = parent[:, selected].detach()
    observed_scale = torch.maximum(
        observed_reference.float().square().mean().sqrt(),
        parent_observed.float().square().mean().sqrt(),
    ).clamp_min(1.0e-3)
    physics_weight = max(0.0, float(pde_weight))
    dt = float(time_s[1] - time_s[0])
    cached_context = model.chonknoris_context(
        velocity, source, observed
    ).detach()

    def residual_fn(state: torch.Tensor) -> torch.Tensor:
        raw = model.raw_wavefield_from_state(
            parent,
            velocity,
            source,
            observed,
            time_s,
            state,
            context=cached_context,
        )
        observed_error = F.adaptive_avg_pool2d(
            raw[:, selected] - observed_reference,
            output_size=(spatial_size, spatial_size),
        )
        blocks = [(observed_error / observed_scale).reshape(-1)]
        if physics_weight > 0.0:
            residual = lwc84_residual(
                raw,
                velocity,
                dt=dt,
                dx=10.0,
                dz=10.0,
                observed_indices=observed_indices,
                normalization_scale=pde_normalization_scale,
            )
            sampled = sample_fixed_physics_residual(
                residual, points, observed_indices
            )
            blocks.append(math.sqrt(physics_weight) * sampled.reshape(-1))
        # The weak Tikhonov state residual makes the reduced normal matrix
        # identifiable even when both onset frames are nearly zero.
        blocks.append(state.reshape(-1) * math.sqrt(1.0e-5))
        return torch.cat(blocks)

    return residual_fn


def build_supervised_chonknoris_residual_function(
    model: OnsetAdaptedV5,
    parent: torch.Tensor,
    velocity: torch.Tensor,
    source: torch.Tensor,
    observed: torch.Tensor,
    time_s: torch.Tensor,
    target: torch.Tensor,
    *,
    pool_size: int = 4,
    residual_sampling: str = "average_pool",
) -> callable:
    """Build the train-only flow residual used for Cholesky supervision.

    Future target frames are legal only in offline train episodes.  They define
    a well-conditioned proxy for the curvature of the field objective; the
    target Cholesky factor itself is detached before the predictor is updated.
    """

    truth = torch.as_tensor(target, dtype=parent.dtype, device=parent.device)
    if truth.shape != parent.shape:
        raise ValueError("supervised CHONKNORIS target must match the parent field")
    spatial_size = max(1, min(int(pool_size), *parent.shape[-2:]))
    sampling = str(residual_sampling).strip().lower()
    if sampling not in {"average_pool", "topk"}:
        raise ValueError("supervised residual sampling must be average_pool or topk")
    scale = truth.detach().float().square().mean().sqrt().clamp_min(1.0e-3)
    cached_context = model.chonknoris_context(
        velocity, source, observed
    ).detach()
    high_indices = None
    if sampling == "topk":
        with torch.no_grad():
            initial_raw = model.raw_wavefield_from_state(
                parent,
                velocity,
                source,
                observed,
                time_s,
                model.deployment_state().detach(),
                context=cached_context,
            )
            high_indices = high_residual_spatial_indices(
                initial_raw - truth, points_per_frame=spatial_size * spatial_size
            )

    def residual_fn(state: torch.Tensor) -> torch.Tensor:
        raw = model.raw_wavefield_from_state(
            parent,
            velocity,
            source,
            observed,
            time_s,
            state,
            context=cached_context,
        )
        field_error = raw - truth
        if high_indices is None:
            field_error = F.adaptive_avg_pool2d(
                field_error, output_size=(spatial_size, spatial_size)
            )
        else:
            field_error = torch.gather(
                field_error.flatten(-2), -1, high_indices
            )
        return torch.cat(
            (
                (field_error / scale).reshape(-1),
                state.reshape(-1) * math.sqrt(1.0e-5),
            )
        )

    return residual_fn


def linearize_supervised_chonknoris_batch(
    model: OnsetAdaptedV5,
    parent: torch.Tensor,
    velocity: torch.Tensor,
    source: torch.Tensor,
    observed: torch.Tensor,
    time_s: torch.Tensor,
    target: torch.Tensor,
    *,
    pool_size: int = 4,
    residual_sampling: str = "average_pool",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorize independent train-only reduced Jacobians over real episodes."""

    truth = torch.as_tensor(target, dtype=parent.dtype, device=parent.device)
    if parent.ndim != 4 or truth.shape != parent.shape:
        raise ValueError("batched parent and target must match [batch,time,z,x]")
    batch, saved_times = parent.shape[:2]
    if velocity.shape[:2] != (batch, 1):
        raise ValueError("batched velocity must be [batch,1,z,x]")
    if source.shape != (batch, 5):
        raise ValueError("batched source must be [batch,5]")
    if observed.shape[:2] != (batch, 2):
        raise ValueError("batched observations must be [batch,2,z,x]")
    times = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(batch, -1)
    if times.shape != (batch, saved_times):
        raise ValueError("batched saved times must be [batch,time]")
    spatial_size = max(1, min(int(pool_size), *parent.shape[-2:]))
    sampling = str(residual_sampling).strip().lower()
    if sampling not in {"average_pool", "topk"}:
        raise ValueError("supervised residual sampling must be average_pool or topk")
    contexts = model.chonknoris_context(velocity, source, observed).detach()
    states = model.deployment_state().detach().expand(batch, -1).clone()
    if sampling == "topk":
        with torch.no_grad():
            initial_raw = model.raw_wavefield_from_state(
                parent,
                velocity,
                source,
                observed,
                times,
                states,
                context=contexts,
            )
            high_indices = high_residual_spatial_indices(
                initial_raw - truth, points_per_frame=spatial_size * spatial_size
            )
    else:
        high_indices = torch.zeros(
            (batch, saved_times, spatial_size * spatial_size),
            dtype=torch.long,
            device=parent.device,
        )

    def single_residual(
        state: torch.Tensor,
        parent_i: torch.Tensor,
        velocity_i: torch.Tensor,
        source_i: torch.Tensor,
        observed_i: torch.Tensor,
        time_i: torch.Tensor,
        truth_i: torch.Tensor,
        context_i: torch.Tensor,
        high_indices_i: torch.Tensor,
    ) -> torch.Tensor:
        raw = model.raw_wavefield_from_state(
            parent_i.unsqueeze(0),
            velocity_i.unsqueeze(0),
            source_i.unsqueeze(0),
            observed_i.unsqueeze(0),
            time_i,
            state,
            context=context_i.unsqueeze(0),
            vmap_compatible=True,
        )[0]
        scale = truth_i.detach().float().square().mean().sqrt().clamp_min(1.0e-3)
        field_error = raw - truth_i
        if sampling == "average_pool":
            field_error = F.adaptive_avg_pool2d(
                field_error, output_size=(spatial_size, spatial_size)
            )
        else:
            field_error = torch.gather(
                field_error.flatten(-2), -1, high_indices_i
            )
        return torch.cat(
            (
                (field_error / scale).reshape(-1),
                state.reshape(-1) * math.sqrt(1.0e-5),
            )
        )

    arguments = (
        states,
        parent,
        velocity,
        source,
        observed,
        times,
        truth,
        contexts,
        high_indices,
    )
    residuals = vmap(single_residual)(*arguments)
    jacobians = vmap(jacfwd(single_residual, argnums=0))(*arguments)
    if not torch.isfinite(residuals).all() or not torch.isfinite(jacobians).all():
        raise FloatingPointError("vectorized CHONKNORIS linearization is nonfinite")
    return residuals, jacobians, contexts


def high_residual_spatial_indices(
    residual: torch.Tensor, *, points_per_frame: int
) -> torch.Tensor:
    """Select deterministic largest-|residual| spatial points in every frame."""

    value = torch.as_tensor(residual)
    if value.ndim not in (3, 4):
        raise ValueError("residual must be [time,z,x] or [batch,time,z,x]")
    count = int(points_per_frame)
    available = int(value.shape[-2] * value.shape[-1])
    if count <= 0 or count > available:
        raise ValueError("points_per_frame must lie inside the spatial grid")
    return value.detach().abs().flatten(-2).topk(
        count, dim=-1, largest=True, sorted=True
    ).indices


def project_residual_gate_to_observations(
    model: OnsetAdaptedV5,
    parent: torch.Tensor,
    velocity: torch.Tensor,
    source: torch.Tensor,
    observed: torch.Tensor,
    time_s: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    maximum_absolute_gate: float = 2.0,
) -> float:
    """Analytically minimize onset MSE along the learned residual direction.

    The latent state is held fixed and only the scalar residual gate is
    projected.  Gate zero is the exact frozen parent, so this causal projection
    cannot make the two observed frames worse in exact arithmetic and needs no
    future truth.
    """

    bound = float(maximum_absolute_gate)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("maximum absolute residual gate must be positive")
    state = model.deployment_state().detach().clone()
    unit_state = state.clone()
    unit_state[-1] = 1.0
    with torch.no_grad():
        unit_raw = model.raw_wavefield_from_state(
            parent,
            velocity,
            source,
            observed,
            time_s,
            unit_state,
        )
        selected = list(observed_indices)
        correction = (unit_raw - parent)[:, selected].float()
        baseline_error = (parent[:, selected] - observed).float()
        denominator = correction.square().sum()
        if not torch.isfinite(denominator) or float(denominator) <= 1.0e-20:
            projected = correction.new_zeros(())
        else:
            projected = -(baseline_error * correction).sum() / denominator
            projected = projected.clamp(-bound, bound)
        if not torch.isfinite(projected):
            projected = correction.new_zeros(())
        model.residual_gate.copy_(projected.to(model.residual_gate))
    return float(projected)


def adapt_instance(
    model: OnsetAdaptedV5,
    record: GuardedOnsetRecord,
    parent_field: torch.Tensor,
    *,
    conditioner=None,
    adam_steps: int = 12,
    lbfgs_steps: int = 4,
    learning_rate: float = 2.0e-4,
    seed: int = 17,
    loss_weights: LossWeights = LossWeights(),
    deployment_lora: bool = False,
    hard_project_observed: bool = False,
    deployment_pde_weight: float = 0.0,
    physics_sampling: PhysicsSampling = PhysicsSampling(),
    optimizer_name: str = "adam_lbfgs",
    chonknoris_config: ReducedChonknorisConfig | None = None,
    minimum_physics_improvement: float = 0.01,
    minimum_energy_ratio: float = 0.8,
    maximum_energy_ratio: float = 1.25,
) -> AdaptationResult:
    """Adapt only the small wrapper while reading exactly the two onset frames.

    When ``deployment_lora`` is true the conditioner and residual meta network
    are frozen and only the per-instance ``latent_delta`` is optimized, giving a
    second-scale deployment fine-tune whose safe rollback returns to the frozen
    meta-network prediction (``latent_delta = 0``) rather than a zero residual.
    """
    del conditioner
    if adam_steps < 0 or lbfgs_steps < 0 or learning_rate <= 0.0:
        raise ValueError("adaptation optimizer configuration is invalid")
    physics_improvement = float(minimum_physics_improvement)
    minimum_energy = float(minimum_energy_ratio)
    maximum_energy = float(maximum_energy_ratio)
    if (
        not math.isfinite(physics_improvement)
        or not 0.0 <= physics_improvement < 1.0
        or not math.isfinite(minimum_energy)
        or not math.isfinite(maximum_energy)
        or minimum_energy <= 0.0
        or minimum_energy > maximum_energy
    ):
        raise ValueError("adaptation acceptance-gate configuration is invalid")
    method = str(optimizer_name).strip().lower().replace("-", "_")
    if method not in {"adam_lbfgs", "chonknoris"}:
        raise ValueError("adaptation optimizer must be adam_lbfgs or chonknoris")
    if method == "chonknoris" and not deployment_lora:
        raise ValueError("reduced CHONKNORIS is defined for deployment LoRA mode")
    chonknoris_settings = (
        ReducedChonknorisConfig()
        if chonknoris_config is None
        else chonknoris_config
    )
    started = time.perf_counter()
    record.audit.read(record.observed_indices)
    device = next(model.parameters()).device
    velocity, source, observed, time_s = _batched_record(record, device)
    parent = torch.as_tensor(parent_field, dtype=torch.float32, device=device)
    if parent.ndim != 4 or parent.shape[0] != 1 or parent.shape[1] != len(time_s):
        raise ValueError("parent_field must be [1,saved_time,z,x]")
    indices = record.observed_indices
    physics_point_count = (
        int(chonknoris_settings.physics_point_count)
        if method == "chonknoris"
        else 512
    )
    points = build_fixed_physics_points(
        len(time_s), indices, count=physics_point_count, seed=seed
    )
    # RAD (residual-based adaptive distribution) sampling: draw collocation points
    # from the current PDE-residual field so gradient concentrates on the confirmed
    # failure modes (sharp wavefront + late frames).  ``uniform`` keeps the fixed
    # points above.  A per-instance seeded generator preserves determinism.
    sampling_method = str(physics_sampling.method).strip().lower().replace("-", "_")
    use_rad = sampling_method == "rad"
    use_r3_rams = sampling_method == "r3_rams"
    use_adaptive_sampling = use_rad or use_r3_rams
    rad_generator = (
        torch.Generator(device=device).manual_seed(int(seed))
        if use_adaptive_sampling else None
    )
    retained_r3_points: PhysicsPointSet | None = None
    freeze_points = False  # set before LBFGS so its closure stays fixed

    def _resample_physics(residual_field: torch.Tensor) -> torch.Tensor | PhysicsPointSet:
        nonlocal retained_r3_points
        if use_r3_rams:
            retained_r3_points = build_r3_rams_physics_points(
                residual_field,
                indices,
                count=physics_point_count,
                hard_quantile=float(physics_sampling.hard_quantile),
                release_quantile=float(physics_sampling.release_quantile),
                uniform_fraction=float(physics_sampling.uniform_fraction),
                rams_fraction=float(physics_sampling.rams_fraction),
                k=float(physics_sampling.k),
                c=float(physics_sampling.c),
                time_tilt=float(physics_sampling.time_tilt),
                rams_steps=int(physics_sampling.rams_steps),
                retained_points=retained_r3_points,
                generator=rad_generator,
            )
            return retained_r3_points
        return build_rad_physics_points(
            residual_field,
            indices,
            count=physics_point_count,
            k=float(physics_sampling.k),
            c=float(physics_sampling.c),
            time_tilt=float(physics_sampling.time_tilt),
            generator=rad_generator,
        )

    bridge: BridgeResult = make_onset_bridge(
        velocity,
        source,
        observed,
        indices,
        time_s,
        source_map=record.source_map.to(device).unsqueeze(0),
        steps=min(4, max(0, len(time_s) - indices[1] - 1)),
        device=device,
    )
    if deployment_lora:
        model.set_deployment_mode(True)
        safe_rollback_state = {
            "latent_delta": torch.zeros_like(model.latent_delta.detach()),
            "residual_gate": torch.zeros_like(model.residual_gate.detach()),
        }
    else:
        model._deployment_mode = False
        safe_rollback_state = _zero_residual_state(model)
    parameters = model.active_parameters()
    optimizer = (
        None
        if method == "chonknoris"
        else torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=1.0e-5
        )
    )
    closure_calls = 0
    chonknoris_result: ReducedChonknorisResult | None = None

    # The bridge/phase/energy terms are defined in physical pressure units, so
    # keep them off in normalized deployment space.  The LWC-84 PDE residual,
    # however, is SELF-NORMALIZED (``residual / scale`` in losses.py) and hence
    # scale-invariant — it is valid in the normalized O(1) space where the
    # deployment LoRA is now fit.  Enable it via ``deployment_pde_weight`` so the
    # latent shift is supervised by wave-equation physics on unobserved times,
    # not only by the two real onset frames.
    if deployment_lora:
        loss_weights = LossWeights(
            observed=1.0,
            bridge=0.0,
            pde=max(0.0, float(deployment_pde_weight)),
            phase=0.0,
            energy=0.0,
            anchor=1.0e-5,
        )

    baseline_raw = parent.detach()
    baseline_residual_field, pde_normalization_scale = lwc84_residual(
        baseline_raw,
        velocity,
        dt=float(time_s[1] - time_s[0]),
        dx=10.0,
        dz=10.0,
        observed_indices=indices,
        return_scale=True,
    )
    baseline_residual = float(baseline_residual_field.square().mean())
    baseline_energy = float(baseline_raw.square().mean().clamp_min(1.0e-12))

    def closure() -> torch.Tensor:
        nonlocal closure_calls, points
        if optimizer is None:
            raise RuntimeError("Adam/L-BFGS closure called for CHONKNORIS optimizer")
        # RAD: refresh collocation points from the current residual every
        # ``resample_every`` Adam closures so they track the evolving high-residual
        # regions.  Frozen during LBFGS (its closure must be fixed).
        if use_adaptive_sampling and not freeze_points:
            every = max(1, int(physics_sampling.resample_every))
            if closure_calls % every == 0:
                with torch.no_grad():
                    raw_detached = model.raw_wavefield(
                        parent, velocity, source, observed, time_s
                    ).detach()
                    residual_field = lwc84_residual(
                        raw_detached,
                        velocity,
                        dt=float(time_s[1] - time_s[0]),
                        dx=10.0,
                        dz=10.0,
                        observed_indices=indices,
                        normalization_scale=pde_normalization_scale,
                    )
                    points = _resample_physics(residual_field)
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        raw = model.raw_wavefield(parent, velocity, source, observed, time_s)
        terms = instance_loss_terms(
            raw_prediction=raw,
            target_observed=observed,
            bridge=bridge,
            velocity=velocity,
            source=source,
            points=points,
            weights=loss_weights,
            dt=float(time_s[1] - time_s[0]),
            observed_indices=indices,
            anchor=torch.cat([parameter.reshape(-1) for parameter in parameters]),
            pde_normalization_scale=pde_normalization_scale,
        )
        loss = terms["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite instance-adaptation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        return loss

    # Always measure gates against the frozen parent, never against a possibly
    # harmful pre-trained residual initialization.
    rollback_reason: str | None = None
    try:
        if method == "chonknoris":
            # RAD is sampled once from the initial state and then frozen.  All
            # 3x3 line-search candidates must see exactly the same objective.
            if use_adaptive_sampling and float(deployment_pde_weight) > 0.0:
                with torch.no_grad():
                    initial_raw = model.raw_wavefield(
                        parent, velocity, source, observed, time_s
                    )
                    initial_residual = lwc84_residual(
                        initial_raw,
                        velocity,
                        dt=float(time_s[1] - time_s[0]),
                        dx=10.0,
                        dz=10.0,
                        observed_indices=indices,
                        normalization_scale=pde_normalization_scale,
                    )
                    points = _resample_physics(initial_residual)
            residual_fn = build_chonknoris_residual_function(
                model,
                parent,
                velocity,
                source,
                observed,
                time_s,
                points,
                indices,
                pde_weight=float(deployment_pde_weight),
                pool_size=int(chonknoris_settings.residual_pool_size),
                pde_normalization_scale=pde_normalization_scale,
            )
            context = model.chonknoris_context(
                velocity, source, observed
            ).detach()[0]
            learned_predictor = (
                model.chonknoris
                if bool(getattr(model, "_chonknoris_pretrained", False))
                else None
            )
            chonknoris_result = reduced_chonknoris_solve(
                residual_fn,
                model.deployment_state().detach(),
                config=chonknoris_settings,
                factor_predictor=learned_predictor,
                context=(context if learned_predictor is not None else None),
            )
            model.set_deployment_state_(chonknoris_result.state)
        else:
            assert optimizer is not None
            for _ in range(int(adam_steps)):
                closure()
                optimizer.step()
        if method != "chonknoris" and int(lbfgs_steps) > 0:
            # LBFGS needs a fixed objective — freeze the RAD point set (drawn from
            # the best Adam-stage residual) so the closure is deterministic.
            freeze_points = True
            fixed_optimizer = torch.optim.LBFGS(
                parameters,
                lr=0.5,
                max_iter=1,
                history_size=10,
                line_search_fn="strong_wolfe",
            )

            def fixed_closure():
                loss = closure()
                return loss

            for _ in range(int(lbfgs_steps)):
                fixed_optimizer.step(fixed_closure)
    except (FloatingPointError, RuntimeError) as error:
        rollback_reason = "nonfinite_loss" if isinstance(error, FloatingPointError) else str(error)

    if (
        rollback_reason is None
        and deployment_lora
        and not bool(hard_project_observed)
    ):
        # The reduced solve couples latent and gate.  Re-project the final scalar
        # gate exactly on the only two labels permitted at deployment.  This
        # preserves the physics-informed latent update while guaranteeing that
        # the causal observation guard has gate=0 (the parent) as a feasible
        # fallback instead of rejecting an otherwise useful update due to a
        # small line-search mismatch.
        project_residual_gate_to_observations(
            model,
            parent,
            velocity,
            source,
            observed,
            time_s,
            indices,
        )

    candidate_raw = model.raw_wavefield(parent, velocity, source, observed, time_s).detach()
    candidate_residual = float(
        lwc84_residual(
            candidate_raw,
            velocity,
            dt=float(time_s[1] - time_s[0]),
            dx=10.0,
            dz=10.0,
            observed_indices=indices,
            normalization_scale=pde_normalization_scale,
        ).square().mean()
    )
    energy_ratio = float(candidate_raw.square().mean().clamp_min(1.0e-12) / baseline_energy)
    candidate_observed = float(
        (candidate_raw[:, list(indices)] - observed).square().mean()
    )
    baseline_observed = float((baseline_raw[:, list(indices)] - observed).square().mean())
    finite = bool(torch.isfinite(candidate_raw).all())
    if rollback_reason is None and not finite:
        rollback_reason = "nonfinite_candidate"
    observed_tolerance = max(1.0e-12, baseline_observed * 1.0e-6)
    if (
        rollback_reason is None
        and not bool(hard_project_observed)
        and candidate_observed > baseline_observed + observed_tolerance
    ):
        rollback_reason = "observed_loss_not_improved"
    if (
        rollback_reason is None
        and candidate_residual
        > baseline_residual * (1.0 - physics_improvement)
    ):
        rollback_reason = "physics_residual_not_improved"
    if rollback_reason is None and not minimum_energy <= energy_ratio <= maximum_energy:
        rollback_reason = "energy_gate_failed"
    accepted = rollback_reason is None
    if not accepted:
        if deployment_lora:
            with torch.no_grad():
                model.latent_delta.copy_(
                    safe_rollback_state["latent_delta"].to(model.latent_delta.device)
                )
                model.residual_gate.copy_(
                    safe_rollback_state["residual_gate"].to(model.residual_gate.device)
                )
        else:
            _restore_residual_state(model, safe_rollback_state)
    return AdaptationResult(
        accepted=accepted,
        rollback_reason=rollback_reason,
        steps=(
            chonknoris_result.accepted_iterations
            if chonknoris_result is not None
            else int(adam_steps)
        ),
        lbfgs_closure_calls=closure_calls,
        elapsed_s=time.perf_counter() - started,
        trainable_parameter_count=sum(parameter.numel() for parameter in parameters),
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        accessed_true_indices=tuple(record.audit.requested_indices),
        future_truth_used=bool(record.audit.payload()["future_truth_used"]),
        baseline_physics=baseline_residual,
        candidate_physics=candidate_residual,
        energy_ratio=energy_ratio,
        adapter_state=_adapter_state(model),
        optimizer_name=method,
        residual_history=(
            chonknoris_result.residual_history
            if chonknoris_result is not None
            else ()
        ),
        relaxation_history=(
            chonknoris_result.relaxation_history
            if chonknoris_result is not None
            else ()
        ),
        step_size_history=(
            chonknoris_result.step_size_history
            if chonknoris_result is not None
            else ()
        ),
        contraction_history=(
            chonknoris_result.contraction_history
            if chonknoris_result is not None
            else ()
        ),
        condition_history=(
            chonknoris_result.condition_history
            if chonknoris_result is not None
            else ()
        ),
        learned_factor_steps=(
            chonknoris_result.learned_factor_steps
            if chonknoris_result is not None
            else 0
        ),
        exact_factor_fallbacks=(
            chonknoris_result.exact_factor_fallbacks
            if chonknoris_result is not None
            else 0
        ),
        stopped_reason=(
            chonknoris_result.stopped_reason
            if chonknoris_result is not None
            else None
        ),
    )


__all__ = [
    "AdaptationResult",
    "adapt_instance",
    "build_chonknoris_residual_function",
    "build_supervised_chonknoris_residual_function",
    "high_residual_spatial_indices",
    "linearize_supervised_chonknoris_batch",
    "project_residual_gate_to_observations",
]
