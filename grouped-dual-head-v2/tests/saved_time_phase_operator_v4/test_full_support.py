import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.full_support import (
    adamw_backend_options,
    audit_epoch_schedule,
    audit_family_curriculum_epoch_schedule,
    build_staged_adamw,
    build_full_support_schedule,
    build_family_curriculum_schedule,
    configure_trainable_stage,
    configure_recovery_stage,
    configure_family_expert_stage,
    schedule_digest,
    warmup_cosine_factor,
)
from saved_time_phase_operator_v4.experts import FamilyRoutedResidualExperts


class TinyOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.medium_encoder = nn.Linear(3, 4)
        self.source_encoder = nn.Linear(3, 4)
        self.coordinate_encoder = nn.Linear(3, 4)
        self.travel_branch = nn.Linear(3, 4)
        self.fusion = nn.Linear(4, 4)
        self.dense_decoder = nn.Linear(4, 1)


class TinyTemporalOperator(TinyOperator):
    def __init__(self):
        super().__init__()
        self.dense_decoder = nn.ModuleDict(
            {
                "base": nn.Linear(4, 4),
                "temporal_basis": nn.ModuleDict(
                    {
                        "coefficient": nn.Linear(4, 4),
                        "time_trunk": nn.Linear(4, 4),
                    }
                ),
            }
        )


class TinyExpertOperator(TinyOperator):
    def __init__(self):
        super().__init__()
        self.dense_decoder = nn.ModuleDict(
            {
                "base": nn.Linear(4, 4),
                "temporal_basis": nn.ModuleDict(
                    {
                        "coefficient": nn.Linear(4, 4),
                        "time_trunk": nn.Linear(4, 4),
                    }
                ),
                "family_experts": FamilyRoutedResidualExperts(width=4, rank=2),
            }
        )


def test_full_support_epoch_covers_2240_records_with_batch48_padding():
    schedule = build_full_support_schedule(
        record_count=2240,
        epochs=2,
        macro_records=12,
        macros_per_update=4,
        seed=307,
    )

    assert len(schedule) == 2 * 188
    audit = audit_epoch_schedule(schedule, epoch=0, record_count=2240)
    assert audit.record_count == 2240
    assert audit.appearances == 2256
    assert audit.minimum_appearances == 1
    assert audit.maximum_appearances == 2
    assert audit.optimizer_updates == 47


def test_full_support_schedule_is_deterministic_and_epoch_distinct():
    first = build_full_support_schedule(
        48, epochs=3, macro_records=12, macros_per_update=4, seed=11
    )
    repeated = build_full_support_schedule(
        48, epochs=3, macro_records=12, macros_per_update=4, seed=11
    )
    changed = build_full_support_schedule(
        48, epochs=3, macro_records=12, macros_per_update=4, seed=12
    )

    assert first == repeated
    assert first != changed
    assert schedule_digest(first) == schedule_digest(repeated)
    first_epoch = tuple(spec.record_indices for spec in first if spec.epoch == 0)
    second_epoch = tuple(spec.record_indices for spec in first if spec.epoch == 1)
    assert first_epoch != second_epoch


def test_family_curriculum_schedule_is_easy_to_hard_with_replay():
    families = ("uniform",) * 12 + ("layered",) * 12 + ("marmousi",) * 12
    stages = (
        {"epochs": 1, "macro_pattern": ("uniform",)},
        {"epochs": 1, "macro_pattern": ("layered", "layered", "layered", "uniform")},
        {"epochs": 1, "macro_pattern": ("marmousi", "marmousi", "marmousi", "uniform", "layered")},
    )

    schedule = build_family_curriculum_schedule(
        families,
        stages=stages,
        epochs=3,
        macro_records=3,
        macros_per_update=4,
        seed=41,
    )
    repeated = build_family_curriculum_schedule(
        families,
        stages=stages,
        epochs=3,
        macro_records=3,
        macros_per_update=4,
        seed=41,
    )

    assert schedule == repeated
    assert len(schedule) == 3 * 12
    by_epoch = {
        epoch: [families[index] for spec in schedule if spec.epoch == epoch for index in spec.record_indices]
        for epoch in range(3)
    }
    assert set(by_epoch[0]) == {"uniform"}
    assert set(by_epoch[1]) == {"uniform", "layered"}
    assert set(by_epoch[2]) == {"uniform", "layered", "marmousi"}
    assert by_epoch[1].count("layered") == 3 * by_epoch[1].count("uniform")
    audit = audit_family_curriculum_epoch_schedule(
        schedule,
        epoch=1,
        record_families=families,
        macros_per_update=4,
    )
    assert audit.family_appearances == {"layered": 27, "uniform": 9}
    assert audit.optimizer_updates == 3


def test_family_curriculum_rejects_incomplete_or_unknown_stages():
    families = ("uniform", "layered", "marmousi") * 4

    with pytest.raises(ValueError, match="epoch count"):
        build_family_curriculum_schedule(
            families,
            stages=({"epochs": 1, "macro_pattern": ("uniform",)},),
            epochs=2,
            macro_records=3,
            macros_per_update=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="unknown family"):
        build_family_curriculum_schedule(
            families,
            stages=({"epochs": 1, "macro_pattern": ("anomaly",)},),
            epochs=1,
            macro_records=3,
            macros_per_update=1,
            seed=1,
        )


def test_appearance_indices_advance_for_padding_repeats_and_later_epochs():
    schedule = build_full_support_schedule(
        49, epochs=2, macro_records=12, macros_per_update=4, seed=19
    )
    observed: dict[int, list[int]] = {index: [] for index in range(49)}
    for spec in schedule:
        for record_index, appearance in zip(
            spec.record_indices, spec.appearance_indices, strict=True
        ):
            observed[record_index].append(appearance)

    assert all(values == list(range(len(values))) for values in observed.values())


def test_full_support_schedule_inherits_parent_time_appearance_offset():
    baseline = build_full_support_schedule(
        49, epochs=2, macro_records=12, macros_per_update=4, seed=19
    )
    inherited = build_full_support_schedule(
        49,
        epochs=2,
        macro_records=12,
        macros_per_update=4,
        seed=19,
        appearance_offset=3,
    )

    assert tuple(spec.record_indices for spec in inherited) == tuple(
        spec.record_indices for spec in baseline
    )
    assert tuple(spec.appearance_indices for spec in inherited) == tuple(
        tuple(value + 3 for value in spec.appearance_indices) for spec in baseline
    )

    with pytest.raises(ValueError, match="appearance offset"):
        build_full_support_schedule(
            48,
            epochs=1,
            macro_records=12,
            macros_per_update=4,
            seed=19,
            appearance_offset=-1,
        )


def test_full_support_schedule_epoch_offset_exactly_continues_parent_schedule():
    complete = build_full_support_schedule(
        49, epochs=5, macro_records=12, macros_per_update=4, seed=29
    )
    continued = build_full_support_schedule(
        49,
        epochs=2,
        macro_records=12,
        macros_per_update=4,
        seed=29,
        epoch_offset=3,
    )
    expected = tuple(spec for spec in complete if spec.epoch >= 3)

    assert tuple(spec.record_indices for spec in continued) == tuple(
        spec.record_indices for spec in expected
    )
    assert tuple(spec.appearance_indices for spec in continued) == tuple(
        spec.appearance_indices for spec in expected
    )
    assert tuple(spec.epoch for spec in continued) == (0,) * 8 + (1,) * 8
    assert tuple(spec.step for spec in continued) == tuple(spec.step for spec in expected)

    with pytest.raises(ValueError, match="epoch offset"):
        build_full_support_schedule(
            48,
            epochs=1,
            macro_records=12,
            macros_per_update=4,
            seed=29,
            epoch_offset=-1,
        )


def test_family_curriculum_schedule_inherits_time_appearance_offset():
    families = ("uniform", "layered", "marmousi") * 4
    stages = ({"epochs": 1, "macro_pattern": ("uniform",)},)
    schedule = build_family_curriculum_schedule(
        families,
        stages=stages,
        epochs=1,
        macro_records=3,
        macros_per_update=1,
        seed=5,
        appearance_offset=7,
    )

    assert min(value for spec in schedule for value in spec.appearance_indices) >= 7


def test_trainable_stages_expand_to_the_complete_model():
    model = TinyOperator()

    stage1 = configure_trainable_stage(model, epoch=1)
    assert stage1.trainable_prefixes == (
        "dense_decoder",
        "local_field",
        "source_encoder",
        "fusion",
    )
    stage3 = configure_trainable_stage(model, epoch=3)
    assert "coordinate_encoder" in stage3.trainable_prefixes
    assert "travel_branch" in stage3.trainable_prefixes
    stage6 = configure_trainable_stage(model, epoch=6)
    assert stage6.trainable_parameters == sum(p.numel() for p in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_recovery_stage_trains_only_the_dense_decoder_for_two_epochs():
    model = TinyOperator()

    stage1 = configure_recovery_stage(model, epoch=1, decoder_only_epochs=2)
    assert stage1.trainable_prefixes == ("dense_decoder",)
    assert model.dense_decoder.weight.requires_grad
    assert not model.source_encoder.weight.requires_grad

    stage3 = configure_recovery_stage(model, epoch=3, decoder_only_epochs=2)
    assert stage3.trainable_prefixes == (
        "dense_decoder",
        "local_field",
        "source_encoder",
        "fusion",
    )


def test_pinned_stage_holds_a_fixed_trainable_set_and_freezes_the_parent():
    from saved_time_phase_operator_v4.full_support import configure_pinned_stage

    model = TinyOperator()
    prefixes = ("dense_decoder", "source_encoder", "fusion")
    stage = configure_pinned_stage(model, prefixes=prefixes)

    assert stage.trainable_prefixes == prefixes
    # dense set trainable
    assert model.dense_decoder.weight.requires_grad
    assert model.source_encoder.weight.requires_grad
    assert model.fusion.weight.requires_grad
    # geometry + backbone frozen (this is the epoch-1/2 memory profile, held for all epochs)
    assert not model.coordinate_encoder.weight.requires_grad
    assert not model.travel_branch.weight.requires_grad
    assert not model.medium_encoder.weight.requires_grad
    expected_trainable = sum(
        p.numel()
        for m in (model.dense_decoder, model.source_encoder, model.fusion)
        for p in m.parameters()
    )
    assert stage.trainable_parameters == expected_trainable


def test_pinned_stage_rejects_unknown_or_empty_prefixes():
    from saved_time_phase_operator_v4.full_support import configure_pinned_stage

    model = TinyOperator()
    with pytest.raises(ValueError):
        configure_pinned_stage(model, prefixes=())
    with pytest.raises(ValueError):
        configure_pinned_stage(model, prefixes=("not_a_real_prefix",))


def test_optimizer_groups_are_exclusive_and_exhaustive():
    model = TinyOperator()

    optimizer = build_staged_adamw(
        model,
        dense_lr=2.0e-4,
        geometry_lr=1.0e-4,
        backbone_lr=2.0e-5,
        weight_decay=1.0e-6,
    )

    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    assert {group["group_name"] for group in optimizer.param_groups} == {
        "dense_decay",
        "dense_no_decay",
        "geometry_decay",
        "geometry_no_decay",
        "backbone_decay",
        "backbone_no_decay",
    }
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names_by_id[id(parameter)]
            if parameter.ndim <= 1 or name.endswith("bias"):
                assert group["weight_decay"] == 0.0


def test_temporal_basis_can_use_a_smaller_adam_step_after_gate_absorption():
    model = TinyTemporalOperator()

    optimizer = build_staged_adamw(
        model,
        dense_lr=1.0e-4,
        temporal_basis_lr=1.0e-5,
        geometry_lr=1.0e-4,
        backbone_lr=2.0e-5,
        weight_decay=1.0e-6,
    )

    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    learning_rate_by_name = {
        names_by_id[id(parameter)]: float(group["lr"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert learning_rate_by_name["dense_decoder.base.weight"] == pytest.approx(1.0e-4)
    assert learning_rate_by_name[
        "dense_decoder.temporal_basis.coefficient.weight"
    ] == pytest.approx(1.0e-5)
    assert learning_rate_by_name[
        "dense_decoder.temporal_basis.time_trunk.bias"
    ] == pytest.approx(1.0e-5)
    assert len(learning_rate_by_name) == len(tuple(model.parameters()))


def test_expert_optimizer_groups_are_exclusive_and_exhaustive():
    model = TinyExpertOperator()

    optimizer = build_staged_adamw(
        model,
        dense_lr=1.0e-5,
        expert_lr=1.0e-4,
        temporal_basis_lr=1.0e-5,
        geometry_lr=2.0e-6,
        backbone_lr=1.0e-6,
        weight_decay=1.0e-4,
    )

    names = {group["group_name"] for group in optimizer.param_groups}
    assert {"expert_decay", "expert_no_decay"} <= names
    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}


def test_adamw_implementation_is_explicit_and_rejects_unknown_backends():
    model = TinyOperator()
    common = {
        "dense_lr": 1.0e-4,
        "geometry_lr": 1.0e-5,
        "backbone_lr": 1.0e-6,
        "weight_decay": 1.0e-6,
    }

    single = build_staged_adamw(model, implementation="single_tensor", **common)

    assert single.defaults["foreach"] is False
    assert adamw_backend_options("fused") == {"fused": True}
    with pytest.raises(ValueError, match="AdamW implementation"):
        adamw_backend_options("unknown")


def test_adamw_moment_hyperparameters_are_explicit_and_validated():
    model = TinyOperator()
    common = {
        "dense_lr": 1.0e-4,
        "geometry_lr": 1.0e-5,
        "backbone_lr": 1.0e-6,
        "weight_decay": 1.0e-6,
    }

    optimizer = build_staged_adamw(
        model,
        betas=(0.9, 0.99),
        eps=1.0e-7,
        **common,
    )

    assert optimizer.defaults["betas"] == (0.9, 0.99)
    assert optimizer.defaults["eps"] == pytest.approx(1.0e-7)
    with pytest.raises(ValueError, match="betas"):
        build_staged_adamw(model, betas=(0.9, 1.0), **common)
    with pytest.raises(ValueError, match="eps"):
        build_staged_adamw(model, eps=0.0, **common)


def test_family_expert_stages_freeze_shared_operator_and_expand_experts():
    model = TinyExpertOperator()

    stage1 = configure_family_expert_stage(model, epoch=1, head_only_epochs=1)

    named = dict(model.named_parameters())
    assert stage1.trainable_prefixes == ("dense_decoder.family_experts",)
    assert named["dense_decoder.family_experts.router.network.0.weight"].requires_grad
    assert named["dense_decoder.family_experts.experts.0.output.weight"].requires_grad
    assert named["dense_decoder.family_experts.experts.0.time.0.weight"].requires_grad
    assert not named["dense_decoder.family_experts.experts.0.down.weight"].requires_grad
    assert not named["dense_decoder.temporal_basis.coefficient.weight"].requires_grad
    assert not named["source_encoder.weight"].requires_grad

    stage2 = configure_family_expert_stage(model, epoch=2, head_only_epochs=1)

    assert stage2.trainable_prefixes == (
        "dense_decoder.family_experts",
        "dense_decoder.temporal_basis",
    )
    assert named["dense_decoder.family_experts.experts.0.down.weight"].requires_grad
    assert named["dense_decoder.temporal_basis.coefficient.weight"].requires_grad
    assert not named["dense_decoder.base.weight"].requires_grad

    stage3 = configure_family_expert_stage(
        model,
        epoch=3,
        head_only_epochs=1,
        dense_unfreeze_epoch=3,
    )

    assert stage3.trainable_prefixes == ("dense_decoder",)
    assert named["dense_decoder.base.weight"].requires_grad
    assert named["dense_decoder.family_experts.experts.0.down.weight"].requires_grad
    assert named["dense_decoder.temporal_basis.coefficient.weight"].requires_grad
    assert not named["source_encoder.weight"].requires_grad


def test_family_expert_dense_unfreeze_must_follow_head_warmup():
    with pytest.raises(ValueError, match="dense_unfreeze_epoch"):
        configure_family_expert_stage(
            TinyExpertOperator(),
            epoch=1,
            head_only_epochs=1,
            dense_unfreeze_epoch=1,
        )


def test_family_expert_stages_progressively_unfreeze_shared_operator():
    model = TinyExpertOperator()
    schedule = {
        "head_only_epochs": 1,
        "dense_unfreeze_epoch": 3,
        "shared_unfreeze_epoch": 4,
        "geometry_unfreeze_epoch": 5,
        "backbone_unfreeze_epoch": 6,
    }

    shared = configure_family_expert_stage(model, epoch=4, **schedule)
    assert shared.trainable_prefixes == (
        "dense_decoder",
        "local_field",
        "source_encoder",
        "fusion",
    )
    assert model.source_encoder.weight.requires_grad
    assert not model.coordinate_encoder.weight.requires_grad

    geometry = configure_family_expert_stage(model, epoch=5, **schedule)
    assert geometry.trainable_prefixes == (
        "dense_decoder",
        "local_field",
        "source_encoder",
        "fusion",
        "coordinate_encoder",
        "travel_branch",
    )
    assert model.coordinate_encoder.weight.requires_grad
    assert not model.medium_encoder.weight.requires_grad

    backbone = configure_family_expert_stage(model, epoch=6, **schedule)
    assert backbone.trainable_parameters == sum(p.numel() for p in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_family_expert_progressive_unfreeze_epochs_must_be_ordered():
    with pytest.raises(ValueError, match="unfreeze epochs"):
        configure_family_expert_stage(
            TinyExpertOperator(),
            epoch=6,
            head_only_epochs=1,
            dense_unfreeze_epoch=3,
            shared_unfreeze_epoch=5,
            geometry_unfreeze_epoch=4,
            backbone_unfreeze_epoch=6,
        )


def test_warmup_cosine_factor_warms_then_decays_to_registered_floor():
    assert warmup_cosine_factor(0, total_epochs=50, warmup_epochs=3, minimum_factor=0.05) == pytest.approx(1 / 3)
    assert warmup_cosine_factor(2, total_epochs=50, warmup_epochs=3, minimum_factor=0.05) == pytest.approx(1.0)
    assert warmup_cosine_factor(49, total_epochs=50, warmup_epochs=3, minimum_factor=0.05) == pytest.approx(0.05)


import numpy as np
from saved_time_phase_operator_v4.full_support import (
    build_full_support_schedule,
    rad_record_weights,
)


def _appearance_counts(schedule, record_count):
    counts = np.zeros(record_count, dtype=np.int64)
    for spec in schedule:
        for r in spec.record_indices:
            counts[r] += 1
    return counts


def test_rad_record_weights_favor_high_error_and_normalize():
    errors = np.full(50, 0.1)
    errors[:5] = 1.0
    w = rad_record_weights(errors, k=1.0, c=1.0)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert float(w[0]) > float(w[10])
    assert np.allclose(rad_record_weights(np.ones(8)), 1.0 / 8)


def test_record_oversample_recurs_hard_records_but_covers_all():
    errors = np.full(64, 0.1)
    errors[:8] = 1.0
    w = rad_record_weights(errors, k=1.0, c=1.0)
    schedule = build_full_support_schedule(
        64, epochs=1, macro_records=8, macros_per_update=2, seed=372,
        record_weights=w, record_oversample=1.5,
    )
    counts = _appearance_counts(schedule, 64)
    assert (counts > 0).all()  # coverage contract
    assert counts[:8].mean() > counts[8:].mean()  # hard records recur


def test_record_weighting_is_noop_at_oversample_one():
    errors = np.full(64, 0.1)
    errors[:8] = 1.0
    w = rad_record_weights(errors)
    base = build_full_support_schedule(
        64, epochs=2, macro_records=8, macros_per_update=2, seed=372
    )
    same = build_full_support_schedule(
        64, epochs=2, macro_records=8, macros_per_update=2, seed=372,
        record_weights=w, record_oversample=1.0,
    )
    assert [s.record_indices for s in base] == [s.record_indices for s in same]


def test_record_weighted_schedule_is_deterministic():
    w = rad_record_weights(np.linspace(0.1, 1.0, 64))
    a = build_full_support_schedule(
        64, epochs=2, macro_records=8, macros_per_update=2, seed=372,
        record_weights=w, record_oversample=2.0,
    )
    b = build_full_support_schedule(
        64, epochs=2, macro_records=8, macros_per_update=2, seed=372,
        record_weights=w, record_oversample=2.0,
    )
    assert [s.record_indices for s in a] == [s.record_indices for s in b]


from saved_time_phase_operator_v4.local_field import LocalPropagationFieldGenerator as _LFG
from saved_time_phase_operator_v4.full_support import build_staged_adamw as _bsa


class _TempOpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0, temporal_operator_rank=8,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def _lr_of(opt, params):
    ids = set(id(p) for p in params)
    return {g["lr"] for g in opt.param_groups if ids & set(id(p) for p in g["params"])}


def test_temporal_operator_gets_its_own_lr_group():
    m = _TempOpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, temporal_operator_lr=5e-5)
    to_params = [p for n, p in m.named_parameters() if n.startswith("local_field.temporal_operator.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.temporal_operator.")]
    assert _lr_of(opt, to_params) == {5e-5}          # temporal operator develops at its own lr
    assert _lr_of(opt, lf_other) == {5e-6}           # rest of local_field near-frozen (dense group)


def test_temporal_operator_lr_none_is_backward_compatible():
    m = _TempOpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    # no separate 5e-5 group; all local_field (incl temporal_operator) in dense group
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    to_params = [p for n, p in m.named_parameters() if n.startswith("local_field.temporal_operator.")]
    assert _lr_of(opt, to_params) == {5e-6}


class _WarpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0, warp=True,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def test_warp_gets_its_own_lr_group():
    m = _WarpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, warp_lr=5e-5)
    warp_params = [p for n, p in m.named_parameters() if n.startswith("local_field.warp.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.warp.")]
    assert warp_params                                # the warp actually has params
    assert _lr_of(opt, warp_params) == {5e-5}         # warp develops at its own lr
    assert _lr_of(opt, lf_other) == {5e-6}            # rest of local_field near-frozen


def test_warp_lr_none_is_backward_compatible():
    m = _WarpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    warp_params = [p for n, p in m.named_parameters() if n.startswith("local_field.warp.")]
    assert _lr_of(opt, warp_params) == {5e-6}         # falls back to dense group


class _GreenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0,
            green_kernel=True, green_kernel_size=5, green_dilations=(1, 2, 4),
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def test_green_kernel_gets_its_own_lr_group():
    m = _GreenModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, green_kernel_lr=5e-5)
    green_params = [p for n, p in m.named_parameters() if n.startswith("local_field.green_kernel.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.green_kernel.")]
    assert green_params                               # the green kernel actually has params
    assert _lr_of(opt, green_params) == {5e-5}        # green kernel develops at its own lr
    assert _lr_of(opt, lf_other) == {5e-6}            # rest of local_field near-frozen


def test_green_kernel_lr_none_is_backward_compatible():
    m = _GreenModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    green_params = [p for n, p in m.named_parameters() if n.startswith("local_field.green_kernel.")]
    assert _lr_of(opt, green_params) == {5e-6}        # falls back to dense group


class _MultiArrivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0,
            multi_arrival=True, multi_arrival_paths=3,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def test_multi_arrival_gets_its_own_lr_group():
    # A4 (class-A) must stage exactly like warp/green_kernel: the fresh multi-arrival
    # paths develop at their own lr while the warm-started parent local_field stays
    # near-frozen (dense group).  This is the frozen-parent contract that keeps an A4
    # pilot from B1-style ep1->2 degradation.
    m = _MultiArrivalModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, multi_arrival_lr=5e-5)
    ma_params = [p for n, p in m.named_parameters() if n.startswith("local_field.multi_arrival.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.multi_arrival.")]
    assert ma_params                                  # the multi-arrival warp actually has params
    assert _lr_of(opt, ma_params) == {5e-5}           # A4 paths develop at their own lr
    assert _lr_of(opt, lf_other) == {5e-6}            # rest of local_field near-frozen


def test_multi_arrival_lr_none_is_backward_compatible():
    m = _MultiArrivalModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    ma_params = [p for n, p in m.named_parameters() if n.startswith("local_field.multi_arrival.")]
    assert _lr_of(opt, ma_params) == {5e-6}           # falls back to dense group


def test_local_field_gets_its_own_lr_group():
    # Upstream-unfreeze lever: train the coarse-field U-Net at a real lr WITHOUT
    # hammering the (capacity-ladder-proven-useless) dense_decoder.
    m = _TempOpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, local_field_lr=1e-4)
    lf_params = [p for n, p in m.named_parameters() if n.startswith("local_field.")]
    dec_params = [p for n, p in m.named_parameters() if n.startswith("dense_decoder.")]
    assert lf_params
    assert _lr_of(opt, lf_params) == {1e-4}           # coarse field develops at its own lr
    assert _lr_of(opt, dec_params) == {5e-6}          # decoder stays near-frozen (dense group)


def test_local_field_lr_precedence_keeps_warp_subtree_split():
    # With both split lrs, local_field.warp.* must stay in the warp group; the rest
    # of local_field goes to the local_field group; neither leaks into the other.
    m = _WarpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, warp_lr=5e-5, local_field_lr=1e-4)
    warp_params = [p for n, p in m.named_parameters() if n.startswith("local_field.warp.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.warp.")]
    assert warp_params and lf_other
    assert _lr_of(opt, warp_params) == {5e-5}         # warp keeps its own group
    assert _lr_of(opt, lf_other) == {1e-4}            # rest of local_field in local_field group


def test_local_field_lr_none_is_backward_compatible():
    m = _TempOpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 1e-4) > 1e-12 for g in opt.param_groups)
    lf_params = [p for n, p in m.named_parameters() if n.startswith("local_field.")]
    assert _lr_of(opt, lf_params) == {5e-6}           # all local_field falls back to dense group


class _TemporalLatentModel(nn.Module):
    """A3 (class-A): continuous temporal latent basis enriches the coarse field."""

    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0,
            temporal_latent_basis=True, temporal_latent_rank=8, temporal_latent_harmonics=4,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def test_temporal_latent_gets_its_own_lr_group():
    m = _TemporalLatentModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, temporal_latent_lr=5e-5)
    tl_params = [p for n, p in m.named_parameters() if n.startswith("local_field.temporal_latent.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.temporal_latent.")]
    assert tl_params                                   # the temporal latent basis actually has params
    assert _lr_of(opt, tl_params) == {5e-5}            # A3 develops at its own lr (fresh adapter warmup)
    assert _lr_of(opt, lf_other) == {5e-6}             # rest of local_field near-frozen (dense group)


def test_temporal_latent_lr_none_is_backward_compatible():
    m = _TemporalLatentModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    tl_params = [p for n, p in m.named_parameters() if n.startswith("local_field.temporal_latent.")]
    assert _lr_of(opt, tl_params) == {5e-6}            # falls back to dense group


class _DispersiveModalModel(nn.Module):
    """A5 (class-A escalation): per-pixel learned-dispersion modal coarse field."""

    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0,
            dispersive_modal=True, dispersive_modal_modes=16, dispersive_modal_max_frequency=8.0,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def test_dispersive_modal_gets_its_own_lr_group():
    m = _DispersiveModalModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
              weight_decay=1e-6, dispersive_modal_lr=5e-5)
    dm_params = [p for n, p in m.named_parameters() if n.startswith("local_field.dispersive_modal.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.") and not n.startswith("local_field.dispersive_modal.")]
    assert dm_params                                   # A5 module actually has params
    assert _lr_of(opt, dm_params) == {5e-5}            # develops at its own lr (fresh adapter warmup)
    assert _lr_of(opt, lf_other) == {5e-6}             # rest of local_field near-frozen (dense group)


def test_dispersive_modal_lr_none_is_backward_compatible():
    m = _DispersiveModalModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6, weight_decay=1e-6)
    assert all(abs(g["lr"] - 5e-5) > 1e-12 for g in opt.param_groups)
    dm_params = [p for n, p in m.named_parameters() if n.startswith("local_field.dispersive_modal.")]
    assert _lr_of(opt, dm_params) == {5e-6}            # falls back to dense group



class _WarpTempOpModel(nn.Module):
    """B1 (r5 stack): warp AND temporal operator both enabled on the coarse field."""

    def __init__(self):
        super().__init__()
        self.local_field = _LFG(
            width=8, pyramid_levels=2, saved_time_count=401,
            domain_t_s=1.0, domain_diagonal_m=2828.0,
            temporal_operator_rank=8, temporal_operator_spatial_kernel=3, warp=True,
        )
        self.dense_decoder = nn.Conv2d(1, 1, 1)
        self.source_encoder = nn.Conv2d(1, 1, 1)
        self.fusion = nn.Conv2d(1, 1, 1)
        self.medium_encoder = nn.Conv2d(1, 1, 1)
        self.coordinate_encoder = nn.Conv2d(1, 1, 1)
        self.travel_branch = nn.Conv2d(1, 1, 1)


def _group_names_of(opt, params):
    ids = set(id(p) for p in params)
    return {g["group_name"] for g in opt.param_groups if ids & set(id(p) for p in g["params"])}


def test_b1_warp_and_temporal_operator_split_into_distinct_groups():
    """B1's optimizer config splits BOTH warp and operator, each into its own group,
    while the converged rest-of-local_field stays near-frozen in the dense group. The
    two adapter groups must be DISTINCT (independent clipping/scheduling) even at equal
    lr -- this is the exact wiring local_field_w128_tempop_warp_b1.yaml relies on."""
    m = _WarpTempOpModel()
    opt = _bsa(m, dense_lr=5e-6, geometry_lr=1e-5, backbone_lr=2e-6,
               weight_decay=1e-6, warp_lr=5e-5, temporal_operator_lr=5e-5)
    warp_params = [p for n, p in m.named_parameters() if n.startswith("local_field.warp.")]
    op_params = [p for n, p in m.named_parameters()
                 if n.startswith("local_field.temporal_operator.")]
    lf_other = [p for n, p in m.named_parameters()
                if n.startswith("local_field.")
                and not n.startswith("local_field.warp.")
                and not n.startswith("local_field.temporal_operator.")]
    assert warp_params and op_params and lf_other
    assert _lr_of(opt, warp_params) == {5e-5}          # warp develops
    assert _lr_of(opt, op_params) == {5e-5}            # operator develops
    assert _lr_of(opt, lf_other) == {5e-6}             # converged parent near-frozen
    # distinct groups despite equal lr (independent clip/schedule):
    assert _group_names_of(opt, warp_params).isdisjoint(_group_names_of(opt, op_params))
