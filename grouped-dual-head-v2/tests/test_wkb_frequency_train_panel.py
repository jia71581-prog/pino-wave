from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_wkb_frequency_train_panel import (
    ScatteringFullTraceStore,
    build_balanced_panel_schedule,
    metrics_meet_panel_target,
    configure_trainable_scope,
    required_gradient_prefixes,
    select_anchor_group_source_panels,
    select_disjoint_family_panels,
)


def _records(per_family: int = 6):
    return tuple(
        SimpleNamespace(split="train", medium_type=family, sample_id=f"{family}:{i}")
        for family in ("uniform", "layered", "marmousi")
        for i in range(per_family)
    )


def test_family_panels_are_disjoint_and_skip_overfit_anchor() -> None:
    fit, calibration, confirm = select_disjoint_family_panels(
        _records(),
        split="train",
        fit_per_family=1,
        calibration_per_family=2,
        confirm_per_family=2,
        skip_per_family=1,
    )
    assert len(fit) == 3
    assert len(calibration) == len(confirm) == 6
    assert not set(fit) & set(calibration)
    assert not set(fit) & set(confirm)
    assert not set(calibration) & set(confirm)
    assert 0 not in fit and 6 not in fit and 12 not in fit


def test_balanced_schedule_has_one_record_per_family_and_tracks_appearance() -> None:
    records = _records()
    fit, _, _ = select_disjoint_family_panels(
        records,
        split="train",
        fit_per_family=2,
        calibration_per_family=1,
        confirm_per_family=1,
        skip_per_family=1,
    )
    schedule = build_balanced_panel_schedule(
        records, fit, split="train", updates=7, seed=372
    )
    assert len(schedule) == 7
    for spec in schedule:
        families = tuple(records[index].medium_type for index in spec.record_indices)
        assert families == ("uniform", "layered", "marmousi")
    assert schedule == build_balanced_panel_schedule(
        records, fit, split="train", updates=7, seed=372
    )


def test_single_family_panels_and_schedule_remain_disjoint() -> None:
    records = _records()
    fit, calibration, confirm = select_disjoint_family_panels(
        records,
        split="train",
        fit_per_family=2,
        calibration_per_family=1,
        confirm_per_family=1,
        skip_per_family=1,
        families=("marmousi",),
    )
    assert len(fit) == 2 and len(calibration) == len(confirm) == 1
    assert not set(fit) & set(calibration)
    schedule = build_balanced_panel_schedule(
        records,
        fit,
        split="train",
        updates=5,
        seed=372,
        families=("marmousi",),
    )
    assert all(len(spec.record_indices) == 1 for spec in schedule)
    assert all(records[spec.record_indices[0]].medium_type == "marmousi" for spec in schedule)


def test_panel_target_checks_only_registered_families() -> None:
    metrics = {
        "aggregate_relative_l2": 0.04,
        "family_relative_l2": {"marmousi": 0.049},
    }
    assert metrics_meet_panel_target(
        metrics, families=("marmousi",), maximum=0.05
    )
    assert not metrics_meet_panel_target(
        metrics, families=("uniform", "marmousi"), maximum=0.05
    )


def test_stratified_random_panels_are_deterministic_and_exclude_anchor() -> None:
    records = _records(per_family=10)
    kwargs = dict(
        split="train",
        fit_per_family=2,
        calibration_per_family=2,
        confirm_per_family=2,
        skip_per_family=1,
        families=("marmousi",),
        selection="stratified_random",
        seed=372,
    )
    first = select_disjoint_family_panels(records, **kwargs)
    second = select_disjoint_family_panels(records, **kwargs)
    assert first == second
    fit, calibration, confirm = first
    assert not set(fit) & set(calibration)
    assert not set(fit) & set(confirm)
    assert not set(calibration) & set(confirm)
    marmousi_anchor = 20
    assert marmousi_anchor not in set(fit + calibration + confirm)


def test_anchor_group_source_holdout_uses_last_two_sources_for_gates() -> None:
    records = tuple(
        SimpleNamespace(
            split="train",
            medium_type=family,
            sample_id=f"{family}:{group}:{source}",
            group_id=f"{family}:{group}",
        )
        for family, group, source_count in (
            ("layered", 0, 4),
            ("layered", 1, 4),
            ("marmousi", 0, 5),
            ("marmousi", 1, 5),
        )
        for source in range(source_count)
    )
    fit, calibration, confirm = select_anchor_group_source_panels(
        records,
        split="train",
        anchor_sample_ids=("layered:0:0", "marmousi:0:0"),
        families=("layered", "marmousi"),
    )
    assert tuple(records[index].sample_id for index in fit) == (
        "layered:0:0",
        "layered:0:1",
        "marmousi:0:0",
        "marmousi:0:1",
        "marmousi:0:2",
    )
    assert tuple(records[index].sample_id for index in calibration) == (
        "layered:0:2",
        "marmousi:0:3",
    )
    assert tuple(records[index].sample_id for index in confirm) == (
        "layered:0:3",
        "marmousi:0:4",
    )
    assert not set(fit) & set(calibration + confirm)


def test_scattering_store_subtracts_fixed_background_and_closes_truth():
    class Truth:
        def __init__(self):
            self.closed = False

        def __getitem__(self, sample_id):
            assert sample_id == "sample"
            return np.full((3, 2, 2), 5.0, dtype=np.float32)

        def close(self):
            self.closed = True

    class Provider:
        def full_physical(self, sample_ids, *, device):
            assert sample_ids == ("sample",) and device == "cpu"
            return torch.full((1, 3, 2, 2), 2.0)

    truth = Truth()
    store = ScatteringFullTraceStore(truth, Provider())
    np.testing.assert_allclose(store["sample"], 3.0)
    store.close()
    assert truth.closed


def test_zero_head_first_update_gradient_contract_is_local_only():
    assert required_gradient_prefixes(zero_head=True, update=1) == ("local_field",)
    assert required_gradient_prefixes(zero_head=True, update=2) == (
        "local_field",
        "medium_encoder",
        "source_encoder",
    )
    assert required_gradient_prefixes(zero_head=False, update=1) == (
        "local_field",
        "medium_encoder",
        "source_encoder",
    )
    assert required_gradient_prefixes(
        zero_head=True, update=2, trainable_scope="helmholtz_head"
    ) == ("local_field",)


def test_helmholtz_head_scope_freezes_every_other_parameter():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.local_field = torch.nn.Module()
            self.local_field.helmholtz_synthesis = torch.nn.Module()
            self.local_field.helmholtz_synthesis.head = torch.nn.Conv2d(4, 6, 1)

    model = Model()
    report = configure_trainable_scope(model, "helmholtz_head")
    assert report["parameter_count"] == 30
    assert report["parameter_names"] == (
        "local_field.helmholtz_synthesis.head.weight",
        "local_field.helmholtz_synthesis.head.bias",
    )
    assert not model.encoder.weight.requires_grad
