from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import fno_acoustic.ais_screen as ais_screen_module
from fno_acoustic.ais_screen import (
    CANDIDATE_IDS,
    HIGHER_GUARDS,
    LOWER_GUARDS,
    CandidateGateRecord,
    ScreenDecision,
    gate_h1,
    gate_h2,
    gate_h3,
    gate_o,
    gate_o_candidate,
    ranking_score,
)


CATEGORIES = ("uniform", "layered", "marmousi")


def _sha(candidate_id: str) -> str:
    if len(candidate_id) == 2 and candidate_id[1].isdigit():
        return f"{int(candidate_id[1]) + 1:x}" * 64
    return "a" * 64


def _category_metrics(**overrides: float) -> dict[str, dict[str, float]]:
    metrics = {
        "relative_l2": 0.20,
        "relative_l2_q4": 0.30,
        "prediction_target_norm_ratio": 1.0,
        "prediction_target_pearson": 0.90,
    }
    metrics.update(overrides)
    return {category: dict(metrics) for category in CATEGORIES}


def _record(
    candidate_id: str,
    *,
    update: int = 400,
    finite: bool = True,
    metrics: dict[str, dict[str, float]] | None = None,
    checkpoint_sha256: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> CandidateGateRecord:
    return CandidateGateRecord(
        candidate_id=candidate_id,
        update=update,
        checkpoint_sha256=checkpoint_sha256 or _sha(candidate_id),
        category_metrics=metrics or _category_metrics(),
        finite=finite,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def _gate_o_records(
    advanced: tuple[str, ...] = CANDIDATE_IDS,
) -> list[CandidateGateRecord]:
    selected = set(advanced)
    return [
        _record(candidate_id, finite=candidate_id in selected)
        for candidate_id in CANDIDATE_IDS
    ]


def _gate_o_prior(advanced: tuple[str, ...] = CANDIDATE_IDS) -> ScreenDecision:
    return gate_o(_gate_o_records(advanced))


def test_gate_o_accepts_every_inclusive_boundary() -> None:
    record = _record(
        "N0",
        metrics=_category_metrics(
            relative_l2=0.35,
            relative_l2_q4=0.50,
            prediction_target_norm_ratio=0.50,
            prediction_target_pearson=0.80,
        ),
    )

    assert gate_o_candidate(record).passed is True

    record = _record(
        "N0", metrics=_category_metrics(prediction_target_norm_ratio=1.50)
    )
    assert gate_o_candidate(record).passed is True


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("relative_l2", 0.3500000001),
        ("relative_l2_q4", 0.5000000001),
        ("prediction_target_norm_ratio", 0.4999999999),
        ("prediction_target_norm_ratio", 1.5000000001),
        ("prediction_target_pearson", 0.7999999999),
    ),
)
def test_gate_o_rejects_epsilon_threshold_failures(metric: str, value: float) -> None:
    record = _record("N0", metrics=_category_metrics(**{metric: value}))

    assert gate_o_candidate(record).passed is False


def test_gate_o_stops_when_n0_fails_before_count_check() -> None:
    records = [_record(candidate_id, finite=candidate_id != "N0") for candidate_id in CANDIDATE_IDS]

    decision = gate_o(records)

    assert decision.status == "stop"
    assert decision.reason == "n0_not_representable"
    assert decision.advanced == ()
    assert set(decision.rejected) == set(CANDIDATE_IDS)
    assert decision.rejected["N1"] == ("gate_stopped_n0_not_representable",)


def test_gate_o_stops_when_fewer_than_four_candidates_pass() -> None:
    records = [
        _record(candidate_id, finite=candidate_id in {"N0", "N1", "N2"})
        for candidate_id in CANDIDATE_IDS
    ]

    decision = gate_o(records)

    assert decision.status == "stop"
    assert decision.reason == "fewer_than_four_representable_candidates"
    assert set(decision.rejected) == set(CANDIDATE_IDS)
    assert decision.rejected["N0"] == ("gate_stopped_insufficient_representable",)


def test_gate_o_is_deterministic_and_records_rejections_and_hashes() -> None:
    records = [
        _record(candidate_id, finite=candidate_id != "N7")
        for candidate_id in reversed(CANDIDATE_IDS)
    ]

    decision = gate_o(records)

    assert decision.status == "advance"
    assert decision.advanced == ("N0", "N1", "N2", "N3", "N4", "N5", "N6")
    assert decision.rejected == {"N7": ("nonfinite_state",)}
    assert decision.checkpoint_sha256s["N7"] == _sha("N7")


def test_record_and_decision_deep_freeze_caller_mappings() -> None:
    source = _category_metrics()
    record = _record("N0", metrics=source)
    source["uniform"]["relative_l2"] = 99.0
    source["new"] = {}

    assert record.category_metrics["uniform"]["relative_l2"] == pytest.approx(0.20)
    assert set(record.category_metrics) == set(CATEGORIES)
    with pytest.raises(TypeError):
        record.category_metrics["uniform"]["relative_l2"] = 2.0
    with pytest.raises(FrozenInstanceError):
        record.update = 401

    decision = gate_o([record, *[_record(f"N{i}") for i in range(1, 8)]])
    with pytest.raises(TypeError):
        decision.checkpoint_sha256s["N0"] = "f" * 64


@pytest.mark.parametrize("candidate_id", ("N8", "n0", "B1", ""))
def test_record_rejects_unknown_candidate_ids(candidate_id: str) -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        _record(candidate_id)


@pytest.mark.parametrize("checkpoint_sha256", ("a" * 63, "g" * 64, "", 7))
def test_record_rejects_invalid_checkpoint_sha256(checkpoint_sha256: object) -> None:
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        CandidateGateRecord("N0", 400, checkpoint_sha256, _category_metrics(), True)


def test_record_rejects_boolean_update_and_nonboolean_finite_flag() -> None:
    with pytest.raises(ValueError, match="update"):
        CandidateGateRecord("N0", True, "a" * 64, _category_metrics(), True)
    with pytest.raises(ValueError, match="finite must be a boolean"):
        CandidateGateRecord("N0", 400, "a" * 64, _category_metrics(), 1)


def test_gate_o_rejects_duplicate_candidates_and_wrong_update() -> None:
    with pytest.raises(ValueError, match="unique"):
        gate_o([_record("N0"), *[_record(f"N{i}") for i in range(7)]])
    with pytest.raises(ValueError, match="update 400"):
        gate_o([_record("N0", update=399), *[_record(f"N{i}") for i in range(1, 8)]])


@pytest.mark.parametrize(
    "bad_metrics",
    (
        {"uniform": {}, "layered": {}, "marmousi": {}, "unknown": {}},
        {"uniform": {}, "layered": {}},
    ),
)
def test_record_rejects_extra_or_missing_categories(
    bad_metrics: dict[str, dict[str, float]],
) -> None:
    with pytest.raises(ValueError, match="categories"):
        _record("N0", metrics=bad_metrics)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_record_rejects_nonfinite_and_boolean_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="finite real"):
        _record("N0", metrics=_category_metrics(relative_l2=value))


def test_gate_o_rejects_missing_and_extra_metrics() -> None:
    missing = _category_metrics()
    missing["uniform"].pop("relative_l2")
    with pytest.raises(ValueError, match="metrics"):
        gate_o([_record("N0", metrics=missing), *[_record(f"N{i}") for i in range(1, 8)]])

    extra = _category_metrics(extra_metric=1.0)
    with pytest.raises(ValueError, match="metrics"):
        gate_o([_record("N0", metrics=extra), *[_record(f"N{i}") for i in range(1, 8)]])


def _halving_metrics(
    *,
    relative_l2: float = 0.30,
    relative_l2_q4: float = 0.40,
    norm_ratio: float = 1.0,
    zero_relative_l2: float = 1.0,
    zero_relative_l2_q4: float = 1.0,
) -> dict[str, dict[str, float]]:
    return {
        category: {
            "relative_l2": relative_l2,
            "relative_l2_q4": relative_l2_q4,
            "prediction_target_norm_ratio": norm_ratio,
            "zero_relative_l2": zero_relative_l2,
            "zero_relative_l2_q4": zero_relative_l2_q4,
        }
        for category in CATEGORIES
    }


def _halving_record(
    candidate_id: str,
    update: int,
    *,
    relative_l2: float = 0.30,
    relative_l2_q4: float = 0.40,
    norm_ratio: float = 1.0,
    zero_relative_l2: float = 1.0,
    zero_relative_l2_q4: float = 1.0,
    finite: bool = True,
    parent_checkpoint_sha256: str | None = None,
) -> CandidateGateRecord:
    return _record(
        candidate_id,
        update=update,
        finite=finite,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        metrics=_halving_metrics(
            relative_l2=relative_l2,
            relative_l2_q4=relative_l2_q4,
            norm_ratio=norm_ratio,
            zero_relative_l2=zero_relative_l2,
            zero_relative_l2_q4=zero_relative_l2_q4,
        ),
    )


def _h1_history(
    candidate_ids: tuple[str, ...],
) -> tuple[list[CandidateGateRecord], list[CandidateGateRecord], ScreenDecision]:
    gate_o_records = _gate_o_records(candidate_ids)
    h1_records = [_halving_record(candidate_id, 600) for candidate_id in candidate_ids]
    return h1_records, gate_o_records, gate_h1(h1_records, gate_o_records)


def _h1_prior(candidate_ids: tuple[str, ...]) -> ScreenDecision:
    return _h1_history(candidate_ids)[2]


def _h2_history(
    finalists: tuple[str, str] = ("N0", "N1"),
) -> tuple[
    list[CandidateGateRecord],
    list[CandidateGateRecord],
    list[CandidateGateRecord],
    ScreenDecision,
]:
    extras = tuple(candidate for candidate in CANDIDATE_IDS if candidate not in finalists)[:2]
    h1_records, gate_o_records, h1_prior = _h1_history((*finalists, *extras))
    h2_records = [
        _halving_record(
            candidate_id,
            1500,
            relative_l2=0.10 if candidate_id in finalists else 0.30,
            relative_l2_q4=0.10 if candidate_id in finalists else 0.30,
            parent_checkpoint_sha256=h1_prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in h1_prior.advanced
    ]
    return (
        h2_records,
        h1_records,
        gate_o_records,
        gate_h2(h2_records, h1_records, gate_o_records),
    )


def test_h1_uses_inclusive_exclusion_boundaries_and_ranks_deterministically() -> None:
    records = [
        _halving_record("N0", 600, relative_l2=1.05, relative_l2_q4=1.05),
        _halving_record("N7", 600, relative_l2=0.10, relative_l2_q4=0.10),
        _halving_record("N5", 600, relative_l2=0.20, relative_l2_q4=0.20),
        _halving_record("N3", 600, relative_l2=0.30, relative_l2_q4=0.30),
        _halving_record("N1", 600, relative_l2=1.05, relative_l2_q4=1.05, norm_ratio=0.25),
        _halving_record("N2", 600, relative_l2=1.05, relative_l2_q4=1.05, norm_ratio=2.00),
    ]

    gate_o_records = _gate_o_records(("N0", "N1", "N2", "N3", "N5", "N7"))
    decision = gate_h1(records, gate_o_records)

    assert decision.status == "advance"
    assert decision.advanced == ("N7", "N5", "N3", "N0")
    assert decision.rejected == {
        "N1": ("not_selected_by_ranking",),
        "N2": ("not_selected_by_ranking",),
    }
    assert decision.checkpoint_sha256s["N2"] == _sha("N2")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"relative_l2": 1.0500000001},
        {"relative_l2_q4": 1.0500000001},
        {"norm_ratio": 0.2499999999},
        {"norm_ratio": 2.0000000001},
        {"finite": False},
    ),
)
def test_h1_excludes_epsilon_and_nonfinite_failures(kwargs: dict[str, object]) -> None:
    records = [_halving_record(f"N{index}", 600) for index in range(4)]
    records.append(_halving_record("N7", 600, **kwargs))

    decision = gate_h1(records, _gate_o_records(("N0", "N1", "N2", "N3", "N7")))

    assert "N7" in decision.rejected
    assert "N7" not in decision.advanced


def test_h1_uses_recorded_positive_zero_baselines() -> None:
    passing = _halving_record(
        "N0",
        600,
        relative_l2=0.21,
        relative_l2_q4=0.42,
        zero_relative_l2=0.20,
        zero_relative_l2_q4=0.40,
    )
    failing = _halving_record(
        "N1",
        600,
        relative_l2=0.2100000001,
        relative_l2_q4=0.4200000001,
        zero_relative_l2=0.20,
        zero_relative_l2_q4=0.40,
    )

    records = [passing, *[_halving_record(f"N{i}", 600) for i in range(2, 5)]]
    assert "N0" not in gate_h1(records, _gate_o_records(("N0", "N2", "N3", "N4"))).rejected
    records = [
        failing,
        _halving_record("N0", 600),
        *[_halving_record(f"N{i}", 600) for i in range(2, 5)],
    ]
    assert "N1" in gate_h1(
        records, _gate_o_records(("N0", "N1", "N2", "N3", "N4"))
    ).rejected
    with pytest.raises(ValueError, match="zero_relative_l2.*positive"):
        gate_h1(
            [_halving_record("N0", 600, zero_relative_l2=0.0), *[_halving_record(f"N{i}", 600) for i in range(1, 4)]],
            _gate_o_records(("N0", "N1", "N2", "N3")),
        )
    with pytest.raises(ValueError, match="zero_relative_l2_q4.*positive"):
        gate_h1(
            [_halving_record("N0", 600, zero_relative_l2_q4=0.0), *[_halving_record(f"N{i}", 600) for i in range(1, 4)]],
            _gate_o_records(("N0", "N1", "N2", "N3")),
        )


def test_h1_stops_below_four_after_exclusions_and_orders_ties_by_id() -> None:
    stopped = gate_h1(
        [
            _halving_record("N0", 600),
            _halving_record("N1", 600),
            _halving_record("N2", 600),
            _halving_record("N3", 600, finite=False),
        ],
        _gate_o_records(("N0", "N1", "N2", "N3")),
    )
    assert stopped.status == "stop"
    assert stopped.reason == "fewer_than_four_eligible_candidates"
    assert stopped.advanced == ()
    assert stopped.rejected["N0"] == ("gate_stopped_insufficient_eligible",)
    assert set(stopped.rejected) == {"N0", "N1", "N2", "N3"}

    tied = gate_h1(
        [
            _halving_record("N0", 600, relative_l2=1.05, relative_l2_q4=1.05),
            *[_halving_record(f"N{i}", 600) for i in (7, 5, 3, 1, 6)],
        ],
        _gate_o_records(("N0", "N1", "N3", "N5", "N6", "N7")),
    )
    assert tied.advanced == ("N1", "N3", "N5", "N6")
    assert tied.rejected == {
        "N0": ("not_selected_by_ranking",),
        "N7": ("not_selected_by_ranking",),
    }


def test_h1_rejects_wrong_update_and_metric_schema() -> None:
    gate_o_records = _gate_o_records(("N0", "N1", "N2", "N3"))
    with pytest.raises(ValueError, match="update 600"):
        gate_h1(
            [_halving_record("N0", 599), *[_halving_record(f"N{i}", 600) for i in range(1, 4)]],
            gate_o_records,
        )
    metrics = _halving_metrics()
    metrics["uniform"]["extra"] = 1.0
    with pytest.raises(ValueError, match="metrics"):
        gate_h1(
            [_record("N0", update=600, metrics=metrics), *[_halving_record(f"N{i}", 600) for i in range(1, 4)]],
            gate_o_records,
        )


def test_h2_requires_exact_h1_survivors_and_update_1500() -> None:
    survivors = ("N0", "N3", "N5", "N7")
    h1_records, gate_o_records, prior = _h1_history(survivors)
    records = [
        _halving_record(
            candidate,
            1500,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate],
        )
        for candidate in reversed(survivors)
    ]

    decision = gate_h2(records, h1_records, gate_o_records)

    assert decision.status == "advance"
    assert decision.advanced == ("N0", "N3")
    assert decision.rejected == {
        "N5": ("not_selected_by_ranking",),
        "N7": ("not_selected_by_ranking",),
    }
    with pytest.raises(ValueError, match="H1 advanced"):
        gate_h2(records[:-1], h1_records, gate_o_records)
    with pytest.raises(ValueError, match="update 1500"):
        gate_h2(
            [
                _halving_record(
                    candidate,
                    1499,
                    parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate],
                )
                for candidate in survivors
            ],
            h1_records,
            gate_o_records,
        )


def test_h2_stops_if_exclusions_leave_fewer_than_two() -> None:
    survivors = ("N0", "N1", "N2", "N3")
    h1_records, gate_o_records, prior = _h1_history(survivors)
    records = [
        _halving_record("N0", 1500, parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
        _halving_record("N1", 1500, finite=False, parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
        _halving_record("N2", 1500, norm_ratio=2.1, parent_checkpoint_sha256=prior.checkpoint_sha256s["N2"]),
        _halving_record("N3", 1500, relative_l2=1.1, parent_checkpoint_sha256=prior.checkpoint_sha256s["N3"]),
    ]

    decision = gate_h2(records, h1_records, gate_o_records)

    assert decision.status == "stop"
    assert decision.reason == "fewer_than_two_eligible_candidates"
    assert decision.advanced == ()
    assert decision.rejected["N0"] == ("gate_stopped_insufficient_eligible",)
    assert set(decision.rejected) == set(survivors)


def _baseline_metrics() -> dict[str, dict[str, float]]:
    return {
        category: {
            "relative_l2": 1.0,
            "relative_l2_q4": 1.0,
            **{metric: 1.0 for metric in LOWER_GUARDS},
            **{metric: 0.8 for metric in HIGHER_GUARDS},
        }
        for category in CATEGORIES
    }


def test_final_guard_names_are_the_exact_registered_sets() -> None:
    assert LOWER_GUARDS == (
        "receiver_relative_l2",
        "receiver_relative_l2_q1",
        "receiver_relative_l2_q2",
        "receiver_relative_l2_q3",
        "receiver_relative_l2_q4",
        "arrival_mae_s",
        "arrival_miss_rate",
        "receiver_lag_abs_s",
        "receiver_phase_error",
        "energy_log_ratio",
        "komega_relative_l2",
        "komega_high",
        "komega_relative_l2_q4",
        "komega_high_q4",
    )
    assert HIGHER_GUARDS == ("receiver_xcorr_peak", "receiver_phase_coherence")


def _final_metrics(**overrides: float) -> dict[str, dict[str, float]]:
    metrics = {
        "relative_l2": 0.70,
        "relative_l2_q4": 0.70,
        **{metric: 1.05 for metric in LOWER_GUARDS},
        **{metric: 0.76 for metric in HIGHER_GUARDS},
        "prediction_target_norm_ratio": 0.50,
        "prediction_target_pearson": 0.80,
    }
    metrics.update(overrides)
    return {category: dict(metrics) for category in CATEGORIES}


def _final_record(
    candidate_id: str,
    *,
    finite: bool = True,
    metrics: dict[str, dict[str, float]] | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> CandidateGateRecord:
    return _record(
        candidate_id,
        update=3000,
        finite=finite,
        metrics=metrics or _final_metrics(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def test_h3_accepts_exact_30_and_5_percent_boundaries_and_upper_norm() -> None:
    metrics = _final_metrics(prediction_target_norm_ratio=1.50)
    h2_records, h1_records, gate_o_records, prior = _h2_history(("N1", "N3"))

    decision = gate_h3(
        [
            _final_record("N3", metrics=metrics, parent_checkpoint_sha256=prior.checkpoint_sha256s["N3"]),
            _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
        ],
        h2_records,
        h1_records,
        gate_o_records,
        _baseline_metrics(),
    )

    assert decision.status == "promote"
    assert decision.promoted_recipe == "N1"
    assert decision.advanced == ("N1",)
    assert decision.rejected == {"N3": ("not_selected_by_ranking",)}


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("relative_l2", 0.7000000001),
        ("relative_l2_q4", 0.7000000001),
        (LOWER_GUARDS[0], 1.0500000001),
        (HIGHER_GUARDS[0], 0.7599999999),
        ("prediction_target_norm_ratio", 0.4999999999),
        ("prediction_target_norm_ratio", 1.5000000001),
        ("prediction_target_pearson", 0.7999999999),
    ),
)
def test_h3_rejects_epsilon_final_guard_failures(metric: str, value: float) -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    failing = _final_record(
        "N0",
        metrics=_final_metrics(**{metric: value}),
        parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"],
    )

    decision = gate_h3(
        [
            failing,
            _final_record(
                "N1",
                finite=False,
                parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"],
            ),
        ],
        h2_records,
        h1_records,
        gate_o_records,
        _baseline_metrics(),
    )

    assert decision.status == "stop"
    assert decision.reason == "no_candidate_satisfies_final_accuracy_gate"
    assert decision.promoted_recipe is None
    assert "N0" in decision.rejected


def test_h3_guard_absolute_floor_is_inclusive() -> None:
    baseline = _baseline_metrics()
    lower, higher = LOWER_GUARDS[0], HIGHER_GUARDS[0]
    for category in CATEGORIES:
        baseline[category][lower] = 0.0
        baseline[category][higher] = 0.0
    passing_metrics = _final_metrics(**{lower: 1e-6, higher: -1e-6})
    failing_metrics = _final_metrics(**{lower: 1.0000001e-6, higher: -1.0000001e-6})
    h2_records, h1_records, gate_o_records, prior = _h2_history()

    passing = gate_h3(
        [
            _final_record("N0", metrics=passing_metrics, parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
            _final_record("N1", finite=False, parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
        ],
        h2_records,
        h1_records,
        gate_o_records,
        baseline,
    )
    failing = gate_h3(
        [
            _final_record("N0", metrics=failing_metrics, parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
            _final_record("N1", finite=False, parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
        ],
        h2_records,
        h1_records,
        gate_o_records,
        baseline,
    )

    assert passing.promoted_recipe == "N0"
    assert failing.promoted_recipe is None


def test_h3_tie_breaks_by_candidate_id_independent_of_input_order() -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history(("N2", "N7"))
    decision = gate_h3(
        [
            _final_record("N7", parent_checkpoint_sha256=prior.checkpoint_sha256s["N7"]),
            _final_record("N2", parent_checkpoint_sha256=prior.checkpoint_sha256s["N2"]),
        ],
        h2_records,
        h1_records,
        gate_o_records,
        _baseline_metrics(),
    )

    assert ranking_score(_final_record("N2").category_metrics) == pytest.approx(1.05)
    assert decision.promoted_recipe == "N2"


def test_h3_native_accuracy_is_subject_to_independent_screen64_ranking() -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history(("N2", "N7"))
    native = [
        _final_record("N2", parent_checkpoint_sha256=prior.checkpoint_sha256s["N2"]),
        _final_record("N7", parent_checkpoint_sha256=prior.checkpoint_sha256s["N7"]),
    ]
    ranking = [
        _record(
            "N2", update=3000,
            metrics=_halving_metrics(relative_l2=0.4, relative_l2_q4=0.4),
            parent_checkpoint_sha256=prior.checkpoint_sha256s["N2"],
        ),
        _record(
            "N7", update=3000,
            metrics=_halving_metrics(relative_l2=0.2, relative_l2_q4=0.2),
            parent_checkpoint_sha256=prior.checkpoint_sha256s["N7"],
        ),
    ]

    decision = gate_h3(
        native, h2_records, h1_records, gate_o_records, _baseline_metrics(),
        ranking_records=ranking,
    )

    assert decision.promoted_recipe == "N7"


def test_h3_requires_exact_h2_finalists_update_and_strict_schemas() -> None:
    baseline = _baseline_metrics()
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    with pytest.raises(ValueError, match="H2 advanced"):
        gate_h3(
            [_final_record("N0", parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"])],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )
    wrong_update = _record(
        "N0",
        update=2999,
        metrics=_final_metrics(),
        parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"],
    )
    with pytest.raises(ValueError, match="update 3000"):
        gate_h3(
            [wrong_update, _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"])],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )

    extra = _final_metrics(extra=1.0)
    with pytest.raises(ValueError, match="metrics"):
        gate_h3(
            [
                _final_record("N0", metrics=extra, parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
                _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
            ],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )

    del baseline["marmousi"]
    with pytest.raises(ValueError, match="categories"):
        gate_h3(
            [
                _final_record("N0", parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
                _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
            ],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )

    baseline = _baseline_metrics()
    baseline["uniform"].pop("arrival_mae_s")
    with pytest.raises(ValueError, match="baseline.*metrics"):
        gate_h3(
            [
                _final_record("N0", parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
                _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
            ],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, -0.1))
def test_h3_rejects_invalid_baseline_error_values(value: float) -> None:
    baseline = _baseline_metrics()
    baseline["uniform"]["relative_l2"] = value
    h2_records, h1_records, gate_o_records, prior = _h2_history()

    with pytest.raises(ValueError, match="baseline|nonnegative|finite real"):
        gate_h3(
            [
                _final_record("N0", parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"]),
                _final_record("N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]),
            ],
            h2_records,
            h1_records,
            gate_o_records,
            baseline,
        )


def test_gate_o_requires_exactly_all_eight_registered_candidates() -> None:
    with pytest.raises(ValueError, match="exactly N0-N7"):
        gate_o([_record(f"N{index}") for index in range(7)])


def test_candidate_record_binds_optional_parent_checkpoint_sha256() -> None:
    record = CandidateGateRecord(
        candidate_id="N0",
        update=1500,
        checkpoint_sha256="a" * 64,
        category_metrics=_halving_metrics(),
        finite=True,
        parent_checkpoint_sha256="b" * 64,
    )

    assert record.parent_checkpoint_sha256 == "b" * 64


def test_h1_recomputes_an_advancing_gate_o_history() -> None:
    gate_o_records = _gate_o_records()
    records = [_halving_record(candidate, 600) for candidate in CANDIDATE_IDS]

    decision = gate_h1(records, gate_o_records)

    assert decision.gate == "H1"


def test_gate_o_and_h1_require_restart_records_without_parents() -> None:
    gate_o_records = [_record(candidate_id) for candidate_id in CANDIDATE_IDS]
    gate_o_records[0] = _record("N0", parent_checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="Gate O.*parent"):
        gate_o(gate_o_records)
    with pytest.raises(ValueError, match="Gate O.*parent"):
        gate_o_candidate(gate_o_records[0])

    valid_gate_o_records = _gate_o_records(("N0", "N1", "N2", "N3"))
    h1_records = [_halving_record(candidate_id, 600) for candidate_id in ("N0", "N1", "N2", "N3")]
    h1_records[0] = _halving_record(
        "N0", 600, parent_checkpoint_sha256=_sha("N0")
    )
    with pytest.raises(ValueError, match="H1.*restart.*parent"):
        gate_h1(h1_records, valid_gate_o_records)


def test_h1_rejects_gate_o_history_that_recomputes_to_stop() -> None:
    stopped_records = _gate_o_records(("N1", "N2", "N3", "N4"))

    with pytest.raises(ValueError, match="recomputed Gate O.*advance"):
        gate_h1(
            [_halving_record(candidate_id, 600) for candidate_id in ("N1", "N2", "N3", "N4")],
            stopped_records,
        )


def test_h1_gate_o_metric_tamper_is_detected_by_recomputation() -> None:
    gate_o_records = _gate_o_records(("N0", "N1", "N2", "N3"))
    gate_o_records[0] = _record(
        "N0", metrics=_category_metrics(relative_l2=0.3500000001)
    )

    with pytest.raises(ValueError, match="recomputed Gate O.*advance"):
        gate_h1(
            [_halving_record(candidate_id, 600) for candidate_id in ("N0", "N1", "N2", "N3")],
            gate_o_records,
        )


def test_h2_and_h3_require_exact_parent_last_checkpoint_sha() -> None:
    h1_records, gate_o_records, h1_prior = _h1_history(("N0", "N1", "N2", "N3"))
    h2_records = [
        _halving_record(
            candidate_id,
            1500,
            parent_checkpoint_sha256=h1_prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in h1_prior.advanced
    ]
    h2_records[0] = _halving_record("N0", 1500, parent_checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        gate_h2(h2_records, h1_records, gate_o_records)

    valid_h2_records, h1_records, gate_o_records, h2_prior = _h2_history()
    h3_records = [
        _final_record(
            candidate_id,
            parent_checkpoint_sha256=h2_prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in h2_prior.advanced
    ]
    h3_records[0] = _final_record("N0", parent_checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        gate_h3(
            h3_records,
            valid_h2_records,
            h1_records,
            gate_o_records,
            _baseline_metrics(),
        )


def test_h2_rejects_tampered_h1_history_checkpoint() -> None:
    h1_records, gate_o_records, prior = _h1_history(("N0", "N1", "N2", "N3"))
    h2_records = [
        _halving_record(
            candidate_id,
            1500,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in prior.advanced
    ]
    h1_records[0] = _record(
        "N0",
        update=600,
        checkpoint_sha256="f" * 64,
        metrics=_halving_metrics(),
    )

    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        gate_h2(h2_records, h1_records, gate_o_records)


def test_h2_rejects_h1_history_that_recomputes_to_stop() -> None:
    h1_records, gate_o_records, prior = _h1_history(("N0", "N1", "N2", "N3"))
    h2_records = [
        _halving_record(
            candidate_id,
            1500,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in prior.advanced
    ]
    h1_records[0] = _halving_record("N0", 600, finite=False)

    with pytest.raises(ValueError, match="recomputed Gate H1.*advance"):
        gate_h2(h2_records, h1_records, gate_o_records)


def test_all_decision_rejection_reasons_are_nonempty_and_immutable() -> None:
    prior = _h1_prior(("N0", "N1", "N2", "N3", "N4"))

    assert set(prior.rejected) == {"N4"}
    assert prior.rejected["N4"] == ("not_selected_by_ranking",)
    assert all(reasons for reasons in prior.rejected.values())
    with pytest.raises(TypeError):
        prior.rejected["N4"] = ("tampered",)


def test_screen_decision_is_an_immutable_validated_output_dto() -> None:
    rejected = {"N7": ["nonfinite_state"]}
    hashes = {candidate_id: _sha(candidate_id).upper() for candidate_id in CANDIDATE_IDS}
    decision = ScreenDecision(
        "O",
        "advance",
        "representable_candidates_advanced",
        tuple(candidate_id for candidate_id in CANDIDATE_IDS if candidate_id != "N7"),
        rejected,
        hashes,
    )
    rejected["N7"].append("tampered")
    hashes["N0"] = "f" * 64

    assert decision.rejected["N7"] == ("nonfinite_state",)
    assert decision.checkpoint_sha256s["N0"] == _sha("N0")
    assert decision.to_dict()["rejected"]["N7"] == ["nonfinite_state"]


def test_object_new_forged_screen_decision_cannot_be_lineage_authority() -> None:
    prior = _gate_o_prior(("N0", "N1", "N2", "N3"))
    forged = object.__new__(ScreenDecision)
    for field in (
        "gate",
        "status",
        "reason",
        "advanced",
        "rejected",
        "checkpoint_sha256s",
        "promoted_recipe",
    ):
        object.__setattr__(forged, field, getattr(prior, field))
    records = [_halving_record(candidate_id, 600) for candidate_id in prior.advanced]

    with pytest.raises(TypeError, match="CandidateGateRecord|records"):
        gate_h1(records, forged)


def test_method_overridden_decision_cannot_be_lineage_authority() -> None:
    decision = _gate_o_prior(("N0", "N1", "N2", "N3"))
    object.__setattr__(decision, "to_dict", lambda: {"status": "advance"})
    records = [_halving_record(candidate_id, 600) for candidate_id in decision.advanced]

    with pytest.raises(TypeError, match="CandidateGateRecord|records"):
        gate_h1(records, decision)


def test_module_has_no_private_decision_maker_or_attestation_api() -> None:
    assert not hasattr(ais_screen_module, "_make_screen_decision")


@pytest.mark.parametrize(
    "overrides",
    (
        {"reason": "not_canonical"},
        {"rejected": {"N0": ["invalid_partition"]}},
        {"rejected": {"N7": []}},
        {"rejected": {"N7": [""]}},
    ),
)
def test_screen_decision_rejects_invalid_reason_or_partition(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "gate": "O",
        "status": "advance",
        "reason": "representable_candidates_advanced",
        "advanced": tuple(candidate_id for candidate_id in CANDIDATE_IDS if candidate_id != "N7"),
        "rejected": {"N7": ["nonfinite_state"]},
        "checkpoint_sha256s": {
            candidate_id: _sha(candidate_id) for candidate_id in CANDIDATE_IDS
        },
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        ScreenDecision(**values)


@pytest.mark.parametrize(
    "overrides",
    (
        {"relative_l2": -1e-12},
        {"relative_l2_q4": -1e-12},
        {"prediction_target_norm_ratio": -1e-12},
        {"prediction_target_pearson": -1.0000001},
        {"prediction_target_pearson": 1.0000001},
    ),
)
def test_gate_o_rejects_metrics_outside_physical_domain(
    overrides: dict[str, float],
) -> None:
    records = [_record(candidate_id) for candidate_id in CANDIDATE_IDS]
    records[0] = _record("N0", metrics=_category_metrics(**overrides))

    with pytest.raises(ValueError, match="nonnegative|Pearson|norm ratio"):
        gate_o(records)


def test_candidate_record_canonicalizes_uppercase_checkpoint_hashes() -> None:
    record = _record(
        "N0",
        checkpoint_sha256="A" * 64,
        parent_checkpoint_sha256="B" * 64,
    )

    assert record.checkpoint_sha256 == "a" * 64
    assert record.parent_checkpoint_sha256 == "b" * 64


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("receiver_xcorr_peak", -1.0000001),
        ("receiver_xcorr_peak", 1.0000001),
        ("receiver_phase_coherence", -1e-12),
        ("receiver_phase_coherence", 1.0000001),
        ("prediction_target_pearson", -1.0000001),
        ("prediction_target_pearson", 1.0000001),
    ),
)
def test_h3_rejects_candidate_higher_metrics_outside_physical_domain(
    metric: str, value: float
) -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    records = [
        _final_record(
            "N0",
            metrics=_final_metrics(**{metric: value}),
            parent_checkpoint_sha256=prior.checkpoint_sha256s["N0"],
        ),
        _final_record(
            "N1", parent_checkpoint_sha256=prior.checkpoint_sha256s["N1"]
        ),
    ]

    with pytest.raises(ValueError, match="range|Pearson"):
        gate_h3(
            records,
            h2_records,
            h1_records,
            gate_o_records,
            _baseline_metrics(),
        )


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("receiver_xcorr_peak", -1.0000001),
        ("receiver_xcorr_peak", 1.0000001),
        ("receiver_phase_coherence", -1e-12),
        ("receiver_phase_coherence", 1.0000001),
    ),
)
def test_h3_rejects_baseline_higher_metrics_outside_physical_domain(
    metric: str, value: float
) -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    baseline = _baseline_metrics()
    baseline["uniform"][metric] = value
    records = [
        _final_record(
            candidate_id,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in prior.advanced
    ]

    with pytest.raises(ValueError, match="range"):
        gate_h3(records, h2_records, h1_records, gate_o_records, baseline)


def test_followup_gates_reject_screen_decisions_as_lineage_authority() -> None:
    gate_o_records = [_record(candidate_id) for candidate_id in CANDIDATE_IDS]
    decision = gate_o(gate_o_records)
    h1_records = [_halving_record(candidate_id, 600) for candidate_id in decision.advanced]

    with pytest.raises(TypeError, match="CandidateGateRecord|records"):
        gate_h1(h1_records, decision)

    _, _, h1_decision = _h1_history(("N0", "N1", "N2", "N3"))
    h2_records = [
        _halving_record(
            candidate_id,
            1500,
            parent_checkpoint_sha256=h1_decision.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in h1_decision.advanced
    ]
    with pytest.raises(TypeError, match="CandidateGateRecord|records"):
        gate_h2(h2_records, h1_decision, gate_o_records)

    _, h1_history, gate_o_records, h2_decision = _h2_history()
    h3_records = [
        _final_record(
            candidate_id,
            parent_checkpoint_sha256=h2_decision.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in h2_decision.advanced
    ]
    with pytest.raises(TypeError, match="CandidateGateRecord|records"):
        gate_h3(
            h3_records,
            h2_decision,
            h1_history,
            gate_o_records,
            _baseline_metrics(),
        )


def test_gate_h3_recomputes_full_history_and_is_deterministic() -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    records = [
        _final_record(
            candidate_id,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in reversed(prior.advanced)
    ]

    first = gate_h3(
        records, h2_records, h1_records, gate_o_records, _baseline_metrics()
    )
    second = gate_h3(
        records, h2_records, h1_records, gate_o_records, _baseline_metrics()
    )

    assert first.to_dict() == second.to_dict()


def test_gate_h3_stops_when_tampered_h2_metrics_recompute_to_stop() -> None:
    h2_records, h1_records, gate_o_records, prior = _h2_history()
    records = [
        _final_record(
            candidate_id,
            parent_checkpoint_sha256=prior.checkpoint_sha256s[candidate_id],
        )
        for candidate_id in prior.advanced
    ]
    h2_records[0] = _halving_record(
        h2_records[0].candidate_id,
        1500,
        finite=False,
        parent_checkpoint_sha256=h2_records[0].parent_checkpoint_sha256,
    )
    h2_records[1] = _halving_record(
        h2_records[1].candidate_id,
        1500,
        finite=False,
        parent_checkpoint_sha256=h2_records[1].parent_checkpoint_sha256,
    )
    h2_records[2] = _halving_record(
        h2_records[2].candidate_id,
        1500,
        finite=False,
        parent_checkpoint_sha256=h2_records[2].parent_checkpoint_sha256,
    )

    with pytest.raises(ValueError, match="recomputed Gate H2.*advance"):
        gate_h3(
            records,
            h2_records,
            h1_records,
            gate_o_records,
            _baseline_metrics(),
        )
