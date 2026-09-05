import pytest

from saved_time_phase_operator_v4.probe import (
    ProbeVariant,
    probe_variants,
    select_probe_candidate,
)


def test_probe_defines_capacity_and_phase_as_independent_axes():
    variants = probe_variants()
    assert tuple(variants) == (
        "shallow_no_phase",
        "deep_no_phase",
        "shallow_phase",
        "deep_phase",
    )
    assert variants["shallow_no_phase"].depth == variants["shallow_phase"].depth == 2
    assert variants["deep_no_phase"].depth == variants["deep_phase"].depth == 8
    assert not variants["deep_no_phase"].use_local_phase
    assert variants["deep_phase"].use_local_phase
    assert {value.spectral_rank for value in variants.values()} == {112}
    assert not any(value.coupled_axes for value in variants.values())


def test_probe_variant_can_enable_checkpoint_compatible_coupled_axes():
    candidate = ProbeVariant(depth=8, use_local_phase=True, coupled_axes=True)

    assert candidate.coupled_axes


def test_probe_selection_chooses_best_eligible_variant():
    scores = {
        "shallow_no_phase": 0.50,
        "deep_no_phase": 0.42,
        "shallow_phase": 0.31,
        "deep_phase": 0.28,
    }
    family_scores = {
        name: {"uniform": score, "layered": score, "marmousi": score}
        for name, score in scores.items()
    }
    assert select_probe_candidate(scores, family_scores) == "deep_phase"


def test_probe_selection_rejects_family_regression():
    scores = {"shallow_no_phase": 0.50, "deep_no_phase": 0.40}
    family_scores = {
        "shallow_no_phase": {"uniform": 0.5, "layered": 0.5, "marmousi": 0.5},
        "deep_no_phase": {"uniform": 0.3, "layered": 0.3, "marmousi": 0.6},
    }
    with pytest.raises(RuntimeError, match="no V4 probe"):
        select_probe_candidate(scores, family_scores)
