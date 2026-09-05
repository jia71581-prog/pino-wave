from __future__ import annotations

import pytest

from grouped_ufno_mionet_v3.training.refinement_control import (
    RefinementControlState,
    advance_refinement_control,
)
from grouped_ufno_mionet_v3.training.curriculum import StageDecision


def test_ordinary_plateaus_continue_and_reduce_lr_only_after_patience():
    state = RefinementControlState()
    plateau = StageDecision(False, ("balanced validation score did not improve",))

    first = advance_refinement_control(
        state,
        plateau,
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )
    second = advance_refinement_control(
        first.state,
        plateau,
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )
    third = advance_refinement_control(
        second.state,
        plateau,
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )
    fourth = advance_refinement_control(
        third.state,
        plateau,
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )

    assert not first.rollback and not second.rollback
    assert not third.rollback and not fourth.rollback
    assert not first.reduce_learning_rate and not second.reduce_learning_rate
    assert third.reduce_learning_rate
    assert not fourth.reduce_learning_rate
    assert not fourth.stop
    assert fourth.state.plateau_epochs == 4
    assert fourth.state.lr_reductions == 1


def test_safety_regression_requests_rollback_and_lr_reduction():
    unsafe = StageDecision(
        False,
        ("family regression tolerance exceeded for uniform",),
    )
    action = advance_refinement_control(
        RefinementControlState(),
        unsafe,
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )

    assert action.rollback
    assert action.reduce_learning_rate
    assert not action.stop
    assert action.state.safety_rollbacks == 1


def test_new_best_resets_plateau_without_losing_lifetime_counters():
    state = RefinementControlState(
        plateau_epochs=5,
        lr_wait_epochs=2,
        safety_rollbacks=1,
        lr_reductions=2,
    )
    action = advance_refinement_control(
        state,
        StageDecision(True, ()),
        lr_patience_epochs=3,
        early_stopping_patience_epochs=0,
    )

    assert action.new_best
    assert action.state == RefinementControlState(
        safety_rollbacks=1,
        lr_reductions=2,
    )


def test_positive_early_stopping_patience_remains_available():
    state = RefinementControlState(plateau_epochs=4, lr_wait_epochs=1)
    action = advance_refinement_control(
        state,
        StageDecision(False, ("balanced validation score did not improve",)),
        lr_patience_epochs=3,
        early_stopping_patience_epochs=5,
    )
    assert action.stop


@pytest.mark.parametrize(
    ("lr_patience", "early_patience"),
    ((0, 0), (1, -1)),
)
def test_invalid_patience_values_are_rejected(lr_patience, early_patience):
    with pytest.raises(ValueError, match="patience"):
        advance_refinement_control(
            RefinementControlState(),
            StageDecision(False, ("balanced validation score did not improve",)),
            lr_patience_epochs=lr_patience,
            early_stopping_patience_epochs=early_patience,
        )
