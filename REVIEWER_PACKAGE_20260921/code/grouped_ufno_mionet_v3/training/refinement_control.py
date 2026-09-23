"""Pure control policy for guarded long-running refinement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


PLATEAU_FAILURE = "balanced validation score did not improve"


class RefinementDecision(Protocol):
    accepted: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RefinementControlState:
    plateau_epochs: int = 0
    lr_wait_epochs: int = 0
    safety_rollbacks: int = 0
    lr_reductions: int = 0


@dataclass(frozen=True)
class RefinementControlAction:
    state: RefinementControlState
    new_best: bool
    rollback: bool
    reduce_learning_rate: bool
    stop: bool


def advance_refinement_control(
    state: RefinementControlState,
    decision: RefinementDecision,
    *,
    lr_patience_epochs: int,
    early_stopping_patience_epochs: int,
) -> RefinementControlAction:
    """Advance plateau/recovery state from one guarded validation decision."""
    if lr_patience_epochs < 1 or early_stopping_patience_epochs < 0:
        raise ValueError("refinement patience values are invalid")
    if min(
        state.plateau_epochs,
        state.lr_wait_epochs,
        state.safety_rollbacks,
        state.lr_reductions,
    ) < 0:
        raise ValueError("refinement control counters must be nonnegative")

    failures = tuple(decision.failures)
    if decision.accepted:
        if failures:
            raise ValueError("an accepted decision cannot contain failures")
        return RefinementControlAction(
            state=RefinementControlState(
                safety_rollbacks=state.safety_rollbacks,
                lr_reductions=state.lr_reductions,
            ),
            new_best=True,
            rollback=False,
            reduce_learning_rate=False,
            stop=False,
        )
    if not failures:
        raise ValueError("a rejected decision must explain its failures")

    plateau_epochs = state.plateau_epochs + 1
    stop = (
        early_stopping_patience_epochs > 0
        and plateau_epochs >= early_stopping_patience_epochs
    )
    if failures == (PLATEAU_FAILURE,):
        wait = state.lr_wait_epochs + 1
        reduce_lr = wait >= lr_patience_epochs
        return RefinementControlAction(
            state=RefinementControlState(
                plateau_epochs=plateau_epochs,
                lr_wait_epochs=0 if reduce_lr else wait,
                safety_rollbacks=state.safety_rollbacks,
                lr_reductions=state.lr_reductions + int(reduce_lr),
            ),
            new_best=False,
            rollback=False,
            reduce_learning_rate=reduce_lr,
            stop=stop,
        )

    return RefinementControlAction(
        state=RefinementControlState(
            plateau_epochs=plateau_epochs,
            lr_wait_epochs=0,
            safety_rollbacks=state.safety_rollbacks + 1,
            lr_reductions=state.lr_reductions + 1,
        ),
        new_best=False,
        rollback=True,
        reduce_learning_rate=True,
        stop=stop,
    )


__all__ = [
    "PLATEAU_FAILURE",
    "RefinementControlAction",
    "RefinementControlState",
    "advance_refinement_control",
]
