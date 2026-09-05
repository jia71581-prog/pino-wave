# Long-Refinement Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fixed-length V3 refinement continue across ordinary validation plateaus while preserving guarded best-checkpoint selection and valid inherited-checkpoint evaluation.

**Architecture:** Add a pure refinement-control state machine that distinguishes ordinary plateaus from safety regressions. Integrate it into the existing trainer so the live model and AdamW state continue on plateaus, while `best.pt` remains guarded; extend evaluator identity validation to the verified direct parent digest.

**Tech Stack:** Python 3.13, PyTorch, pytest, YAML, atomic PyTorch checkpoints.

---

### Task 1: Refinement control policy

**Files:**
- Create: `grouped_ufno_mionet_v3/training/refinement_control.py`
- Create: `tests/grouped_ufno_mionet_v3/test_refinement_control.py`

- [ ] **Step 1: Write failing policy tests**

```python
from scripts.train_grouped_v3_curriculum import StageDecision
from grouped_ufno_mionet_v3.training.refinement_control import (
    RefinementControlState,
    advance_refinement_control,
)

def test_ordinary_plateaus_continue_and_reduce_lr_only_after_patience():
    state = RefinementControlState()
    plateau = StageDecision(False, ("balanced validation score did not improve",))
    first = advance_refinement_control(state, plateau, lr_patience_epochs=3,
                                       early_stopping_patience_epochs=0)
    second = advance_refinement_control(first.state, plateau, lr_patience_epochs=3,
                                        early_stopping_patience_epochs=0)
    third = advance_refinement_control(second.state, plateau, lr_patience_epochs=3,
                                       early_stopping_patience_epochs=0)
    assert not first.rollback and not second.rollback and not third.rollback
    assert not first.reduce_learning_rate and not second.reduce_learning_rate
    assert third.reduce_learning_rate
    assert not third.stop

def test_safety_regression_requests_rollback():
    unsafe = StageDecision(False, ("family regression tolerance exceeded for uniform",))
    action = advance_refinement_control(RefinementControlState(), unsafe,
                                        lr_patience_epochs=3,
                                        early_stopping_patience_epochs=0)
    assert action.rollback
    assert action.reduce_learning_rate
```

- [ ] **Step 2: Run policy tests and verify RED**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_refinement_control.py`

Expected: collection failure because `refinement_control` does not exist.

- [ ] **Step 3: Implement the pure policy**

```python
from __future__ import annotations
from dataclasses import dataclass

PLATEAU_FAILURE = "balanced validation score did not improve"

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

def advance_refinement_control(state, decision, *, lr_patience_epochs,
                               early_stopping_patience_epochs):
    if lr_patience_epochs < 1 or early_stopping_patience_epochs < 0:
        raise ValueError("refinement patience values are invalid")
    failures = tuple(decision.failures)
    if decision.accepted:
        if failures:
            raise ValueError("an accepted decision cannot contain failures")
        return RefinementControlAction(
            RefinementControlState(
                safety_rollbacks=state.safety_rollbacks,
                lr_reductions=state.lr_reductions,
            ),
            new_best=True, rollback=False, reduce_learning_rate=False, stop=False,
        )
    plateau_epochs = state.plateau_epochs + 1
    stop = (
        early_stopping_patience_epochs > 0
        and plateau_epochs >= early_stopping_patience_epochs
    )
    if failures == (PLATEAU_FAILURE,):
        wait = state.lr_wait_epochs + 1
        reduce_lr = wait >= lr_patience_epochs
        return RefinementControlAction(
            RefinementControlState(
                plateau_epochs=plateau_epochs,
                lr_wait_epochs=0 if reduce_lr else wait,
                safety_rollbacks=state.safety_rollbacks,
                lr_reductions=state.lr_reductions + int(reduce_lr),
            ),
            new_best=False, rollback=False,
            reduce_learning_rate=reduce_lr, stop=stop,
        )
    return RefinementControlAction(
        RefinementControlState(
            plateau_epochs=plateau_epochs,
            lr_wait_epochs=0,
            safety_rollbacks=state.safety_rollbacks + 1,
            lr_reductions=state.lr_reductions + 1,
        ),
        new_best=False, rollback=True, reduce_learning_rate=True, stop=stop,
    )
```

Validate non-negative patience values and reject accepted decisions containing failures.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_refinement_control.py`

Expected: all tests pass.

### Task 2: Continuous trainer integration

**Files:**
- Modify: `scripts/train_grouped_v3_stable_refinement.py`
- Modify: `tests/grouped_ufno_mionet_v3/test_stable_refinement.py`

- [ ] **Step 1: Write failing optimizer-state and mode tests**

```python
def test_reduce_optimizer_learning_rate_preserves_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    (parameter.square()).backward(); optimizer.step()
    state_id = id(optimizer.state[parameter])
    reduced = reduce_optimizer_learning_rate(optimizer, factor=0.5, minimum=1e-5)
    assert reduced == pytest.approx(5e-4)
    assert id(optimizer.state[parameter]) == state_id

def test_continuous_mode_is_explicitly_configured():
    values = refinement_control_values({"training_control": {
        "mode": "continuous", "lr_patience_epochs": 3,
        "early_stopping_patience_epochs": 0}})
    assert values == ("continuous", 3, 0)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_stable_refinement.py`

Expected: imports fail because the new helpers do not exist.

- [ ] **Step 3: Add configuration and LR helpers**

Add `refinement_control_values()` with an explicit `legacy_guarded` default so existing configs retain previous semantics. Add `reduce_optimizer_learning_rate()` that updates optimizer parameter groups in place and clamps at the configured floor.

- [ ] **Step 4: Integrate continuous state transitions**

For `training_control.mode == "continuous"`:

```python
action = advance_refinement_control(
    control_state,
    decision,
    lr_patience_epochs=lr_patience_epochs,
    early_stopping_patience_epochs=early_stopping_patience_epochs,
)
if action.new_best:
    anchor_checkpoint = checkpoint
    _atomic_hardlink(checkpoint, artifact_dir / "best.pt")
elif action.rollback:
    restore model and optimizer from anchor_checkpoint
if action.reduce_learning_rate:
    learning_rate = reduce_optimizer_learning_rate(
        optimizer,
        factor=float(recovery_values["learning_rate_factor"]),
        minimum=float(recovery_values["minimum_learning_rate"]),
    )
current_checkpoint = anchor_checkpoint if action.rollback else checkpoint
_atomic_hardlink(current_checkpoint, artifact_dir / "latest.pt")
if action.stop:
    stopped_early = True
    break
```

Restore optimizer state from an accepted checkpoint during safety rollback before applying the reduced learning rate. Keep the old branch unchanged for legacy configs.

- [ ] **Step 5: Extend terminal diagnostics**

Report `latest_checkpoint`, `plateau_epochs`, `safety_rollbacks`, and `learning_rate_reductions`. Preserve existing terminal keys.

- [ ] **Step 6: Run focused trainer tests**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_refinement_control.py tests/grouped_ufno_mionet_v3/test_stable_refinement.py`

Expected: all tests pass.

### Task 3: Parent-aware evaluator identity

**Files:**
- Modify: `scripts/evaluate_grouped_v3_pilot.py`
- Modify: `tests/grouped_ufno_mionet_v3/test_pilot_evaluation.py`

- [ ] **Step 1: Write failing parent identity tests**

```python
def test_evaluation_accepts_verified_direct_parent_checkpoint():
    identity = {"manifest_digest": "manifest", "run_digest": "child",
                "parent": {"run_digest": "parent"}}
    checkpoint = {"format": CHECKPOINT_FORMAT, "manifest_digest": "manifest",
                  "config_digest": "parent"}
    assert validate_evaluation_identity(identity, checkpoint,
        manifest_digest="manifest", model_config_digest="base") == "parent"

def test_evaluation_rejects_unrelated_checkpoint():
    identity = {"manifest_digest": "manifest", "run_digest": "child",
                "parent": {"run_digest": "parent"}}
    checkpoint = {"format": CHECKPOINT_FORMAT, "manifest_digest": "manifest",
                  "config_digest": "unrelated"}
    with pytest.raises(ValueError, match="run mismatch"):
        validate_evaluation_identity(
            identity,
            checkpoint,
            manifest_digest="manifest",
            model_config_digest="base",
        )
```

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_pilot_evaluation.py`

Expected: parent checkpoint is rejected and the current function returns `None`.

- [ ] **Step 3: Implement bounded parent acceptance**

Return the matched origin digest. The allowed set contains the current `run_digest` and, only when it is a mapping containing a non-empty digest, `identity["parent"]["run_digest"]`. Do not recursively trust arbitrary ancestry.

- [ ] **Step 4: Record origin in evaluation report**

Set `checkpoint_origin_run_digest` to the value returned by identity validation while keeping `run_digest` as the evaluated run identity.

- [ ] **Step 5: Run evaluator tests and verify GREEN**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3/test_pilot_evaluation.py`

Expected: all tests pass.

### Task 4: Production and smoke configurations

**Files:**
- Modify: `configs/grouped_v3/dual_head_long_refinement.yaml`
- Create: `configs/grouped_v3/dual_head_continuous_smoke.yaml`

- [ ] **Step 1: Enable continuous fixed-length control**

Add to the production config:

```yaml
training_control:
  mode: continuous
  lr_patience_epochs: 3
  early_stopping_patience_epochs: 0
```

Remove `max_consecutive_rejections` from the production recovery block because it no longer controls continuous mode.

- [ ] **Step 2: Add a two-epoch smoke config**

Use the same parent, loss, and safety settings but `epochs: 2`, `steps_per_epoch: 2`, a unique seed, and `/data/jiayh/v3_dual_head_continuous_smoke` artifact path.

- [ ] **Step 3: Validate YAML contracts**

Run a Python YAML check asserting production has 20 × 187 steps, smoke has 2 × 2 steps, both use continuous mode, and production early stopping is disabled.

Expected: prints `CONFIG_OK`.

### Task 5: Verification and launch

**Files:**
- Modify: `docs/research/2026-07-17-v3-training-stability-optimization.md`

- [ ] **Step 1: Run the complete V3 test suite**

Run: `PYTHONPATH=. python -m pytest -q tests/grouped_ufno_mionet_v3`

Expected: zero failures.

- [ ] **Step 2: Run CUDA smoke training**

Remove only a stale smoke artifact if its `run_identity.json` matches the smoke configuration, then run the smoke config in the foreground. Verify two terminal epochs, finite metrics, `stopped_early: false`, and an independently updated `latest.pt` when no new best occurs.

- [ ] **Step 3: Evaluate an inherited best checkpoint**

Run the evaluator with the smoke run identity and smoke `best.pt`. Expected: successful report containing `checkpoint_origin_run_digest` equal to the parent digest.

- [ ] **Step 4: Update research record**

Document the root cause, selected state separation, test evidence, smoke metrics, and research links. Do not claim accuracy improvement from a control-flow smoke test.

- [ ] **Step 5: Commit implementation**

Stage only the new policy, trainer/evaluator changes, tests, configs, plan, and research note. Preserve unrelated untracked artifacts. Commit message:

```text
fix(v3): continue refinement across validation plateaus
```

- [ ] **Step 6: Launch production long refinement**

Use a new artifact directory `/data/jiayh/v3_dual_head_long_refinement_v2`, start with `nohup`/`setsid` without sudo, and log to `logs/train.log`. Confirm the process advances, loss is finite, GPU utilization reaches steady compute, and install a 20-minute local monitor that evaluates the final best checkpoint.
