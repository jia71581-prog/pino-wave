# Long-Refinement Continuation Design

## Context

The first long dual-head refinement was configured for 20 epochs but stopped after four.
Every candidate was only slightly worse than the inherited score, so the trainer rolled back
to the same parent checkpoint, recreated AdamW with a halved learning rate, and incremented
the rejection counter. Four repetitions triggered `max_consecutive_rejections`. The run did
not crash; its control policy prevented a continuous optimization trajectory.

The post-run evaluator then rejected the inherited `best.pt`: its checkpoint run digest
belonged to the verified parent run while the supplied identity belonged to the child run.

## Evidence and Research Basis

- The observed candidate scores were 2.116596, 2.114235, 2.114027, and 2.114222 versus an
  inherited best of 2.113160. All were finite and within the configured family and head
  consistency guards; aggregate non-improvement alone caused every rollback.
- PyTorch's documented `ReduceLROnPlateau` policy waits for a configurable patience interval
  before reducing learning rate, and does not restore weights after every ordinary plateau:
  <https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>.
- Weight-averaging research supports treating the optimization trajectory separately from
  the selected inference weights. EMA is intentionally deferred so this change isolates the
  control-policy fix: <https://openreview.net/pdf?id=2M9CUnYnBA>.

## Selected Design

### 1. Separate three states

- **Training state:** the most recent safe model and optimizer state; continues across an
  ordinary validation plateau.
- **Best state:** the checkpoint with the lowest guarded balanced validation score; used for
  final selection and evaluation.
- **Recovery state:** the best accepted checkpoint, restored only for non-finite metrics or
  a family/head-consistency safety regression.

An aggregate score that does not improve is a plateau event, not a safety failure.

### 2. Plateau and early-stop policy

- `lr_patience_epochs` counts consecutive epochs without a new best.
- At patience, multiply all optimizer parameter-group learning rates by the configured factor
  without recreating AdamW, then reset the LR patience counter.
- `early_stopping_patience_epochs: 0` disables early stopping for a fixed-length production
  refinement. A positive value remains available for bounded experiments.
- The learning rate floor remains enforced.

### 3. Checkpoint semantics

- Save every epoch checkpoint as before.
- `latest.pt` points to the actual current safe training state, not always to `best.pt`.
- `best.pt` changes only after a guarded aggregate improvement.
- Terminal output reports both `selected_checkpoint` and `latest_checkpoint`, plus plateau,
  rollback, and learning-rate-reduction counters.

### 4. Evaluation identity

Evaluation accepts a checkpoint digest only when it matches either the current run digest or
the direct verified parent run digest embedded in the current identity. It still requires the
checkpoint format and manifest digest to match. Arbitrary unrelated checkpoints remain
rejected. The report records the actual checkpoint-origin digest.

## Failure Handling

- Non-finite validation metrics: restore the accepted best checkpoint and its optimizer state,
  reduce learning rate, and increment the safety rollback counter.
- Family or head-consistency guard violation: same recovery behavior.
- Ordinary non-improvement: retain current state, count plateau, and continue.
- Missing or mismatched parent identity: fail before evaluation or training begins.

## Verification

1. Unit tests distinguish plateau-only rejection from safety rollback.
2. A state-machine test proves four plateau epochs do not terminate a fixed 20-epoch run.
3. A learning-rate test proves optimizer identity/state is preserved during plateau reduction.
4. Evaluation tests accept the verified parent digest and reject unrelated digests.
5. Existing V3 tests run without regressions.
6. A short CUDA run must advance beyond the former rejection boundary, save `latest.pt`
   independently, and maintain finite loss/gradients.

## Scope

This change fixes refinement control flow and inherited-checkpoint evaluation. It does not
alter the operator architecture, loss weights, dataset, validation records, or introduce
EMA/SWA into the production result.
