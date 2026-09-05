# V3 training stability diagnosis and optimization

## Empirical diagnosis

The 20-epoch balanced pilot improved the fixed validation score from 2.856826 to
2.214690, but only 13 of 19 epoch transitions improved. The training loss was not
stationary or monotone: mean 4.5578, median 3.1171, standard deviation 6.5982,
maximum 162.3296, with 23 updates above 40.

The instability is localized. Correlation with total loss was 0.9988 for the
per-frame relative L2 term and 0.9944 for the complex-spectrum relative term,
while all other component correlations were at most 0.256. Both terms divide each
frame error by that frame's target norm. Near-silent frames therefore dominate an
update even though the denominator is clamped at `1e-8`.

The easy-to-hard curriculum improved the selected score from 2.214690 to 2.139053.
Uniform specialization was rejected, layered specialization improved to 2.177458,
and Marmousi specialization improved to 2.139053. Intermediate validation
reversals and a final score slightly worse than epoch 9 (2.135950) show that target
family improvement alone is not identical to balanced full-field improvement.

## Evidence review

- NeuralOperator merged an epsilon denominator fix specifically because relative
  losses become unstable when the target norm approaches zero. This prevents
  non-finite values but does not bound the influence of a physically near-silent
  frame: <https://github.com/neuraloperator/neuraloperator/pull/554>.
- GradNorm dynamically balances gradient magnitudes across tasks, but requires
  task-gradient measurements: <https://proceedings.mlr.press/v80/chen18a.html>.
- ReLoBRaLo balances relative loss progress with random lookback and reports lower
  overhead than gradient-based balancing on PINN objectives:
  <https://arxiv.org/abs/2110.09813>.
- FAMO amortizes multi-task loss balancing to constant extra space/time and is the
  preferred next comparison if component/family gradient conflict remains after
  removing the numerical outliers: <https://github.com/Cranial-XIX/FAMO> and
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf>.
- PCGrad and CAGrad explicitly modify conflicting task gradients, but their
  multiple-task gradient cost is a poor first choice for the current 21 GB update:
  <https://github.com/tianheyu927/PCGrad> and
  <https://github.com/Cranial-XIX/CAGrad>.
- Incremental FNO training is relevant to a future architecture-scale run, but it
  changes the spectral-resolution curriculum rather than fixing the observed
  low-energy loss pathology: <https://openreview.net/pdf?id=xI6cPQObp0>.

## Initial implemented training policy

1. Replace the fixed absolute denominator floor for frame and complex-spectrum
   relative losses with
   `max(frame_norm, fraction * max_frame_norm_for_same_record, eps)`.
   This remains invariant to pressure scaling and does not mix sources.
2. Use a balanced 4-uniform/4-layered/4-Marmousi macro-batch with the existing
   80% wavefield-energy + 20% uniform query sampling and 25% interpolated times.
3. Promote a checkpoint only if the balanced validation score improves and no
   family regresses by more than 2%.
4. On rejection, immediately restore the last accepted model, recreate AdamW, and
   halve the learning rate. Stop after two consecutive rejected epochs.
5. Save every epoch candidate independently; `best.pt` and `latest.pt` always point
   to the accepted anchor.

## Controlled 30-update ablation

All variants used the same curriculum checkpoint, schedule, seed, batches, AdamW
learning rate, and clipping threshold.

| Relative energy floor | Loss SD | Max loss | Max pre-clip gradient norm | Max frame relative error | Validation score |
|---:|---:|---:|---:|---:|---:|
| 0% | 1.1512 | 7.8822 | 2741.38 | 82.26 | 2.134708 |
| 1% | 0.2747 | 3.3564 | 222.21 | 4.38 | 2.134851 |
| 5% | 0.2271 | 3.0398 | 50.39 | 1.37 | 2.137983 |

The 1% floor preserves the no-floor short-run validation result within 0.007%
while reducing the worst gradient norm by 12.3x and the worst per-frame relative
error by 18.8x. It is therefore selected for the formal refinement. The 5% floor
is rejected because its additional stabilization slightly harms short-run accuracy.

## Deferred method

FAMO/ReLoBRaLo is intentionally deferred until the robust-denominator refinement
is evaluated. Adaptive weighting before removing the outliers can amplify a
numerical artifact rather than resolve genuine task conflict. If the formal run
still shows family trade-offs after rollback and learning-rate reduction, the next
controlled experiment will use FAMO over the three family aggregate objectives,
not over seven highly correlated component losses.

## Long-refinement control correction

The first nominal 20-epoch dual-head long refinement stopped after epoch 4. Its
four candidate scores were 2.116596, 2.114235, 2.114027, and 2.114222 versus the
inherited best score of 2.113160. All candidates were finite. Aggregate
non-improvement caused the trainer to restore the same parent checkpoint, discard
AdamW state, halve the learning rate, and increment the rejection counter after
every epoch. This made the run four independent short attempts rather than one
continuous trajectory; the configured rejection limit then ended the run.

The corrected control policy separates the live training state from checkpoint
selection:

1. Aggregate-only non-improvement is a plateau event. Training continues from the
   current candidate and retains AdamW moments.
2. `best.pt` remains guarded and changes only on a balanced-score improvement with
   no family or dual-head consistency regression.
3. The learning rate is reduced in place only after a configurable plateau
   patience. This follows the semantics of PyTorch `ReduceLROnPlateau`, which waits
   for patience rather than restoring weights after every non-improving epoch:
   <https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>.
4. Non-finite, family-regression, or consistency-regression candidates remain
   safety failures and restore both model and optimizer state from the guarded
   anchor.
5. Fixed-length production refinement disables ordinary early stopping while
   preserving every epoch checkpoint and the final best selection.

Weight averaging remains a later controlled experiment. EMA evidence supports
separating trajectory and inference weights, but adding EMA in the same run would
confound this control-flow correction:
<https://openreview.net/pdf?id=2M9CUnYnBA>.

The CUDA smoke run completed two epochs and four updates with finite losses and
gradients, `stopped_early=false`, and improved its selected validation score from
2.113160 to 2.112156. This is a control-flow verification, not evidence of a
statistically significant accuracy gain. The complete V3 regression suite passed
136 tests. Parent-aware evaluation also reproduced and fixed the previous
`evaluation checkpoint run mismatch`: a child identity can evaluate only its
explicitly verified direct-parent checkpoint, while unrelated run digests remain
rejected.
