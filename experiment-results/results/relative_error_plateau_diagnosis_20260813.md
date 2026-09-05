# Relative-error plateau diagnosis (2026-08-13)

## Scope and safety

This audit used code inspection, existing logs, CPU/static tests, schedule audits,
and read-only GPU/process queries. It did not stop, restart, signal, renice, or
otherwise control the four-GPU r5c job. r5c exited naturally; the wait-only
supervisor then launched the preregistered four-GPU r5d branch. Validation and
test_id were not opened.

## Verified metric contract

- Inputs must be real scalar `[record,time,z,x]` tensors. Shape mismatches,
  complex tensors, and implicit channel dimensions are rejected.
- Per-record numerator and denominator sum squared error/target over `time,z,x`.
- Per-record relative error is
  `sqrt(sum((prediction-target)^2)) / sqrt(max(sum(target^2), 1e-16))`.
- `aggregate_relative_l2` is the arithmetic mean of the per-record ratios, not a
  globally pooled pixel/energy ratio.
- Time-bin and spectrum diagnostics pool energy within each bin before taking a
  ratio; they are diagnostics, not the checkpoint-selection aggregate.
- The norm-domain denominator epsilon is therefore `1e-8`.
- `energy_floor_fraction` is reported but is not applied to the primary aggregate.
  That naming/telemetry ambiguity should not be changed inside an active lineage;
  it is not the plateau cause because all 48 records have nonzero total energy and
  r5c update-0 exactly reproduces r5b.

## Verified normalization and training mechanics

- Pressure encoding and decoding are exact scalar inverses:
  `p_norm = p / (pressure_scale * source_amplitude)` and the inverse multiplies by
  the same two factors. Metadata are bound to the train manifest.
- Evaluation compares normalized prediction with normalized target. No decode is
  required because per-record relative L2 is invariant to that per-record scalar.
- AdamW contains every adapter parameter exactly once and partitions feature/output
  parameters into registered learning-rate groups.
- `zero_grad(set_to_none=True)` occurs once per effective optimizer update. Ragged
  microbatches are weighted by their record count, backward accumulates over them,
  DDP gradients are averaged, then gradients are clipped and `optimizer.step()` is
  called once.
- Training sets both wrappers to `train()`; fixed metrics use `inference_mode()` and
  `eval()`.
- No GradScaler or outer autocast is active. Spectral critical sections explicitly
  disable autocast, so mixed-precision underflow is not the explanation.
- Epoch retry rollback restores model, optimizer, RNG, global step, and coverage.
  The identical first loss in attempts 1--3 (`0.7204923331737518`) confirms replay.

## Direct evidence and conclusion

r5c update-0 fixed-train aggregate equals the r5b parent exactly:
`0.2772543673381241`. This proves transfer identity, normalization, mask, and metric
plumbing on the actual run.

Epoch-1 attempts 1--8 were rejected at `0.27728080477282147`,
`0.27730061823701074`, `0.27728749937679215`, `0.2772680236883441`,
`0.27725846027382994`, `0.2772575429385903`, and `0.2772561814881532`,
and `0.27725488634210116`, respectively. Their candidate-minus-parent deltas were `+2.6437e-5`,
`+4.6251e-5`, `+3.3132e-5`, `+1.3656e-5`, `+4.0929e-6`, `+3.1756e-6`, and
`+1.8142e-6`, and `+5.1900e-7` as the learning-rate multiplier was successively reduced from
`1` to `1/128`. The maximum-attempt gate was exhausted, r5c terminated naturally
with no accepted new epoch, and the immutable r5b parent remained the selected
checkpoint. The first two attempts
also show that slightly lower mixed training loss can coexist with a worse fixed
record-relative aggregate (`0.8956576407 -> 0.8954800761`), which directly exposes
the selection mismatch rather than an absent optimizer step.

The direct cause is objective/sampling misalignment, not a relative-error code
defect:

1. Training uses 24 appearance/RAD-selected exact frames per record; the gate uses
   32 fixed exact frames. For the actual 48 gate records, one training appearance
   overlaps only `5.92296511627907%` of gate frames on average. The union across the
   entire first epoch overlaps only `9.765625%` on average (range `0%`--`28.125%`).
2. Training optimizes record-relative L2 plus `0.5*delta + 0.1*temporal +
   0.1*gradient + 0.1*spectrum`, whereas selection uses only record-relative L2.
3. The fixed 48-record gate is 29.17% Uniform, 50% Layered, 20.83% Marmousi. RAD
   epoch exposure is about 15.5%, 48.2%, 36.3%, respectively. Marmousi is therefore
   weighted about 1.7x relative to the gate and Uniform only about 0.53x.
4. The hard residual remains concentrated in Marmousi, late time, and middle/high
   spectrum, consistent with a capacity/conditioning plateau. Adapter gradients are
   finite and optimizer steps occur, but clipping is frequent. In r5d attempt 1,
   132/140 updates (94.3%) clipped; the pre-clip gradient norm had median 368.8 and
   maximum $1.13\times10^6$. The attempt nevertheless completed all 140 steps and
   worsened the fixed gate by $6.74\times10^{-5}$. This is evidence of spiky,
   objective-misaligned gradients, not a missing optimizer step.

The optimizer's learning-rate backoff therefore reduces the magnitude of the
regression toward zero but does not change its sign. This is expected when the
sampled composite objective is not the quantity used to accept a checkpoint.

## Exact code locations

- Primary metric formula and dimensions: `saved_time_phase_operator_v4/streaming_metrics.py:210-238`.
- Norm epsilon: `streaming_metrics.py:210-212` (`sqrt(1e-16) = 1e-8`).
- Normalization/inverse: `grouped_ufno_mionet_v3/normalization.py:112-136`.
- Training record-relative denominator: `saved_time_phase_operator_v4/losses.py:435-453`.
- Composite recovery weights: `losses.py:550-675` and the active YAML `loss` block.
- Fixed checkpoint gate: `scripts/train_saved_time_v4_full_support.py:404-422`.
- Gradient zeroing and stepping: `train_saved_time_v4_full_support.py:2249` and `:3740-3777`.
- Evaluation mode: `train_saved_time_v4_full_support.py:2922-2985`.
- Metric-aligned selector: `saved_time_phase_operator_v4/data.py:205-225` and
  `sampling.py:316-379`; its identical coverage selector is audited in
  `tests/saved_time_phase_operator_v4/test_full_support_runner.py:118-153`.

## Safe changes prepared

- Configurable multiscale-adapter `Dropout2d`, default `p=0`; the proposed branch
  uses `p=0.05`. Evaluation is deterministic and the zero-output adapter remains
  exact-parent identity even in training mode.
- Configurable `record_axis_strategy: uniform`, defaulting to legacy `rad`. Uniform
  replay keeps 4,480 appearances and 140 updates per epoch; only replay probability
  changes.
- A `fixed_train_gate` training-time policy that uses 24 fixed gate-aligned frames
  for every training record. It keeps the original record schedule and compute and
  raises exact overlap with the 32-frame gate from 5.92% to exactly 75%.

The intended single-variable `fixed_train_gate` branch, r5d, started after r5c's
natural exit. A post-launch selector audit found that r5d did not in fact align
the time samples: training uses seed 372, whereas `_evaluate` historically added
7919. Across the actual 48 gate records, the 24 training frames intersect the 32
gate frames in only 0--4 frames (mean and median 2). This is 8.33% of the training
set or 6.25% of the gate set, rather than the intended 24/32 = 75%. With the same
seed, every one of the 48 records has all 24 training frames inside its 32 gate
frames. r5d is therefore diagnostic and cannot test the claimed aligned-time
hypothesis, irrespective of whether a retry happens to pass.

r5d subsequently finished naturally. All eight attempts were rejected. The final
candidate score was `0.27725556830198084`, which remained
`1.2009638567511693e-6` above the immutable r5b score
`0.2772543673381241`; maximum-attempt exhaustion therefore left r5b selected.

The safe correction adds the content-bound
`epoch_validation_control.time_selector_seed_offset` field. Legacy configurations
retain offset 7919. A train-only `fixed_train_gate` run now fails before training
unless it explicitly binds offset 0. The unlaunched r5e configuration binds zero
and changes no other conceptual variable from r5d. Its CPU audit reconstructs all
2,240 records, 4,480 appearances, and 140 updates per epoch. The random-replay and
dropout branches remain unlaunched so that variables are not conflated.

## Corrected aligned-time outcome and gradient mechanism

The corrected r5e control subsequently completed naturally. It enforced the
24-of-32 subset contract on every record, but all eight candidates remained above
the corrected r5b baseline of `0.27717799602124693`. The best candidate was
`0.27718377962083424` at the `1/128` multiplier, a regression of
`5.783599587305677e-6`. Selector mismatch was therefore a real protocol defect but
not the sole cause of the optimization plateau.

Across the 1,120 r5e optimizer updates, 1,026 updates (91.607%) triggered adapter
gradient clipping. Ninety-seven updates had temporal-difference loss greater than
10, and every one was clipped. Their median pre-clip gradient norm was
`210629.61`, compared with `175.38` when temporal loss was at most 1. The temporal
loss and gradient norm have log-scale correlation `0.422`; 16 of the 20 largest
gradient norms coincide with temporal loss above 10. The active temporal term
normalizes pooled consecutive-frame differences only by
`norm(target[t+1]-target[t])`, clamped at the absolute `1e-8` floor. Unlike the
frame and delta terms, it has no energy-relative denominator floor. Nearly static
consecutive targets can therefore create very large auxiliary-loss gradients that
are absent from the primary checkpoint metric.

This pairing does not prove that every clipped update is caused by the temporal
term, because four of the 20 largest norms have small temporal loss. It does show
that the temporal denominator is a concrete, code-level instability and a better
next one-factor target than adding more capacity. The running r5f/r5g experiments
must remain unchanged. After both terminate naturally, the next train-only test
should compare the current objective with either `temporal_difference: 0` or a
temporal denominator floored relative to the complete-record target energy. Those
two remedies should not be combined in the first test.

All three candidate branches remain diagnostic-only because their parent lineage
and time-axis RAD evidence include historical validation-derived choices.

## Recommended experiment order

1. Retain the naturally completed r5d result as diagnostic only; its selector
   mismatch makes it an invalid aligned-time ablation and it did not beat r5b.
2. Retain the naturally completed r5e result as evidence that exact time alignment
   alone does not resolve the plateau.
3. Allow the active equal-compute uniform-random replay and queued dropout branches
   to terminate naturally without source changes.
4. Then run one train-only temporal-objective intervention, first disabling the
   unstable temporal auxiliary term or, in a separate run, adding an
   energy-relative temporal denominator floor.
5. Test an information-preserving late-coda output path only after the objective
   intervention, because the complete fixed-position panel shows strong late and
   high-band failure.
6. Any promotion claim requires a clean train-only-selected ancestor and train-bound
   adaptive evidence before opening validation/test_id.
