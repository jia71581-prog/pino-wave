# V61 multiscale spectral band-adapter design

Date: 2026-07-20

## Decision context

V60 proved that a frozen V49 anchor plus an architecturally band-limited
increment can improve the complete 401-stored-time wavefield without damaging
the registered high spatial band.  Its best three-record continuation reached
aggregate relative L2 `0.3632131252`, a cumulative `10.9928%` reduction from
the exact V49 anchor `0.4080715670`.  The high-band anchor delta remained only
`5.642e-6`.

The result does not pass the registered 20% three-record overfit gate.  This is
not explained by optimization instability: the fixed-panel error decreased
monotonically, late gradient norms stayed finite, and the prefix clip never
activated.  It is also not explained by the low-rank bottleneck alone: raising
the V60 expert rank from 16 to 64 made the all-time result slightly worse.  V60
therefore remains useful evidence for the hard spectral guard and sparse family
routing, but its one local depthwise convolution is rejected as the production
adapter.

## Requirements

- Preserve the exact V49 prediction at initialization and preserve every V49
  coefficient in the registered high spatial-frequency band thereafter.
- Keep the pressure-release top row exact and keep the other three CPML/data
  semantics unchanged.
- Keep one source per record, encode a medium once, condition on source
  location/frequency without receiver inputs, and output a 201x201 complete
  wavefield at only the 401 stored time indices.
- Reuse the existing shared V4 decoder tensor.  This retains the current
  multiscale medium encoder, source branch, MIONet/DeepONet-style multiplicative
  fusion, attention, coarse-field lift, phase features, and time conditioning.
- Execute only the selected family expert for exact one-hot routes; do not
  restore dense three-expert activation memory.
- Increase low/mid-band spatial capacity materially while keeping the frozen
  anchor and the final band projection as independent safety layers.
- Require a successful three-record all-401-time gate before any long four-GPU
  training.

## Considered designs

### A. Sparse multiscale spectral residual adapter (selected)

Replace each shallow V60 expert with a compact U-FNO-like residual expert.  It
uses a full-resolution factorized spectral/local stack and a pooled coarse-scale
stack, upsamples the coarse result, fuses both scales, and applies exact-time
FiLM before a zero-initialized scalar wavefield head.  The existing outer
low/mid-band projection remains unchanged.

This directly targets wavefront propagation and long-range phase coupling while
preserving V60's verified high-band invariant.  It reuses the already computed
shared representation instead of encoding the medium or source again.

### B. Frozen V49 plus a complete second decoder

A full second decoder followed by frequency splicing has the most capacity, but
nearly duplicates decoder compute and activation memory.  The measured V49/V59
splice improved aggregate error by only `1.80%`, so duplication is not justified
before testing the smaller multiscale residual path.

### C. Continue widening or deepening the local V60 expert

Rank 64 already failed to beat rank 16.  More channel rank without nonlocal
spatial propagation does not address the observed plateau and is rejected.

## Architecture

`BandLimitedFamilyAdapter` gains a backwards-compatible architecture selector.
The default remains the existing V60 low-rank expert, so old checkpoints and
parameter names are unchanged.  V61 selects `multiscale_spectral` and constructs
three `MultiscaleSpectralResidualExpert` instances under the same
`dense_decoder.band_limited_adapter.experts` prefix.

For one selected family, the expert performs:

1. project the shared decoder tensor from parent width to a configurable latent
   width with a 1x1 convolution;
2. run a checkpointed full-resolution `FactorizedComplexResidualStack`;
3. average-pool the projected tensor by two, run a separate checkpointed
   spectral/local stack, and bilinearly upsample it to the exact original size;
4. concatenate full and coarse tensors and fuse them with a 1x1 convolution;
5. generate scale and shift from the existing five exact-time/source features,
   apply FiLM and GELU, and map to one channel;
6. initialize the final output weight and bias to exact zero.

The first diagnostic configuration uses latent width 32, spectral rank 32,
32 retained modes, full depth 4, and coarse depth 2.  These values are
configuration fields rather than hidden constants.  The implementation records
the exact adapter parameter count and peak allocated memory before promotion.

The operator-level safety path is unchanged:

```text
anchor_surface = free_surface(V49 prediction)
raw_increment = routed_multiscale_expert(shared, exact_time_features)
increment_surface = free_surface(raw_increment)
safe_increment = project_low_mid_increment(increment_surface)
prediction = anchor_surface + safe_increment
```

`project_low_mid_increment` removes every normalized radial mode above `2/3`
and enforces a zero top row within the retained band.  Consequently adapter
capacity cannot alter the anchor's registered high band.  No receiver tensor is
introduced anywhere in this path.

## Configuration and checkpoint contract

The probe/model configuration adds explicit adapter architecture, latent width,
spectral rank, modes, full depth, coarse depth, and activation-checkpointing
fields.  All fields have legacy-safe defaults.  Invalid combinations, such as a
multiscale architecture with zero depth or without the existing family router,
fail before model construction.

V61 transfers the exact V49 parent strictly.  Missing keys are allowed only
under the registered adapter prefix, every inherited tensor must match, and the
identity audit must prove:

- the candidate prediction exactly matches V49 before its first update;
- all new output heads are zero;
- the parent checkpoint/run/identity digests match the registered values;
- only adapter parameters are trainable in the initial stage.

Optimizer state is intentionally fresh because the trainable parameter set is
new.  AdamW remains the diagnostic optimizer; an optimizer comparison is only
admitted after the architecture passes the overfit gate, so optimizer and
capacity effects are not confounded.

## Validation and promotion gates

1. **Red/green unit tests:** cover configuration validation, shape/finite
   forward, exact zero-init identity, nonzero internal gradients after the
   output head wakes up, exact one-hot sparse dispatch, soft-route equivalence,
   unchanged high-band coefficients, and exact free-surface top row.
2. **Transfer audit:** reject any unregistered checkpoint mismatch and bind the
   identity report to the exact V49 checkpoint/run digests.
3. **Three-record overfit:** train one uniform, one corrected layered, and one
   Marmousi record with staggered exact-time appearances.  Restore the selected
   checkpoint and evaluate all 401 stored times.  Require at least 20% aggregate
   reduction from V49, all three families improved, and high-band delta no more
   than `2e-5`.
4. **Same-panel validation:** require improvement on the identity-bound
   48-record panel using both its registered sampled-time panel and its complete
   401-time evaluation.  A three-record pass alone is never promotion evidence.
5. **Four-GPU resource smoke:** use measured capacity, not a requested batch
   number.  The established production candidate is physical/macro batch 24 per
   rank, 24 frames, four macros per update, global effective batch 96.  It may be
   reduced only on measured OOM; peak allocation must remain below 23 GiB per
   4090D and every rank must execute useful work.
6. **Short curriculum pilot:** uniform, then corrected layered, then Marmousi,
   with balanced replay, per-epoch checkpoints, intermediate-step metrics, and
   an external gate.  Only a passing complete epoch may seed long training.

The project completion gate remains independent held-out aggregate relative L2
below 10%, with uniform, layered, and Marmousi each below 12%.  Numerical solver
outputs cannot substitute for neural-operator predictions.

## Failure and recovery

- A nonzero adapter high-band increment or top-row violation is an
  implementation failure, not a tunable metric.
- An overfit reduction below 20% rejects this capacity choice; its best
  checkpoint and evidence are retained, but no long run starts.
- OOM triggers only a measured microbatch/frame adjustment while preserving the
  effective batch and appearance schedule where possible.
- Non-finite loss, zero required gradients, missing epoch checkpoints, digest
  mismatch, or family regression stops promotion and keeps the last verified
  parent.
- All remote checkpoints, logs, metrics, and gate reports are synchronized into
  the local project artifact tree before a remote run is treated as complete.

## Evidence-driven optimizer amendment

The first CUDA smoke measured a fixed-panel jump from `0.4024023289` to
`1.3697543506` after one AdamW update at `3e-3`, even though the adapter gradient
norm was finite at `2.4387`, clipping scale was exactly one, and the high-band
delta remained safe.  One-step controls at `1e-4`, `3e-4`, and `1e-3` showed
increasing regressions.  At 20 updates, the best global-rate all-401 result was
only `0.4077054222` at `1e-5`, a `0.0897%` reduction from the registered
`0.4080715670` anchor; `3e-5` reached `0.4077523446`, while `1e-4` regressed to
`0.4082296591` and `2e-4` selected the unchanged anchor.  Adam's first update magnitude is
approximately the learning rate for every nonzero-gradient parameter; the
zero-initialized output head therefore needs a much smaller step than the
2.44-million-parameter spectral feature path.

Three remedies were considered.  Keeping a single global learning rate is safe
only at a value too small for efficient feature learning.  Adding an output
normalization factor would alter the architecture after its identity tests and
would still couple feature and head step sizes.  The selected remedy is a
parameter-disjoint AdamW split: every
`dense_decoder.band_limited_adapter.experts.*.output.*` tensor uses an explicit
output-head learning rate, while all other adapter tensors use an explicit
feature learning rate.  The first registered split uses output `1e-5` and sweeps
feature `1e-4`, `3e-4`, `1e-3`, and `3e-3` over a short fixed window.  Both groups retain
the same weight-decay classification and epoch schedule factor.

This amendment changes optimizer state only.  It preserves exact initialization,
state-dict names, sparse routing, full-field output, frequency projection, data,
and boundary contracts.  Unit tests must prove disjoint/exhaustive parameter
membership and exact learning rates before the split is used remotely.  The
user's standing approval for automatically verified stages applies.

## Approval

The user previously authorized automatic execution of every verified next stage
without repeated approval.  This design is therefore approved under that
standing instruction.  Self-review found no placeholders, receiver dependence,
time interpolation, boundary change, data-contract change, FWI work, or GitHub
publication in scope.
