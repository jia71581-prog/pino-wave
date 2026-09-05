# CLFC Efficient Two-Frame Instance Adaptation Design

Date: 2026-07-24

Project:
`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project`

Parent artifact:
`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long`

## 1. Decision

Implement **Causal Local-Field Closed-Form Calibration (CLFC)** on top of the
completed epoch-40 local-field checkpoint. The parent model remains frozen and
unchanged. Each validation instance exposes exactly the first two saved
wavefield frames at or after source onset. CLFC estimates twelve bounded scalar
coefficients from those two frames and applies the resulting correction to the
cached parent prediction.

The initial evaluation is staged:

1. one independent validation medium from each of Uniform, Layered, and
   Marmousi;
2. expand to three independent media per family only if the three-instance gate
   passes.

## 2. Bound parent and data

The only permitted parent checkpoint is:

`pretraining/gate4_long/run/best.pt`

Its registered identity is:

- epoch 40, global step 2800;
- checkpoint SHA-256
  `0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f`;
- manifest digest
  `a20c9a65abbc65294062af443e2ceae241ead66450f940f652e2d95aaa0aa92b`;
- config digest
  `f2f54922adc5be4d96245334516cff0e338c62a05f31a7d785968df4aff1b8c3`.

The local VDS and travel-time files are:

- `/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`;
- `/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5`.

Loading must fail closed if any checkpoint, config, manifest, time-axis, or
travel-time identity differs.

## 3. Goals and non-goals

### Goals

- preserve the original pretrained weights byte for byte;
- restrict each instance to twelve fitted scalars;
- avoid optimizer steps and parent-model backpropagation;
- add only seconds of work beyond the parent forward pass;
- prevent future-truth leakage structurally, not just by convention;
- produce paired parent-versus-adapted full-field and receiver metrics.

### Non-goals

- no supervised use of the remaining 399 saved frames during adaptation;
- no full-decoder, local-field U-Net, or encoder fine-tuning;
- no use of the coarse saved-time PDE residual as an acceptance oracle;
- no claim that the three-instance smoke set is a final benchmark;
- no modification or overwrite of `best.pt`.

## 4. Leakage boundary

For source onset time `t0` and registered saved-time axis `time_s`:

```text
k0 = min{k | time_s[k] >= t0}
k1 = k0 + 1
```

The adaptation process may read:

- velocity;
- source coordinates, frequency, onset, amplitude, and source map;
- spatial and temporal coordinates;
- travel-time features derived without target future wavefields;
- true wavefield frames `k0` and `k1`;
- the frozen parent's predictions and internal activations at arbitrary query
  times.

It may not open, cache, derive from, or receive a handle capable of reading any
true target frame after `k1`. A guarded dataset returns only the two observed
frames. A separate evaluator process opens future truth only after the
calibration state, gate decision, and SHA-256 state digest have been written.

A mutation test must replace every target frame after `k1` and demonstrate
identical coefficients, gate decisions, access logs, and state hashes.

## 5. Parent decomposition cache

Run the frozen parent once for all 401 registered times and cache:

- final normalized parent field `P(t,z,x)`;
- residual local-propagation contribution `L(t,z,x)` captured from
  `model.local_field`;
- registered time axis and free-surface factor;
- the two raw parent observation errors.

The captured local field is multiplied by the same free-surface factor used by
the operator before constructing calibration bases. Existing eikonal causality
is already contained in `L`; the calibration must not introduce support outside
that causal envelope.

The cache is detached, stored as float32, and cannot expose an autograd graph.
No parent forward is repeated during coefficient fitting.

## 6. Twelve-parameter calibration

Split `L` into three deterministic spatial-frequency bands with cosine-tapered
radial masks:

- low;
- middle;
- high.

For each band `b`, construct two physical response directions:

- `L_b(t)`, representing amplitude correction;
- centered `dL_b/dt`, representing a first-order phase or arrival-time
  correction.

Multiply both directions by the exact eikonal causal gate captured for the same
query time and by the free-surface factor. This explicit second application is
required because a centered time derivative near first arrival could otherwise
create small support immediately before the parent's gated arrival.

Give each direction a constant and a normalized post-onset linear temporal
coefficient. With

```text
s(t) = clip((t - time_s[k1]) / (time_s[-1] - time_s[k1]), 0, 1)
```

the adapted field is:

```text
P_alpha(t) = P(t)
  + sum_b [
      (a_b0 + a_b1 s(t)) L_b(t)
      + (tau_b0 + tau_b1 s(t)) dL_b/dt
    ].
```

This yields `3 bands × 2 directions × 2 temporal terms = 12` fitted scalars.
The zero vector is exactly the frozen parent prediction.

## 7. Closed-form fit and trust region

Build the design matrix only from spatial samples in true frames `k0` and
`k1`. Normalize each column by a detached RMS scale. Solve ridge systems over a
fixed regularization grid using float64:

```text
lambda in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10}
```

Select regularization by symmetric two-frame cross-fitting:

- fit on `k0`, score on `k1`;
- fit on `k1`, score on `k0`;
- choose the lowest mean held-out observation loss;
- refit on both frames.

Apply a deterministic shrinkage grid
`eta in {1, 1/2, 1/4, 1/8, 0}` and select the largest candidate satisfying all
pre-evaluation gates. The final output hard-projects frames `k0/k1` only after
the raw-fit diagnostics have been recorded.

## 8. Acceptance and rollback

A candidate is accepted only when:

1. both cross-fitted held-out observation losses are finite and no worse than
   the parent;
2. the normalized ridge system condition number is at most `1e6`;
3. each effective band-amplitude correction stays within `±15%`;
4. each phase shift stays within half one saved-time interval;
5. the full-time correction norm is at most `25%` of the parent norm;
6. full-time energy ratio lies in `[0.75, 1.25]`;
7. the correction is finite and preserves the parent's causal support and
   free-surface zero row;
8. the access audit contains only `k0` and `k1`.

If no nonzero shrinkage candidate passes, store a rejected calibration and use
the exact parent field. Rejection is a valid safe result, not an experiment
failure.

## 9. Three-instance smoke evaluation

Before opening future truth, select one independent validation medium per
family with seed 372. Selection is by medium group, not by source record, and is
sealed in `instance_manifest.json`. No sample may be replaced after seeing
adapted future errors.

For every instance, report parent and accepted-or-rolled-back CLFC values for:

- future full-field relative L2 over `k > k1`;
- early, middle, and late future relative L2;
- low, middle, and high spatial-spectrum relative L2;
- phase correlation;
- per-time relative-L2 P50 and P95;
- nine fixed near-surface virtual-receiver relative L2 and normalized RMSE;
- parent forward time, cache construction time, fit time, total inference time,
  peak CPU RAM, and peak CUDA memory.

Store metrics, access audit, coefficients, parent identity, state hash, full
fields, and comparison figures under:

`artifacts/clfc_two_frame_epoch40/smoke3`

The smoke set advances to nine instances only if:

- no sample's future full-field relative L2 regresses by more than 1%;
- aggregate future full-field relative L2 improves by at least 3%;
- all access audits pass;
- all outputs are finite;
- median calibration overhead, excluding the unavoidable parent forward, is
  below 10 seconds.

Failure of the research-improvement gate stops expansion but does not invalidate
the parent baseline or leakage audit.

## 10. Components and interfaces

Add focused modules under
`saved_time_phase_operator_v4/instance_adaptation`:

- `local_field_cache.py`: capture and validate detached parent/local-field
  decomposition;
- `clfc.py`: construct bands and bases, solve ridge systems, apply constraints,
  and return an immutable calibration result;
- reuse `contracts.py` and `data_guard.py` for the two-frame boundary;
- `sealed_evaluation.py`: verify a sealed state before opening future truth.

Add:

- `scripts/run_clfc_two_frame_evaluation.py`;
- `configs/saved_time_v5/clfc_epoch40_smoke3.yaml`;
- isolated tests for identity, closed-form recovery, cross-fitting, constraints,
  deterministic selection, mutation invariance, strict parent loading, and
  sealed evaluation.

The calibration result interface records coefficients, band scales,
regularization, shrinkage, condition number, gate diagnostics, timing, parent
identity, observed indices, accessed indices, acceptance, rollback reason, and
state digest.

## 11. Error handling and reproducibility

- Missing or mismatched identities abort before prediction.
- Singular or ill-conditioned systems roll back to the parent.
- Nonfinite cache, bases, coefficients, or fields roll back and are logged.
- A missing free-surface or causal-support check is a hard error.
- Every selection and numerical operation uses registered deterministic seeds.
- Adapter and evaluator artifacts are written atomically.
- Evaluation does not rewrite any file in the downloaded pretraining artifact.

The downloaded project has no `.git` metadata, so this specification cannot be
committed without inventing repository history. The file itself, source-tree
aggregate hash, checkpoint hashes, resolved config, and output manifests provide
the durable provenance boundary.

## 12. Verification order

1. unit tests for basis construction, solver, trust bounds, and identity;
2. two-frame data-guard and future-mutation invariance tests;
3. strict local CPU checkpoint-load smoke;
4. one GPU single-instance dry run with future truth still closed;
5. sealed one-instance post-hoc evaluation;
6. three-family smoke evaluation;
7. expansion to nine instances only after the registered gate passes.
