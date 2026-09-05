# AIS-MQFNO Zero-Collapse Recovery and Eight-Network Screen

## Status and scope

This design replaces the failed unnormalized 64-grid B1/S screen with a
versioned normalization contract and an eight-network successive-halving
screen.  It does not open or tune on the sealed test split.  Existing B1 and S
artifacts remain immutable diagnostic controls:

- B1 completed 3000 updates with validation relative L2 `0.999970341682434`.
- S was stopped atomically at update 436 after showing the same zero-collapse
  trend.

The new screen restarts every network from random initialization.  No old B1 or
S model, optimizer, scheduler, sampler, or RNG state may initialize a new run.

## Diagnosis to address

The relative L2 metric is one for an all-zero prediction.  The old B1 result is
therefore statistically indistinguishable from the zero predictor.  Three
observed causes define the required changes:

1. The AIS path declared normalization statistics but fed raw velocity,
   velocity gradients, slowness squared, and raw wavefield targets to the
   network.
2. Raw feature magnitudes span roughly ten orders of magnitude, allowing the
   velocity channel to dominate source, time, and slowness information.
3. A 64-grid representation has about 1.9 points per wavelength at 25 Hz and
   1500 m/s.  The screen therefore needs explicit multiscale and
   dispersion-aware candidates rather than only width changes.

Source frequency, amplitude, onset, and Gaussian width are constant in the
registered train and validation splits.  They are recorded in provenance but
are not additional model inputs in this screen.

## Normalization contract v2

The frozen train-only statistics are the sole normalization authority.  Let
`mu_v`, `sigma_v`, `mu_u`, and `sigma_u` denote the registered velocity and
wavefield statistics, with `eps` from the same file.

For normalized coordinates `xi,zeta` in `[-1,1]`, the five static channels are:

1. `v_hat = (v - mu_v) / max(sigma_v, eps)`;
2. `source_hat = source / max(abs(source).max(), eps)`;
3. `dv_dxi = derivative(v_hat, xi)`;
4. `dv_dzeta = derivative(v_hat, zeta)`;
5. `slow_contrast = (mu_v / max(v, eps))**2 - 1`.

The sixth global feature is normalized physical time
`tau = (t - t0) / (t_last - t0)`.  Native and global branches receive the same
five static definitions.  No per-scene target normalization is allowed.

The model predicts standardized wavefield
`u_hat = (u - mu_u) / max(sigma_u, eps)`.  The field objective combines:

- Hansen--Hurwitz MSE on `u_hat`, including registered late-time weights; and
- Hansen--Hurwitz relative L2 after decoding both prediction and target to
  physical wavefield units.

Receiver, phase, local-spectrum, energy, validation, residual-sampler updates,
figures, and seismic gathers operate in physical units.  This prevents the MSE
scale from vanishing without changing the meaning of reported accuracy.

The statistics file SHA-256 and the literal feature-contract identifier
`ais_normalization_v2` are stored in every checkpoint.  The checkpoint schema
is incremented.  Resume, initialization, evaluation, census, and sealed-test
preflight reject missing or different normalization bindings.  Consequently,
old checkpoints are intentionally incompatible with every v2 training,
initialization, candidate census, and sealed-test path.

There is one narrow read-only exception named `legacy_raw_b1_baseline`.  It
accepts only the immutable B1 checkpoint SHA-256
`91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653`,
forces the original raw-input/raw-output semantics, and emits validation
baseline metrics.  The exception cannot return training state and is forbidden
for resume, initialization, candidate ranking inputs other than the baseline,
test census, recipe freezing, authorization, and sealed-test preflight.

## Eight registered network candidates

All candidates use full 160-step traces, 2048 spatial queries per scene,
identical scene order, identical registered validation sites, AdamW, and the
same seed.  Parameter count and peak GPU memory are recorded before screening.

| ID | Candidate | Change from normalized control |
|---|---|---|
| N0 | Norm-MQFNO | Existing 24-wide, 24 spatial modes, 32 temporal modes, patch 17 control |
| N1 | Wide-MQFNO | Spatial width 32, local width 24, fusion width 48 |
| N2 | Spatial-MQFNO | Spatial width 32 and 32 retained spatial modes |
| N3 | Temporal-MQFNO | 48 temporal modes and fusion width 48 |
| N4 | LargeLocal-MQFNO | Patch 25 and local width 24 |
| N5 | MultiLocal-MQFNO | Parallel patch-9/local-width-12 and patch-25/local-width-12 encoders; concatenated 24-wide local embedding |
| N6 | DispersionHead-MQFNO | N0 plus the 24-mode dispersion residual head defined below |
| N7 | MultiDispersion-MQFNO | N5's 24-wide multiscale embedding plus N6's dispersion residual head |

N5 and N7 use two independent local encoders; neither may emulate multiscale
behavior by resizing a single embedding.  Their 12-wide outputs are
concatenated without summation and passed as the 24-wide local input to the
ordinary fusion layer.  N5 uses fusion width 48.  N7 uses the same fusion width
48 for both its ordinary trajectory path and dispersion head.

The dispersion head in N6 and N7 uses exactly 24 positive temporal Fourier
modes.  For duration `T`, mode `m` has basis frequency `f_m=m/T`; `f_m` is not
the 25 Hz source frequency.  At each query, physical center velocity is
recovered as `v_q=mu_v+sigma_v*v_hat_q`.  For each `(query,m)` the shared MLP
receives the local embedding, `m/24`, `f_m*dx/v_q`, and `f_m*dz/v_q`, where
`dx,dz` are the current stage's physical grid spacings.  It emits two real
coefficients `(a_qm,b_qm)`.  The standardized-wavefield residual is
`sum_m (a_qm*cos(2*pi*f_m*(t-t0)) + b_qm*sin(...))/sqrt(24)` and is added to
the ordinary trajectory prediction before physical decoding.  The MLP hidden
width is 48 and its two-output final affine layer is initialized with zero
weights and zero bias.  Normalization constants are immutable model buffers
bound by the checkpoint normalization hash.

## Successive-halving protocol

### Gate O: one-scene representability

Every candidate trains independently on the same registered training scene for
400 optimizer updates.  Evaluation uses 2048 fixed unique sites not redrawn
during the measurement.  A candidate proceeds only if all conditions hold:

- physical relative L2 is at most `0.35`;
- late-time Q4 relative L2 is at most `0.50`;
- prediction/target norm ratio is in `[0.5, 1.5]`;
- physical prediction-target Pearson correlation is at least `0.80`;
- every loss, gradient, parameter, and prediction is finite.

If N0 fails, the entire screen stops because the normalization path is still
invalid.  If fewer than four candidates pass, the screen stops for design
review rather than relaxing the gate.

### Gate H1: survivor screen

All Gate O survivors restart from the same registered random seed and train for
600 optimizer updates on the same scene sequence.  Gate O weights are never
reused.  Up to eight candidates enter this gate, and the best four advance by
the registered ranking score:

`max_category_relative_l2 + 0.5 * max_category_q4_relative_l2`.

Any candidate with a nonfinite value is eliminated before ranking.  For every
category separately, its prediction/target norm ratio must lie in
`[0.25,2.0]`, its field relative L2 must be at most
`1.05 * zero_relative_l2`, and its field Q4 relative L2 must be at most
`1.05 * zero_relative_l2_q4`.  The reporter computes both zero-baseline fields
from a physical all-zero tensor on the identical category samples; both are
mathematically one for nonzero targets, but the recorded reporter values are
the gate authority.

If hard exclusions leave fewer than four candidates, the experiment stops and
reports the rejected candidates; it does not promote a smaller field.  Ranking
uses the exact update-600 `last.pt`, never an earlier interval-best checkpoint.

### Gate H2: four-network continuation

The four survivors resume exactly from update 600 and continue to a total of
1500 updates.  The same hard exclusions and ranking score select two finalists.
If fewer than two remain after exclusions, the experiment stops.  Ranking uses
the exact update-1500 `last.pt`.

### Gate H3: equal-budget final screen

The two finalists resume exactly and continue to a total of 3000 updates.  Both
therefore have identical final optimizer, scene-draw, query-site, full160 label,
and validation budgets.  The selected network minimizes the registered ranking
score at the exact update-3000 `last.pt`, subject to the accuracy gate below.

## Validation and accuracy gates

Before ranking new candidates, the immutable B1 best checkpoint is evaluated
once with the new category-aware validation reporter but its original raw-input
semantics.  This produces baseline values for `uniform`, `layered`, and
`marmousi` without retraining or test access.

Every formal validation reports, globally and per category:

- field relative L2 and Q1--Q4 relative L2;
- receiver relative L2 and Q1--Q4 relative L2;
- prediction/target norm ratio and Pearson correlation;
- arrival lag, phase error, high-k spatial error, and k--omega error;
- sampler ESS, duplicate fraction, coverage, and inverse-weight diagnostics.

The final candidate must improve both field relative L2 and field Q4 relative
L2 by at least 30% relative to B1 in every category.  Equivalently, for metric
`m` and category `c`, `candidate[m,c] <= 0.70 * B1[m,c]`.  It must also avoid a
greater-than-5% regression against B1 in the same category for this exact
lower-is-better set:

- `receiver_relative_l2` and `receiver_relative_l2_q1` through `_q4`;
- `arrival_mae_s`, `arrival_miss_rate`, and `receiver_lag_abs_s`;
- `receiver_phase_error` and `energy_log_ratio`;
- `komega_relative_l2`, `komega_high`, `komega_relative_l2_q4`, and
  `komega_high_q4`.

There is no additional local-spectrum promotion field in this screen:
`komega_high` and `komega_high_q4` are the registered high-k spatial-spectrum
errors.  Patch-local spectrum remains a training-loss component and diagnostic
only, so it cannot independently accept or reject a candidate.

For each nonnegative error `e`, the rule is
`candidate_e <= B1_e + max(0.05*B1_e, 1e-6)`.  The two higher-is-better guards
are `receiver_xcorr_peak` and `receiver_phase_coherence`, with rule
`candidate_h >= B1_h - max(0.05*abs(B1_h), 1e-6)`.  The final norm ratio must be
in `[0.5,1.5]` and final Pearson correlation must be at least `0.80` in every
category; the wider H1 exclusion bounds do not apply to final promotion.
`arrival_target_coverage`, output shapes, and finite/nonzero flags are
diagnostics or validity checks, not 5% comparison metrics.  Failing any final
condition produces no promoted recipe.

Validation remains restricted to the registered validation split.  Raw test
IDs remain sealed until the existing authorization gate is satisfied.

## Artifacts and provenance

Generated configurations live under `configs/ais_zero_collapse_v2/`.  Runs live
under `artifacts/ais_mqfno_zero_collapse_v2_20260714/` with one directory per
candidate and gate.  The experiment manifest records:

- command and source commit;
- config, split, normalization, checkpoint, and metrics SHA-256 values;
- runtime seed, candidate ID, parent checkpoint, and exact update budget;
- parameter count, peak allocated/reserved memory, wall time, and GPU model;
- per-gate decision and immutable rejection reason.

Writes use the existing atomic checkpoint and JSON publication contracts.
Interrupted candidates resume only from their own schema-compatible `last.pt`.
No artifact from one candidate initializes another.

## Testing and execution order

Implementation follows test-driven development in this order:

1. normalization feature and physical decode unit tests;
2. standardized-HH gradient and zero-error tests;
3. checkpoint normalization-binding rejection tests;
4. multiscale-local and dispersion-head shape/zero-init/gradient tests;
5. config generation and equal-budget contract tests;
6. category-aware validation and ranking/gate tests;
7. CPU end-to-end smoke and exact-resume tests;
8. serial CUDA finite-gradient and memory smokes for all eight candidates;
9. Gate O, then H1, H2, and H3 without test-split access.

Each implementation task receives independent specification and code-quality
review.  A failed gate is reported as a research result; thresholds are not
changed after observing candidate validation results.
