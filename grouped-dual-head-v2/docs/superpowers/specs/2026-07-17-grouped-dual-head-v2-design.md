# Grouped Dual-Head Wave Operator V2 Design

Date: 2026-07-17

Project: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation`

Implementation root: `grouped_ufno_mionet_v2/`

## 1. Objective

Replace the failed grouped V1 training path with a numerically scaled, genuinely dual-head
single-source acoustic operator that:

- predicts complete `[time,z,x]` wavefields on the saved `401×201×201` grid;
- queries pressure at arbitrary receiver coordinates and continuous times;
- encodes a shared velocity model once for independent shots in the same medium;
- keeps interfaces compatible with possible future velocity differentiation, without implementing FWI in this work;
- supports onset-aligned two-snapshot instance adaptation only after the base model passes accuracy gates;
- uses useful GPU work rather than artificial compute, while preserving reproducible validation.

Every record remains one velocity model, one source, and one numerical solution. No target ever
contains a superposition of different sources.

## 2. Root-Cause Evidence From V1

The stopped production run is retained as a failed baseline at:

`artifacts/grouped_ufno_mionet/production_w96_sparse_cache_batch48/checkpoints/last.pt`

Its final durable state is step 9466, epoch 136. It is not a valid warm start by default.

Measured failures:

- the eight dense-head parameters and four source-map projection parameters have no optimizer
  state, proving that those paths never received gradients;
- the dense full-wavefield test has relative L2 approximately `1.39e8`;
- the trained query head has receiver relative L2 approximately `368` on `test_id`;
- the query head remains wrong on its own cached train coordinates, with relative L2 between
  approximately `41` and `87` in the audited examples;
- a checkpoint prediction has 16.3 times the configured objective of a zero prediction on the
  audited cached train record;
- physical targets have RMS near `1e-9`, while AMP-era outputs resolve steps near `6e-8`;
- the configured spectral loss performs a 1-D FFT over a flattened sequence of unrelated random
  receivers and dominates raw MSE by roughly seven orders of magnitude;
- dense loss, dense/query consistency, residual importance sampling, source-map input, stage
  scheduling, and PDE loss exist as unused components rather than an executed training path;
- training records only `last.pt`; no validation-based best model is selected.

The V2 design fixes these causes at their source. It does not continue V1 optimization.

## 3. Alternatives and Decision

### 3.1 Selected: normalized dual-head U-FNO–MIONet

A shared medium/source representation feeds a continuous MIONet-attention query head and a
time-conditioned multiscale dense decoder. The heads are supervised independently and aligned on
identical grid coordinates.

This is selected because it jointly supports fast full fields and arbitrary receivers/times while
avoiding an architectural dead end for a later, separately designed FWI project.

### 3.2 Rejected as primary: query-only implicit decoder

A single query decoder avoids dual-head disagreement but requires about 16.2 million evaluations
for one complete saved wavefield. It remains a correctness oracle and fallback renderer, not the
production full-field path.

### 3.3 Rejected as primary: dense-only temporal U-FNO

Dense-only output can be fast but makes arbitrary receiver/time queries dependent on interpolation,
weakening off-grid accuracy.

## 4. Numerical Contract

### 4.1 Input normalization

Velocity is transformed using fixed training-split statistics stored in every checkpoint. A robust
affine transform maps the expected velocity range to order one; validation and test data never
contribute statistics.

Coordinates use explicit physical-domain normalization. Source position, frequency, delay, and
amplitude each have named scales in the configuration and checkpoint.

### 4.2 Pressure normalization

A single global robust pressure scale `p_scale` is fitted on the train split only, using a
streaming high percentile of absolute nonzero pressure. It must be independent of the target
record at inference. Training targets are

```text
p_normalized = p_physical / (p_scale * source_amplitude)
```

where source amplitude is known and applied exactly once. The network predicts normalized pressure at
order-one scale. Conversion to physical pressure is FP32 even when hidden layers use AMP.

The scale, fitting manifest digest, percentile, epsilon, and amplitude convention are checkpointed.
Resume or inference fails on a mismatch.

### 4.3 Precision policy

Convolutions and MLP hidden activations may use BF16 or FP16 after an accuracy smoke test.
Normalized output projections, losses, FFTs, physical-unit restoration, and PDE residuals use
FP32. BF16 is preferred when it gives equivalent throughput because of its wider exponent range.

## 5. Data and Cache V2

V2 uses separate train and validation caches. Each record stores:

- normalized or raw velocity with recorded normalization metadata;
- all five source parameters and the unit-mass source map;
- onset-aligned, early, middle, and late saved-time indices;
- structured dense spatial tiles or full frames with their exact `[time,z,x]` indices;
- fixed receiver trajectories `[receiver,time]` for meaningful temporal spectra;
- continuous query coordinates and targets;
- sampling probabilities, sample/group IDs, medium family, and source dataset digest.

Sampling combines:

- at least 20% uniform spatial/time queries;
- wavefront/energy-aware queries derived from training targets;
- residual-adaptive sampling after predictions become available;
- inverse-probability weights so importance sampling does not change the target distribution.

The same medium may appear with several independent sources in a macro-batch, but source targets
remain separate. Validation sampling is deterministic and never updated from residuals.

## 6. Architecture

### 6.1 Shared medium encoder

A multiscale 2-D U-FNO encodes each unique velocity once. It outputs local pyramids, global tokens,
and medium coefficients. It must retain spatial features; a global vector cannot be the sole
medium representation.

### 6.2 Source encoder

The source encoder consumes normalized `[x_s,z_s,f_0,t_0,A]`, local medium features sampled at
the source, and the unit-mass source map. The data packer and trainer must pass `source_map`
explicitly. A gradient audit verifies that every source-map projection parameter is updated.

### 6.3 Continuous query head

The query head retains the MIONet product plus source-conditioned attention/local residual. It
returns normalized pressure for arbitrary `(x,z,t)`. Chunked and unchunked results must agree
within configured precision tolerance.

### 6.4 Time-conditioned dense decoder

V1's fixed rank-64 spatial-basis/time contraction is removed. It is too restrictive for translating,
reflecting wavefronts across 401 times.

The V2 dense decoder takes the shared multiscale pyramid, source latent, and a block of 8–16 time
embeddings. Time-conditioned FiLM/AdaGN modulates U-FNO decoder blocks, which produce one pressure
map per requested time. Time blocks bound memory while executing all spatial points in parallel.

The decoder supports arbitrary saved-time subsets during training and streams the full 401 frames
at inference.

### 6.5 Cross-head consistency

At dense-frame grid locations, the query head is evaluated on a deterministic subset. Both heads
receive target supervision; consistency is symmetric in normalized units and never substitutes
for a missing target loss.

## 7. Losses

The stage-1 supervised objective is:

```text
L = L_query
  + lambda_dense * L_dense
  + lambda_consistency * L_consistency
  + lambda_gradient * L_gradient
  + lambda_spatial_fft * L_spatial_fft
  + lambda_trace_fft * L_trace_fft
```

- query and dense data losses combine normalized Huber and per-record relative L2;
- relative denominators use a recorded epsilon and active-energy mask;
- gradient loss compares spatial wavefront derivatives on structured frames;
- spatial FFT is 2-D and only applied to structured grid tiles/frames;
- trace FFT is 1-D along time at fixed receiver coordinates;
- no FFT is allowed over a flattened random-query axis.

Loss components are logged separately before weighting. An automated dominance audit stops the
run if one auxiliary loss exceeds the sum of data losses by more than a configured sustained ratio.

PDE/LWC-84 loss is disabled until both heads pass validation data gates. It is then ramped in FP32
using correctly scaled physical coordinates, source terms, and future/onset masks while supervised
loss remains active.

## 8. Training Protocol

### Stage 0: contract and overfit gate

Before production training:

1. validate cache/source/checkpoint digests and split isolation;
2. run forward/backward and verify finite, nonzero gradients for every required module;
3. overfit eight records containing nonzero early/middle/late wavefields;
4. require both heads to beat the zero predictor;
5. require train relative L2 below 0.10 for both heads;
6. render wavefield and receiver comparisons.

Failure stops the launch.

### Stage 1: supervised dual-head pretraining

Train query and dense heads jointly. Use grouped media, one-source targets, mixed precision under
the precision contract, bounded prefetch, and deterministic validation. Save `last.pt` for
recovery and `best.pt` using a composite validation score.

### Stage 2: physical refinement

Enable a gradual PDE residual only after supervised gates pass. Validation data accuracy may not
degrade beyond 2%; otherwise roll back to the best supervised checkpoint.

### Stage 3: instance adaptation

Only a base checkpoint that passes full-field and receiver gates may initialize the onset-aligned
two-snapshot adaptation system. The adaptation stage remains causally isolated from future labels.
FWI inversion, velocity-gradient acceptance tests, and FWI performance benchmarks are explicitly
deferred to a separate design and implementation cycle.

## 9. Validation, Monitoring, and Failure Policy

Every epoch reports train and validation:

- query relative L2 and normalized Huber;
- dense full-frame relative L2;
- receiver-line relative L2 and correlation;
- per-time P50/P95 relative error;
- spatial/temporal spectral error;
- energy ratio and LWC-84 residual when enabled;
- dense/query consistency;
- samples/s, full-field frames/s, GPU utilization, memory, and I/O wait;
- gradient presence/norm for every module.

Production launch gates:

- zero-predictor baseline computed and stored;
- both heads improve over zero on train and validation;
- no required parameter lacks gradient/optimizer state;
- validation composite score improves during the bounded pilot;
- generated comparison figures show nonzero, spatially varying predictions.

The job terminates with an actionable report on nonfinite values, loss dominance, missing gradients,
scale mismatch, split leakage, validation regression, or checkpoint incompatibility.

The three `ood_canonical` records are reported individually and never tune hyperparameters.

## 10. Accuracy and Performance Gates

Initial base-model gates on `test_id`:

- full-wavefield relative L2 at most 0.30;
- receiver-line relative L2 at most 0.20;
- receiver correlation at least 0.90;
- spectral error at most 0.35;
- no medium family may collapse to a zero or constant output.

Performance is benchmarked after accuracy passes. Dense full-field inference is compared with the
same-grid CPU and GPU LWC-84 solvers using warmup, CUDA synchronization, repeated runs, and separate
compute/transfer/HDF5-write timing. No acceleration claim is made unless measured.

FWI runtime, inversion accuracy, acquisition geometry, regularization, and velocity reconstruction
are not part of the V2 completion criteria.

## 11. Artifacts and Rollout

V2 code, caches, checkpoints, logs, figures, and reports use new paths:

- `grouped_ufno_mionet_v2/`;
- `artifacts/grouped_ufno_mionet_v2/`;
- `/home/jiayh/Data/data/processed/grouped_dual_head_v2_*.h5`.

V1 code and the stopped checkpoint remain immutable diagnostic baselines. V2 never resumes the V1
optimizer or silently migrates its output heads. A medium-encoder warm-start is allowed only as a
separate ablation after the from-scratch V2 pilot passes.

No GitHub push is part of this design.

## 12. Definition of Done

The solution is complete when:

- all numerical, data, architecture, loss, and split contracts have automated tests;
- the eight-record overfit gate passes for both heads;
- a bounded train/validation pilot beats zero and produces credible figures;
- production training runs from a fresh V2 artifact directory with best/last checkpoints;
- test and OOD reports state every accuracy/performance gate with measured evidence;
- onset-aligned instance adaptation is enabled only from a base checkpoint that passes these gates.

FWI is not required for this definition of done.
