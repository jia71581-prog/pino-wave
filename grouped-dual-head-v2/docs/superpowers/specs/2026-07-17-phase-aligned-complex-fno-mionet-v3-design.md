# Phase-Aligned Complex-FNO Multi-Input DeepONet V3 Design

Date: 2026-07-17

Project: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation`

Implementation root: `grouped_ufno_mionet_v3/`

## 1. Objective

V3 replaces the under-expressive V2 operator with a phase-capable, source-isolated
acoustic neural operator that:

- encodes each velocity field once and supports independent single-source records;
- queries normalized pressure at arbitrary `(x,z,t)` coordinates;
- renders complete `201×201` wavefield snapshots at arbitrary requested times;
- uses no receiver observations as model inputs;
- remains differentiable with respect to velocity, source parameters, and query coordinates;
- excludes the `anomaly` medium family from training, validation, gates, and reports;
- does not implement FWI, instance adaptation, or a production launch in this design cycle;
- must pass accuracy gates before any larger pilot or production run starts.

Every target is the solution of exactly one source in one velocity model. V3 never
superposes independent sources in a training target.

## 2. Root-Cause Evidence

### 2.1 The V2 spectral block is not a learned Fourier operator

V2's `SpectralResidualBlock` truncates the input FFT with a fixed binary mask, applies
an inverse FFT, and then uses a spatial `1×1` convolution. It has no mode-dependent
learned complex weights. Its parameters are only local convolution, point convolution,
and normalization tensors. Consequently it is a fixed low-pass CNN residual rather
than an FNO kernel capable of learning mode-dependent propagation phase.

The complete V2 model has 421,507 parameters. Its dense and query heads contain only
47,489 and 37,954 parameters respectively. Increasing optimizer effort cannot create
the missing complex spectral operator.

This differs from the Fourier Neural Operator definition and official implementation,
which contract retained Fourier coefficients with learned complex mode weights:

- Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces With
  Applications to PDEs*, JMLR 24 (2023):
  https://www.jmlr.org/papers/v24/21-1524.html
- Official NeuralOperator `SpectralConv` implementation:
  https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py

### 2.2 The residual error is propagation phase, not global amplitude

The best V2 L-BFGS checkpoint was audited on the exact gate records. Mean results were:

- dense relative L2: 0.167 in the independent diagnostic calculation;
- query relative L2: 0.213;
- dense correlation: 0.984;
- query correlation: 0.976;
- optimal post-hoc amplitude factors: approximately 1.0;
- predicted/target energy above radial frequency 16: 0.157/0.164;
- early four-frame dense error: 0.080;
- middle-frame dense error: 0.139;
- late four-frame dense error: 0.255.

Amplitude rescaling did not reduce the error. The global high-frequency energy was also
close to the target. In uniform media, however, late-time radial energy centroids drifted
by approximately 36–39 m. The dominant failure is therefore local phase and wavefront
position accumulated during propagation.

### 2.3 V2 lacks a propagation-aligned coordinate

V2 encodes `t-t0`, spatial displacement, radius, and separate Fourier features, but it
does not explicitly encode the wave-propagation variable

```text
tau(x,z,t) = t - t0 - T(x,z).
```

It asks shallow GELU networks to learn the multiplicative time-distance coupling and
localized oscillatory envelope from scratch. Published work on Fourier features, SIREN,
and Gabor wavefield representations identifies this spectral bias and shows the value of
periodic or wave-adapted coordinates:

- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low
  Dimensional Domains*, NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html
- Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions*,
  NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html
- Abedi, Pardo, and Alkhalifah, *Gabor-Enhanced Physics-Informed Neural Networks for
  Fast Simulations of Acoustic Wavefields*:
  https://arxiv.org/abs/2502.17134

### 2.4 Long-time dynamics require an explicit architectural response

Recent acoustic-wave operator work reports long-term degradation as a central failure
mode and motivates latent dynamics rather than static time conditioning. V3 does not
adopt a recursive Koopman rollout because the required base operator must query arbitrary
continuous times without receiving early wavefield frames. It instead incorporates phase
alignment and time-conditioned complex spectral refinement in a direct operator:

- *Solving acoustic wave equation with Koopman neural operator*, GJI 246 (2026):
  https://academic.oup.com/gji/article/246/1/ggag196/8688499

### 2.5 V2 strengths retained in V3

V3 is a targeted correction, not a wholesale removal of V2. The following V2 contracts
and useful inductive biases remain first-class behavior:

- one velocity encoding is reused by multiple independent single-source records through
  `record_to_medium`, without superposing their targets;
- a multiscale medium pyramid supplies both global rank features and point-sampled local
  features;
- the source branch combines physical source parameters, source-map features, and the
  local medium at the source;
- the query path keeps a direct local-geometry residual in parallel with global low-rank
  fusion and medium-token attention;
- the dense path keeps source-conditioned FiLM-style scale/bias modulation and multiscale
  local features before its new learned-complex spectral correction;
- query and dense heads both receive true wavefield supervision, and their agreement is an
  auxiliary constraint rather than a replacement target;
- the normalized pressure/source-amplitude convention, unit-mass source-map check, exact
  free-surface output factor, time blocking, and physical decoding boundary remain explicit.

The V2 behaviors deliberately removed are the fixed FFT low-pass block, deletion of source
position/amplitude inside the source parameter branch, and a time-only phase coordinate that
does not account for propagation travel time.

## 3. Alternatives Considered

### 3.1 Selected: phase-aligned complex-FNO multi-input DeepONet

A learned complex FNO encodes the medium and refines dense fields. Independent velocity,
source, approximate travel-time, and coordinate branches are fused multiplicatively as a
multi-input DeepONet/MIONet. A shared phase-coordinate core serves arbitrary queries and
dense grid rendering.

This option directly addresses the missing phase operator while retaining fast parallel
full-field output and continuous point queries.

### 3.2 Rejected as the base model: recursive Koopman propagator

A Koopman model can improve long-time stability but requires a wavefield state to advance,
introduces rollout error and cost, and does not naturally provide direct continuous-time
queries. It remains a future comparison after V3 passes the supervised gates.
### 3.3 Rejected as the production renderer: query-only SIREN/Gabor field

A unified implicit network is expressive and continuous, but evaluating all
`401×201×201` output values point by point is likely slower than a grid-parallel spectral
renderer. Periodic/Gabor features are retained inside V3's coordinate trunk instead.

## 4. Data Contract

### 4.1 Source dataset and allowed medium families

V3 reads the existing virtual dataset:

`/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`

It uses only:

```text
allowed_medium_types = [uniform, layered, marmousi]
```

The current source/cache census is:

| split | uniform | layered | marmousi | anomaly | V3 total |
|---|---:|---:|---:|---:|---:|
| train | 420 | 1120 | 700 | 560 excluded | 2240 |
| validation | 90 | 240 | 150 | 120 excluded | 480 |

Filtering occurs in the V3 index layer. The source HDF5 and V2 caches are not modified.
Startup fails if the allowed family set or expected counts do not match the recorded
manifest.

### 4.2 Manifest and split integrity

The V3 manifest records:

- source HDF5 SHA-256 and source manifest hash;
- allowed and excluded medium families;
- counts before and after filtering for every split;
- sample IDs, group IDs, split IDs, and per-sample SHA-256 values;
- normalization metadata and amplitude convention;
- time axis, spatial axes, interpolation convention, and code/config digests.

Train-derived normalization never observes validation or test records. The checkpoint
stores the V3 manifest digest and refuses a mismatch.

### 4.3 Continuous-time targets

The source VDS contains full wavefields with shape `[record,401,201,201]`. V3 samples:

- exact saved times for the majority of supervision;
- continuous times between adjacent saved steps for a bounded fraction of supervision;
- the continuous target by linear interpolation between the two adjacent physical frames.

Each interpolated example records its left/right time indices and interpolation weight.
Exact-time validation remains the primary accuracy gate; interpolated-time validation is
reported separately so interpolation cannot make the gate easier.

## 5. Numerical Representation

### 5.1 Physical normalization

V3 preserves the V2 order-one pressure convention:

```text
p_normalized = p_physical / (p_scale * source_amplitude).
```

Velocity and source parameters use train-only fixed scales. V3 recomputes the normalization
manifest after applying the three-family filter. It may reuse the numerical scale only if a
streaming audit proves it matches the filtered train distribution within the recorded
tolerance; otherwise a new scale is fitted.

### 5.2 Precision

- FFT, learned complex contraction, output projection, losses, normalization, and physical
  decoding run in FP32.
- Local convolutions, attention, and branch MLPs may use BF16 after equivalence tests.
- Learned complex weights are stored as paired real tensors with a final size-two
  real/imaginary axis and converted for contraction. This gives standard real-parameter
  AdamW semantics and avoids optimizer ambiguity for native complex parameters.

## 6. Architecture

The model is named `PhaseAlignedComplexFNOMIONet`.

### 6.1 Learned complex spectral convolution

`LearnedComplexSpectralConv2d` performs:

1. FP32 `rfft2` on an odd or even spatial grid;
2. separate learned contractions for positive and negative vertical modes;
3. mode-dependent complex channel mixing;
4. zero-fill outside retained modes;
5. FP32 `irfft2` to the exact input shape;
6. a local spatial residual and normalized nonlinear channel mixer.

The initial configuration uses width 64, spectral channel rank 40, and modes
`[20,16,12,8]`. Low-rank input/output channel projections bound the expected complete
model size near 10–20 million real parameters while retaining a learned complex matrix for
each retained mode.

Unit tests compare the result with explicit Fourier-domain multiplication and cover both
vertical frequency halves, odd `201×201` shapes, gradients, serialization, and chunking.

### 6.2 Velocity branch

The velocity branch is a multiscale complex U-FNO. It outputs:

- local feature pyramids;
- spatial medium tokens;
- a global low-rank velocity branch vector.

Every medium token contains explicit normalized `(x,z)` Fourier position features before
cross-attention. Tokens therefore preserve where an interface or heterogeneity occurs.
The multiscale pyramid/global-rank/local-sampling decomposition is retained from V2, while
its fixed low-pass residual is replaced by the learned complex operator.

### 6.3 Source branch

The source branch independently encodes:

```text
[x_s, z_s, f0, t0, amplitude, source_map local features].
```

It outputs source hidden features and a rank vector. Position is not silently removed from
the parameter branch; translation structure is enforced through relative coordinates and
tests rather than by deleting the source position.

As in V2, source-map features and velocity features sampled at the physical source location
enter the branch. V3 additionally keeps amplitude in the parameter branch while enforcing
that physical pressure decoding applies amplitude exactly once.

### 6.4 Differentiable travel-time branch

For each query point `y=(x,z)`, V3 samples `K` points along the straight segment from the
source to `y` using differentiable bilinear `grid_sample` on the raw velocity field:

```text
T_ray(y) = distance(source,y) * mean_k(1 / v(ray_k)).
tau(y,t) = t - t0 - T_ray(y).
```

The initial value is `K=12`, configurable and tested. For a dense grid, `T_ray` is computed
once per source and reused across all requested times. For point queries it is computed only
for requested coordinates.

`T_ray` is an approximate differentiable inductive feature, not an exact eikonal solution.
Refraction, reflection, scattering, and boundary effects remain learned residuals. Uniform
media must satisfy `T_ray=r/v` to numerical tolerance.

The travel branch encodes `T_ray`, local/path velocity summaries, `tau`, causal-cone
features, and the source-frequency phase `2*pi*f0*tau`.

### 6.5 Periodic coordinate trunk

The coordinate trunk consumes:

```text
[x,z,t, dx,dz,r, t-t0, T_ray, tau, f0,
 Fourier(x,z,tau), sin/cos(2*pi*f0*tau), Gabor(tau,f0)].
```

It uses periodic/SIREN-style layers with their required initialization and a bounded Gabor
bank. Raw features have a residual path so periodic activations cannot erase low-frequency
information.

### 6.6 Multi-input DeepONet/MIONet fusion

The coarse continuous operator is

```text
p_coarse(y,t) = sum_r Bv_r(v,y) * Bs_r(source)
                      * Btau_r(v,source,y,t) * Trunk_r(y,t)
                  + R_local(y,t).
```

`Bv` combines global velocity rank and local velocity features. `Bs`, `Btau`, and `Trunk`
are independent branches. `R_local` uses sampled local medium features and cross-attention
to position-encoded medium tokens. It preserves V2's direct
`query + sampled-local + attended-token` residual path so small-scale geometry is not forced
through a global rank bottleneck. Gradient audits require every branch and the spectral
weights to participate.

### 6.7 Query path

The query path evaluates the shared fusion for arbitrary batched `(x,z,t)` coordinates.
Chunked and unchunked results must agree. It returns normalized pressure and can be decoded
to physical units with source amplitude applied exactly once.

### 6.8 Dense renderer

For each requested time block, the dense renderer:

1. vectorizes the shared branches across the complete `201×201` grid;
2. forms coarse rank-channel phase-aligned fields;
3. concatenates multiscale local velocity and source-map features;
4. applies the retained V2 source/time FiLM scale-bias modulation;
5. applies time-conditioned learned complex spectral residual blocks;
6. projects to one normalized pressure field per time.

The grid travel-time field and medium encoding are cached per medium/source. Time blocks
bound memory. The renderer supports arbitrary time subsets and streams all 401 saved frames
when requested.

Query and dense paths share the physical coordinate/travel-time branches but retain a dense
spectral correction for speed and spatial coherence. They both receive true target
supervision; consistency never replaces a target loss.

Both paths apply the same explicit top free-surface factor used by V2, with the factor
tested independently so the learned operator does not spend capacity rediscovering the
known pressure boundary condition.
## 7. Losses

The supervised objective is

```text
L = lambda_point * L_point
  + lambda_frame * L_frame
  + lambda_complex * L_complex_spectrum
  + lambda_phase * L_spectral_phase
  + lambda_grad * L_spatial_gradient
  + lambda_dt * L_time_difference
  + lambda_consistency * L_query_dense_consistency.
```

- `L_point` is inverse-probability-corrected query Huber plus per-record relative L2.
- `L_frame` computes relative L2 independently for every time frame before averaging, so
  high-energy frames cannot hide early or late errors.
- `L_complex_spectrum` compares normalized complex 2-D FFT coefficients, not only their
  magnitudes.
- `L_spectral_phase` compares unit complex phase only on target modes above a recorded
  energy threshold; near-zero modes are masked.
- `L_spatial_gradient` compares wavefront derivatives.
- `L_time_difference` compares adjacent-time pressure differences on exact or interpolated
  time pairs.
- `L_query_dense_consistency` is evaluated at identical grid/time points with symmetric
  scaling.

Receiver traces are not a training objective and are never inputs. They may be derived from
query predictions only for diagnostic reports.

Every component is logged before and after weighting. Nonfinite phase values, an empty phase
mask, or sustained auxiliary-loss dominance stops training.

## 8. Training Curriculum

### 8.1 Structural smoke

Run CPU and CUDA shape/gradient tests on synthetic uniform and layered examples. Verify all
branches, complex weights, dense renderer, and query path receive finite nonzero gradients.

### 8.2 One-record uniform-medium gate

Overfit one energetic uniform record with exact early, middle, and late frames. Requirements:

- early-frame mean relative L2 below 0.05;
- middle-frame mean relative L2 below 0.05;
- late-frame mean relative L2 below 0.05;
- query relative L2 below 0.05;
- radial energy-centroid error below one spatial grid cell where the metric is defined.

Failure blocks the mixed-family gate and produces a phase/centroid diagnostic report.

### 8.3 Nine-record balanced gate

Select exactly three energetic records each from `uniform`, `layered`, and `marmousi`.
The gate requires:

- aggregate query relative L2 below 0.10;
- aggregate dense relative L2 below 0.10;
- each medium family's query and dense relative L2 below 0.10;
- late four-frame mean relative L2 below 0.10 for every family;
- no required parameter missing gradients;
- all metrics beat the zero-prediction baseline.

Training begins with balanced early/middle/late exact frames, then increases retained spectral
modes and the continuous-time interpolation fraction. AdamW performs the main fit. A
deterministic full-batch L-BFGS phase is allowed only after an AdamW plateau and uses fresh
curvature history.

### 8.4 Pilot and production policy

Only a passed nine-record gate can authorize a bounded three-family train/validation pilot.
The pilot must improve a fixed validation composite and show credible nonzero wavefields.
Production launch is outside the initial V3 implementation and requires separate explicit
authorization after pilot evidence.

## 9. Validation and Diagnostics

Reports contain:

- per-record and per-family query/dense relative L2;
- per-time early/middle/late and P50/P95 frame errors;
- normalized complex-spectrum and phase errors;
- spatial-gradient and time-difference errors;
- radial wavefront energy-centroid displacement;
- query/dense same-point disagreement;
- prediction/target energy ratio and optimal amplitude-rescaling baseline;
- exact-time and interpolated-time metrics separately;
- full branch and complex spectral gradient audit;
- peak memory, records/s, frames/s, GPU utilization, and I/O wait.

Allowed report families are exactly `uniform`, `layered`, and `marmousi`. Encountering
`anomaly` in a V3 train, validation, gate, or report batch is a hard error.

## 10. Failure Handling

Training stops with an atomic diagnostic report on:

- source manifest, family allowlist, split, or normalization mismatch;
- a V2 checkpoint passed to a V3 restore path;
- nonfinite complex contractions, phase losses, targets, or gradients;
- missing gradients in any velocity/source/travel/trunk/dense/query branch;
- query/dense target-coordinate disagreement;
- auxiliary-loss dominance;
- an empty or invalid continuous-time interpolation pair;
- CUDA out-of-memory after one bounded automatic reduction of the dense time block.

Automatic memory recovery may reduce only the time block. It may not change the record
distribution, medium-family balance, loss definition, or accuracy gate.

## 11. Artifacts and Versioning

New paths are:

- code: `grouped_ufno_mionet_v3/`;
- configs: `configs/grouped_v3/`;
- scripts: V3-specific build/train/evaluate commands under `scripts/`;
- artifacts: `artifacts/grouped_ufno_mionet_v3/`;
- manifest/indices: `/home/jiayh/Data/data/processed/grouped_v3_*`.

Checkpoint format is:

```text
phase_aligned_complex_fno_mionet_v3
```

V3 refuses V2 model and optimizer state. V2 code, checkpoints, gates, and failure reports
remain immutable baselines.

Changes are committed to the existing local branch. No GitHub push is authorized.

## 12. Test-Driven Implementation Order

1. Three-family index filtering and manifest contract.
2. Learned complex spectral contraction and odd-grid behavior.
3. Differentiable ray travel time and gradients.
4. Periodic coordinate/travel-time feature encoder.
5. Position-aware velocity tokens and source branch.
6. Four-input DeepONet/MIONet fusion.
7. Query path and chunk equivalence.
8. Dense renderer and arbitrary-time streaming.
9. Per-frame, complex phase, derivative, and consistency losses.
10. V3 checkpoint incompatibility and atomic restore/save.
11. Structural smoke, one-record gate, then nine-record gate.
12. Pilot implementation only after all prerequisite gates pass.

Every behavior is introduced by a failing test, followed by the minimum implementation and
full relevant-suite verification.

## 13. Definition of Done

The initial V3 implementation is complete when:

- all V3 data, numerical, architecture, loss, checkpoint, and gate contracts have tests;
- anomaly records are demonstrably absent from every V3 data path;
- the one-record uniform gate passes all early/middle/late/query thresholds;
- the balanced nine-record three-family gate passes query, dense, late-time, and per-family
  thresholds below 0.10;
- full wavefield snapshots can be rendered at arbitrary requested times;
- arbitrary point queries and dense rendering share the approved phase-aligned multi-input
  operator inputs;
- comparison figures and an evidence report are written;
- no pilot or production run is launched from a failed gate.

FWI, receiver-conditioned prediction, anomaly-medium training, instance adaptation, and a
production launch are explicitly outside this definition of done.
