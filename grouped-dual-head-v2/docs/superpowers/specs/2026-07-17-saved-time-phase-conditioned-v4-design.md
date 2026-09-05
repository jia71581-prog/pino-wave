# Saved-Time Phase-Conditioned Multi-Input F-FNO V4 Design

Date: 2026-07-17

## Goal and fixed scope

Build a forward acoustic neural operator that:

- encodes a 201 x 201 velocity model once;
- accepts one point source per simulation while allowing multiple source locations and frequencies for the same medium;
- returns the complete 201 x 201 pressure field at any of the 401 times stored in the source HDF5 file;
- preserves the medium/source decoupling and arbitrary-point query API from V3;
- is judged only at stored times. Midpoint/interpolated-time accuracy is explicitly out of scope;
- uses held-out relative L2 below 10% as a hard target, reported by medium family and by early/middle/late time. Training is not declared successful until a sealed evaluation passes.

FWI and anomaly-medium support are out of scope.

## Evidence from the current V3 run

The current model has 10,126,581 trainable parameters:

- medium encoder: 5,701,664;
- source branch: 59,904;
- coordinate branch: 20,416;
- travel-time branch: 13,120;
- fusion/query head: 78,595;
- dense correction head: 4,252,882.

The latest long-refinement checkpoint is
`/data/jiayh/v3_dual_head_long_refinement_v2/best.pt`. Its fixed 12-record stored-time validation metric is 50.667% dense relative L2. Across 20 epochs, this metric moved only from 51.010% to 50.667%, while the learning rate fell from 1.25e-6 to 3.125e-7.

A broader read-only audit evaluated 96 seen training records and 240 held-out records at stored times. On the seen records, corrected dense-field error was 55.87% (early 18.94%, middle 55.76%, late 92.92%). On held-out records it was 83.48% mean and 53.31% median; low-energy late uniform frames make the un-floored mean unstable. Removing the dense correction changed the seen-record mean from 55.87% to 60.37%, so the 4.25M-parameter correction head supplies only a modest gain.

V3 can nevertheless fit nine records to 5.99% complete-field error. Therefore it has enough raw capacity for a small support, but it does not learn the full dataset operator.

The training-time audit exposes a contract mismatch:

- the HDF5 time axis has 401 stored samples from 0 to 1 s at 0.0025 s spacing;
- each record appeared only 13-36 times (median 17) during the 3,740-step run;
- only three frames were requested per appearance;
- each record therefore covered only about 25 distinct requested time positions (median), roughly 6.2% of the stored axis;
- sampling was hard-limited to `source_t0 + 0.60 s`;
- 25% of requests were assigned to midpoint interpolation.

The current dense decoder also receives only global time, relative time, source frequency and source phase. Unlike the coarse point-query head, it does not receive the local travel time, distance, path velocity, endpoint velocity or local causal phase at each grid point. It has only two spectral residual blocks. The observed error grows sharply with propagation time, while a global amplitude rescaling or small rigid spatial shift gives negligible improvement. This is a phase/wavefront and coverage failure, not a simple output-scale failure.

## Research basis

- MIFNO uses separate geology and source branches for source-dependent wave propagation, matching the required multi-input factorization. Its paper reports that 16 layers outperform 8 layers, while also identifying high-frequency/coda underestimation and greater error after longer propagation paths. See Lehmann, Gatti and Clouteau, 2024/2025: <https://arxiv.org/abs/2404.10115>.
- Factorized FNO replaces dense multidimensional spectral contractions with separable spectral layers, allowing deeper networks and improved residual connections at lower cost. See Tran et al., ICLR 2023: <https://arxiv.org/abs/2111.13802>.
- PDE-Refiner attributes long-horizon deterioration to neglected low-amplitude spatial-frequency components and improves them through iterative residual refinement. This supports a residual spectrum curriculum, but its autoregressive diffusion-style rollout is not required for the first implementation. See Lippe et al., NeurIPS 2023: <https://openreview.net/forum?id=Qv6468llWS>.
- Prior seismic FNO/PFNO work trains separate source/frequency-conditioned mappings and reports strong accuracy on variable velocity models. It supports explicit source/frequency conditioning rather than hiding these variables in a generic time MLP. See Li et al., 2023: <https://arxiv.org/abs/2209.12340>.

These papers support additional depth, but they do not show that parameter count alone fixes this repository's failure. The local audits require coverage and propagation conditioning to be fixed first.

## Alternatives considered

### A. Widen V3 only

Increasing width, spectral rank and dense block count is the smallest code change. A four-block dense decoder would increase the model from 10.1M to 14.8M parameters; an eight-block version to 23.8M. This is a useful capacity-control experiment, not the preferred final design. It retains the missing local propagation features and the incorrect time-sampling contract. Current peak training memory is already about 21.2 GB on a 24 GB RTX 3090, so naive widening would also exceed memory.

### B. Full 3D space-time FNO

A 3D operator over all 401 x 201 x 201 output values would enforce temporal coherence and use the exact time grid naturally. It is rejected as the first implementation because one record contains over 16 million output values, activation memory would be excessive, and a single saved-time query would unnecessarily compute the whole trajectory.

### C. Saved-time phase-conditioned multi-input F-FNO (selected)

Retain the V3 medium/source/query branches, but replace the weak dense correction path with a deeper factorized spatial spectral corrector that receives local travel/phase fields. It predicts one or a small block of requested stored frames directly, so errors do not autoregressively accumulate and arbitrary saved-time snapshots remain efficient.

## Selected architecture

### Inputs and reusable state

`encode_medium(velocity)` remains reusable across sources. Each simulation record contains exactly one source with `(x_s, z_s, f0, t0, amplitude)`. `prepare_sources` may batch several sources that map to the same encoded medium or ordinary sources from different media.

### Stored-time contract

The public full-field API accepts physical times, but V4 validates that every requested value is within a small tolerance of the HDF5 time grid and maps it to an integer saved-time index. No target interpolation is performed. A learned saved-index embedding is combined with continuous physical time features, retaining physical meaning without spending capacity on unobserved times.

### Local propagation bundle

For every grid point and requested time, compute from the already cached ray features:

- travel time `T(x,z)`;
- local retarded time `tau = t - t0 - T`;
- distance, path-average velocity and endpoint velocity;
- causal gate around `tau = 0`;
- `sin(2 pi f0 tau)` and `cos(2 pi f0 tau)`;
- multi-scale Gabor envelopes in `tau`.

Project these fields to the decoder width and fuse them with the medium pyramid, source vector, source map and the V3 coarse MIONet field. The dense head must therefore see the same propagation-aligned information that already benefits the point-query head.

### Deep factorized spectral corrector

Use 8 factorized complex spectral residual blocks at width 64. Each block performs separate x- and z-axis spectral mixing plus the V2 local depthwise/pointwise residual path. Use pre-normalization, residual scaling and periodic residual forks every two blocks. This increases effective depth while avoiding the quadratic 2D mode tensor of the current block.

The initial capacity target is 18-25M parameters. Depth and local phase conditioning are independently switchable so a controlled 2 x 2 experiment can distinguish their contributions:

1. V3 depth, no local phase (control);
2. deeper factorized head, no local phase (capacity effect);
3. V3 depth with local phase (conditioning effect);
4. deeper factorized head with local phase (selected candidate).

Activation checkpointing is enabled inside the deep correction stack. If memory still exceeds 23 GB, reduce physical records per microbatch and preserve the effective batch with gradient accumulation; do not silently reduce the number of medium/source examples per optimizer update.

## Data sampling and optimization

- Set interpolation fraction to zero everywhere.
- Sample only indices from the 401-point stored time axis.
- Cover the complete 0-1 s range, including pre-onset zero frames and the late coda. Use stratified bins: pre-onset, onset/early, middle, late.
- Draw at least one energetic frame and one uniformly sampled frame per record update. Use energy-floor-aware importance weights so nearly zero frames do not dominate relative error, while retaining a separate absolute-error zero-frame constraint.
- Track per-record time-index coverage. Before the long run, the scheduler must prove that every training record reaches a configured minimum coverage over the planned epochs.
- Start with AdamW and cosine decay at a fresh-head learning rate, using a lower learning rate for transferred V3 branches. L-BFGS is reserved for small deterministic polishing gates because full-batch closure cost is unsuitable for this stochastic full dataset.
- Train by the existing easy-to-hard medium curriculum, but make time coverage independent of medium stage. Late-time examples are introduced from the start at lower weight and raised progressively.
- Add residual-spectrum loss bands so low-amplitude/high-spatial-frequency coda is not ignored. Keep direct frame relative L2 as the primary objective.

## Evaluation and gates

Interpolation metrics are removed from checkpoint selection. The sealed validation suite must use stored frames only and report:

- energy-floored per-record relative L2 over all selected frames;
- un-floored relative L2 and absolute RMSE, with zero/near-zero frames reported separately;
- early, middle and late error;
- uniform, layered and Marmousi error;
- point-query/full-field consistency at grid points;
- phase correlation and spatial spectrum error;
- train/validation gap on a fixed census larger than one 12-record macro-batch.

Development gates:

1. contract tests: every request is a stored index; midpoint requests fail;
2. one-record overfit below 2%;
3. nine-record balanced overfit below 8%;
4. 48-record capacity probe: selected candidate must beat the V3-depth control by at least 15% relative without family regression;
5. full held-out stored-time aggregate below 10%, with every family below 12% and late-time aggregate below 15%;
6. only after gate 5, benchmark full-field inference against LWC-84.

The 10% outcome cannot be guaranteed before experiments. It is a stop gate: failure triggers the next controlled ablation rather than being reported as success.

## Failure handling and artifacts

Every epoch writes an atomic checkpoint and exact-time validation JSON. The run identity binds the source HDF5 digest, 401-point time axis, model configuration, parent checkpoint and sampling schedule. NaN, OOM and process-exit diagnostics remain explicit. Long training starts only after the 48-record capacity probe chooses a candidate.

