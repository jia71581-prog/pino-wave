# Grouped Single-Source U-FNO–MIONet–Attention Operator Design

Date: 2026-07-17

Project: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation`

New implementation root: `grouped_ufno_mionet/`

## 1. Objective

Build a new acoustic-wave neural operator that:

- preserves the physical contract that every training record and every predicted wavefield correspond to exactly one source;
- treats changes in source position, frequency, delay, or amplitude as different training records;
- encodes a shared velocity model once when several independent single-source records use that medium;
- supports arbitrary receiver coordinates and arbitrary continuous query times;
- remains differentiable from predicted pressure to the velocity field for future FWI;
- provides a fast dense output path for a complete `[401,201,201]` wavefield;
- increases useful GPU work per HDF5 read without adding artificial burn computations;
- is trained from scratch while retaining `continuous_wave_operator/` and its checkpoints as the baseline.

The implementation will live under `grouped_ufno_mionet/`. It must not overwrite baseline code, checkpoints, metrics, or logs.

## 2. Evidence and Constraints

The unified dataset is:

`/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`

Its split inventory is:

| Split | Records |
|---|---:|
| `train` | 2800 |
| `validation` | 600 |
| `test_id` | 600 |
| `ood_canonical` | 3 |

The training split has 1120 unique `group_id` values. The number of single-source records per group is 1, 2, 4, or 5, with a mean of 2.5. Grouping can therefore remove repeated medium reads and repeated medium encoding for most non-uniform records.

The VDS maps 501 source shards. Source wavefields use `[1,1,201,201]` LZF-compressed chunks. Profiling showed HDF5 workers frequently waiting in `folio_wait_bit_common`, while the GPU reached 100% utilization during kernels but was idle for roughly 67–78% of sampled wall time. The design must improve effective queries per disk read; GPU board power is not itself an optimization objective.

## 3. Non-Negotiable Physical Contract

1. One record is one numerical solution with one source.
2. Sources are never superposed to create a training target.
3. Each source has independent parameters, queries, targets, residuals, and loss.
4. Grouping is an internal batching optimization only.
5. A group may share a medium read and medium latent, but never a target wavefield.
6. The predicted pressure must have finite, nonzero gradients with respect to the velocity input.
7. Full-wavefield axis order is always `[time,z,x]`.

## 4. Model Architecture

The model is named `GroupedSingleSourceUFNOMIONetOperator` and contains one shared encoder with two output interfaces.

### 4.1 U-FNO medium branch

The medium branch receives one velocity tensor per unique medium:

`velocity_mps: float32[M,1,201,201]`

It uses four U-shaped spectral levels with default width 96 and Fourier modes `[24,16,12,8]`. Each level contains a spectral residual block and a local convolutional path. Downsampling builds global context; upsampling and skip connections restore local interface and wavefront information.

The branch returns:

- a multiscale local feature pyramid;
- global medium tokens for cross-attention;
- rank-64 medium coefficients for MIONet fusion;
- differentiable source-location sampling features.

No global pooling result is allowed to be the only representation of the velocity field.

### 4.2 Single-source branch

For every independent source record, the source branch encodes:

- `source_x_m`;
- `source_z_m`;
- `source_f0_hz`;
- `source_t0_s`;
- `source_amplitude`;
- local U-FNO features sampled around that source.

It emits one source latent and one rank-64 source coefficient vector per record. Source amplitude remains an explicit physical scaling input and is not absorbed silently into normalization.

### 4.3 Continuous coordinate trunk

The trunk accepts arbitrary physical `(x,z,t)` coordinates. Fourier coordinate features are projected to:

- rank-64 MIONet trunk coefficients;
- four-head attention queries;
- local feature sampling coordinates.

The same trunk serves grid points, off-grid receivers, and arbitrary times.

### 4.4 Multiple-input DeepONet base

The primary continuous prediction follows an MIONet-style low-rank product:

\[
p_{\mathrm{base}}(q;v,s)
=\sum_{r=1}^{64}
b^{(v)}_r(v)\,b^{(s)}_r(s)\,\tau_r(q),
\qquad q=(x,z,t).
\]

MIONet motivates using separate branches for multiple operator inputs and a shared output-domain trunk: Pengzhan Jin, Shuai Meng, and Lu Lu, “MIONet: Learning multiple-input operators via tensor product,” arXiv:2202.06137, <https://arxiv.org/abs/2202.06137>.

### 4.5 Attention residual query head

The arbitrary-query head adds a source-conditioned cross-attention residual. It attends to global medium tokens and concatenates differentiably sampled local U-FNO features. Attention corrects interface reflections, scattering, caustics, wavefront detail, and high-frequency phase that may not fit a rank-64 product.

\[
p_{\mathrm{query}}(q;v,s)
=g(q,s)\left[p_{\mathrm{base}}(q;v,s)+p_{\mathrm{attn}}(q;v,s)\right],
\]

where `g` applies the initial-time and free-surface hard constraints and explicit source-amplitude scaling.

### 4.6 Fast dense-wavefield head

The fast head avoids running attention for all 16.2 million full-grid queries. A source-conditioned U-FNO decoder creates rank-64 spatial basis maps

`spatial_basis: [M,S,64,201,201]`,

and the continuous time trunk creates

`time_basis: [M,S,T,64]`.

The complete field is reconstructed using one tensor contraction:

```python
wavefield = torch.einsum("mstr,msrzx->mstzx", time_basis, spatial_basis)
```

The dense result receives the same initial-time gate, free-surface gate, and explicit source-amplitude scaling as the query head. For one record, the public output is `[T,201,201]`. With all saved times, `T=401`. The fast head and arbitrary-query head share the U-FNO and source branches but have separate output projections.

## 5. Grouped Data Model

A macro-batch contains at most 24 independent single-source records, packed by `group_id`. The batch carries:

- unique velocity tensors `[M,1,201,201]`;
- a record-to-medium index `[S_total]`;
- source parameters and source maps for every record;
- per-record target frames;
- per-record masks for variable group sizes;
- group, sample, and medium-type metadata.

Before grouping, all velocity arrays associated with a `group_id` must be verified equal. A mismatch is fatal and must identify the group and sample IDs.

## 6. Read-Once Query Reuse

Each single-source record reads 32 complete `201×201` frames once. The in-memory frames generate four independently sampled receiver blocks. Each block contains 512 receivers per frame, so one disk read supports:

`32 × 512 × 4 = 65,536` continuous supervision queries per source.

The four blocks share the same medium and source encoding but retain independent query coordinates and targets. They execute sequentially to bound activation memory. Their gradients are accumulated and the optimizer updates once per macro-batch. The shared branch graph remains valid until the final block; no optimizer update may occur between blocks.

Eight process-local HDF5 readers feed a bounded four-macro-batch prefetch queue. Completed CPU tensors use pinned memory and non-blocking host-to-device transfers. Prefetch depth is bounded and must not grow with training duration.

## 7. Adaptive Sampling and Weighting

Sampling is hierarchical:

1. balance medium families;
2. sample medium groups;
3. sample independent single-source records within groups;
4. select 32 time frames with early, middle, and late-time coverage;
5. sample four spatial receiver blocks using residual-based importance probabilities.

Residual EMA state remains record-specific. Losses are reduced in this order:

```text
query mean → single-source mean → group mean → medium-family mean
```

This prevents a five-source group from receiving five times the weight of a one-source group.

## 8. Training Losses

The default objective is:

\[
L=L_{\mathrm{query}}
+0.5L_{\mathrm{dense}}
+0.1L_{\mathrm{consistency}}
+0.05L_{\mathrm{spectral}}
+\lambda_{\mathrm{pde}}L_{\mathrm{pde}}.
\]

- `L_query`: importance-weighted pressure loss from the MIONet-attention query head.
- `L_dense`: dense spatial loss from the fast head on eight of the already loaded frames.
- `L_consistency`: agreement between both heads at identical grid coordinates.
- `L_spectral`: temporal and spatial Fourier error for phase and high-frequency fidelity.
- `L_pde`: source-aware two-dimensional acoustic-wave residual.

Eight dense frames are selected from the same 32 loaded frames using stratified early/middle/late coverage. This adds no disk reads.

Training uses three stages:

1. **Data pretraining:** query and dense data losses; no PDE loss.
2. **Dual-head alignment:** enable attention, consistency, and spectral losses.
3. **Physics fine-tuning:** ramp `lambda_pde` from zero while retaining data, consistency, and spectral supervision.

Stage transitions, weights, and sampler state are checkpointed. They must not be inferred from wall time.

## 9. Inference Interfaces

### 9.1 Arbitrary receiver/time query

`query_pressure(velocity, source, coords, chunk_size)` returns pressure at arbitrary physical coordinates. It supports receiver gathers and future FWI. Query chunking must be numerically equivalent to an unchunked call within configured floating-point tolerance.

### 9.2 Fast complete wavefield

`predict_wavefield(velocity, source, time_s)` returns `[T,201,201]` using the rank-64 dense head. The default saved-time grid has 401 entries.

### 9.3 Streaming HDF5 export

`export_wavefield_h5(...)` writes:

- `pressure[T,201,201]`;
- `time_s[T]`;
- `z_m[201]`;
- `x_m[201]`;
- velocity and source metadata;
- config, checkpoint, and dataset digests.

Output is written to a temporary file and atomically renamed only after all chunks and metadata validate. Inference defaults to `torch.no_grad()`. FWI uses the arbitrary-query interface and preserves autograd.

## 10. Failure Handling

The implementation must stop with an actionable error for:

- mixed or unequal velocity fields inside a group;
- more than one source in a record;
- invalid source mass, coordinates, frequency, delay, or amplitude;
- nonfinite velocity, target, prediction, loss, or gradient;
- split leakage or missing VDS shards;
- config or dataset digest mismatch during resume;
- dense/query axis disagreement;
- incomplete full-wavefield export.

CUDA OOM handling may reduce only query chunk size after a dry-run autotune. It must never silently reduce record count, frame count, receiver count, query reuse, dense supervision, or model width.

## 11. Verification Matrix

### 11.1 Unit tests

- U-FNO scale shapes, spectral blocks, upsampling, and skip connections.
- MIONet branch/trunk rank and multiplicative fusion.
- source isolation and source-amplitude scaling.
- grouped and ungrouped forward/loss equivalence.
- one medium-encoder call per unique medium.
- arbitrary-query chunk equivalence.
- dense/query grid consistency.
- finite, nonzero velocity gradients.
- rank-64 dense reconstruction shape and axis order.
- atomic HDF5 output and interrupted-export recovery.

### 11.2 Integration tests

- tiny CPU two-step training with checkpoint resume;
- real-VDS multiprocessing and four-block query reuse smoke test;
- CUDA forward/backward smoke test;
- 24 GB memory-budget dry run;
- deterministic fixed-seed sampler and resume test;
- `validation`, `test_id`, and `ood_canonical` split isolation.

### 11.3 Accuracy gates

On `test_id`:

- full-wavefield relative L2 at most 0.30;
- receiver-trace correlation at least 0.90;
- spectral error at most 0.35;
- no medium family may silently collapse to near-zero prediction.

The three `ood_canonical` records are reported individually and are not used for tuning or aggregate pass/fail claims.

### 11.4 Performance gates

On the same RTX 3090 and identical output grid:

- effective supervision queries per second at least 2× the current baseline;
- steady-state sampled GPU idle fraction at most 35%;
- peak allocated GPU memory below 24 GB;
- fast-head full-wavefield compute faster than same-grid GPU LWC-84;
- end-to-end full-wavefield output at least 10× faster than same-grid CPU LWC-84.

Benchmarks require warmup, CUDA synchronization, at least ten timed repetitions, and median plus percentile reporting. Compute-only, host transfer, and HDF5 write times are reported separately. If the model does not beat GPU LWC-84, the result is reported as a failed speed gate and no acceleration claim is made.

## 12. Reproducibility and Rollout

- New checkpoints and logs use a separate `artifacts/grouped_ufno_mionet/` root.
- Config, dataset binding, git commit, RNG, sampler state, optimizer, scheduler, stage, and query-reuse state are checkpointed.
- The current baseline continues untouched until the new CUDA smoke and equivalence gates pass.
- The new model trains from scratch; no partial checkpoint migration is assumed.
- No GitHub push is part of this design or its implementation unless separately requested.

## 13. Definition of Done

The implementation is complete only when all unit and integration tests pass, the full waveform can be exported with exact `[401,201,201]` shape, gradients to velocity are verified, source isolation is proven, and accuracy/performance reports state every gate with measured evidence. A high GPU power number without improved useful throughput does not count as success.
