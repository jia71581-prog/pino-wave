# Parameter-Matched Patch-DeepONet Baseline Design

**Date:** 2026-07-24  
**Status:** Approved in chat; written specification awaiting user review  
**Scope:** Train an independent DeepONet baseline on the exact active acoustic-wave dataset and compare it with the epoch-40 parent model and the strict two-post-onset-frame CLFC method.

## 1. Goal

Build a strong, parameter-matched Patch-DeepONet baseline that:

1. uses the same records, split membership, source/medium information, training-frame policy, optimizer-update budget, and random seed as the current gate-4 parent run;
2. is trained from random initialization without teacher predictions, parent weights, validation truth, or receiver truth;
3. avoids the previously observed near-zero-prediction failure through local spatial conditioning, arrival-aware query sampling, and an explicit early descent gate;
4. produces directly comparable field, receiver, phase, spectrum, runtime, and memory results on the same uniform, layered, and Marmousi validation instances.

This baseline is intended to answer two separate questions:

- **Architecture comparison:** raw epoch-40 parent versus raw DeepONet.
- **Deployment comparison:** strict two-frame CLFC versus raw DeepONet.

These comparisons must be reported separately so that CLFC adaptation gains are not incorrectly attributed to the parent architecture alone.

## 2. Fixed Provenance

| Item | Fixed value |
|---|---|
| Editable workspace | `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project` |
| Historical DeepONet reference, read-only | `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation` |
| Dataset | `/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5` |
| Registered travel-time data | `/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5` |
| Dataset geometry | `201 x 201`, 401 saved time frames |
| Split sizes | 2240 train, 480 validation, 480 test-ID |
| Parent checkpoint | `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/best.pt` |
| Parent checkpoint SHA-256 | `0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f` |
| Parent trainable parameters | 32,294,258 |
| Main-run seed | 372 |
| Main-run budget | 40 epochs, 2800 optimizer updates |
| Effective records per update | 32 |
| Training time policy | `appearance16`, 16 frames per record |
| Strict comparison instances | `validation_uniform_00003`, `validation_layered_00082`, `validation_marmousi_00087` |
| CLFC observation contract | exactly the first two stored frames at or after source onset |

Before any training launch, a machine-readable provenance audit must resolve these paths, record hashes and split indices, and fail closed if they differ.

## 3. Considered Approaches

### 3.1 Selected: parameter- and update-matched Patch-DeepONet

Use a local convolutional branch, a Fourier time trunk, and query-local feature fusion. Match the parent parameter count within 0.1% and use the same 40-epoch/2800-update budget.

This is selected because it provides the strongest independent DeepONet comparison without using parent predictions.

### 3.2 Rejected: smaller compute-matched Patch-DeepONet

A roughly 16.7-million-parameter historical configuration would train faster, but it would have about half the parent capacity. A poor result would therefore be ambiguous and unsuitable as the primary comparison.

### 3.3 Rejected: teacher-distilled DeepONet

Distillation could improve the DeepONet result, but the resulting model would no longer be independent of the proposed method. It may be studied later, but it is outside this baseline.

## 4. Model Architecture

### 4.1 Static local branch

For each record, construct four target-free dense channels on the `201 x 201` grid:

1. normalized velocity;
2. normalized source map;
3. normalized registered eikonal travel time;
4. normalized slowness contrast derived only from velocity.

A local CNN maps these channels to a latent spatial field. The latent vector for a query is obtained by bilinear sampling at its spatial coordinate. This retains local medium structure and avoids the global-pooling failure mode of earlier DeepONet experiments.

### 4.2 Time/source trunk

The trunk receives target-free query descriptors:

- normalized `x`, `z`, and saved-time coordinate;
- the five normalized source parameters;
- local travel time;
- dimensionless arrival phase `(t - t0 - travel_time) * f0`;
- Fourier features of time and arrival phase.

The source descriptors come from the same record fields available to the parent model. No measured or simulated pressure value is a trunk input.

### 4.3 Query fusion and output

At each query point:

1. sample the local branch latent;
2. compute the time/source trunk latent;
3. compute a small query-local feature residual from the static channels and descriptors;
4. fuse them using a scaled elementwise DeepONet interaction;
5. decode one normalized pressure value.

Dense fields are produced by chunked evaluation of query coordinates; query chunk size changes memory use only and must not change outputs beyond floating-point tolerance.

The decoder uses a target-free Ricker first-arrival residual:

- its nonzero learned residual weights are initialized at 0.1 times the
  framework default;
- its scalar bias parameter multiplies the analytic Ricker value evaluated at
  `(t - t0 - travel_time) * f0`, rather than acting as a spatially constant
  offset;
- that scalar starts at `0.1` normalized-pressure units and remains trainable.

This correction was added after the three-family overfit gate showed that the
original constant decoder bias converged to the zero predictor despite falling
loss. It uses only source parameters and registered travel time, adds no
parameters, does not inspect pressure targets at inference, and preserves the
hard causal mask.

### 4.4 Capacity target

The initial capacity candidate uses:

- branch width: 48;
- latent dimension: 1984;
- trunk hidden dimension: 2752;
- trunk depth: 5;
- Fourier temporal trunk enabled;
- no teacher or parent backbone.

The historical four-channel core counts 32,294,209 parameters, 49 fewer than the 32,294,258-parameter parent. The final source-descriptor projection may alter this count slightly. A deterministic parameter search may adjust only the final trunk width or a projection width so that the final trainable count remains within 0.1% of the parent. The resolved count and configuration must be written to the run identity before training.

## 5. Data and Leakage Contract

The baseline must use the existing dataset split arrays directly. It must not recreate, reshuffle, or infer a new split.

Allowed inputs:

- velocity and deterministic transforms of velocity;
- source map and the five source parameters;
- stored coordinates and saved-time axis;
- registered travel-time data;
- training targets only at sampled training queries.

Forbidden inputs:

- validation or test wavefield values during training or hyperparameter selection;
- receiver traces as model inputs or training losses;
- parent or CLFC predictions;
- frames outside the selected 16 training frames when forming a training update;
- either of the two deployment observation frames as a special DeepONet input.

Target-energy query sampling is permitted only inside the training split. It selects which known training labels are supervised; it is never used on validation/test records and is not exposed to the model.

A leakage test must verify:

- train, validation, and test-ID index sets are pairwise disjoint;
- query samplers cannot open wavefield data for non-training records;
- evaluation tensors are created only after checkpoint selection;
- source/travel features are independent of pressure targets;
- all strict comparison instance IDs belong to validation.

## 6. Efficient Training Protocol

### 6.1 Record and frame schedule

The main run uses:

- 40 epochs;
- all 2240 training records once per epoch;
- effective macro batch of 32 records;
- 70 optimizer updates per epoch;
- 2800 optimizer updates total;
- seed 372;
- the same deterministic `appearance16` time-frame policy as the parent.

Physical microbatch is one record on the local RTX 3090. Gradients are accumulated without changing the effective record batch.

### 6.2 Query budget

For each selected record, supervise 65,536 space-time queries:

- 4096 spatial queries for each of the 16 selected frames;
- process them as four 16,384-query chunks;
- accumulate chunk gradients before advancing to the next record.

The query set is drawn from a fixed mixture:

- 50% spatially uniform;
- 25% training-target energy weighted;
- 25% arrival-band weighted using eikonal travel time and source frequency.

The sampler records the mixture probability for every selected query. The data loss uses self-normalized inverse-probability weights capped at 20 times the batch median to control variance. Spatial-gradient supervision is formed from explicit adjacent-point query pairs, not from unrelated sampled points.

Use the largest passing `query_chunk_size` from 65,536, 32,768, 16,384,
8,192, 4,096, 2,048, and 1,024.  If memory is insufficient, only
`query_chunk_size` may be reduced. The 65,536-query record budget, selected
query IDs, effective batch, loss, and update count must remain unchanged.

### 6.3 Loss

Use the parent run's main supervised semantics:

- robust normalized pressure error with `delta=0.5`;
- energy floor fraction `0.01`;
- hard causality before physically possible arrival;
- spatial-gradient weight `0.1`;
- no teacher loss;
- no receiver loss;
- no spectrum loss in the primary baseline.

Each logged loss must include its unweighted data, spatial-gradient, causality, and total components.

### 6.4 Optimizer selection

Use AdamW with fused CUDA implementation where available, weight decay `1e-6`, gradient norm clipping at `1.0`, two warm-up epochs, and cosine decay to 5% of the initial learning rate.

The learning rate is selected before the main run using fixed 200-update pilots at `1e-4`, `3e-4`, and `1e-3`. All pilots use identical train records, query IDs, and a fixed validation panel. The selected learning rate is the one with the lowest finite fixed-panel validation relative L2; ties are broken by lower gradient-norm variance. The main model is then reinitialized with seed 372 and trained from update zero.

## 7. Long-Run Entry Gate

Before the full 40-epoch launch:

1. run unit tests and the leakage audit;
2. run one forward/backward GPU dry run;
3. overfit one record from each medium family to prove non-zero learnability;
4. complete the learning-rate pilots;
5. require the selected 200-update pilot to:
   - contain no NaN or Inf;
   - have a lower final-window validation relative L2 than its initial window;
   - beat the zero-prediction relative-L2 baseline on the fixed panel;
   - produce non-collapsed prediction RMS.

If the gate fails, the full run must not start blindly. The process must write a terminal diagnostic identifying whether the failure came from normalization, sampling, optimization, capacity, or resource limits.

## 8. Checkpointing and Restart

The main run writes:

- `updates.jsonl` after every optimizer update;
- `metrics.jsonl` at every validation event;
- `checkpoints/latest.pt` at every validation event;
- `checkpoints/best.pt` when the fixed-panel validation metric improves;
- `terminal.json` on completion or controlled failure;
- RNG, sampler, optimizer, scheduler, scaler, epoch, record cursor, query cursor, and accumulated-gradient state required for exact resume.

A resume is accepted only when dataset identity, split hash, model identity, parameter count, frame schedule, query budget, and optimizer schedule match. A mismatch aborts rather than silently starting a scientifically different run.

The detached launcher must use a durable process host and record PID/PGID, command, environment, CUDA device, log path, and start time. Launch success requires a live Python process, non-zero GPU utilization, early `updates.jsonl`, and a writable checkpoint directory.

## 9. Evaluation and Comparison

### 9.1 Checkpoint selection

Select DeepONet `best.pt` using only the fixed validation panel of 48 records and 32 deterministic frames per record. Final evaluation must not influence checkpoint selection.

### 9.2 Evaluation panels

Report two panels:

1. **Raw architecture panel:** epoch-40 parent versus DeepONet.
2. **Deployment panel:** strict two-post-onset-frame CLFC versus DeepONet.

The DeepONet receives no deployment observation frames. Applying a separate DeepONet instance adapter is outside this baseline.

### 9.3 Required instances and outputs

Evaluate exactly:

- `validation_uniform_00003`;
- `validation_layered_00082`;
- `validation_marmousi_00087`.

For each method and instance, save:

- wavefield truth/prediction/error snapshots on identical time indices and color limits;
- receiver waveform overlays at identical receiver positions;
- full-space relative L2;
- active-energy relative L2;
- post-onset framewise relative-L2 median and 95th percentile;
- receiver normalized L2 and maximum cross-correlation lag;
- spectral-amplitude error and phase error;
- inference latency and peak CUDA memory.

Also report parameter count, checkpoint hash, split hash, normalization identity, and whether any adaptation was used.

## 10. Artifact Layout

All new outputs live under:

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/deeponet_same_vds_parammatched_seed372`

Required layout:

```text
audit/
  provenance.json
  split_manifest.json
  leakage_audit.json
pilot/
  lr_1e-4/
  lr_3e-4/
  lr_1e-3/
run/
  resolved_config.yaml
  run_identity.json
  updates.jsonl
  metrics.jsonl
  checkpoints/
  terminal.json
evaluation/
  smoke3/
  comparison_summary.json
  comparison_table.csv
logs/
  launch.log
  train.log
```

Existing parent, CLFC, historical DeepONet, and pristine-project artifacts must not be modified.

## 11. Implementation Boundaries

The editable project does not currently contain the historical `PatchDeepONetResidual` implementation. Implementation may consult the read-only historical file:

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/src/fno_acoustic/transfer/patch_deeponet.py`

Only the minimal coordinate helpers and local-query ideas required by this design may be adapted into the editable workspace. Historical training scripts, old 400-by-400 datasets, teacher-distillation paths, and unrelated transfer-learning code must not be copied wholesale.

The editable workspace is not a Git repository. Therefore this specification and subsequent implementation cannot be committed locally unless repository metadata is restored by the user. File-level provenance and SHA-256 manifests are required instead.

## 12. Tests

Required focused tests:

- flat index to `(z, x, t)` query-coordinate conversion;
- deterministic query selection under seed 372;
- mixture-probability and capped inverse-weight calculation;
- adjacent-pair spatial-gradient targets;
- chunked versus unchunked inference equivalence;
- parameter-count tolerance;
- forbidden-input and split-disjointness audit;
- checkpoint round-trip and exact next-update resume;
- one-record GPU forward/backward memory smoke;
- three-record overfit showing clear improvement over zero prediction;
- evaluator compatibility with the existing three-instance manifests.

## 13. Acceptance Criteria

The implementation is ready for the long run only when:

1. all focused tests and the leakage audit pass;
2. final trainable parameters differ from 32,294,258 by no more than 0.1%;
3. the audit confirms exactly 2240/480/480 records and no split overlap;
4. the resolved main schedule is exactly 40 epochs and 2800 updates;
5. the selected 200-update pilot passes the long-run entry gate;
6. the detached process, GPU activity, update log, and checkpoint path are verified live;
7. resume metadata is complete and tested;
8. final evaluation can produce the required fields, receiver figures, and metric tables for all three fixed validation instances.

Completion of a process without passing the descent and non-collapse checks is a failed baseline, not a successful training result.
