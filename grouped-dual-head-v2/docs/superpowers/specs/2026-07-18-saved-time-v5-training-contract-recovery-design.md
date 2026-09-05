# Saved-Time V5 Training-Contract Recovery Design

Date: 2026-07-18

## Objective

Recover a trustworthy full-support training baseline before changing the wavefield architecture. The baseline must train on every non-anomaly training record, cover a broad deterministic subset of the 401 stored time indices for every record, allow the transferred representation to adapt, and report validation metrics that are representative of all held-out media and source instances.

This stage does not implement FWI, time interpolation, receiver conditioning, a new temporal decoder, or a new PDE loss. It does not interrupt the active V4 ASAM run.

## Evidence and root cause

The V4 parent runner builds a pool containing only `train_macro_steps` macros and repeats that pool for all epochs. With the production configuration this exposes 48 of 2,240 training records. The full-support ASAM runner corrects record coverage for one short pass, but performs only 94 updates, samples four frames per record appearance, and freezes every parameter outside the dense decoder. Its best fixed-validation relative L2 changes from 0.5583051 to 0.5577154, which is not meaningful progress toward the 0.10 target.

The failure is therefore attributed first to the training contract, not to insufficient GPU occupancy or an established model-capacity limit.

## Considered approaches

### A. Continue dense-only ASAM for more updates

This preserves the current implementation and high GPU utilization. It cannot correct errors in the medium, source, travel-time, or fusion representations, and the observed validation curve is already flat. Rejected.

### B. Replace the architecture immediately

A direct multiscale spatiotemporal decoder is likely the correct long-term architecture, but changing data exposure, optimizer behavior, validation, and architecture simultaneously would prevent causal attribution. Deferred until the recovered baseline is measured.

### C. Recover the full-support AdamW baseline first

Build an epoch-aware schedule, deterministic time-coverage sampler, staged unfreezing, and streamed validation around the existing V4 model. This is the selected approach because it tests whether the existing representation can learn when given a valid training signal and creates a fair baseline for the later V5 architecture ablation.

## Training schedule contract

An epoch is defined by coverage, not by a small fixed macro count.

- Every one of the 2,240 training records must appear at least once in each epoch.
- Any padding needed to complete a 12-record macro may repeat records, but the maximum and minimum per-epoch appearance counts may differ by at most one.
- The schedule is deterministic from the run seed and epoch index and is stored in the run identity by digest rather than as an unbounded JSON list.
- Resume reconstructs exactly the same schedule and continues at an epoch boundary. A checkpoint is written at every epoch.
- The default production run uses 50 epochs, effective batch 48, and four 12-record macros per optimizer update. Physical microbatch is stage-aware: 12 while training dense/source/fusion, 8 after coordinate/travel unfreeze, and 4 after medium-backbone unfreeze. CUDA gates showed that a fixed larger microbatch exceeds 24 GiB as additional point-query branches enter the backward graph; gradient accumulation preserves effective batch 48.
- With 2,240 records, one epoch contains 47 optimizer updates after deterministic padding; 50 epochs contain 2,350 updates.

The runner must fail before allocating the model if record coverage, balance, split membership, or schedule determinism is violated.

## Exact stored-time coverage contract

Targets are restricted to the 401 stored HDF5 time indices. No interpolation is introduced.

- Each record appearance samples four exact time indices.
- One slot is onset-aware, one samples the first active third, one the middle active third, and one the late active third.
- A deterministic per-record permutation/cursor prevents repeatedly selecting the same indices across epochs.
- Pre-onset frames are capped at five percent of all sampled frames and are retained only to audit causality.
- At epoch 30, every training record must have seen at least 120 unique stored time indices. At epoch 50, coverage must not decrease and the median should exceed 180 indices.
- A coverage ledger is checkpointed compactly and verified on resume.

Coverage tests use synthetic manifests and do not read the production HDF5 file.

## Optimization and staged unfreezing

The baseline uses AdamW with linear warmup followed by cosine decay.

- Epochs 1–2: train the dense decoder, source encoder, and fusion modules.
- Epochs 3–5: additionally unfreeze the coordinate and travel-time modules.
- Epoch 6 onward: unfreeze the complete model, using a lower learning rate for the transferred medium backbone.
- Default peak learning rates are `2e-4` for newly initialized/dense parameters and `2e-5` for the transferred backbone, with `1e-6` weight decay and gradient clipping at 1.0.
- Optimizer parameter groups must be mutually exclusive, exhaustive for the intended stage, and logged with trainable parameter counts.

ASAM and L-BFGS are excluded from this baseline. ASAM may be tested only after the recovered AdamW baseline shows stable held-out descent.

## Validation and selection

Validation is separated into development feedback and final sealed measurement.

- Every epoch evaluates a deterministic rotating panel of 48 held-out records and 16 exact stored times per record.
- Every fifth epoch evaluates all 480 held-out records at the same 16 stratified exact times.
- Model selection uses the all-record evaluation when available and otherwise uses the rotating panel only for observability.
- At completion, the selected checkpoint is streamed over all 480 held-out records and all 401 stored times in bounded time blocks. Predictions do not need to be retained in memory.
- Metrics include aggregate and per-family relative L2, per-medium and per-source summaries, active-time energy bins, spatial spectrum bands, phase correlation, RMSE, and exact counts of records and time indices.
- The final evaluator rejects interpolated times, missing records, duplicate final coverage, or a checkpoint/manifest/config identity mismatch.

The validation split remains sealed from optimizer updates and curriculum decisions.

## Observability and failure handling

Each epoch appends one JSONL record containing loss components, learning rates, parameter-group counts, record/time coverage, validation scope, GPU peak memory, elapsed time, and checkpoint path. Writes and stable checkpoint links are atomic.

The run stops with an explicit terminal state on non-finite loss, incomplete coverage, resume identity mismatch, unexpected trainable parameters, or CUDA allocation above the registered limit. A process exit without a terminal record is treated as interrupted and is resumable from the last complete epoch.

GPU snapshots are taken during a training step and after validation so idle validation snapshots are not reported as training utilization.

## Verification gates

Implementation proceeds through these gates:

1. Unit tests prove balanced all-record epoch coverage, deterministic schedules, exact-time-only sampling, the 120-index minimum, staged parameter groups, and resume equivalence.
2. A CPU/synthetic runner test proves checkpoint and coverage-ledger behavior.
3. A one-update CUDA smoke test stays below 23 GiB and produces finite gradients for every trainable group.
4. A one-epoch production-manifest audit proves all 2,240 training records are covered before long training starts.
5. A three-epoch pilot must show non-increasing best held-out relative L2 and no family regression above three percent. Failure returns to diagnosis instead of automatically launching 50 epochs.
6. The 50-epoch run targets aggregate relative L2 below 0.10, with every family below 0.12. These are acceptance targets, not guaranteed outcomes.

## Follow-on architecture decision

After this baseline, architecture work uses equal data exposure, optimizer steps, seed, and validation support. The next ablation compares the recovered V4 baseline with a direct dense decoder, then adds consecutive-time blocks and finally a U-shaped multiscale medium/source decoder. MIONet remains an optional arbitrary-point auxiliary head rather than the primary full-grid path.
