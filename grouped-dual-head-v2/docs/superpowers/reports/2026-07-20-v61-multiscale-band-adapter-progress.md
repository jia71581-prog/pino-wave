# V61 multiscale band-adapter progress

Date: 2026-07-20

## Admission decision

V60 is not admitted to long training.  Its final rank-16 learning-rate sweep
selected `3e-3`, with three-record all-401-time aggregate relative L2
`0.3632131252` against the exact V49 anchor `0.4080715670`.  The cumulative
reduction is `10.9928%`, below the registered `20%` overfit gate.  Its high-band
anchor delta `5.6422e-6` passes the `2e-5` safety gate.  Rank 64 was slightly
worse than rank 16, late gradients remained finite, and clipping never
activated, so V60's shallow local expert is treated as a representation
bottleneck rather than a batch, optimizer, or gradient-stability failure.

V61 is registered as the next bounded experiment.  It retains the frozen V49
anchor, sparse one-hot family dispatch, free-surface-aware low/mid projection,
and exact-time/full-field API.  Each family expert adds a full-resolution
factorized spectral/local stack, a pooled spectral/local stack, multiscale
fusion, and exact-time FiLM.

The initial V61 optimizer smoke isolated a zero-head step-size bottleneck.  A
single `3e-3` AdamW update moved the fixed-panel score from `0.4024023289` to
`1.3697543506`, despite a finite `2.4387` gradient norm and no clipping.  The
completed 20-update global-rate controls gave these all-401-time scores:

- `1e-5`: `0.4077054222` (best, only `0.0897%` below the V49 anchor);
- `3e-5`: `0.4077523446`;
- `1e-4`: `0.4082296591` (regression);
- `2e-4`: selected the unchanged `0.4080715670` anchor.

V61 therefore now uses disjoint AdamW groups: each zero-initialized expert
output head uses `1e-5`, while spectral/local feature parameters use a separately
swept rate.  This changes optimizer state only and leaves the model function,
state-dict names, high-band projection, and parent identity unchanged.

## Architecture and identity

- parent checkpoint SHA256:
  `3b8569790ae9f03883d3a9a1aed0c71f6138a611d9f33392b90ccd5914fa19bc`;
- parent run digest:
  `f71dbac27c09bb94244c5866c8c02b370724a2a56904f2a66d0913c17a36cc63`;
- parent identity SHA256:
  `8d787f15fc6eb077309de46e297d2e4e71726e1a9561453b6a1e9f97f0b9b21b`;
- manifest digest:
  `a20c9a65abbc65294062af443e2ceae241ead66450f940f652e2d95aaa0aa92b`;
- time-axis SHA256:
  `c9561c394a77954aecefcedb0dc6ec3f11d8280d18bdd6bd89999a8304fd7af9`;
- adapter architecture: `multiscale_spectral`;
- latent width / spectral rank / modes: `32 / 32 / 32`;
- full/coarse depth: `4 / 2`;
- adapter parameter count: `2,442,741`, exactly `814,247` per family expert;
- activation checkpointing: enabled;
- output heads: exact zero at transfer.

The hard output path remains:

```text
prediction = free_surface(V49 anchor)
           + project_low_mid_increment(free_surface(V61 raw increment))
```

No receiver input, time interpolation, solver change, boundary change, FWI, or
GitHub publication is introduced.

## Data and batch contract

The identity-bound dataset contains 4,003 records and exactly one source per
record.  The numerical grid remains 401x401, saved wavefields remain 201x201,
and there are 401 exact stored times.  The top is the node-centred
pressure-release surface with eighth-order odd ghost extension; the left,
right, and bottom boundaries remain unsplit CFS-CPML/ADE.

The generated diagnostic/pilot configuration registers physical and macro
batch 24 per rank, four macros per update, effective global batch 96, and 24
training frames per record.  This is the requested throughput target, not yet a
V61 memory claim.  V60 measured `22,415,729,664` peak allocated bytes at this
geometry; V61 must pass a new four-GPU smoke below 23 GiB before promotion.

## Verification evidence

- V61 design commit: `e0714fe`;
- implementation plan commit: `fe59a59`;
- schema commit: `eb352f1`;
- multiscale expert commit: `2df9a6c`;
- operator plumbing commit: `69decd6`;
- transfer/config commit: `6e6f276`;
- optimizer-split commit: `346f4d1`;
- local V4 plus grouped-V3 regression: 720 collected tests, all passed;
- remote architecture-focused regression: 128 collected tests, all passed;
- remote optimizer-split regression: 76 collected tests, all passed;
- local/remote SHA comparison for the initial 11 and optimizer-split 9
  transferred source/test/documentation files: exact;
- generated config:
  `configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_overfit_r1.yaml`;
- generated config SHA256:
  `8140c6ec96933392be582175537186336c295160aa1c7f3080859e8547409657`;
- registered feature/output learning rates: `3e-4 / 1e-5`;
- parent-selection evidence:
  `docs/superpowers/reports/2026-07-20-v61-parent-selection.json`.

Unit and operator tests prove legacy V60 compatibility, V61 exact zero-init
identity, post-wakeup internal gradients, exact one-hot sparse dispatch, soft
route equivalence, strict transfer prefixing, full-field shape, zero registered
high-band increment, and a zero free-surface top row.

## Next gate

The next command is a four-way, same-anchor 20-update three-record sweep using
one uniform, one corrected vertical layered, and one Marmousi training record;
staggered 64-frame exact-time appearances; output-head rate `1e-5`; and feature
rates `1e-4`, `3e-4`, `1e-3`, and `3e-3`.  The best finite candidate then
continues to 200 updates with evaluation every 20 updates and terminal
evaluation over all 401 stored times.  Promotion requires all of:

1. cumulative aggregate reduction from V49 at least `20%`;
2. every family improved over its V49 anchor;
3. high-band anchor delta no more than `2e-5`;
4. selected-checkpoint identity bound to the terminal all-time metrics;
5. subsequent 48-record same-panel and V61 four-GPU resource gates passing.

No long curriculum run starts before these gates pass.
