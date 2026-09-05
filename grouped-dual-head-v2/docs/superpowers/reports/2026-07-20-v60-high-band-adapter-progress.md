# V60 high-band-safe adapter progress

Date: 2026-07-20

## Registered objective

This stage tests whether a family-routed, band-limited residual adapter can
improve the exact V49 acoustic operator without changing its high spatial
band.  The fixed data contract remains one source per record, no receiver
input, 401 solver grid points per axis, 201 saved grid points per axis, all 401
stored times, a pressure-free top boundary, and CPML on the other three sides.

The registered parent is V49 epoch 1:

- checkpoint SHA256:
  `3b8569790ae9f03883d3a9a1aed0c71f6138a611d9f33392b90ccd5914fa19bc`;
- run digest:
  `f71dbac27c09bb94244c5866c8c02b370724a2a56904f2a66d0913c17a36cc63`;
- identity SHA256:
  `8d787f15fc6eb077309de46e297d2e4e71726e1a9561453b6a1e9f97f0b9b21b`.

## Diagnostic findings

The fixed-32-time experiments were rejected as a selection protocol.  Rank-16
with learning rate `3e-3` reduced that fixed panel from `0.402402329` to
`0.330139`, but worsened the 401-time metric to about `1.023`.  Exact-time
appearance panels must therefore rotate, and the terminal decision must use
all 401 saved times.

Sparse dispatch of exact one-hot family routes was implemented in commit
`7a63b74`.  It executes only the selected expert during teacher-forced
training.  Rank-64 then used about 14.3 GiB for a three-record, 64-time update,
instead of OOM at about 23.4 GiB with dense expert dispatch.  Rank-16 used about
6.3 GiB for the same update.  This establishes routing work, not model size, as
the source of the earlier memory waste.

Rank-64 is not the preferred capacity direction.  Its completed 100-update,
64-time run produced a 401-time aggregate relative L2 of `0.386900210`, a
`5.1881%` improvement over the V49 anchor.  The rank-16 100-update reference
produced `0.386176154`, a `5.3656%` improvement.  The larger adapter therefore
did not improve convergence enough to justify its additional parameters.

The current completed best is the rank-16, 64-time, `3e-3` continuation:

- artifact:
  `artifacts/saved_time_v60_high_band_adapter_r1/overfit_appearance64_lr3e3_staggered_sparse_cont1_r1`;
- selected checkpoint: `checkpoints/update_0100.pt`;
- fixed-panel aggregate relative L2: `0.367359009`;
- 401-time aggregate relative L2: `0.374981119`;
- cumulative improvement over V49: `8.10898%`;
- 401-time family relative L2: layered `0.213010688`, Marmousi
  `0.436843423`, uniform `0.475089247`.

This is a development overfit gate, not held-out acceptance evidence.  It is
still below the registered 20% overfit-reduction threshold.

## Provenance correction

Commit `7a4a207` fixed continuation scoring.  Every new diagnostic now saves
the warm-start state as `checkpoints/update_0000.pt`, points `best.pt` at it,
restores the selected best checkpoint before the 401-time evaluation, and
records both stage-local `relative_reduction` and cumulative
`anchor_relative_reduction`.  The evidence gate prioritizes the cumulative
anchor value.  The focused test files passed 42 tests, and the broader
band-adapter/full-support/data selection passed 119 tests.

## Four-GPU batch geometry

Commit `6f74da2` made macro records and effective batch explicit.  The earlier
configuration `macro_records=12`, `macros_per_update=8`, and
`microbatch_records=12` has effective batch 96 but processes two physical
microbatches per rank.  The registered throughput probe instead uses
`24 x 4 = 96`, with `microbatch_records=24`.  This preserves optimizer batch
and update density while allowing one full 24-record pass per rank.

The 16- and 24-frame capacity configurations are committed in `b44f9c3`.
Both passed the deterministic schedule audit: 24 optimizer updates per epoch,
2,304 record appearances, and 768 appearances for each medium family.

## Live remote work

Remote snapshot:
`/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2`.

At registration time, these detached diagnostics were active:

- PID `862002`: 64 frames, `1e-3`, first continuation;
- PID `866039`: 16 frames, `1e-3`, second continuation;
- PID `870067`: 64 frames, `1e-3`, continuation from the former update 80;
- PID `873514`: 64 frames, `3e-3`, continuation from the current update 100.

Detached PID `876345` waits for those exact processes to exit and for all CUDA
compute processes to disappear.  It then runs one four-GPU smoke update for
the 16-frame `24 x 4` configuration, followed by the 24-frame configuration
only if the first succeeds.  Queue evidence is stored under
`artifacts/saved_time_v60_batch24_capacity_queue_r1` on the remote host.

## Promotion conditions

No V60 long run is promoted until all of the following are true:

1. the three-record all-401-time cumulative reduction is at least 20%;
2. the high-band anchor delta remains below `2e-5`;
3. the selected four-GPU physical batch stays below 23 GiB allocated memory;
4. aggregate and every-family pilot metrics improve without changing the data
   or boundary contract;
5. final claims use the independent held-out panel, not the three training
   records or the numerical baseline.
