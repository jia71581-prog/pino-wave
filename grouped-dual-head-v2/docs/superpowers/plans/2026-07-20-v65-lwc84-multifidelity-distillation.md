# V65 LWC-84 multi-fidelity distillation implementation plan

> Execute task by task with focused tests.  Preserve the dirty worktree and the live V63 job until
> the replacement has passed local tests and a numerical-cache smoke check.

## Task 1: Register cache and loss contracts

**Files:**

- Create `saved_time_phase_operator_v4/multifidelity.py`
- Create `tests/saved_time_phase_operator_v4/test_multifidelity.py`

1. Write failing tests for deterministic 64-time selection, cache sample/time lookup, identity
   rejection, exact time-block decomposition, residual energy flooring, and detached targets.
2. Implement immutable cache metadata, a lazy read-only HDF5 lookup, and the decomposable
   multi-fidelity loss/reference objects.
3. Run the focused test file with CUDA disabled.

## Task 2: Add fixed numerical-teacher time sampling

**Files:**

- Modify `saved_time_phase_operator_v4/data.py`
- Modify `scripts/train_saved_time_v4_full_support.py`
- Modify `tests/saved_time_phase_operator_v4/test_data.py`
- Create `tests/saved_time_phase_operator_v4/test_v65_multifidelity.py`

1. Write failing tests for a `numerical_teacher_pool` policy that selects only registered cache
   indices and rotates deterministically across appearances.
2. Pass the pool into dataset construction and make coverage accounting use the identical sampler.
3. Instantiate one lazy numerical cache per DDP rank and reject cache/dataset identity mismatches.
4. Add the high/low/residual loss branch and per-update telemetry without changing legacy paths.
5. Verify focused data, loss, V63-regression, and V65-contract tests.

## Task 3: Build resumable four-shard numerical cache

**Files:**

- Create `scripts/build_lwc84_multifidelity_cache.py`
- Create `scripts/merge_lwc84_multifidelity_cache.py`
- Create `tests/saved_time_phase_operator_v4/test_multifidelity_cache_cli.py`

1. Test shard membership, numerical metadata, completion-mask resume, and VDS index mapping on
   tiny synthetic files.
2. Reconstruct exact 401-grid velocities from the frozen manifest, prove their restricted 201-grid
   values match HDF5 bitwise, and implement the original 401-to-201 LWC-84 generation path.
3. Retain resumable replay shards as audit evidence, then publish an identity-bound zero-copy VDS
   over the already verified source solver fields instead of duplicating 23.2 GB.
4. Compile the CLIs and run the focused tests.

## Task 4: Configure and gate V65

**Files:**

- Create `configs/saved_time_v4/generated/v65_lwc84_multifidelity_pilot_4gpu.yaml`
- Create `configs/saved_time_v4/generated/v66_lwc84_multifidelity_long_4gpu.yaml`
- Create `scripts/gate_saved_time_v65_multifidelity.py`
- Create `scripts/run_remote_v65_multifidelity.sh`
- Extend `tests/saved_time_phase_operator_v4/test_v65_multifidelity.py`

1. Register the 2,240-record cache, four-epoch curriculum, loss weights, full-backbone unfreezing,
   exact-time policies, per-epoch checkpoints, GPU cap, and high-fidelity gates.
2. Test that V66 starts only from V65 best, disables numerical loss, and restores rotating all-time
   appearances.
3. Implement a prospective gate against the same parent panel with family-regression protection.
4. Add an idempotent detached pipeline that generates/merges cache, probes capacity, runs V65,
   gates, and conditionally starts V66.

## Task 5: Verify and launch

1. Run the complete `tests/saved_time_phase_operator_v4` suite, Python compilation, and shell
   syntax checks.
2. Archive source/config/tests under
   `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/algorithm/`.
3. Sync only the changed source/config/test files to the remote worktree.
4. Stop V63 only after the replacement preflight passes; record its last checkpoint and metrics.
5. Start the V65 detached pipeline with `nohup setsid`, verify PID/PGID/logs and four GPUs, then
   monitor the cache completion and first training updates.
6. Report measured cache ETA, selected physical microbatch, GPU memory/utilization/power, first
   validation error, and promotion decision.  Never claim the under-10% neural target before the
   independent validation gate passes.
