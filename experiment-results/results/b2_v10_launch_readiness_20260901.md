# B2-v10 causal-prefix assimilation launch readiness

## Current state

- V9 parent retraining remains active on all four GPUs.
- B2-v10 GPU work is intentionally blocked until the V9 supervisor writes a
  successful terminal record and the frozen two-seed selector passes.
- Validation and `test_id` have not been opened for B2-v10 development.
- The implementation is additive and does not modify protected checkpoints.

## Verified implementation

- Core causal-prefix solver:
  `saved_time_phase_operator_v4/instance_adaptation/b2_v10_prefix_assimilation.py`
- Offline rank-4 family POD fitting:
  `scripts/pretrain_b2_v10_residual_pod.py`
- Prefix-only online evaluator:
  `scripts/evaluate_b2_v10_prefix_assimilation.py`
- Four-GPU calibration runner:
  `scripts/run_b2_v10_four_arm.sh`
- Frozen V9 parent selector:
  `scripts/select_b2_v9_parent.py`
- Frozen B2-v10 arm selector:
  `scripts/select_b2_v10_arm.py`
- Stage preregistration binder:
  `scripts/prepare_b2_v10_prereg.py`
- Twenty-one related CPU/static tests pass.

## Frozen data state

- Calibration: `results/b2_v10_online_calibration_manifest_24rec_20260901.json`
- Confirmation: `results/b2_v10_online_confirmation_manifest_24rec_20260901.json`
- The panels contain 8 records per family, have zero group overlap with each
  other, and have zero group overlap with V6, V7, V8, V9 fit/holdout records.
- Historical limitation: pre-v6 V4 work covered every Marmousi train group, so
  no never-before-opened Marmousi train group exists.  This is disclosed in the
  design record rather than hidden by relabeling the panel as globally fresh.

## Post-V9 execution order

1. Apply `scripts/select_b2_v9_parent.py`.  Stop if neither variant passes both
   seeds.
2. Build calibration and confirmation causal caches from the frozen cache
   preregistration.  Cache construction may use two free GPUs in parallel.
3. Use `scripts/prepare_b2_v10_prereg.py --stage offline` to bind the selected
   checkpoint, V9 fit cache, fit manifest, code, and design digests.
4. Fit the family rank-4 residual POD bundle on one GPU.  Stop if any family has
   nonpositive mean oracle capacity.
5. Use `scripts/prepare_b2_v10_prereg.py --stage calibration` to bind the POD
   bundle and calibration cache.
6. Launch `scripts/run_b2_v10_four_arm.sh` detached.  The four GPUs evaluate
   `peak`, `cycle025`, `cycle050`, and `fixed24` independently.
7. Apply `scripts/select_b2_v10_arm.py`.  An arm must first strictly improve its
   own unobserved suffix; selection then uses the common frames 49:64, with
   smaller information budget as tie-breaker.
8. Freeze a confirmation preregistration and evaluate only the selected arm on
   the second panel.  Stop if aggregate relative L2 is not strictly improved.
9. Only after confirmation passes may a complete frozen validation be opened.
   No algorithm or hyperparameter changes are permitted after that point.

## Required long-job checks

For every detached stage verify supervisor PID, child PID, live log updates,
run identity, checkpoint or candidate creation, terminal semantics,
`nvidia-smi`, and disk headroom.  A process alone is not evidence of a live or
successful run.
